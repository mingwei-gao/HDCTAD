import os
import gc
import time
import argparse
import warnings
import numpy as np
import tensorflow as tf
from functools import wraps
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          路径配置                               ║
# ║                    ★★★ 修改这里即可 ★★★                        ║
# ╚══════════════════════════════════════════════════════════════════╝

PATHS = {
    # ★ 项目根目录（与 randomwalk.py 的 output_base 一致）
    "project_base": "/mnt/sdb/gao/tad_project",

    # Hi-C 接触矩阵 (KR 标准化)
    # 文件名: {chr}_{res}KB.txt
    "hic_base": "/mnt/sde/gmw/GM12878/Hi-C/KR",
}


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         超参数配置                              ║
# ╚══════════════════════════════════════════════════════════════════╝

# ---------- 模型结构 ----------
HGNN_LAYERS    = 1
HGNN_HIDDEN    = 128
TCN_DILATIONS  = [1, 2, 4]
TCN_KERNEL     = 3
TCN_FILTERS    = 64
LATENT_DIM     = 32
DECODER_DIMS   = [128, 256, 128]

# ---------- 训练 ----------
MASK_RATE       = 0.15
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 0.001
PRETRAIN_EPOCHS = 200
JOINT_EPOCHS    = 300
EPS             = 1e-6       # 数值稳定项

# ---------- Loss 权重 ----------
LAMBDA_RECON    = 0.5
LAMBDA_ENTROPY  = 0.01

# ---------- IDEC ----------
UPDATE_INTERVAL = 10

# ---------- K 值搜索 ----------
K_SEARCH_MIN    = 20
K_SEARCH_MAX    = 120
K_SEARCH_STEP   = 5
K_SEARCH_FINE   = 10         # 精搜索半径
K_GLOBAL_MIN    = 10
K_GLOBAL_MAX    = 130

# ---------- TAD 后处理 ----------
MIN_TAD_LENGTH  = 3
MIN_CLUSTERS    = 5
MAX_CLUSTERS    = 250

# ---------- 高质量 TAD 筛选 ----------
MIN_INTERNAL_DENSITY     = 0.5
DENSITY_GAP_RATIO_THRESH = 1.0

# ---------- Hi-C 精修 ----------
REFINE_ENABLED      = True
REFINE_WINDOW       = 5
INSULATION_HALF     = 3
RANKSUM_SIZE        = 5
PVALUE_THRESHOLD    = 0.05
MERGE_COS_THRESHOLD = 0.8
MIN_DIFF            = 10

# ---------- 其他 ----------
PARAM_TAG   = "p2_q0.5"
RESOLUTION  = 25         # KB
RANDOM_SEED = 42


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         路径构建工具                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_chr_dir(chr_name, res=RESOLUTION):
    """单条染色体的数据目录（与 randomwalk.py 输出一致）"""
    chr_num = chr_name.replace("chr", "")
    return os.path.join(PATHS["project_base"], f"res{res}", f"chr{chr_num}")


def get_hic_path(chr_name, res=RESOLUTION):
    """Hi-C 矩阵路径"""
    return os.path.join(PATHS["hic_base"], f"{res}KB", f"{chr_name}_{res}KB.txt")


def get_hypergraph_path(chr_name, res=RESOLUTION):
    """超图 NPZ 路径（randomwalk.py 自动生成）"""
    return os.path.join(get_chr_dir(chr_name, res), "hypergraph.npz")


def get_embed_dir(chr_name, res=RESOLUTION):
    """节点嵌入 & 相似度矩阵所在目录"""
    return get_chr_dir(chr_name, res)


def get_output_dir(chr_name, res=RESOLUTION):
    """输出目录"""
    return get_chr_dir(chr_name, res)


def get_result_path(chr_name, res=RESOLUTION):
    """原始 TAD 结果路径"""
    return os.path.join(get_output_dir(chr_name, res), "result", "original_TAD.txt")


def get_refine_output_dir(chr_name, res=RESOLUTION):
    """精修输出目录"""
    return os.path.join(get_output_dir(chr_name, res), "result")


def get_refine_output_path(chr_name, res=RESOLUTION):
    """精修后 TAD 路径"""
    return os.path.join(get_refine_output_dir(chr_name, res), "TADs.txt")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        GPU & 随机种子                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU x{len(gpus)}")
    else:
        print("⚠️ 未检测到 GPU，使用 CPU")
    return "GPU" if gpus else "CPU"


DEVICE = setup_gpu()

import random as _random
_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         通用工具函数                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def catch_exception(default=None, prefix=""):
    """装饰器：捕获异常并打印警告"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ {prefix}失败: {e}")
                return default
        return wrapper
    return decorator


def release_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def save_tad_file(path, tads, header_lines=None):
    """保存 TAD 边界文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in (header_lines or []):
            f.write(f"# {line}\n")
        f.write("Start\tEnd\n")
        for s, e in tads:
            f.write(f"{s}\t{e}\n")
    print(f"  💾 {path} ({len(tads)} 个 TAD)")


def load_tad_file(path):
    """加载 TAD 边界文件 → list[(start, end)]"""
    tads = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("Start"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                s, e = int(parts[0]), int(parts[1])
                if s < e:
                    tads.append((s, e))
    return tads


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    缺失 bin 检测 & Mask                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_missing_mask(embeddings):
    """检测全零向量（缺失 bin）"""
    norms = tf.reduce_sum(tf.abs(embeddings), axis=1)
    mask = tf.equal(norms, 0.0)
    n = tf.reduce_sum(tf.cast(mask, tf.int32))
    if n > 0:
        tf.print(f"   检测到 {n} 个缺失 bin")
    return mask


def apply_input_mask(X, missing_mask, mask_rate=MASK_RATE):
    """在缺失 bin + 随机子集上施加 mask"""
    N = tf.shape(X)[0]
    normal_indices = tf.where(tf.logical_not(missing_mask))
    n_normal = tf.shape(normal_indices)[0]
    n_mask = tf.maximum(tf.cast(tf.cast(n_normal, tf.float32) * mask_rate, tf.int32), 0)
    random_mask = tf.scatter_nd(
        tf.random.shuffle(normal_indices)[:n_mask],
        tf.ones([n_mask], dtype=tf.bool), [N])
    total_mask = tf.logical_or(missing_mask, random_mask)
    masked_X = X * (1.0 - tf.expand_dims(tf.cast(total_mask, tf.float32), 1))
    return masked_X, total_mask


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        HGNN 模块                                ║
# ╚══════════════════════════════════════════════════════════════════╝

class HypergraphConv(tf.keras.layers.Layer):
    """超图卷积层: D_v^{-1} H W D_e^{-1} H^T X"""
    def __init__(self, out_channels, dropout=DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(out_channels)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, X, H, training=False):
        is_sparse = isinstance(H, tf.SparseTensor)
        if is_sparse:
            W = tf.sparse.reduce_sum(H, axis=0)
            W = W / (tf.reduce_max(W) + EPS)
            H_w = tf.SparseTensor(H.indices, H.values * tf.gather(W, H.indices[:, 1]), H.dense_shape)
            D_v_inv = 1.0 / (tf.sparse.reduce_sum(H_w, axis=1) + EPS)
            D_e_inv = 1.0 / (tf.sparse.reduce_sum(H_w, axis=0) + EPS)
            tmp = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(H_w), X)
            agg = tf.sparse.sparse_dense_matmul(H_w, tf.expand_dims(D_e_inv, 1) * tmp)
        else:
            W = tf.reduce_sum(H, axis=0)
            W = W / (tf.reduce_max(W) + EPS)
            H_w = H * tf.expand_dims(W, 0)
            D_v_inv = 1.0 / (tf.reduce_sum(H_w, axis=1) + EPS)
            D_e_inv = 1.0 / (tf.reduce_sum(H_w, axis=0) + EPS)
            agg = tf.matmul(H_w, tf.expand_dims(D_e_inv, 1) * tf.matmul(tf.transpose(H_w), X))
        out = self.fc(tf.expand_dims(D_v_inv, 1) * agg)
        return self.dropout(tf.nn.gelu(out), training=training)


class HGNNEncoder(tf.keras.layers.Layer):
    """多层 HGNN 编码器"""
    def __init__(self, out_dim=HGNN_HIDDEN, dropout=DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        self.convs = [HypergraphConv(out_dim, dropout) for _ in range(HGNN_LAYERS)]

    def call(self, X, H, training=False):
        h = X
        for conv in self.convs:
            h = conv(h, H, training=training)
        return h


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        TCN 模块                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

class TCNBlock(tf.keras.layers.Layer):
    """TCN 残差块: Conv1D + GELU + Dropout + Residual"""
    def __init__(self, filters, kernel_size, dilation_rate, dropout=DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size,
                                            dilation_rate=dilation_rate, padding="same")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        return x + self.dropout(tf.nn.gelu(self.conv(x)), training=training)


class TCNEncoder(tf.keras.layers.Layer):
    """多尺度 TCN 编码器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj_in = tf.keras.layers.Conv1D(TCN_FILTERS, kernel_size=1)
        self.blocks = [TCNBlock(TCN_FILTERS, TCN_KERNEL, d) for d in TCN_DILATIONS]
        self.proj_out = tf.keras.layers.Dense(LATENT_DIM)

    def call(self, x, training=False):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, training=training)
        return self.proj_out(x)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      Decoder & 组合 Encoder                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class Decoder(tf.keras.layers.Layer):
    """全连接解码器"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(DECODER_DIMS[0])
        self.fc2 = tf.keras.layers.Dense(DECODER_DIMS[1])
        self.fc3 = tf.keras.layers.Dense(input_dim)

    def call(self, z, training=False):
        return self.fc3(tf.nn.gelu(self.fc2(tf.nn.gelu(self.fc1(z)))))


class HGNN_TCN_Encoder(tf.keras.layers.Layer):
    """HGNN → expand → TCN → squeeze"""
    def __init__(self, input_dim, hypergraph_H, **kwargs):
        super().__init__(**kwargs)
        self.hypergraph_H = hypergraph_H
        self.hgnn = HGNNEncoder()
        self.tcn = TCNEncoder()

    def call(self, X, training=False):
        h = self.hgnn(X, self.hypergraph_H, training=training)
        z = self.tcn(tf.expand_dims(h, 0), training=training)
        return tf.squeeze(z, 0)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    Masked AutoEncoder (预训练)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

class MaskedAutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, hypergraph_H):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = HGNN_TCN_Encoder(input_dim, hypergraph_H)
        self.decoder = Decoder(input_dim)

    def call(self, X, missing_mask, training=False):
        masked_X, mask = apply_input_mask(X, missing_mask)
        z = self.encoder(masked_X, training=training)
        return self.decoder(z, training=training), mask, z

    def pretrain_step(self, X, missing_mask, optimizer):
        with tf.GradientTape() as tape:
            recon, mask, _ = self(X, missing_mask, training=True)
            diff = tf.square(X - recon)
            mask_f = tf.cast(mask, tf.float32)
            missing_f = tf.cast(missing_mask, tf.float32)
            observed_f = 1.0 - mask_f
            # 观测部分 loss
            loss_obs = tf.reduce_sum(diff * tf.expand_dims(observed_f, 1)) / (
                tf.reduce_sum(observed_f) * tf.cast(self.input_dim, tf.float32) + EPS)
            # 随机 mask 部分 loss
            rand_f = mask_f - missing_f
            n_rand = tf.reduce_sum(rand_f)
            loss_mask = tf.cond(
                n_rand > 0,
                lambda: tf.reduce_sum(diff * tf.expand_dims(rand_f, 1)) / (
                    n_rand * tf.cast(self.input_dim, tf.float32) + EPS),
                lambda: tf.constant(0.0))
            loss = loss_obs + loss_mask
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, loss_obs, loss_mask


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        IDEC 聚类模型                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class IDECModel(tf.keras.Model):
    def __init__(self, input_dim, hypergraph_H, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.encoder = HGNN_TCN_Encoder(input_dim, hypergraph_H)
        self.decoder = Decoder(input_dim)
        self.cluster_layer = tf.keras.layers.Dense(n_clusters, use_bias=False)
        self.cluster_layer.build([None, LATENT_DIM])

    def get_soft_assignment(self, z):
        """t-分布软分配 q_ij"""
        centers = tf.transpose(self.cluster_layer.weights[0])
        dist = tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - tf.expand_dims(centers, 0)), 2)
        q = tf.pow(1.0 / (1.0 + dist), 2.0)
        return q / (tf.reduce_sum(q, axis=1, keepdims=True) + EPS)

    def target_distribution(self, q):
        """强化目标分布 p"""
        weight = tf.pow(q, 2) / tf.reduce_sum(q, axis=0, keepdims=True)
        return weight / (tf.reduce_sum(weight, axis=1, keepdims=True) + EPS)

    def entropy_loss(self, q):
        """聚类熵正则"""
        p = tf.reduce_mean(q, axis=0)
        return tf.reduce_sum(p * tf.math.log(p + EPS))

    def init_cluster_centers(self, X, missing_mask):
        """KMeans 初始化聚类中心"""
        masked_X, _ = apply_input_mask(X, missing_mask)
        z = self.encoder(masked_X, training=False).numpy()
        km = KMeans(n_clusters=self.n_clusters, random_state=RANDOM_SEED, n_init=10, max_iter=200)
        km.fit(z)
        self.cluster_layer.set_weights([tf.transpose(tf.convert_to_tensor(km.cluster_centers_, tf.float32))])
        print(f"   ✅ 聚类中心初始化: {km.cluster_centers_.shape}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         训练函数                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def pretrain_ae(model, X, missing_mask):
    """预训练 Masked AutoEncoder"""
    print(f"\n📝 预训练 Masked AutoEncoder（{PRETRAIN_EPOCHS} epochs）...")
    lr = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, 50, 0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    for ep in range(PRETRAIN_EPOCHS):
        loss, l_obs, l_mask = model.pretrain_step(X, missing_mask, opt)
        if (ep + 1) % 20 == 0:
            print(f"   Epoch {ep+1}/{PRETRAIN_EPOCHS} | Loss: {loss:.4f} | Obs: {l_obs:.4f} | Mask: {l_mask:.4f}")
    print("✅ 预训练完成")
    return model


def train_idec(idec, X, missing_mask):
    """联合训练 IDEC"""
    print(f"\n📝 联合训练 IDEC（{JOINT_EPOCHS} epochs）...")
    lr = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, 100, 0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    kl_loss_fn = tf.keras.losses.KLDivergence()

    masked_X, _ = apply_input_mask(X, missing_mask)
    q = idec.get_soft_assignment(idec.encoder(masked_X, training=False))
    p_target = idec.target_distribution(q)
    prev_preds = None

    for ep in range(JOINT_EPOCHS):
        with tf.GradientTape() as tape:
            masked_X, _ = apply_input_mask(X, missing_mask)
            z = idec.encoder(masked_X, training=True)
            q = idec.get_soft_assignment(z)
            recon = idec.decoder(z, training=True)
            loss = (kl_loss_fn(p_target + EPS, q + EPS)
                    + LAMBDA_RECON * tf.reduce_mean(tf.square(X - recon))
                    + LAMBDA_ENTROPY * idec.entropy_loss(q))
        grads = tape.gradient(loss, idec.trainable_variables)
        opt.apply_gradients(zip(grads, idec.trainable_variables))

        if (ep + 1) % UPDATE_INTERVAL == 0:
            masked_X_u, _ = apply_input_mask(X, missing_mask)
            z_u = idec.encoder(masked_X_u, training=False)
            p_target = idec.target_distribution(idec.get_soft_assignment(z_u))

        if (ep + 1) % 30 == 0:
            preds = tf.argmax(q, 1).numpy()
            n_cls = len(np.unique(preds))
            chg = np.mean(preds != prev_preds) if prev_preds is not None else 0.0
            prev_preds = preds
            print(f"   Epoch {ep+1}/{JOINT_EPOCHS} | Loss: {loss:.4f} | Clusters: {n_cls} | Change: {chg:.3f}")

    z_f = idec.encoder(apply_input_mask(X, missing_mask)[0], training=False)
    preds = tf.argmax(idec.get_soft_assignment(z_f), 1).numpy()
    print(f"✅ 联合训练完成，最终聚类数: {len(np.unique(preds))}")
    return preds, z_f


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       K 值自动搜索                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def estimate_k(ae, X, missing_mask):
    """粗搜 + 精搜确定最优聚类数 K"""
    masked_X, _ = apply_input_mask(X, missing_mask)
    z = ae.encoder(masked_X, training=False).numpy()

    # Stage 1: 粗搜
    print(f"\n🔍 粗搜索: K ∈ [{K_SEARCH_MIN}, {K_SEARCH_MAX}], 步长 {K_SEARCH_STEP}")
    best_score, best_k = -np.inf, K_SEARCH_MIN
    for k in range(K_SEARCH_MIN, K_SEARCH_MAX + 1, K_SEARCH_STEP):
        preds = KMeans(k, random_state=RANDOM_SEED, n_init=10, max_iter=300).fit_predict(z)
        sil = silhouette_score(z, preds)
        print(f"   K={k:3d} | Silhouette={sil:.4f}")
        if sil > best_score:
            best_score, best_k = sil, k
    print(f"   粗搜最优: K={best_k} (Silhouette={best_score:.4f})")

    # Stage 2: 精搜
    lo = max(K_GLOBAL_MIN, best_k - K_SEARCH_FINE)
    hi = min(K_GLOBAL_MAX, best_k + K_SEARCH_FINE)
    print(f"\n🔍 精搜索: K ∈ [{lo}, {hi}], 步长 1")
    for k in range(lo, hi + 1):
        preds = KMeans(k, random_state=RANDOM_SEED, n_init=10, max_iter=300).fit_predict(z)
        sil = silhouette_score(z, preds)
        if sil > best_score:
            best_score, best_k = sil, k
    print(f"✅ 最优 K={best_k} (Silhouette={best_score:.4f})")
    return best_k


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  TAD 高质量筛选 & 后处理                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def block_internal_density(sim, s, e):
    """TAD 内部平均余弦相似度"""
    block = sim[s:e+1, s:e+1].astype(np.float32)
    cos_sim = 1 - pairwise_distances(block, metric="cosine")
    mask = np.ones_like(cos_sim) - np.eye(len(cos_sim))
    valid = cos_sim[mask.astype(bool)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def gap_density(sim, interval, prev=None, nxt=None):
    """TAD 与相邻 TAD 之间的平均相似度"""
    s, e = interval
    vals = []
    if prev:
        ps, pe = prev
        vals.extend(sim[s:e+1, ps:pe+1].flatten())
    if nxt:
        ns, ne = nxt
        vals.extend(sim[s:e+1, ns:ne+1].flatten())
    return float(np.mean(vals)) if vals else 0.01


def filter_tads(tads, sim):
    """筛选: 内部密度 ≥ 阈值 且 密度/间隙比 ≥ 阈值"""
    print(f"\n🔧 TAD 筛选（密度≥{MIN_INTERNAL_DENSITY}，比值≥{DENSITY_GAP_RATIO_THRESH}）")
    kept = []
    for i, (s, e) in enumerate(tads):
        density = block_internal_density(sim, s, e)
        if density < MIN_INTERNAL_DENSITY:
            print(f"   ✗ [{s}:{e}] 密度={density:.3f} < {MIN_INTERNAL_DENSITY}")
            continue
        gap = gap_density(sim, (s, e),
                          tads[i-1] if i > 0 else None,
                          tads[i+1] if i < len(tads)-1 else None)
        ratio = density / gap
        if ratio >= DENSITY_GAP_RATIO_THRESH:
            kept.append((s, e))
            print(f"   ✓ [{s}:{e}] 密度={density:.3f} 比值={ratio:.3f}")
        else:
            print(f"   ✗ [{s}:{e}] 比值={ratio:.3f} < {DENSITY_GAP_RATIO_THRESH}")
    print(f"   {len(tads)} → {len(kept)}")
    return kept


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      Hi-C 精修流水线                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def insulation_score(pos, hic, half_w):
    """位置 pos 的 insulation score（越小越可能是边界）"""
    N = hic.shape[0]
    s, e = max(0, pos - half_w), min(N, pos + half_w)
    if s >= pos or pos >= e:
        return 1.0
    cross = hic[s:pos, pos:e]
    return float(np.mean(cross)) if cross.size > 0 else 1.0


def ranksum_pvalue(hic, pos, size):
    """对角线区域 vs 上下游的秩和检验 p 值"""
    N = hic.shape[0]
    # 对角线值
    dia = []
    for k in range(size):
        r = pos - k
        if 0 <= r < N - 1:
            for c in range(pos + 1, min(N, pos + size + 1)):
                dia.append(hic[r, c])
    # 上游 + 下游
    ref = []
    for (start, end) in [(max(0, pos-size), pos+1), (pos+1, min(N, pos+size+1))]:
        if end - start >= 2:
            b = hic[start:end, start:end]
            for r in range(b.shape[0]):
                for c in range(r + 1, b.shape[1]):
                    ref.append(b[r, c])
    x, y = np.array(dia, np.float64), np.array(ref, np.float64)
    if len(x) < 2 or len(y) < 2:
        return 1.0
    try:
        _, p = mannwhitneyu(x, y, alternative="less", method="asymptotic")
        return p
    except Exception:
        return 1.0


def find_refined_boundary(pos, hic, window, half_w, rs_size, p_thresh):
    """在 pos 附近搜索 insulation 最小值，并用秩和检验验证"""
    N = hic.shape[0]
    lo, hi = max(0, pos - window), min(N - 1, pos + window)
    if lo > hi:
        return None
    scores = [(c, insulation_score(c, hic, half_w)) for c in range(lo, hi + 1)]
    # 找局部极小值
    minima = []
    for i, (c, ins) in enumerate(scores):
        if (i == 0 or scores[i-1][1] > ins) and (i == len(scores)-1 or scores[i+1][1] > ins):
            minima.append((c, ins))
    if not minima:
        return None
    # 秩和检验过滤
    passed = [(c, ins, ranksum_pvalue(hic, c, rs_size))
              for c, ins in minima
              if ranksum_pvalue(hic, c, rs_size) < p_thresh]
    if not passed:
        return None
    passed.sort(key=lambda x: (abs(x[0] - pos), x[1]))
    return passed[0][0]


def refine_tads(hic, tads_1based):
    """阶段1: 极小值 + 秩和检验精修"""
    refined = []
    stats = {"no_minima": 0, "no_pval": 0, "collapse": 0, "shifted": 0, "unchanged": 0}
    for s, e in tads_1based:
        new_s = find_refined_boundary(s-1, hic, REFINE_WINDOW, INSULATION_HALF, RANKSUM_SIZE, PVALUE_THRESHOLD)
        new_e = find_refined_boundary(e-1, hic, REFINE_WINDOW, INSULATION_HALF, RANKSUM_SIZE, PVALUE_THRESHOLD)
        if new_s is None and new_e is None:
            stats["no_minima"] += 1; continue
        if new_s is None or new_e is None:
            stats["no_pval"] += 1; continue
        if new_s >= new_e:
            stats["collapse"] += 1; continue
        if new_s != s-1 or new_e != e-1:
            stats["shifted"] += 1
        else:
            stats["unchanged"] += 1
        refined.append((new_s + 1, new_e + 1))
    return refined, stats


def cosine_sim(a, b, all_tads, hic):
    """两个 TAD 之间的余弦相似度（基于与其他 TAD 的交互 profile）"""
    s1, e1 = a; s2, e2 = b
    v1, v2 = [], []
    for os_, oe_ in all_tads:
        if (os_, oe_) == a or (os_, oe_) == b:
            continue
        v1.append(np.mean(hic[s1:e1+1, os_:oe_+1]) / (np.mean(hic[s1:e1+1, s1:e1+1]) + 1e-10))
        v2.append(np.mean(hic[s2:e2+1, os_:oe_+1]) / (np.mean(hic[s2:e2+1, s2:e2+1]) + 1e-10))
    a_arr, b_arr = np.array(v1), np.array(v2)
    if np.all(a_arr == 0) or np.all(b_arr == 0):
        return 1.0 if np.array_equal(a_arr, b_arr) else 0.0
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))


def merge_adjacent(tads_1based, hic):
    """阶段2: 基于余弦相似度合并相邻 TAD"""
    if len(tads_1based) < 2:
        return list(tads_1based)
    tads_0 = [(s-1, e-1) for s, e in tads_1based]
    avg = float(np.mean(hic[hic > 0])) if np.any(hic > 0) else 0.0
    merged, i = [], 0
    while i < len(tads_0):
        cur_s, cur_e = tads_0[i]
        j = i + 1
        while j < len(tads_0):
            ns, ne = tads_0[j]
            cs = cosine_sim((cur_s, cur_e), (ns, ne), tads_0, hic)
            if cs >= MERGE_COS_THRESHOLD and np.mean(hic[cur_s:ne+1, cur_s:ne+1]) >= avg:
                cur_e = ne; j += 1
            else:
                break
        merged.append((cur_s, cur_e)); i = j
    return [(s+1, e+1) for s, e in merged]


def final_filter(tads_1based):
    """阶段3: 过滤过短 TAD"""
    return [(s, e) for s, e in tads_1based if e - s >= MIN_DIFF]


def run_refine_pipeline(chr_name, tads_1based, hic, res=RESOLUTION):
    """完整精修流水线"""
    print(f"\n{'='*50}")
    print(f"Hi-C 精修: {chr_name}")
    print(f"{'='*50}")

    # 阶段 1
    print(f"\n┌─ 极小值+秩和检验 ...", end=" ", flush=True)
    t0 = time.time()
    refined, stats = refine_tads(hic, tads_1based)
    print(f"({time.time()-t0:.1f}s) {len(tads_1based)} → {len(refined)}")

    # 阶段 2
    print(f"├─ 相邻合并 ...", end=" ", flush=True)
    t0 = time.time()
    merged = merge_adjacent(refined, hic)
    print(f"({time.time()-t0:.1f}s) {len(refined)} → {len(merged)}")

    # 阶段 3
    print(f"└─ 最终过滤 ...", end=" ", flush=True)
    t0 = time.time()
    final = final_filter(merged)
    print(f"({time.time()-t0:.1f}s) {len(merged)} → {len(final)}")

    # 保存
    out_path = get_refine_output_path(chr_name, res)
    save_tad_file(out_path, final, [
        f"{chr_name} 精修后 TAD",
        f"window={REFINE_WINDOW} half_w={INSULATION_HALF} ranksum={RANKSUM_SIZE} "
        f"p<{PVALUE_THRESHOLD} cos>{MERGE_COS_THRESHOLD} min_diff={MIN_DIFF}",
    ])
    print(f"\n✅ 精修完成: {len(tads_1based)} → {len(final)}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        TAD 检测器                               ║
# ╚══════════════════════════════════════════════════════════════════╝

class TADDetector:
    """单条染色体的 TAD 检测"""

    def __init__(self, chr_name, res=RESOLUTION):
        self.chr_name = chr_name
        self.res = res
        self.embed_dir = get_embed_dir(chr_name, res)
        self.output_dir = get_output_dir(chr_name, res)

        # 加载数据
        self.hypergraph = self._load_hypergraph()
        self.sim_matrix = self._load_similarity()
        self._load_and_align_embeddings()

        self.num_nodes = self.sim_matrix.shape[0]
        self.embed_dim = self.node_embeds.shape[1]
        self.num_hyperedges = self.hypergraph.shape[1]

        self.missing_mask = build_missing_mask(tf.convert_to_tensor(self.node_embeds, tf.float32))
        print(f"✅ {chr_name} 对齐完成: sim={self.sim_matrix.shape} embed={self.node_embeds.shape} hg={self.hypergraph.shape}")

    def _load_hypergraph(self):
        path = get_hypergraph_path(self.chr_name, self.res)
        z = np.load(path)
        m = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=z["shape"])
        z.close()
        m = m.maximum(0).astype(np.float32)
        m.eliminate_zeros()
        print(f"📌 超图: {m.shape}, nnz={m.nnz}")
        return m

    def _load_similarity(self):
        path = os.path.join(self.embed_dir, f"optimize_node_similarity_{PARAM_TAG}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"相似度矩阵不存在: {path}")
        sim = np.loadtxt(path, dtype=np.float32)
        # 自适应阈值
        thresh = np.clip(max(np.mean(sim) + np.std(sim), np.percentile(sim, 90)), 0.3, 0.7)
        sim[sim < thresh] = 0.0
        print(f"📌 相似度阈值: {thresh:.3f}")
        return sim

    def _load_and_align_embeddings(self):
        path = os.path.join(self.embed_dir, f"hypergraph_node_embeddings_{PARAM_TAG}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"节点嵌入不存在: {path}")
        data = np.loadtxt(path, dtype=np.float32)
        bin_ids, embeds = data[:, 0].astype(np.int32), data[:, 1:]
        base_num = self.sim_matrix.shape[0]
        aligned = np.zeros((base_num, embeds.shape[1]), dtype=np.float32)
        n_ok, n_skip = 0, 0
        for i, bid in enumerate(bin_ids):
            idx = bid - 1
            if 0 <= idx < base_num:
                aligned[idx] = embeds[i]; n_ok += 1
            else:
                n_skip += 1
        if n_skip:
            print(f"⚠️ {n_skip} 个 bin 超出范围")
        self.node_embeds = aligned
        # 对齐超图行数
        if self.hypergraph.shape[0] != base_num:
            pad_n = base_num - self.hypergraph.shape[0]
            if hasattr(self.hypergraph, "toarray"):
                from scipy.sparse import vstack as sp_vstack
                self.hypergraph = sp_vstack([self.hypergraph, csr_matrix((pad_n, self.hypergraph.shape[1]))])
            else:
                self.hypergraph = np.vstack([self.hypergraph, np.zeros((pad_n, self.hypergraph.shape[1]), np.float32)])

    def detect(self):
        """完整检测流程: 预训练 → K值搜索 → IDEC → TAD提取 → 筛选"""
        X = tf.convert_to_tensor(StandardScaler().fit_transform(self.node_embeds), tf.float32)

        # 超图 → TF 张量
        if hasattr(self.hypergraph, "toarray"):
            coo = self.hypergraph.tocoo()
            H = tf.SparseTensor(
                np.array([coo.row, coo.col], np.int64).T,
                coo.data.astype(np.float32), coo.shape)
            H = tf.sparse.reorder(H)
        else:
            H = tf.convert_to_tensor(self.hypergraph, tf.float32)

        # 阶段 1: 预训练
        print(f"\n{'='*60}\n阶段1: 预训练 AE ({self.chr_name})\n{'='*60}")
        ae = MaskedAutoEncoder(self.embed_dim, H)
        _ = ae(X, self.missing_mask, training=False)
        ae = pretrain_ae(ae, X, self.missing_mask)

        # 阶段 2: K 值搜索
        print(f"\n{'='*60}\n阶段2: K 值搜索 ({self.chr_name})\n{'='*60}")
        n_clusters = max(MIN_CLUSTERS, min(estimate_k(ae, X, self.missing_mask), MAX_CLUSTERS))

        # 阶段 3: IDEC 聚类
        print(f"\n{'='*60}\n阶段3: IDEC 聚类 K={n_clusters} ({self.chr_name})\n{'='*60}")
        idec = IDECModel(self.embed_dim, H, n_clusters)
        for w_i, w_a in zip(idec.encoder.trainable_variables, ae.encoder.trainable_variables):
            w_i.assign(w_a)
        for w_i, w_a in zip(idec.decoder.trainable_variables, ae.decoder.trainable_variables):
            w_i.assign(w_a)
        idec.init_cluster_centers(X, self.missing_mask)
        preds, z = train_idec(idec, X, self.missing_mask)

        # 阶段 4: TAD 提取
        print(f"\n{'='*60}\n阶段4: TAD 提取 ({self.chr_name})\n{'='*60}")
        tads = self._extract_tads(preds)
        tads = filter_tads(tads, self.sim_matrix)
        self._save_results(tads, preds, n_clusters)
        # 转成 1-based,与文件坐标一致(full 模式直接进 refine_tads,不再减 1)
        return [(s + 1, e + 1) for s, e in tads]

    def _extract_tads(self, preds):
        """聚类标签 → TAD 区间"""
        tads = []
        start = 0
        for i in range(1, len(preds)):
            if preds[i] != preds[start]:
                if i - start >= MIN_TAD_LENGTH:
                    tads.append((start, i - 1))
                start = i
        if len(preds) - start >= MIN_TAD_LENGTH:
            tads.append((start, len(preds) - 1))
        print(f"   原始区间数: {len(tads)}")
        return tads

    def _save_results(self, tads, preds, n_clusters):
        """保存检测结果"""
        path = get_result_path(self.chr_name, self.res)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        densities = [block_internal_density(self.sim_matrix, s, e) for s, e in tads]
        with open(path, "w") as f:
            f.write(f"# Device: {DEVICE}\n")
            f.write(f"# Arch: HGNN + TCN + IDEC\n")
            f.write(f"# Filter: density>={MIN_INTERNAL_DENSITY} ratio>={DENSITY_GAP_RATIO_THRESH}\n")
            f.write(f"# Hypergraph: nodes={self.num_nodes} edges={self.num_hyperedges}\n")
            f.write("Start\tEnd\tInternal_Density\n")
            for (s, e), d in zip(tads, densities):
                f.write(f"{s+1}\t{e+1}\t{d:.4f}\n")
        print(f"💾 结果保存: {path}")
        return path


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          主入口                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="HGNN+TCN+IDEC TAD 检测 + Hi-C 精修")
    parser.add_argument("--chr",   type=int, required=True, help="染色体编号")
    parser.add_argument("--res",   type=int, default=RESOLUTION, help="分辨率 KB")
    parser.add_argument("--mode",  choices=["full", "detect_only", "refine_only"], default="full")
    parser.add_argument("--tag",   type=str, default=PARAM_TAG, help="参数标签")
    args = parser.parse_args()

    chr_name = f"chr{args.chr}"

    # 检测
    if args.mode in ("full", "detect_only"):
        print(f"\n{'='*80}\n📌 TAD 检测: {chr_name} ({DEVICE})\n{'='*80}")
        detector = TADDetector(chr_name, args.res)
        tads = detector.detect()
        result_path = get_result_path(chr_name, args.res)
        del detector
        release_memory()

    # 精修
    if args.mode in ("full", "refine_only"):
        hic_path = get_hic_path(chr_name, args.res)
        if not os.path.exists(hic_path):
            print(f"⚠️ Hi-C 不存在，跳过精修: {hic_path}")
            return
        if args.mode == "refine_only":
            result_path = get_result_path(chr_name, args.res)
            tads = load_tad_file(result_path)
        hic = np.loadtxt(hic_path, dtype=np.float32)
        run_refine_pipeline(chr_name, tads, hic, args.res)
        del hic

    print(f"\n✅ {chr_name} 全部完成")


if __name__ == "__main__":
    main()

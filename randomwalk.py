import os
import time
import random
import warnings
import argparse
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          路径配置                               ║
# ║                    ★★★ 修改这里即可 ★★★                        ║
# ╚══════════════════════════════════════════════════════════════════╝

PATHS = {
    # 输入: 稠密超图矩阵 (txt)
    # 文件名: chr{N}_{res}_VE_matrix.txt
    "input_dir": "/mnt/sde/gmw/GM12878/hypergraph",

    # 输出根目录 (所有中间产物和最终结果都在此下)
    "output_base": "/mnt/sdb/gao/tad_project",
}

# 输出子目录结构（一般不用改）
# {output_base}/res{res}/chr{N}/hypergraph_node_embeddings_{tag}.txt
# {output_base}/res{res}/chr{N}/optimize_node_similarity_{tag}.txt
# {output_base}/res{res}/chr{N}/hypergraph.npz


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         超参数配置                              ║
# ╚══════════════════════════════════════════════════════════════════╝

# walk_length 按分辨率自适应
RES_TO_WALK_LENGTH = {25: 80, 50: 40, 100: 20}

NUM_WALKS   = 1000
EMBED_DIM   = 128
W2V_WINDOW  = 3
W2V_EPOCHS  = 10
W2V_WORKERS = 4
RANDOM_SEED = 42


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        路径构建工具                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_input_matrix_path(chr_num, res):
    """输入稠密矩阵路径"""
    return os.path.join(PATHS["input_dir"], f"chr{chr_num}_{res}_VE_matrix.txt")


def get_chr_output_dir(chr_num, res):
    """单条染色体的输出目录"""
    return os.path.join(PATHS["output_base"], f"res{res}", f"chr{chr_num}")


def get_embed_path(chr_num, res, tag):
    """节点嵌入输出路径"""
    return os.path.join(get_chr_output_dir(chr_num, res), f"hypergraph_node_embeddings_{tag}.txt")


def get_similarity_path(chr_num, res, tag):
    """相似度矩阵输出路径"""
    return os.path.join(get_chr_output_dir(chr_num, res), f"optimize_node_similarity_{tag}.txt")


def get_npz_path(chr_num, res):
    """稀疏超图 NPZ 输出路径"""
    return os.path.join(get_chr_output_dir(chr_num, res), "hypergraph.npz")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  超图随机游走 (Node2Vec 二阶偏置)                 ║
# ╚══════════════════════════════════════════════════════════════════╝

class HypergraphRandomWalk:
    """超图上的二阶偏置随机游走"""

    def __init__(self, matrix, p, q):
        self.matrix = np.array(matrix, dtype=np.float32)
        self.num_nodes = self.matrix.shape[0]
        self.num_edges = self.matrix.shape[1]
        self.p = p
        self.q = q

        # 预计算: 节点-超边关系
        self.node_edges = [set() for _ in range(self.num_nodes + 1)]  # 1-indexed
        self.edge_nodes = []
        self.edge_weight = np.zeros(self.num_edges, dtype=np.float32)
        self.edge_size = np.zeros(self.num_edges, dtype=np.int32)

        for e in range(self.num_edges):
            nodes = set(np.where(self.matrix[:, e] > 0)[0] + 1)
            self.edge_nodes.append(nodes)
            if nodes:
                self.edge_weight[e] = self.matrix[next(iter(nodes)) - 1, e]
            self.edge_size[e] = len(nodes)
            for n in nodes:
                self.node_edges[n].add(e)

        # 预计算: 一阶转移概率 CSR
        rows, cols, vals = [], [], []
        for e in range(self.num_edges):
            n_list = list(self.edge_nodes[e])
            sz = self.edge_size[e]
            if sz < 2:
                continue
            contrib = self.edge_weight[e] / sz
            for u in n_list:
                for v in n_list:
                    if u != v:
                        rows.append(u - 1)
                        cols.append(v - 1)
                        vals.append(contrib)
        self.first_order = csr_matrix(
            (vals, (rows, cols)),
            shape=(self.num_nodes, self.num_nodes), dtype=np.float32)

    def _bias(self, target, prev, current):
        """二阶偏置: 返回/前行/远离"""
        cur_set = self.node_edges[current]
        prev_set = self.node_edges[prev]
        tgt_set = self.node_edges[target]
        if cur_set & prev_set & tgt_set:
            return 1.0 / self.p
        if (cur_set & tgt_set) and (cur_set & tgt_set).isdisjoint(prev_set):
            return 1.0
        return 1.0 / self.q

    def _step_probs(self, prev, current):
        """计算下一步转移概率"""
        row = self.first_order[current - 1]
        idx, data = row.indices, row.data
        if len(idx) == 0:
            return np.array([1.0], np.float32), np.array([current - 1])
        probs = np.array([data[i] * self._bias(idx[i] + 1, prev, current)
                          for i in range(len(idx))], np.float32)
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs, idx

    def walk(self, start, length):
        """从 start 节点出发的一条随机游走"""
        path = [str(start)]
        # 第一步: 无偏
        if length > 1:
            row = self.first_order[start - 1]
            if len(row.indices) == 0:
                path.append(str(start))
            else:
                d = row.data / row.data.sum()
                path.append(str(np.random.choice(row.indices, p=d) + 1))
        # 后续: 二阶偏置
        while len(path) < length:
            prev, curr = int(path[-2]), int(path[-1])
            probs, idx = self._step_probs(prev, curr)
            if len(idx) == 0 or probs.sum() == 0:
                path.append(str(curr))
            else:
                path.append(str(np.random.choice(idx, p=probs) + 1))
        return path


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     训练 & 保存                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def train_embedding(matrix, p, q, num_walks, walk_length, embed_dim):
    """随机游走 + Word2Vec 训练节点嵌入"""
    hrw = HypergraphRandomWalk(matrix, p, q)

    # 随机游走
    t0 = time.time()
    walks = []
    for _ in range(num_walks):
        start = np.random.randint(1, hrw.num_nodes + 1)
        walks.append(hrw.walk(start, walk_length))
    print(f"  ⏱ 随机游走: {num_walks} 条 × {walk_length} 步 = {time.time()-t0:.1f}s")

    # Word2Vec
    t0 = time.time()
    model = Word2Vec(sentences=walks, vector_size=embed_dim, window=W2V_WINDOW,
                     min_count=1, sg=1, workers=W2V_WORKERS, epochs=W2V_EPOCHS)
    print(f"  ⏱ Word2Vec: {time.time()-t0:.1f}s")

    embeddings = {}
    for i in range(1, hrw.num_nodes + 1):
        key = str(i)
        if key in model.wv:
            embeddings[i] = model.wv[key]
    return embeddings, model, hrw.num_nodes


def save_similarity(model, num_nodes, save_path):
    """保存节点相似度矩阵"""
    sim = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    keys = [str(i) for i in range(1, num_nodes + 1)]
    for i in range(num_nodes):
        if keys[i] not in model.wv:
            continue
        for j in range(i, num_nodes):
            if keys[j] not in model.wv:
                continue
            s = float(model.wv.similarity(keys[i], keys[j]))
            sim[i, j] = s
            sim[j, i] = s
    np.savetxt(save_path, sim, fmt="%.4f")
    print(f"  💾 相似度: {save_path}")


def save_embeddings(model, save_path):
    """保存节点嵌入（bin_id + embedding 列）"""
    temp = save_path + ".tmp"
    model.wv.save_word2vec_format(temp)
    with open(temp) as f:
        next(f)  # 跳过头行
        lines = sorted([l.strip() for l in f if l.strip()], key=lambda x: int(x.split()[0]))
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.remove(temp)
    print(f"  💾 嵌入: {save_path}")


def save_hypergraph_npz(matrix, save_path):
    """将稠密矩阵转为稀疏 NPZ（供 pipeline 使用）"""
    sparse = csr_matrix(matrix.astype(np.float32))
    sparse.eliminate_zeros()
    np.savez(save_path,
             data=sparse.data, indices=sparse.indices,
             indptr=sparse.indptr, shape=sparse.shape)
    print(f"  💾 超图NPZ: {save_path} (nnz={sparse.nnz})")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                           主流程                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def run(chr_num, res, p, q):
    """单条染色体: 加载矩阵 → 随机游走 → 训练 → 保存"""
    chr_name = f"chr{chr_num}"
    tag = f"p{int(p) if p == int(p) else p}_q{int(q) if q == int(q) else q}"
    walk_length = RES_TO_WALK_LENGTH.get(res, 40)

    # 路径
    matrix_path = get_input_matrix_path(chr_num, res)
    out_dir = get_chr_output_dir(chr_num, res)
    embed_path = get_embed_path(chr_num, res, tag)
    sim_path = get_similarity_path(chr_num, res, tag)
    npz_path = get_npz_path(chr_num, res)

    print(f"\n{'='*60}")
    print(f"  {chr_name} | res={res}KB | p={p} q={q} | walk={walk_length}")
    print(f"  输入: {matrix_path}")
    print(f"  输出: {out_dir}/")
    print(f"{'='*60}")

    # 加载矩阵
    if not os.path.exists(matrix_path):
        print(f"❌ 矩阵文件不存在: {matrix_path}")
        return
    matrix = np.loadtxt(matrix_path, dtype=np.float32)
    print(f"  矩阵: {matrix.shape}")

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 训练
    t0 = time.time()
    embeddings, model, num_nodes = train_embedding(
        matrix, p, q, NUM_WALKS, walk_length, EMBED_DIM)
    print(f"  ⏱ 总训练: {time.time()-t0:.1f}s")

    # 保存
    save_embeddings(model, embed_path)
    save_similarity(model, num_nodes, sim_path)
    save_hypergraph_npz(matrix, npz_path)

    print(f"\n✅ {chr_name} 完成")


def main():
    parser = argparse.ArgumentParser(description="超图随机游走 → 节点嵌入 & 相似度")
    parser.add_argument("--chr", type=int, required=True, help="染色体编号")
    parser.add_argument("--res", type=int, default=25, help="分辨率 KB (默认 25)")
    parser.add_argument("--p",   type=float, default=2.0, help="Node2Vec 返回参数 (默认 2.0)")
    parser.add_argument("--q",   type=float, default=0.5, help="Node2Vec 前行参数 (默认 0.5)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    run(args.chr, args.res, args.p, args.q)


if __name__ == "__main__":
    main()
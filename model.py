import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EPS = 1e-6
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
PRETRAIN_EPOCHS = 500
JOINT_EPOCHS = 1000
JOINT_LR = 5e-4
K_SIL_FRAC = 0.98
K_MAX_DIV = 5
LAMBDA_RECON = 1.0
KL_WEIGHT = 0.1
LAMBDA_ENTROPY = 0.01
UPDATE_INTERVAL = 10
REINIT_INTERVAL = 100
HGNN_LAYERS = 2
HGNN_HIDDEN = 128
TCN_KERNEL = 3
TCN_FILTERS = 64
TCN_STAGE_DILATIONS = [(1, 2, 3), (1, 2, 3)]
LATENT_DIM = 64
DECODER_DIMS = [128, 256, 128]
RANDOM_SEED = 42


def build_missing_mask(embeddings):
    norms = tf.reduce_sum(tf.abs(embeddings), axis=1)
    return tf.equal(norms, 0.0)


def edge_weights(h):
    c = h.tocsc()
    w = np.ones(c.shape[1], dtype=np.float32)
    for e in range(c.shape[1]):
        s, t = c.indptr[e], c.indptr[e + 1]
        if t > s:
            w[e] = c.data[s]
    if np.allclose(w, w[0]):
        w[:] = 1.0
    return tf.constant(w / w.mean(), tf.float32)


class HypergraphConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, H, edge_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.H = H
        self.edge_weight = edge_weight
        self.fc = tf.keras.layers.Dense(out_channels)
        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE)

    def call(self, X, training=False):
        H = self.H
        D_e_inv = 1.0 / (tf.sparse.reduce_sum(H, axis=0) + EPS)
        D_v_isqrt = 1.0 / tf.sqrt(tf.sparse.reduce_sum(H, axis=1) + EPS)
        Xs = X * tf.expand_dims(D_v_isqrt, 1)
        tmp = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(H), Xs)
        tmp = tmp * tf.expand_dims(D_e_inv, 1)
        if self.edge_weight is not None:
            tmp = tmp * tf.expand_dims(self.edge_weight, 1)
        agg = tf.sparse.sparse_dense_matmul(H, tmp)
        out = self.fc(tf.expand_dims(D_v_isqrt, 1) * agg)
        return self.dropout(tf.nn.gelu(out), training=training)


class HGNNEncoder(tf.keras.layers.Layer):
    def __init__(self, H, out_dim=HGNN_HIDDEN, edge_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.convs = [HypergraphConv(out_dim, H, edge_weight) for _ in range(HGNN_LAYERS)]

    def call(self, X, training=False):
        h = X
        for conv in self.convs:
            h = conv(h, training=training)
        return h


class TCNStage(tf.keras.layers.Layer):
    def __init__(self, dilations, **kwargs):
        super().__init__(**kwargs)
        self.convs = [tf.keras.layers.Conv1D(TCN_FILTERS, TCN_KERNEL,
                                             dilation_rate=d, padding="same")
                      for d in dilations]
        self.fuse = tf.keras.layers.Conv1D(TCN_FILTERS, 1, padding="same")
        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE)

    def call(self, x, training=False):
        outs = [tf.nn.gelu(c(x)) for c in self.convs]
        h = self.fuse(tf.concat(outs, axis=-1))
        return x + self.dropout(tf.nn.gelu(h), training=training)


class TCNEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj_in = tf.keras.layers.Dense(TCN_FILTERS)
        self.stages = [TCNStage(d) for d in TCN_STAGE_DILATIONS]

    def call(self, x, training=False):
        h = tf.nn.gelu(self.proj_in(x))
        for stage in self.stages:
            h = stage(h, training=training)
        return h


class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(DECODER_DIMS[0])
        self.fc2 = tf.keras.layers.Dense(DECODER_DIMS[1])
        self.fc3 = tf.keras.layers.Dense(input_dim)

    def call(self, z, training=False):
        return self.fc3(tf.nn.gelu(self.fc2(tf.nn.gelu(self.fc1(z)))))


class HGNN_TCN_Encoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, H, edge_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.hgnn = HGNNEncoder(H, edge_weight=edge_weight)
        self.tcn = TCNEncoder()

    def call(self, X, training=False):
        h = self.hgnn(X, training=training)
        z = self.tcn(tf.expand_dims(h, 0), training=training)
        return tf.squeeze(z, 0)


class MaskedAutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, H, edge_weight=None):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = HGNN_TCN_Encoder(input_dim, H, edge_weight)
        self.decoder = Decoder(input_dim)

    def call(self, X, missing_mask, training=False):
        z = self.encoder(X, training=training)
        return self.decoder(z, training=training), missing_mask, z

    def pretrain_step(self, X, missing_mask, optimizer):
        with tf.GradientTape() as tape:
            recon, _, _ = self(X, missing_mask, training=True)
            diff = tf.square(X - recon)
            observed = 1.0 - tf.cast(missing_mask, tf.float32)
            loss = tf.reduce_sum(diff * tf.expand_dims(observed, 1)) / (
                tf.reduce_sum(observed) * tf.cast(self.input_dim, tf.float32) + EPS)
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


class IDECModel(tf.keras.Model):
    def __init__(self, input_dim, H, n_clusters, edge_weight=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.encoder = HGNN_TCN_Encoder(input_dim, H, edge_weight)
        self.decoder = Decoder(input_dim)
        self.cluster_layer = tf.keras.layers.Dense(n_clusters, use_bias=False)
        self.cluster_layer.build([None, LATENT_DIM])

    def get_soft_assignment(self, z):
        centers = tf.transpose(self.cluster_layer.weights[0])
        dist = tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - tf.expand_dims(centers, 0)), 2)
        q = 1.0 / (1.0 + dist)
        return q / (tf.reduce_sum(q, axis=1, keepdims=True) + EPS)

    def target_distribution(self, q):
        weight = tf.pow(q, 2) / tf.reduce_sum(q, axis=0, keepdims=True)
        return weight / (tf.reduce_sum(weight, axis=1, keepdims=True) + EPS)

    def entropy_loss(self, q):
        p = tf.reduce_mean(q, axis=0)
        return tf.reduce_sum(p * tf.math.log(p + EPS))

    def init_cluster_centers(self, X):
        z = self.encoder(X, training=False).numpy()
        km = KMeans(n_clusters=self.n_clusters, random_state=RANDOM_SEED,
                    n_init=10, max_iter=200).fit(z)
        self.cluster_layer.set_weights(
            [tf.transpose(tf.convert_to_tensor(km.cluster_centers_, tf.float32))])


def pretrain_ae(model, X, missing_mask):
    lr = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, 50, 0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    best, best_w = float("inf"), None
    for ep in range(PRETRAIN_EPOCHS):
        loss = model.pretrain_step(X, missing_mask, opt)
        l = float(loss)
        if l < best:
            best = l
            best_w = [v.numpy() for v in model.trainable_variables]
        if (ep + 1) % 50 == 0:
            print(f"pretrain {ep+1}/{PRETRAIN_EPOCHS} loss={l:.4f} best={best:.4f}", flush=True)
    if best_w is not None:
        for v, w in zip(model.trainable_variables, best_w):
            v.assign(w)
    return model


def reinit_empty_clusters(idec, z, q):
    assign = tf.argmax(q, axis=1).numpy()
    counts = np.bincount(assign, minlength=idec.n_clusters)
    dead = np.where(counts == 0)[0]
    if len(dead) == 0:
        return 0
    centers = idec.cluster_layer.weights[0].numpy()
    zc = z.numpy()
    own = centers[:, assign].T
    far = np.argsort(-np.sum((zc - own) ** 2, axis=1))[:len(dead)]
    centers[:, dead] = zc[far].T
    idec.cluster_layer.set_weights([centers.astype(np.float32)])
    return len(dead)


def train_idec(idec, X, missing_mask):
    lr = tf.keras.optimizers.schedules.ExponentialDecay(JOINT_LR, 100, 0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    kl_loss_fn = tf.keras.losses.KLDivergence()
    q = idec.get_soft_assignment(idec.encoder(X, training=False))
    p_target = idec.target_distribution(q)
    best, best_w = float("inf"), None
    for ep in range(JOINT_EPOCHS):
        with tf.GradientTape() as tape:
            z = idec.encoder(X, training=False)
            q = idec.get_soft_assignment(z)
            recon = idec.decoder(z, training=True)
            diff = tf.square(X - recon)
            observed = 1.0 - tf.cast(missing_mask, tf.float32)
            recon_loss = tf.reduce_sum(diff * tf.expand_dims(observed, 1)) / (
                tf.reduce_sum(observed) * tf.cast(X.shape[1], tf.float32) + EPS)
            kl = kl_loss_fn(p_target + EPS, q + EPS)
            ent = idec.entropy_loss(q)
            loss = LAMBDA_RECON * recon_loss + KL_WEIGHT * kl + LAMBDA_ENTROPY * ent
        grads = tape.gradient(loss, idec.trainable_variables)
        opt.apply_gradients(zip(grads, idec.trainable_variables))
        if (ep + 1) % UPDATE_INTERVAL == 0:
            z_u = idec.encoder(X, training=False)
            q_u = idec.get_soft_assignment(z_u)
            p_target = idec.target_distribution(q_u)
            if (ep + 1) % REINIT_INTERVAL == 0:
                if reinit_empty_clusters(idec, z_u, q_u):
                    q_u = idec.get_soft_assignment(z_u)
                    p_target = idec.target_distribution(q_u)
        l = float(loss)
        if l < best:
            best = l
            best_w = [v.numpy() for v in idec.trainable_variables]
        if (ep + 1) % 50 == 0:
            print(f"idec {ep+1}/{JOINT_EPOCHS} total={l:.4f} recon={recon_loss:.4f} "
                  f"kl={kl:.4f} ent={ent:.4f} clusters={len(np.unique(np.argmax(q, 1)))} "
                  f"best={best:.4f}", flush=True)
    if best_w is not None:
        for v, w in zip(idec.trainable_variables, best_w):
            v.assign(w)
    z_f = idec.encoder(X, training=False)
    km = KMeans(n_clusters=idec.n_clusters, random_state=RANDOM_SEED,
                n_init=10, max_iter=300)
    preds = km.fit_predict(z_f.numpy())
    return preds, z_f


def kmeans_silhouette_k(z, k_max, frac=K_SIL_FRAC, step=5, seed=RANDOM_SEED):
    ks = [k for k in range(2, k_max + 1, step) if k < len(z)]
    sil = []
    for k in ks:
        lab = KMeans(k, random_state=seed, n_init=10, max_iter=300).fit_predict(z)
        sil.append(float(silhouette_score(z, lab)))
    sil = np.array(sil)
    s_max = sil.max()
    ok = np.where(sil >= frac * s_max)[0]
    return int(ks[ok[-1]]), np.array(ks), sil


def estimate_k(ae, X):
    z = ae.encoder(X, training=False).numpy()
    k_max = max(5, len(z) // K_MAX_DIV)
    k, _, _ = kmeans_silhouette_k(z, k_max)
    return k

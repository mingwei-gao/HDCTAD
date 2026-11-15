import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EPS = 1e-6
MASK_RATE = 0.15
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
PRETRAIN_EPOCHS = 200
JOINT_EPOCHS = 300
LAMBDA_RECON = 0.5
LAMBDA_ENTROPY = 0.01
UPDATE_INTERVAL = 10
HGNN_LAYERS = 1
HGNN_HIDDEN = 128
TCN_DILATIONS = [1, 2, 4]
TCN_KERNEL = 3
TCN_FILTERS = 64
LATENT_DIM = 32
DECODER_DIMS = [128, 256, 128]
K_SEARCH_MIN = 20
K_SEARCH_MAX = 120
K_SEARCH_STEP = 5
K_SEARCH_FINE = 10
K_GLOBAL_MIN = 10
K_GLOBAL_MAX = 130
RANDOM_SEED = 42


def build_missing_mask(embeddings):
    norms = tf.reduce_sum(tf.abs(embeddings), axis=1)
    return tf.equal(norms, 0.0)


def apply_input_mask(X, missing_mask, mask_rate=None):
    if mask_rate is None:
        mask_rate = MASK_RATE
    N = tf.shape(X)[0]
    normal_indices = tf.where(tf.logical_not(missing_mask))
    n_normal = tf.shape(normal_indices)[0]
    n_mask = tf.maximum(tf.cast(tf.cast(n_normal, tf.float32) * mask_rate, tf.int32), 0)
    random_mask = tf.scatter_nd(tf.random.shuffle(normal_indices)[:n_mask],
                                tf.ones([n_mask], dtype=tf.bool), [N])
    total_mask = tf.logical_or(missing_mask, random_mask)
    masked_X = X * (1.0 - tf.expand_dims(tf.cast(total_mask, tf.float32), 1))
    return masked_X, total_mask


class HypergraphConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, hypergraph_H, dropout=DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        self.H = hypergraph_H
        self.fc = tf.keras.layers.Dense(out_channels)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, X, training=False):
        H = self.H
        if isinstance(H, (list, tuple)):
            deg_v = tf.add_n([tf.sparse.reduce_sum(h, axis=1) for h in H])
            D_v_isqrt = 1.0 / tf.sqrt(deg_v + EPS)
            Xs = X * tf.expand_dims(D_v_isqrt, 1)
            agg = tf.zeros_like(Xs)
            for h in H:
                D_e_inv = 1.0 / (tf.sparse.reduce_sum(h, axis=0) + EPS)
                tmp = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(h), Xs)
                agg += tf.sparse.sparse_dense_matmul(h, tmp * tf.expand_dims(D_e_inv, 1))
        elif isinstance(H, tf.SparseTensor):
            D_e_inv = 1.0 / (tf.sparse.reduce_sum(H, axis=0) + EPS)
            D_v_isqrt = 1.0 / tf.sqrt(tf.sparse.reduce_sum(H, axis=1) + EPS)
            Xs = X * tf.expand_dims(D_v_isqrt, 1)
            tmp = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(H), Xs)
            agg = tf.sparse.sparse_dense_matmul(H, tmp * tf.expand_dims(D_e_inv, 1))
        else:
            D_e_inv = 1.0 / (tf.reduce_sum(H, axis=0) + EPS)
            D_v_isqrt = 1.0 / tf.sqrt(tf.reduce_sum(H, axis=1) + EPS)
            Xs = tf.expand_dims(D_v_isqrt, 1) * X
            agg = tf.matmul(H, tf.expand_dims(D_e_inv, 1) * tf.matmul(tf.transpose(H), Xs))
        out = self.fc(tf.expand_dims(D_v_isqrt, 1) * agg)
        return self.dropout(tf.nn.gelu(out), training=training)


class HGNNEncoder(tf.keras.layers.Layer):
    def __init__(self, hypergraph_H, out_dim=HGNN_HIDDEN,
                 dropout=DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        self.convs = [HypergraphConv(out_dim, hypergraph_H, dropout)
                      for _ in range(HGNN_LAYERS)]

    def call(self, X, training=False):
        h = X
        for conv in self.convs:
            h = conv(h, training=training)
        return h


class TCNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout=DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size,
                                           dilation_rate=dilation_rate, padding="same")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        return x + self.dropout(tf.nn.gelu(self.conv(x)), training=training)


class TCNEncoder(tf.keras.layers.Layer):
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


class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(DECODER_DIMS[0])
        self.fc2 = tf.keras.layers.Dense(DECODER_DIMS[1])
        self.fc3 = tf.keras.layers.Dense(input_dim)

    def call(self, z, training=False):
        return self.fc3(tf.nn.gelu(self.fc2(tf.nn.gelu(self.fc1(z)))))


class HGNN_TCN_Encoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, hypergraph_H, **kwargs):
        super().__init__(**kwargs)
        self.hypergraph_H = hypergraph_H
        self.hgnn = HGNNEncoder(hypergraph_H)
        self.tcn = TCNEncoder()

    def call(self, X, training=False):
        h = self.hgnn(X, training=training)
        z = self.tcn(tf.expand_dims(h, 0), training=training)
        return tf.squeeze(z, 0)


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
            loss_obs = tf.reduce_sum(diff * tf.expand_dims(observed_f, 1)) / (
                tf.reduce_sum(observed_f) * tf.cast(self.input_dim, tf.float32) + EPS)
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


class IDECModel(tf.keras.Model):
    def __init__(self, input_dim, hypergraph_H, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.encoder = HGNN_TCN_Encoder(input_dim, hypergraph_H)
        self.decoder = Decoder(input_dim)
        self.cluster_layer = tf.keras.layers.Dense(n_clusters, use_bias=False)
        self.cluster_layer.build([None, LATENT_DIM])

    def get_soft_assignment(self, z):
        centers = tf.transpose(self.cluster_layer.weights[0])
        dist = tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - tf.expand_dims(centers, 0)), 2)
        q = tf.pow(1.0 / (1.0 + dist), 2.0)
        return q / (tf.reduce_sum(q, axis=1, keepdims=True) + EPS)

    def target_distribution(self, q):
        weight = tf.pow(q, 2) / tf.reduce_sum(q, axis=0, keepdims=True)
        return weight / (tf.reduce_sum(weight, axis=1, keepdims=True) + EPS)

    def entropy_loss(self, q):
        p = tf.reduce_mean(q, axis=0)
        return tf.reduce_sum(p * tf.math.log(p + EPS))

    def init_cluster_centers(self, X, missing_mask):
        masked_X, _ = apply_input_mask(X, missing_mask)
        z = self.encoder(masked_X, training=False).numpy()
        km = KMeans(n_clusters=self.n_clusters, random_state=RANDOM_SEED, n_init=10, max_iter=200)
        km.fit(z)
        self.cluster_layer.set_weights([tf.transpose(tf.convert_to_tensor(km.cluster_centers_, tf.float32))])


def pretrain_ae(model, X, missing_mask):
    lr = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, 50, 0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    for ep in range(PRETRAIN_EPOCHS):
        loss, l_obs, l_mask = model.pretrain_step(X, missing_mask, opt)
        if (ep + 1) % 20 == 0:
            print(f"pretrain {ep+1}/{PRETRAIN_EPOCHS} loss={loss:.4f}", flush=True)
    return model


def train_idec(idec, X, missing_mask):
    lr = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, 100, 0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    kl_loss_fn = tf.keras.losses.KLDivergence()
    masked_X, _ = apply_input_mask(X, missing_mask)
    q = idec.get_soft_assignment(idec.encoder(masked_X, training=False))
    p_target = idec.target_distribution(q)
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
            print(f"idec {ep+1}/{JOINT_EPOCHS} loss={loss:.4f} "
                  f"clusters={len(np.unique(np.argmax(q, 1)))}", flush=True)
    z_f = idec.encoder(apply_input_mask(X, missing_mask)[0], training=False)
    preds = tf.argmax(idec.get_soft_assignment(z_f), 1).numpy()
    return preds, z_f


def estimate_k(ae, X, missing_mask):
    masked_X, _ = apply_input_mask(X, missing_mask)
    z = ae.encoder(masked_X, training=False).numpy()
    best_score, best_k = -np.inf, K_SEARCH_MIN
    for k in range(K_SEARCH_MIN, K_SEARCH_MAX + 1, K_SEARCH_STEP):
        preds = KMeans(k, random_state=RANDOM_SEED, n_init=10, max_iter=300).fit_predict(z)
        sil = silhouette_score(z, preds)
        if sil > best_score:
            best_score, best_k = sil, k
    lo = max(K_GLOBAL_MIN, best_k - K_SEARCH_FINE)
    hi = min(K_GLOBAL_MAX, best_k + K_SEARCH_FINE)
    for k in range(lo, hi + 1):
        preds = KMeans(k, random_state=RANDOM_SEED, n_init=10, max_iter=300).fit_predict(z)
        sil = silhouette_score(z, preds)
        if sil > best_score:
            best_score, best_k = sil, k
    return best_k

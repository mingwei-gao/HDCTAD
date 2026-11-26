import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, vstack as sp_vstack
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import embed
import model
import postprocess as P


def load_sim(path):
    s = np.loadtxt(path, dtype=np.float32)
    th = np.clip(max(np.mean(s) + np.std(s), np.percentile(s, 90)), 0.3, 0.7)
    s[s < th] = 0.0
    return s


def align(n, path):
    d = np.loadtxt(path, dtype=np.float32)
    ids, emb = d[:, 0].astype(np.int32), d[:, 1:]
    out = np.zeros((n, emb.shape[1]), dtype=np.float32)
    for i, b in enumerate(ids):
        if 0 <= b - 1 < n:
            out[b - 1] = emb[i]
    return out


def build_H(h):
    coo = h.tocoo()
    return tf.sparse.reorder(tf.SparseTensor(
        np.array([coo.row, coo.col], np.int64).T,
        np.ones_like(coo.data, dtype=np.float32), coo.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ve", default=None,
                    help="hypergraph VE matrix (bin x hyperedge, value = read support)")
    ap.add_argument("--features", default=None,
                    help="existing embedding directory, skip embedding stage")
    ap.add_argument("--chr", type=int, required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--hic", default=None, help="raw Hi-C matrix, enables refinement and nesting")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--q", type=float, default=0.5)
    a = ap.parse_args()
    c, r = a.chr, a.res
    tag = "p%s_q%s" % (int(a.p) if a.p == int(a.p) else a.p,
                       int(a.q) if a.q == int(a.q) else a.q)

    feat_dir = a.features or os.path.join(a.out, "features")
    if a.ve:
        embed.run_embedding(a.ve, feat_dir, r, a.p, a.q)
    elif not os.path.exists(os.path.join(feat_dir, "hypergraph.npz")):
        raise FileNotFoundError("need --ve or a features directory with hypergraph.npz")

    sim = load_sim(os.path.join(feat_dir, "optimize_node_similarity_%s.txt" % tag))
    hg = embed.load_sparse_npz(os.path.join(feat_dir, "hypergraph.npz"))
    hg = hg.maximum(0).astype(np.float32)
    if hg.shape[0] != sim.shape[0]:
        hg = sp_vstack([hg, csr_matrix((sim.shape[0] - hg.shape[0], hg.shape[1]))])
    emb = align(sim.shape[0], os.path.join(feat_dir, "hypergraph_node_embeddings_%s.txt" % tag))

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    np.random.seed(model.RANDOM_SEED)
    tf.random.set_seed(model.RANDOM_SEED)

    missing = model.build_missing_mask(tf.convert_to_tensor(emb, tf.float32))
    X = tf.convert_to_tensor(StandardScaler().fit_transform(emb), tf.float32)
    H = build_H(hg)
    w_e = model.edge_weights(hg)

    ae = model.MaskedAutoEncoder(emb.shape[1], H, w_e)
    _ = ae(X, missing, training=False)
    ae = model.pretrain_ae(ae, X, missing)
    k = max(5, model.estimate_k(ae, X))
    print("k=%d" % k, flush=True)

    idec = model.IDECModel(emb.shape[1], H, k, w_e)
    for w, v in zip(idec.encoder.trainable_variables, ae.encoder.trainable_variables):
        w.assign(v)
    for w, v in zip(idec.decoder.trainable_variables, ae.decoder.trainable_variables):
        w.assign(v)
    idec.init_cluster_centers(X)
    preds, _ = model.train_idec(idec, X, missing)

    t0 = P.to_shared(P.extract_tads(preds))
    merged = P.merge_short(t0)
    hic = np.loadtxt(a.hic, dtype=np.float32) if a.hic else None
    if hic is not None:
        leaves = np.array(P.to_shared(P.refine_sequential(
            hic, np.array(P.to_adjacent(merged), np.int64))), np.int64)
        nested, types = P.build_nested(leaves, hic)
    else:
        leaves = np.array(merged, np.int64)
        nested, types = leaves, np.array(["leaf"] * len(leaves), dtype=object)

    out_path = os.path.join(a.out, "nested_tads.txt")
    os.makedirs(a.out, exist_ok=True)
    P.save_nested(out_path, nested, types)
    print("saved %s (%d TADs, k=%d)" % (out_path, len(nested), k), flush=True)


if __name__ == "__main__":
    main()

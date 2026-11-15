import argparse
import os
import resource
import time

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sp_vstack
from sklearn.preprocessing import StandardScaler

import model
import postprocess
import walk

HIC = "/mnt/sde/gmw/GM12878/Hi-C/KR/25KB/chr{c}_25KB.txt"
H_CHUNK_COLS = 500_000
MIN_CLUSTERS = 5
MAX_CLUSTERS = 250


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return "GPU" if gpus else "CPU"


def load_similarity(path):
    sim = np.loadtxt(path, dtype=np.float32)
    thresh = np.clip(max(np.mean(sim) + np.std(sim), np.percentile(sim, 90)), 0.3, 0.7)
    sim[sim < thresh] = 0.0
    return sim


def load_hypergraph(path):
    z = np.load(path)
    m = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=z["shape"])
    z.close()
    m = m.maximum(0).astype(np.float32)
    m.eliminate_zeros()
    return m


def align_embeddings(sim, path):
    data = np.loadtxt(path, dtype=np.float32)
    bin_ids, embeds = data[:, 0].astype(np.int32), data[:, 1:]
    base_num = sim.shape[0]
    aligned = np.zeros((base_num, embeds.shape[1]), dtype=np.float32)
    for i, bid in enumerate(bin_ids):
        idx = bid - 1
        if 0 <= idx < base_num:
            aligned[idx] = embeds[i]
    return aligned


def build_H(hypergraph):
    n_cols = hypergraph.shape[1]
    if n_cols > H_CHUNK_COLS:
        H = []
        for start in range(0, n_cols, H_CHUNK_COLS):
            blk = hypergraph[:, start:start + H_CHUNK_COLS].tocoo()
            H.append(tf.sparse.reorder(tf.SparseTensor(
                np.array([blk.row, blk.col], np.int64).T,
                np.ones_like(blk.data, dtype=np.float32), blk.shape)))
        return H
    coo = hypergraph.tocoo()
    H = tf.SparseTensor(np.array([coo.row, coo.col], np.int64).T,
                        np.ones_like(coo.data, dtype=np.float32), coo.shape)
    return tf.sparse.reorder(H)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--res", type=int, default=25)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=walk.EMBED_DIM)
    parser.add_argument("--sparse-input", default=None)
    parser.add_argument("--output-base", default=None)
    parser.add_argument("--no-walk", action="store_true")
    args = parser.parse_args()
    if args.output_base:
        walk.OUTPUT_BASE = args.output_base
    device = setup_gpu()
    np.random.seed(model.RANDOM_SEED)
    tf.random.set_seed(model.RANDOM_SEED)
    chr_name = f"chr{args.chr}"
    tag = f"p{int(args.p) if args.p == int(args.p) else args.p}_" \
          f"q{int(args.q) if args.q == int(args.q) else args.q}"
    sim_path = walk.similarity_path(args.chr, args.res, tag)
    emb_path = walk.embed_path(args.chr, args.res, tag)
    hg_path = walk.npz_path(args.chr, args.res)

    t_start = time.perf_counter()
    if not args.no_walk:
        walk.run(args.chr, args.res, args.p, args.q, args.embed_dim, args.sparse_input)
    t_walk = time.perf_counter()

    sim = load_similarity(sim_path)
    hypergraph = load_hypergraph(hg_path)
    embeds = align_embeddings(sim, emb_path)
    if hypergraph.shape[0] != sim.shape[0]:
        pad_n = sim.shape[0] - hypergraph.shape[0]
        hypergraph = sp_vstack([hypergraph, csr_matrix((pad_n, hypergraph.shape[1]))])
    missing_mask = model.build_missing_mask(tf.convert_to_tensor(embeds, tf.float32))
    X = tf.convert_to_tensor(StandardScaler().fit_transform(embeds), tf.float32)
    H = build_H(hypergraph)
    t_load = time.perf_counter()

    ae = model.MaskedAutoEncoder(embeds.shape[1], H)
    _ = ae(X, missing_mask, training=False)
    ae = model.pretrain_ae(ae, X, missing_mask)
    n_clusters = max(MIN_CLUSTERS, min(model.estimate_k(ae, X, missing_mask), MAX_CLUSTERS))
    idec = model.IDECModel(embeds.shape[1], H, n_clusters)
    for w_i, w_a in zip(idec.encoder.trainable_variables, ae.encoder.trainable_variables):
        w_i.assign(w_a)
    for w_i, w_a in zip(idec.decoder.trainable_variables, ae.decoder.trainable_variables):
        w_i.assign(w_a)
    idec.init_cluster_centers(X, missing_mask)
    preds, _ = model.train_idec(idec, X, missing_mask)
    t_model = time.perf_counter()

    tads = postprocess.extract_tads(preds)
    tads = postprocess.filter_tads(tads, sim)
    tads_1based = [(s + 1, e + 1) for s, e in tads]
    hic = np.loadtxt(HIC.format(c=args.chr), dtype=np.float32)
    final = postprocess.refine_sequential(hic, tads_1based)
    out_path = os.path.join(walk.chr_output_dir(args.chr, args.res),
                            "result", "TAD.txt")
    postprocess.save_tads(out_path, final)
    t_end = time.perf_counter()

    print(f"device={device} chr={chr_name} res={args.res} k={n_clusters}", flush=True)
    print(f"walk={t_walk - t_start:.1f}s load={t_load - t_walk:.1f}s "
          f"model={t_model - t_load:.1f}s post={t_end - t_model:.1f}s "
          f"total={t_end - t_start:.1f}s", flush=True)
    print(f"tad_raw={len(tads)} tad_final={len(final)} peak_rss={rss_mb():.0f}MB",
          flush=True)
    print(f"out={out_path}", flush=True)


if __name__ == "__main__":
    main()

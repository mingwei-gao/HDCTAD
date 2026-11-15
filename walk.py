import os
import time
import argparse
import warnings
import numpy as np
from scipy.sparse import csr_matrix, issparse
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

INPUT_DIR = "/mnt/sde/gmw/GM12878/hypergraph"
OUTPUT_BASE = "/mnt/sdb/gao/final-experience/output"
RES_TO_WALK_LENGTH = {25: 80, 50: 40, 100: 20}
WALKS_PER_NODE = 8
EMBED_DIM = 128
W2V_WINDOW = 3
W2V_EPOCHS = 10
W2V_WORKERS = 4
RANDOM_SEED = 42


def input_matrix_path(chr_num, res):
    return os.path.join(INPUT_DIR, f"chr{chr_num}_{res}_VE_matrix.txt")


def chr_output_dir(chr_num, res):
    return os.path.join(OUTPUT_BASE, f"res{res}", f"chr{chr_num}")


def embed_path(chr_num, res, tag):
    return os.path.join(chr_output_dir(chr_num, res), f"hypergraph_node_embeddings_{tag}.txt")


def similarity_path(chr_num, res, tag):
    return os.path.join(chr_output_dir(chr_num, res), f"optimize_node_similarity_{tag}.txt")


def npz_path(chr_num, res):
    return os.path.join(chr_output_dir(chr_num, res), "hypergraph.npz")


def load_sparse_npz(path):
    d = np.load(path)
    return csr_matrix((d["data"].astype(np.float32), d["indices"], d["indptr"]),
                      shape=tuple(d["shape"]))


def csr_from_dense_txt(path):
    rows, cols, vals = [], [], []
    ncol = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            arr = np.fromstring(line, dtype=np.float32, sep=" ")
            ncol = max(ncol, arr.size)
            nz = np.flatnonzero(arr)
            rows.append(np.full(nz.size, i, dtype=np.int32))
            cols.append(nz.astype(np.int32))
            vals.append(arr[nz])
    if not rows:
        return csr_matrix((0, 0), dtype=np.float32)
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(rows), ncol), dtype=np.float32)


def load_matrix(matrix_path, sparse_path=None):
    if sparse_path and os.path.exists(sparse_path):
        return load_sparse_npz(sparse_path)
    return csr_from_dense_txt(matrix_path)


class HypergraphRandomWalk:
    def __init__(self, matrix, p, q):
        self.matrix = matrix.tocsc() if issparse(matrix) else np.array(matrix, dtype=np.float32)
        self.num_nodes = self.matrix.shape[0]
        self.num_edges = self.matrix.shape[1]
        self.p = p
        self.q = q
        self.edge_nodes = []
        self.edge_weight = np.zeros(self.num_edges, dtype=np.float32)
        self.edge_size = np.zeros(self.num_edges, dtype=np.int32)
        if issparse(self.matrix):
            indptr, indices, data = self.matrix.indptr, self.matrix.indices, self.matrix.data
            for e in range(self.num_edges):
                s, t = indptr[e], indptr[e + 1]
                self.edge_nodes.append(set((indices[s:t] + 1).tolist()))
                if t > s:
                    self.edge_weight[e] = data[s]
                self.edge_size[e] = t - s
        else:
            for e in range(self.num_edges):
                nodes = set(np.where(self.matrix[:, e] > 0)[0] + 1)
                self.edge_nodes.append(nodes)
                if nodes:
                    self.edge_weight[e] = self.matrix[next(iter(nodes)) - 1, e]
                self.edge_size[e] = len(nodes)
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
        self.first_order = csr_matrix((vals, (rows, cols)),
                                      shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        self.first_order.sort_indices()

    def step_probs(self, prev, current):
        row = self.first_order[current - 1]
        idx, data = row.indices, row.data
        if len(idx) == 0:
            return np.array([1.0], np.float32), np.array([current - 1])
        bias = np.full(len(idx), 1.0 / self.q, dtype=np.float32)
        pidx = self.first_order[prev - 1].indices
        if len(pidx):
            pos = np.clip(np.searchsorted(pidx, idx), 0, len(pidx) - 1)
            bias[pidx[pos] == idx] = 1.0
        bias[idx == prev - 1] = 1.0 / self.p
        probs = data * bias
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs, idx

    def walk(self, start, length):
        path = [str(start)]
        if length > 1:
            row = self.first_order[start - 1]
            if len(row.indices) == 0:
                path.append(str(start))
            else:
                d = row.data / row.data.sum()
                path.append(str(np.random.choice(row.indices, p=d) + 1))
        while len(path) < length:
            prev, curr = int(path[-2]), int(path[-1])
            probs, idx = self.step_probs(prev, curr)
            if len(idx) == 0 or probs.sum() == 0:
                path.append(str(curr))
            else:
                path.append(str(np.random.choice(idx, p=probs) + 1))
        return path


def train_embedding(matrix, p, q, num_walks, walk_length, embed_dim):
    hrw = HypergraphRandomWalk(matrix, p, q)
    t0 = time.time()
    walks = [hrw.walk(np.random.randint(1, hrw.num_nodes + 1), walk_length)
             for _ in range(num_walks)]
    print(f"walk {num_walks}x{walk_length} {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    model = Word2Vec(sentences=walks, vector_size=embed_dim, window=W2V_WINDOW,
                     min_count=1, sg=1, workers=W2V_WORKERS, epochs=W2V_EPOCHS)
    print(f"word2vec {time.time()-t0:.1f}s", flush=True)
    embeddings = {}
    for i in range(1, hrw.num_nodes + 1):
        key = str(i)
        if key in model.wv:
            embeddings[i] = model.wv[key]
    return embeddings, model, hrw.num_nodes


def save_similarity(model, num_nodes, save_path):
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


def save_embeddings(model, save_path):
    temp = save_path + ".tmp"
    model.wv.save_word2vec_format(temp)
    with open(temp) as f:
        next(f)
        lines = sorted([l.strip() for l in f if l.strip()], key=lambda x: int(x.split()[0]))
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.remove(temp)


def save_hypergraph_npz(matrix, save_path):
    sparse = csr_matrix(matrix.astype(np.float32))
    sparse.eliminate_zeros()
    np.savez(save_path, data=sparse.data, indices=sparse.indices,
             indptr=sparse.indptr, shape=sparse.shape)


def run(chr_num, res, p, q, embed_dim=EMBED_DIM, sparse_input=None):
    tag = f"p{int(p) if p == int(p) else p}_q{int(q) if q == int(q) else q}"
    walk_length = RES_TO_WALK_LENGTH.get(res, 40)
    matrix_path = input_matrix_path(chr_num, res)
    out_dir = chr_output_dir(chr_num, res)
    os.makedirs(out_dir, exist_ok=True)
    matrix = load_matrix(matrix_path, sparse_input)
    print(f"matrix {matrix.shape} nnz={matrix.nnz}", flush=True)
    num_walks = WALKS_PER_NODE * matrix.shape[0]
    embeddings, model, num_nodes = train_embedding(
        matrix, p, q, num_walks, walk_length, embed_dim)
    save_embeddings(model, embed_path(chr_num, res, tag))
    save_similarity(model, num_nodes, similarity_path(chr_num, res, tag))
    save_hypergraph_npz(matrix, npz_path(chr_num, res))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--res", type=int, default=25)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--dim", type=int, default=EMBED_DIM)
    parser.add_argument("--sparse-input", default=None)
    parser.add_argument("--output-base", default=None)
    args = parser.parse_args()
    if args.output_base:
        global OUTPUT_BASE
        OUTPUT_BASE = args.output_base
    np.random.seed(RANDOM_SEED)
    run(args.chr, args.res, args.p, args.q, args.dim, args.sparse_input)


if __name__ == "__main__":
    main()

import os

import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix

EMBED_DIM = 128
W2V_WINDOW = 3
W2V_EPOCHS = 10
W2V_WORKERS = 4
WALKS_PER_NODE = 8
RES_TO_WALK_LENGTH = {25: 80, 50: 40, 100: 20}
RANDOM_SEED = 42


def ve_to_npz(ve_path, npz_path):
    rows, cols, vals = [], [], []
    n_col = 0
    with open(ve_path) as f:
        for i, line in enumerate(f):
            v = np.fromstring(line, sep=" ")
            nz = np.nonzero(v)[0]
            rows.append(np.full(len(nz), i, np.int64))
            cols.append(nz.astype(np.int64))
            vals.append(v[nz].astype(np.float32))
            n_col = max(n_col, len(v))
    m = csr_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(i + 1, n_col))
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, data=m.data, indices=m.indices,
                        indptr=m.indptr, shape=np.array(m.shape))
    return m


def load_sparse_npz(path):
    d = np.load(path)
    return csr_matrix((d["data"].astype(np.float32), d["indices"], d["indptr"]),
                      shape=tuple(d["shape"]))


class HypergraphRandomWalk:
    def __init__(self, matrix, p, q):
        self.matrix = matrix.tocsc()
        self.num_nodes = self.matrix.shape[0]
        self.num_edges = self.matrix.shape[1]
        self.p = p
        self.q = q
        self.edge_nodes = []
        self.edge_weight = np.zeros(self.num_edges, dtype=np.float32)
        self.edge_size = np.zeros(self.num_edges, dtype=np.int32)
        indptr, indices, data = self.matrix.indptr, self.matrix.indices, self.matrix.data
        for e in range(self.num_edges):
            s, t = indptr[e], indptr[e + 1]
            self.edge_nodes.append(set((indices[s:t] + 1).tolist()))
            if t > s:
                self.edge_weight[e] = data[s]
            self.edge_size[e] = t - s
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
                                      shape=(self.num_nodes, self.num_nodes),
                                      dtype=np.float32)
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
    walks = [hrw.walk(np.random.randint(1, hrw.num_nodes + 1), walk_length)
             for _ in range(num_walks)]
    model = Word2Vec(sentences=walks, vector_size=embed_dim, window=W2V_WINDOW,
                     min_count=1, sg=1, workers=W2V_WORKERS, epochs=W2V_EPOCHS)
    embeddings = {}
    for i in range(1, hrw.num_nodes + 1):
        key = str(i)
        if key in model.wv:
            embeddings[i] = model.wv[key]
    return embeddings, model, hrw.num_nodes


def save_embeddings(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    model.wv.save_word2vec_format(temp)
    with open(temp) as f:
        next(f)
        lines = sorted([l.strip() for l in f if l.strip()],
                       key=lambda x: int(x.split()[0]))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.remove(temp)


def save_similarity(model, num_nodes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    np.savetxt(path, sim, fmt="%.4f")


def save_hypergraph_npz(matrix, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sparse = csr_matrix(matrix.astype(np.float32))
    sparse.eliminate_zeros()
    np.savez(path, data=sparse.data, indices=sparse.indices,
             indptr=sparse.indptr, shape=sparse.shape)


def run_embedding(ve_path, out_dir, res, p=2.0, q=0.5, embed_dim=EMBED_DIM):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "hypergraph.npz")
    matrix = ve_to_npz(ve_path, npz_path)
    walk_length = RES_TO_WALK_LENGTH.get(res, 40)
    num_walks = WALKS_PER_NODE * matrix.shape[0]
    embeddings, model, num_nodes = train_embedding(
        matrix, p, q, num_walks, walk_length, embed_dim)
    tag = f"p{int(p) if p == int(p) else p}_q{int(q) if q == int(q) else q}"
    save_embeddings(model, os.path.join(out_dir, f"hypergraph_node_embeddings_{tag}.txt"))
    save_similarity(model, num_nodes,
                    os.path.join(out_dir, f"optimize_node_similarity_{tag}.txt"))
    save_hypergraph_npz(matrix, npz_path)
    return out_dir

# HDCTAD Inference Pipeline

The pipeline from hypergraph matrix to TADs is implemented in four files:
`walk.py`, `model.py`, `postprocess.py`, and `run.py`.
One run processes **one chromosome at one resolution** and writes one TAD file.

## 1. Environment

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10.20 | Runtime |
| numpy | 1.26.4 (**must be < 2.0**) | Matrix operations; `walk.py` uses the deprecated `np.fromstring`, which fails on numpy 2.x |
| scipy | 1.15.3 | Sparse matrices, Mann–Whitney test |
| scikit-learn | 1.7.2 | KMeans, silhouette, scaling, cosine distance |
| tensorflow | 2.17.0 | HGNN / TCN / MAE / IDEC |
| gensim | 4.4.0 | Word2Vec node embeddings |

GPU is optional. If a GPU is available it is used automatically; otherwise the
pipeline falls back to CPU (`setup_gpu()` in `run.py`).

Install:

```bash
conda create -n hdctad python=3.10
conda activate hdctad
pip install "numpy<2" scipy scikit-learn tensorflow==2.17.0 gensim==4.4.0
```

## 2. Input data

| Data | Path | Description |
|---|---|---|
| Hypergraph incidence matrix | `/mnt/sde/gmw/GM12878/hypergraph/chr{N}_{res}_VE_matrix.txt` | Dense text: rows = bins, columns = hyperedges |
| Hypergraph sparse matrix (optional) | same name with `.npz` | Pass via `--sparse-input` to skip parsing the dense text |
| Hi-C matrix | `/mnt/sde/gmw/GM12878/Hi-C/KR/25KB/chr{N}_25KB.txt` | KR-normalized, space-separated dense matrix |

These two paths are hard-coded in `walk.py:11` and `run.py:16`; change them if
your data lives elsewhere.

## 3. File overview

### `walk.py` (236 lines) — random walk and node embeddings

- Loads the hypergraph incidence matrix, builds within-hyperedge node
  transitions, runs biased second-order random walks (p/q), and trains node
  vectors with Word2Vec.
- Has its own entry point and can be run standalone.
- Writes to `output/res{res}/chr{N}/`:

| File | Content |
|---|---|
| `hypergraph_node_embeddings_p2_q0.5.txt` | One line per bin: bin ID + 128-d vector |
| `optimize_node_similarity_p2_q0.5.txt` | Pairwise cosine similarity matrix |
| `hypergraph.npz` | Hypergraph in CSR format |

Key parameters (module top): `WALKS_PER_NODE=8`, `W2V_WINDOW=3`,
`EMBED_DIM=128`, `W2V_EPOCHS=10`.
Walk length depends on resolution: 25 kb → 80 steps, 50 kb → 40, 100 kb → 20.

### `model.py` (267 lines) — model definition

- Provides classes and training functions only; no entry point. Called by `run.py`.
- Components: hypergraph convolution (HGNN) → temporal convolution
  (TCN, dilations 1/2/4) → masked autoencoder (MAE, 200 epochs) → deep
  embedded clustering (IDEC, 300 epochs).
- Input: node embedding matrix, hypergraph matrix, missing mask.
- Output: a cluster label per bin (`preds`) and latent vectors (`z`); the number
  of clusters is selected automatically by `estimate_k` using silhouette score.

### `run.py` (149 lines) — main entry point

Calls the modules above in order; this is the only file that needs to be run:

```
walk.py -> AE pretraining -> cluster-count estimation -> IDEC clustering
        -> postprocess -> save TADs
```

Output: `output/res{res}/chr{N}/result/TAD.txt`

File format:

```
Start	End
10	61
62	100
...
```

Each line holds two integers: the start bin and the end bin of one TAD.

## 4. Run command

Full command:

```bash
cd /mnt/sdb/gao/final-experience
python run.py \
  --chr 20 \
  --res 25 \
  --p 2.0 \
  --q 0.5 \
  --embed-dim 128 \
  --output-base /mnt/sdb/gao/final-experience/output
```

Parameters:

| Parameter | Default | Required | Description |
|---|---|---|---|
| `--chr` | none | yes | Chromosome number, e.g. 20 |
| `--res` | 25 | no | Resolution in kb: 25 / 50 / 100 |
| `--p` | 2.0 | no | Random-walk return parameter |
| `--q` | 0.5 | no | Random-walk in-out parameter |
| `--embed-dim` | 128 | no | Node embedding dimension |
| `--sparse-input` | none | no | Path to the hypergraph `.npz`; skips dense-text parsing |
| `--output-base` | `final-experience/output` | no | Output root directory |
| `--no-walk` | off | no | Skip the walk; reuse existing embeddings and similarity files |

Output: `<output-base>/res{res}/chr{chr}/result/TAD.txt`

## 5. Output directory structure

```
output/
└── res25/
    └── chr20/
        ├── hypergraph_node_embeddings_p2_q0.5.txt
        ├── optimize_node_similarity_p2_q0.5.txt
        ├── hypergraph.npz
        └── result/
            └── TAD.txt
```

## 6. Notes

1. **The Hi-C path in `run.py:16` is hard-coded to 25 KB.** For `--res 50` or
   `--res 100`, change it to the matching matrix; otherwise refinement uses the
   wrong data.
2. **The hypergraph directory in `walk.py:11` is hard-coded to GM12878.** Change
   it for another cell line.
3. **numpy must be < 2.0**, otherwise `np.fromstring` raises an error.
4. **Large matrices**: a 25 kb dense text file can reach several GB (chr1 is
   about 7.8 GB). Convert it to `.npz` and use `--sparse-input`.
5. **No batch mode**: one chromosome and one resolution per run; write your own
   loop for a whole genome.
6. **Random seed is fixed at 42** (`run.py` and `model.py`), so repeated runs on
   the same input are reproducible.

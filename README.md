HDCTAD: a topologically associating domain detection method based on high-order chromatin interactions and deep embedded clustering

## Requirements

```
python=3.10.20
tensorflow=2.17.0      # GPU version, requires CUDA 12.x + cuDNN 9
numpy=1.26.4
scipy=1.15.3
scikit-learn=1.7.2
gensim=4.4.0
pandas=2.3.3
```

Install the dependencies:

```bash
pip install tensorflow==2.17.0 numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 gensim==4.4.0 pandas==2.3.3
```

## Usage

Inputs and arguments:

| Argument | Description |
|---|---|
| `--ve` | Hypergraph VE matrix text; each row is a bin, each column is a hyperedge, and nonzero values are the read support |
| `--hic` | Raw Hi-C contact matrix (optional); when given, boundary refinement and nesting are performed |
| `--chr` | Chromosome number (integer) |
| `--res` | Resolution in kb, one of 25 / 50 / 100; determines the random walk length |
| `--out` | Output directory |
| `--p` / `--q` | Random walk return and in-out parameters, default 2.0 / 0.5 |
| `--features` | Directory of precomputed embeddings (containing `hypergraph.npz` etc.); skips the embedding stage |

Run one chromosome starting from the hypergraph VE matrix:

```bash
python run.py --chr 20 --res 100 \
  --ve /path/chr20_100_VE_matrix.txt \
  --hic /path/chr20_100KB.txt \
  --out /path/out/chr20
```

Reuse precomputed embeddings (skip the random walk):

```bash
python run.py --chr 20 --res 100 \
  --features /path/res100/chr20 \
  --hic /path/chr20_100KB.txt \
  --out /path/out/chr20
```

## Output

```
<out>/
├── features/                        # generated when --ve is used
│   ├── hypergraph.npz                              # weighted hypergraph (CSR)
│   ├── hypergraph_node_embeddings_p2_q0.5.txt      # node embeddings, first column is the bin index
│   └── optimize_node_similarity_p2_q0.5.txt        # cosine similarity matrix
└── nested_tads.txt                  # final output: nested TADs
```

`nested_tads.txt` is tab-separated with three columns:

```
# nested TAD:  [Start, End]; Type = leaf|parent
Start	End	Type
```

## Reproduced results

Model parameters are fixed in the constant block at the top of `model.py`: random seed 42, 500 pretraining epochs, 1000 joint-training epochs, KL weight 0.1, latent dimension 64, `k_max = n//5`, silhouette threshold `0.98·S_max`, random walk p=2 / q=0.5, embedding dimension 128. The random seed is fixed, so repeated runs on the same input give identical results.

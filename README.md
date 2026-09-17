## Environment Dependencies

```
tensorflow
numpy
scipy
scikit-learn
gensim
pandas
seaborn          
matplotlib       
```
## Usage
### Step 1: Generate Node Embeddings via Random Walk
```bash
# Basic usage
python randomwalk.py --chr 22

# Specify resolution and p/q parameters
python randomwalk.py --chr 22 --res 50 --p 2.0 --q 0.5

# Batch processing
for c in 20 21 22; do
    python randomwalk.py --chr $c --res 25
done
```
Argument	               Description	                               Default
--chr	Chromosome number (required)	                    —
--res	         Resolution in KB	                                            25
--p	               Node2Vec return parameter	                    2.0
--q	            Node2Vec in-out parameter	                            0.5
**Input**: Dense hypergraph matrix txt (`chr{N}_{res}_VE_matrix.txt`)
**Output**:
- Node embeddings `hypergraph_node_embeddings_{tag}.txt`
- Similarity matrix `optimize_node_similarity_{tag}.txt`
- Sparse hypergraph `hypergraph.npz` (converted automatically from the dense matrix)



### Step 2: TAD Detection

```bash
python pipeline_refactored.py --chr 22

# Specify resolution
python pipeline_refactored.py --chr 22 --res 50


# Batch processing
for c in 20 21 22; do
    python pipeline_refactored.py --chr $c --res 25
done
```

| Argument | Description                                    | Default |
| -------- | ------------------------------------------ | ------- |
| `--chr`  | Chromosome number (required)      | —       |
| `--res`  | Resolution in KB                                   | 25      |
| `--tag`  | Parameter tag (corresponds to p/q from Step 1) | p2_q0.5 |

**Input**: The three outputs from Step 1 + Hi-C contact matrix

**Output**:

- Raw TAD `result/original_TAD.txt`
- Refined TAD `result/TADs.txt`



### Step 3: Hierarchical Nested Merging

```bash
# Basic usage
python nested_merge.py --chr 22

# Specify resolution and threshold
python nested_merge.py --chr 22 --res 50 --threshold 0.7

# Batch processing
for c in 20 21 22; do
    python nested_merge.py --chr $c --res 25
done
```

| Argument      | Description                       | Default |
| ------------- | --------------------------------- | ------- |
| `--chr`       | Chromosome number (required)      | —       |
| `--res`       | Resolution in KB                  | 25      |
| `--threshold` | Cosine similarity merge threshold | 0.8     |

**Input**: Refined TAD from Step 2 + Hi-C contact matrix

**Output**: Nested TAD `v3_refine_nested/final_tads.txt`



## Full Example

Using chr22 at 25KB resolution as an example:

```bash
# 1. Generate node embeddings
python randomwalk.py --chr 22 --res 25

# 2. TAD detection + refinement
python pipeline_refactored.py --chr 22 --res 25

# 3. Nested merging
python nested_merge.py --chr 22 --res 25

# Final result at:
# /mnt/sde/gmw/tad_project/res25/chr22/v3_refine_nested/final_tads.txt
```



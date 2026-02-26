# Train Generative Models for Perturbation Prediction

Minimal training code for **Diffusion**, **Flow Matching**, and **Schrodinger Bridge** models for single-cell perturbation prediction in PCA latent space.

Given an AnnData `.h5ad` file with control and perturbed cells and gene perturbation embeddings, this code:

1. Applies PCA to reduce gene expression to a low-dimensional latent space
2. Trains a conditioned generative model (DiT backbone) to predict perturbed cell states from control cells
3. Logs training/validation loss and saves checkpoints

## Installation

```bash
# 1. Create environment
uv venv --python=3.11
source .venv/bin/activate

# 2. Install PyTorch (adjust CUDA version as needed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install core dependencies
uv pip install anndata scanpy scikit-learn numpy pandas scipy
uv pip install hydra-core omegaconf
uv pip install einops timm ema-pytorch tqdm transformers

# 4. Install flow_matching (Facebook Research) -- required for Flow and Schrodinger
git clone https://github.com/facebookresearch/flow_matching.git
cd flow_matching && uv pip install -e . && cd ..

# 5. Install conditional-flow-matching (torchcfm) -- required for Schrodinger
git clone https://github.com/atong01/conditional-flow-matching.git
cd conditional-flow-matching && uv pip install -e . && cd ..

# 6. Install SDE/ODE solvers -- required for Schrodinger
uv pip install torchdyn torchsde

# 7. Optional: experiment tracking
uv pip install wandb
```


## Quick Start

```bash
# Flow Matching (default) — random 90/10 gene split
python run.py +experiment=flow \
    data_path=path/to/data.h5ad \
    embeddings_path=path/to/embeddings.pkl \
    cell_line="K-562"

# Diffusion (DDPM)
python run.py +experiment=diffusion \
    data_path=path/to/data.h5ad \
    embeddings_path=path/to/embeddings.pkl

# Schrodinger Bridge
python run.py +experiment=schrodinger \
    data_path=path/to/data.h5ad \
    embeddings_path=path/to/embeddings.pkl

# Override any parameter from CLI
python run.py +experiment=flow \
    data_path=path/to/data.h5ad \
    embeddings_path=path/to/embeddings.pkl \
    training.epochs=50 \
    training.batch_size=128 \
    pca_dim=30 \
    wandb.enabled=true
```



## Data Format

### AnnData `.h5ad`

The `.h5ad` file contains single-cell gene expression data with both **control** (unperturbed) and **perturbed** cells. Each cell has a single-gene perturbation identified by an Ensembl gene ID.

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `.X` | sparse or dense matrix | Gene expression matrix (raw counts). Shape: `(n_cells, n_genes)`. Will be normalized (`scanpy.pp.normalize_total`) and log-transformed (`scanpy.pp.log1p`) before PCA. |
| `.obs["is_control"]` | `bool` | `True` for control (unperturbed) cells, `False` for perturbed cells. |
| `.obs["gene_id"]` | `str` / `category` | Ensembl gene ID of the perturbation applied to this cell (e.g., `"ENSG00000139618"`). For control cells, this should be a placeholder value like `"non-targeting"`. |
| `.obs["cell_line"]` | `str` / `category` | Cell line name (e.g., `"K-562"`). Used to group control cells with their corresponding perturbed cells. Must match the `cell_line` config parameter. |

**Example structure:**

```
AnnData object with n_obs x n_vars = 310049 x 6640
  obs: 'is_control', 'gene_id', 'cell_line', ...

  # is_control  gene_id            cell_line
  # False       ENSG00000145414    K-562        <- perturbed cell (NAF1 knockout)
  # False       ENSG00000169679    K-562        <- perturbed cell (BUB1 knockout)
  # True        non-targeting      K-562        <- control cell
```

The dataset should have many more perturbed cells than controls (e.g., ~300k perturbed vs ~10k controls is typical). Control cells are shared across all perturbation conditions during training.

### Gene Embeddings (`embeddings.pkl`)

A Python pickle file containing a dictionary that maps Ensembl gene IDs to fixed-dimensional embedding vectors. These embeddings condition the generative model on which gene was perturbed. You can prepare such a file using any embedding from [our collection](https://huggingface.co/datasets/genbio-ai/foundation-models-perturbation/tree/main/gene_embeddings).

```python
# Structure:
{
    "ENSG00000000003": [0.123, -0.456, ...],  # list or np.ndarray, length D
    "ENSG00000000005": [0.789, 0.012, ...],
    ...
}
# Type: dict[str, list[float] | np.ndarray]
# All vectors must have the same dimensionality D (e.g., 128).
# Must cover all gene_id values present in the perturbed cells of the adata.
```

The embedding dimensionality should match `dit.z_emb_dim` in the config (default: 128). Any embedding type works (e.g., Gene Ontology, knowledge graph, protein language model).

# Train GNN

Train GNN embeddings on protein interaction graphs using link prediction. 

In particular, this code can be used to reproduce the `STRING GNN` embeddings and variants thereof. 

## Requirements

```bash
pip install torch torch_geometric hydra-core omegaconf wandb tqdm numpy
```

## Data

The graph we used can be found on [HuggingFace](https://huggingface.co/datasets/genbio-ai/foundation-models-perturbation/blob/main/gnn/9606.protein.links.ensembl_900_keep20_adaptive.txt).

This is a filtered version of the original STRINGdb graph from [here](https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz). The filtering logic can be found in `train_gnn/filter_by_score_adaptive.py`. 

## Quick Start

```bash
# Basic run with defaults
python run.py graph_path=/path/to/graph.txt save_dir=/path/to/output

# With custom model settings
python run.py graph_path=/path/to/graph.txt save_dir=/path/to/output \
    model.dim=256 model.num_layers=4 model.layer_type=gcn

# Disable wandb logging
python run.py graph_path=/path/to/graph.txt wandb.enabled=false
```

## Configuration

All settings are in `configs/config.yaml`. Override any setting via command line using Hydra syntax (`key=value`).

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `graph_path` | `graph.txt` | Path to input graph (tab-separated adjacency list) |
| `save_dir` | `saved_models` | Output directory for checkpoints and embeddings |
| `model.dim` | 128 | Embedding dimension |
| `model.num_layers` | 3 | Number of GNN layers |
| `model.layer_type` | `gcn` | GNN type: `gcn` or `gin` |
| `model.dropout` | 0.1 | Dropout rate |
| `training.lr` | 0.001 | Learning rate |
| `training.num_epochs` | 120 | Max epochs |
| `training.patience` | 10 | Early stopping patience |
| `use_weights` | true | Use edge weights (GCN only) |
| `wandb.enabled` | true | Enable W&B logging |
| `wandb.project` | null | W&B project name |

## Input Format

The graph file should be tab-separated with a header row:
```
protein1    protein2    ...    combined_score
ENSP001     ENSP002     ...    850
```

The `combined_score` column (index 9) is used as edge weight when `use_weights=true`.

## Output

Training creates the following in `save_dir`:
```
<run_name>/
├── config.yaml           # Saved configuration
├── best_model.pt         # Best model checkpoint
└── embeddings/
    ├── final_embeddings.npz
    ├── layer0_embeddings.npz
    ├── layer1_embeddings.npz
    └── ...
```

Embeddings are keyed by protein ID (e.g., `ENSP00000123456`).

## Debug Mode

For quick overfitting tests without train/val split:
```bash
python run.py debug_mode=true graph_path=/path/to/graph.txt
```

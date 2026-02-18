# Cross-Context Model Training and Evaluation

This code trains STATE models and baselines on the **Tahoe** and **Essential** datasets. Note that this code focuses on the **cross-context** (i.e., unseen cell line) setting. All other experiments in our paper concern the unseen perturbation setting.

## Overview

### Models
Five models are available in this code: 
1. **STATE** from [Adduri et al. 2025](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2). Trained from scratch.
2. **MLP (Continuous)**: Uses continuous embeddings of perturbations as input.
3. **MLP (One-Hot)**: Uses fixed random embeddings for each perturbation, which is equivalent to a one-hot encoding followed by a random projection matrix. 
4. **Train Mean (Context Mean)**: Predicts the average response to all perturbations in the target cell line.
5. **Cross-Context kNN**: Predicts based on the average effect of a perturbation in the most similar training cell lines.

See the paper for details.

### Datasets
Two datasets are available in this code: 
1.  **Tahoe**: Small molecule perturbations for 50 cell lines. 
2.  **Essential**: CRISPR gene knockouts for 4 cell lines.

## Repository Structure and Setup

We use modified versions of two libraries from the Arc Institute:
* `cell-load` (https://github.com/ArcInstitute/cell-load)
* `state` (https://github.com/ArcInstitute/state) 

### Code Snapshots

To ensure reproducibility and transparency, we provide snapshots of the `cell-load` and `state` libraries.
Inside each library folder (`state/` and `cell-load/`), you will find two archives:
- `code_original.zip`: A snapshot of the unmodified repository as originally cloned.
    - `cell-load`: Commit `1702978bef2c4f5bca938e1a782f7a21a1f46950` (2025-09-21).
    - `state`: Commit `681111b12c302db0b6542b90d0991b92bcf1150d` (2025-09-24).
- `code_modified.zip`: The modified version of the library used for the experiments in our paper.
    - `cell-load` is adapted to support continuous perturbation embeddings alongside one-hot encodings.
    - `state` is extended with additional baselines, Hydra configs, and training/inference utilities.

### Extracting and Inspecting the Code
1. **Extract the modified repositories:**
   Navigate into each directory and extract the modified code archive.
   ```bash
   cd state
   unzip code_modified.zip
   
   cd ../cell-load
   unzip code_modified.zip
   cd ..
   ```

2. **(Optional) Review the changes:**
   To see exactly what was modified compared to the original repositories, you can extract the original snapshots and run a recursive diff within each folder. For example, to identify the changes we made to the `state` library:
   ```bash
   cd state
   unzip code_original.zip -d original_snapshot
   diff -r original_snapshot .
   ```
   Here we assume `code_modified.zip` has already been extracted. 

### Environment Setup
First, create and activate a virtual environment with Python 3.12:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

If you don't have `uv` installed, you can follow the installation instructions here: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/). Afterwards, ensure that both the modified `state` package and the customized `cell-load` library are installed.

```bash
cd state
uv pip install -e .

cd ../cell-load
uv pip install -e .
```

### Directory Layout
Extracting `state/code_modified.zip` creates the `scripts/`, `src/`, `tomls/`, and `data/` directories. The `data/` directory initially contains HVG metadata and drug embeddings. Users must manually download the raw datasets, run the preprocessing pipeline, and place the resulting preprocessed datasets into this folder. The final structure should look like this:

```text
.
├── src/state/       # (Code) The modified STATE Python package
├── scripts/         # (Code) Entrypoints for training and inference
│   ├── paths.py     # Central configuration for file paths
│   └── ...
├── tomls/           # (Config) Pre-generated TOML configuration files
└── data/            # (Data) Partially populated folder
    ├── tahoe/       # (User-Added) Tahoe .h5ad files
    ├── essential/   # (User-Added) Essential .h5ad files
    ├── var_dims.pkl # (Included) HVG metadata
    └── compound_embeddings.h5ad # (Included) Drug embeddings
```

#### Data Directory (`data/`)
The `data/` directory is provided with the code and contains `var_dims.pkl` (HVG metadata) and `compound_embeddings.h5ad` (Drug embeddings).

**Action Required:** Download the Tahoe and Essential datasets and place them in the `data/` directory as follows:

*   **Tahoe Data:** Create `data/tahoe/` and add `Tahoe100M_{cell_line}.h5ad` files.
    *   *Source:* See the accompanying paper for data availability and preprocessing details.
*   **Essential Data:** Create `data/essential/` and add `replogle.h5ad`.
    *   *Source:* See the [STATE Authors' Colab notebook](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN) for preprocessing details. We created a local copy of this notebook on January 23, 2026, and included it in the repository. It can be found in `cross_context/notebooks/`.

#### Path Configuration (`scripts/paths.py`)
All file paths are defined centrally in `scripts/paths.py`. If you choose to store `data` in a different location, you must update this file.

## Step 0: Initialization Checklist

Before running any experiments, verify the following:

1.  **Environment:** You have successfully installed `state` and `cell-load` (which installs all Python dependencies like PyTorch and Scanpy).
2.  **Data:** The `data/` directory exists and contains:
    *   Subdirectories `tahoe/` and `essential/` with their respective `.h5ad` files.
    *   `var_dims.pkl`.
    *   `compound_embeddings.h5ad`.
3.  **Config:** `scripts/paths.py` points to the correct location of the `data` folder.

## Step 1: Data Preparation

Before training, we must augment the `.h5ad` files with perturbation embeddings and prepare the Essential dataset test files.

### 1. Add Embeddings
Continuous perturbation embeddings must be added to the `.h5ad` files.

**Tahoe (Small Molecules)**

```bash
python -m scripts.embedding.add_dose_aware_embeddings \
    data/tahoe \
    --compound-embeddings data/compound_embeddings.h5ad \
    --embedding-key chatgpt \
    --compute-both
```
*This adds `chatgpt_scaled_pca_dose` (continuous) and `onehot_proj_dose` (one-hot) embeddings to `obsm`. See the paper for details.*

**Essential (Gene Knockouts)**

```bash
python -m scripts.embedding.add_gene_embeddings \
    data/essential \
    --embedding-key wavegc \
    --compute-both
```
*This adds `wavegc` (continuous) and `onehot_proj` (one-hot) embeddings to `obsm`. See the paper for details.*

### 2. Essential Dataset: Extract Test Sets and Filter

**Extract Test Sets**
The Essential dataset must be separated into cell-line-specific test sets for the leave-one-out cross-validation.
```bash
python -m scripts.data.extract_replogle_test_sets
```

**Filter for MLP (Continuous)**
WaveGC embeddings are not available for all genes.
1.  **Training**: We filter the training data to exclude perturbations without WaveGC embeddings. This generates a filtered dataset `replogle_train.h5ad` used specifically for the continuous MLP.
    ```bash
    python -m scripts.data.filter_replogle_train
    ```
2.  **Inference**: For perturbations in the test set that lack WaveGC embeddings, the MLP cannot generate a prediction. In these cases, the inference script (`scripts/inference/run_mlp_inference.py`) falls back to predicting the **mean response** observed for that perturbation in the training cell lines.

## Step 2: Training Models

We train STATE from scratch using hyperparameters derived from the authors' released configurations. All training scripts utilize pre-generated TOML configuration files located in `tomls/tahoe/cross_context/` and `tomls/essential/cross_context/` to define data splits.

### Tahoe Dataset

**1. Train STATE Model**
Hyperparameters are adapted from the [ST-Tahoe config](https://huggingface.co/arcinstitute/ST-Tahoe/blob/main/config.yaml).

```bash
python -m scripts.training.train_st_tahoe \
    --toml tomls/tahoe/cross_context/cross_context_st+hvg.toml
```

**2. Train Baselines**
All baseline models (MLP, kNN, Context Mean) are trained using this script. To run a different baseline, simply specify the corresponding model-specific TOML file.

```bash
# MLP (Continuous)
python -m scripts.training.train_baseline_tahoe \
    --toml tomls/tahoe/cross_context/cross_context_mlpcontinuous+hvg.toml

# MLP (One-Hot)
python -m scripts.training.train_baseline_tahoe \
    --toml tomls/tahoe/cross_context/cross_context_mlponehot+hvg.toml

# Context Mean
python -m scripts.training.train_baseline_tahoe \
    --toml tomls/tahoe/cross_context/cross_context_contextmean+hvg.toml

# Cross-Context kNN
python -m scripts.training.train_baseline_tahoe \
    --toml tomls/tahoe/cross_context/cross_context_knn+hvg.toml
```

### Essential Dataset

For the Essential dataset, we use a leave-one-out cross-validation strategy. We train four separate models, where each of the following cell lines serves as the held-out test set once: `hepg2`, `k562`, `rpe1`, and `jurkat`.

**1. Train STATE Model**
Hyperparameters are adapted from the [STATE Authors' Colab Notebook](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN). We created a local copy of this notebook on January 23, 2026, and included it in the repository. It can be found in `cross_context/notebooks/`.

Run the training for each `cell_line` (i.e., one of `hepg2`, `k562`, `rpe1`, `jurkat`):

```bash
python -m scripts.training.train_st_essential \
    --toml tomls/essential/cross_context/{cell_line}/cross_context_st+hvg.toml
```

**2. Train Baselines**
Repeat for each `cell_line`.

```bash
# MLP (Continuous)
python -m scripts.training.train_baseline_essential \
    --toml tomls/essential/cross_context/{cell_line}/cross_context_mlpcontinuous+hvg.toml

# MLP (One-Hot)
python -m scripts.training.train_baseline_essential \
    --toml tomls/essential/cross_context/{cell_line}/cross_context_mlponehot+hvg.toml

# Context Mean
python -m scripts.training.train_baseline_essential \
    --toml tomls/essential/cross_context/{cell_line}/cross_context_contextmean+hvg.toml

# Cross-Context kNN
python -m scripts.training.train_baseline_essential \
    --toml tomls/essential/cross_context/{cell_line}/cross_context_knn+hvg.toml
```

## Step 3: Inference and Evaluation

After training, we run inference on held-out cell lines using `last.ckpt` (the checkpoint with the lowest validation loss) and evaluate performance using [`cell-eval`](https://github.com/ArcInstitute/cell-eval).

### STATE Model Inference

To run inference for the STATE model:

```bash
python -m scripts.inference.run_st_inference \
    --dataset tahoe
```

Arguments:
- `--dataset`: `tahoe` or `essential`.
- `--cell-lines`: Specific cell lines to process (optional, defaults to all test cell lines).

### Baseline Models Inference

To run inference for baseline models (MLP, kNN, Context Mean):

```bash
python -m scripts.inference.baseline_inference_coordinator \
    --dataset tahoe \
    --models mlpcontinuous mlponehot contextmean knn
```

Arguments:
- `--dataset`: `tahoe` or `essential`.
- `--models`: List of models to run (`mlpcontinuous`, `mlponehot`, `contextmean`, `knn`).
- `--cell-lines`: Specific cell lines to process (optional, defaults to all test cell lines).

Both scripts automatically run `cell-eval` after generating predictions. Results are saved in `results/{dataset}/{model_name}`.

**Note**: The kNN model may produce negative gene expression values. It might be necessary to fix these negative counts prior to evaluation using the provided utility script:
```bash
python scripts/data/fix_negative_counts.py --directory /path/to/knn/predictions
```

## Output Structure

Training artifacts are saved to `state/cross_context/{dataset}/{model_name}`.

For Tahoe:
```
state/cross_context/tahoe/
├── cross_context_st+hvg/
│   ├── checkpoints/
│   │   └── last.ckpt  
│   ├── config.yaml
│   └── ...
```

For Essential (per cell line):
```
state/cross_context/essential/
├── cross_context_st+hvg_hepg2/
├── cross_context_st+hvg_k562/
└── ...
```

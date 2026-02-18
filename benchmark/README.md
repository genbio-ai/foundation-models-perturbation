# benchmark

This folder contains code to reproduce the embedding benchmarking results from our paper. 

## Dependencies
### Installation with uv

Use the [uv](https://docs.astral.sh/uv/) package manager.

```bash
cd benchmark

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate

# If you don't install in editable mode you will need to set the BENCHMARK_DATA_ROOT environment variable because the default data paths are defined relative to the paths.py module
uv pip install -e .

# Install PyTorch for GPU-accelerated models
uv pip install -e ".[torch]"

# Install development dependencies (testing, linting)
uv pip install -e ".[dev]"
```

## Data Setup


### Preprocessed Data 

Download the data from [HuggingFace](https://huggingface.co/datasets/genbio-ai/foundation-models-perturbation) and place it into `data/` so that you get something that looks like this:
```
benchmark
├── benchmark/
├── data/
├───── essential/
├───── gene_embeddings/
├───── norman/
├───── sciplex/
├───── tahoe/
├── pyproject.toml
└── README.md
```

If you want to keep the data in a different directory, set the `BENCHMARK_DATA_ROOT` environment variable to point to your data directory:

```bash
export BENCHMARK_DATA_ROOT=/path/to/your/data
```

Having this data allows you to run the benchmarking scripts. We also provide the code to reproduce our data preprocessing. We do not provide the raw data files directly but they can be retrieved as described below.

### Raw Data Sources

#### Tahoe

The data is available at `gs://arc-ctc-tahoe100/2025-02-25/h5ad/`. To run with our script you must resplit this data by cell line instead of by plate and put in the `data/tahoe/raw/` directory.

#### Essential

The original h5ad files can be found at:
* [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667#/) for Jurkat and HepG2
* [Figshare](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387#/) for K562 and RPE1.
To be compatible with our script all data should be merged into a single h5ad at `data/essential/raw_data.h5ad`.

#### Sciplex

The data can be found on the [NCBI website](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378#/).

#### Norman

The data files can be found at the following links: 
* [GEARS_perturb_processed.h5ad](https://dataverse.harvard.edu/api/access/datafile/6154020)
* [NormanWeissman2019_filtered.h5ad](https://zenodo.org/records/13350497)

We use the original Norman data which contains all metadata including GEM groups, but in order to be consistent with GEARS benchmarking, we subset to the same set of genes.


## Running Benchmarks

Benchmark scripts use [Hydra](https://hydra.cc/) for configuration. Use `--multirun` to sweep over all parameter combinations defined in the config.

### From the benchmark script directory

```bash
cd benchmark/benchmark

# Essential DEG
python bench_essential_deg.py --config-name config_essential_deg --multirun
python bench_essential_deg.py --config-name config_essential_deg_baseline --multirun

# Essential LFC
python bench_essential_lfc.py --config-name config_essential_lfc --multirun
python bench_essential_lfc.py --config-name config_essential_lfc_baseline --multirun

# Norman LFC
python bench_norman_lfc.py --config-name config_norman_lfc --multirun
python bench_norman_lfc.py --config-name config_norman_lfc_baseline --multirun

# SciPlex DEG
python bench_sciplex_deg.py --config-name config_sciplex_deg --multirun
python bench_sciplex_deg.py --config-name config_sciplex_deg_baseline --multirun

# SciPlex LFC
python bench_sciplex_lfc.py --config-name config_sciplex_lfc --multirun
python bench_sciplex_lfc.py --config-name config_sciplex_lfc_baseline --multirun

# Tahoe DEG
python bench_tahoe_deg.py --config-name config_tahoe_deg --multirun
python bench_tahoe_deg.py --config-name config_tahoe_deg_baseline --multirun

# Tahoe LFC
python bench_tahoe_lfc.py --config-name config_tahoe_lfc --multirun
python bench_tahoe_lfc.py --config-name config_tahoe_lfc_baseline --multirun
```

### Running a Single Configuration

Override parameters directly on the command line:

```bash
python bench_tahoe_deg.py cell_line=CVCL_0023 fold_id=0 emb_name=random
```

### Parallelization

Hydra supports parallel execution via launchers. Install a launcher plugin:

```bash
uv pip install hydra-joblib-launcher
```

Then run with:

```bash
python bench_tahoe_deg.py --config-name config_tahoe_deg --multirun hydra/launcher=joblib
```

### Configuration

Configs are in `benchmark/benchmark/config/{dataset}/`. Each config defines:

- `model`: Model hyperparameters (e.g., regularization strength `C`)
- `estimator_name`: Model type (`logistic_regression`, `knn`, `lasso`, `no_change`, `prior`, `most_frequent`)
- `task_name`: The benchmark task identifier
- `hydra.sweeper.params`: Parameters to sweep over (fold IDs, cell lines, embeddings)

### Output

Results are saved as JSON files to `submissions/{dataset}/{fold}/{timestamp}.json` containing:

- Evaluation metrics (F1, precision, recall for DEG; L2, MSE, Spearman for LFC)
- Model name and description
- Timestamp and configuration details

## Preprocessing

The scripts to aggregate single-cell data into learning tasks can be found in the `preprocessing` subpackage. You will need to download the raw single cell data which can take up a lot of memory.

### Pseudobulk Aggregation

Aggregate single-cell data to pseudobulk representations:

```bash
cd benchmark/preprocessing/pseudobulk

python pseudobulk_essential.py
python pseudobulk_tahoe.py
python pseudobulk_sciplex3.py
python pseudobulk_norman.py
```

### Differentially Expressed Genes (DEG)

Compute differentially expressed genes for each dataset:

```bash
cd benchmark/preprocessing/differentially_expressed_genes

python deg_essential.py
python deg_tahoe.py
python deg_sciplex3.py
```

## Experimental Error

The scripts to compute experimental error bounds via bootstrapping can be found in the `experimental_error` subpackage. You will need to download the raw single cell data which can take up a lot of memory.

```bash
cd benchmark/experimental_error

# Essential dataset
python run_exp_error_essential.py
python run_exp_error_essential_deg.py

# Tahoe dataset (specify cell line)
python run_exp_error_tahoe_deg.py --dataset-name Tahoe100M_CVCL_0023

# SciPlex dataset (specify cell line)
python run_exp_error_sciplex_deg.py

# Plate-matched analysis
python run_exp_error_plate_matched.py
```

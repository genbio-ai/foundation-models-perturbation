import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from benchmark import paths
from benchmark.preprocessing.pseudobulk import utils

LOGGER = logging.getLogger(__name__)

# GEARS: download via https://dataverse.harvard.edu/api/access/datafile/6154020
# But this was specifically taken from: https://github.com/snap-stanford/GEARS/tree/master
# scperturb: download via https://zenodo.org/record/7041849/files/NormanWeissman2019_filtered.h5ad

## Preprocessing Steps
"""
1. Load the two files 
2. Create the filtered anndata
3. pseudobulk
4. add splits
"""


def create_single_cell_dataset(gears_adata_path, scperturb_adata_path):
    # Load datasets
    scperturb_adata = sc.read_h5ad(scperturb_adata_path)
    gears_adata = sc.read_h5ad(gears_adata_path)

    # Filter scperturb_adata to genes in gears_adata
    # Match gears_adata.var.index (ensembl IDs) to scperturb_adata.var["ensemble_id"]
    common_genes = scperturb_adata.var[
        scperturb_adata.var["ensemble_id"].isin(gears_adata.var.index)
    ]
    scperturb_filtered = scperturb_adata[:, common_genes.index].copy()

    # Filter to single perturbations and controls
    scperturb_filtered = scperturb_filtered[scperturb_filtered.obs["nperts"] <= 1]

    # Build quick symbol --> ensembl lookup
    symbol_to_ensembl = scperturb_adata.var["ensemble_id"]
    symbol_to_ensembl.index = scperturb_adata.var.index  # gene symbols as index

    # Hardcoded mappings for genes not found in the automatic lookup
    manual_mappings = {
        "C19orf26": "ENSG00000099625",
        "KIAA1804": "ENSG00000143674",
        "C3orf72": "ENSG00000206262",
    }

    # Track original perturbations before mapping (convert to string to avoid categorical issues)
    original_perturbations = scperturb_filtered.obs["perturbation"].astype(str).copy()

    # Map to Ensembl IDs; check manual mappings first, then symbol_to_ensembl, keep original if missing
    scperturb_filtered.obs["perturbation"] = scperturb_filtered.obs[
        "perturbation"
    ].apply(lambda x: manual_mappings.get(x, symbol_to_ensembl.get(x, x)))

    # Find unmapped genes (where symbol stayed the same because not found in mapping)
    mapped_perturbations = scperturb_filtered.obs["perturbation"].astype(str)
    unmapped_mask = original_perturbations == mapped_perturbations
    unmapped_genes = original_perturbations[unmapped_mask].unique()
    # Filter out 'ctrl' or empty strings which are expected to not map
    unmapped_genes = [g for g in unmapped_genes if g and g != "ctrl"]

    if unmapped_genes:
        LOGGER.warning(
            f"Found {len(unmapped_genes)} gene(s) that could not be mapped to Ensembl IDs:"
        )
        for gene in sorted(unmapped_genes):
            LOGGER.warning(f"  - {gene}")
    else:
        LOGGER.info("All perturbation genes successfully mapped to Ensembl IDs")

    # Rename perturbation column to gene_id
    scperturb_filtered.obs.rename(columns={"perturbation": "gene_id"}, inplace=True)

    return scperturb_filtered


def add_splits(adata):
    # Create 5 equal (or near-equal) folds
    np.random.seed(42)
    n_perts = adata.n_obs
    indices = np.random.permutation(n_perts)

    # Split indices into 5 equal-sized groups
    fold_arrays = np.array_split(indices, 5)
    test_split = np.empty(n_perts, dtype=int)
    for fold_id, fold_indices in enumerate(fold_arrays):
        test_split[fold_indices] = fold_id

    # Add test_split column
    adata.obs["test_split"] = test_split.astype(str)

    # Print split distribution
    print("\nSplit distribution:")
    print(adata.obs["test_split"].value_counts().sort_index())

    return adata


def pseudobulk(adata_raw_counts: ad.AnnData) -> ad.AnnData:
    # Normalize and log-transform
    LOGGER.info("Preprocessing")
    sc.pp.normalize_total(adata_raw_counts, target_sum=1e4)
    sc.pp.log1p(adata_raw_counts)

    LOGGER.info("Calculating average expression")

    # Average by batch (gemgroup)
    pseudobulk_by_batch = utils.average_by(
        adata_raw_counts, by=["gemgroup", "gene_id", "nperts"]
    )

    # Split control and perturbed
    LOGGER.info("Subtracting controls")
    adata_control = pseudobulk_by_batch[pseudobulk_by_batch.obs["nperts"] == 0].copy()
    adata_pert = pseudobulk_by_batch[pseudobulk_by_batch.obs["nperts"] == 1].copy()

    # Subtract batch-matched controls
    for i in range(adata_pert.n_obs):
        gemgroup = adata_pert.obs["gemgroup"].values[i]
        idx = adata_control.obs["gemgroup"] == gemgroup
        adata_pert.X[i] -= adata_control.X[idx, :].squeeze()

    # Average across batches
    adata_pert = utils.average_by(adata_pert, by=["gene_id"])

    return adata_pert


def main():
    LOGGER.info("Loading Norman dataset")

    LOGGER.info("Creating the Single Cell Dataset")
    adata = create_single_cell_dataset(paths.NORMAN_GEARS, paths.NORMAN_SCPERTURB)

    adata.write_h5ad(paths.NORMAN_SINGLE_CELL)

    LOGGER.info("Computing pseudobulk")
    pseudobulk_deltas = pseudobulk(adata)

    LOGGER.info("Adding Splits")
    pseudobulk_deltas = add_splits(pseudobulk_deltas)

    LOGGER.info("Writing data")
    pseudobulk_deltas.write_h5ad(paths.NORMAN_PSEUDOBULK)

    LOGGER.info(f"Saved pseudobulk deltas: {pseudobulk_deltas.shape}")


if __name__ == "__main__":
    import sys

    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    main()

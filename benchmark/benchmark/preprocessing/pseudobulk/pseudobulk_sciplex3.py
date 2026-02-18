import gzip
import logging
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tqdm

from benchmark import paths
from benchmark.preprocessing.pseudobulk import utils

LOGGER = logging.getLogger(__name__)

DOSES_TO_KEEP = {"10000.0"}


def pseudobulk(adata_raw_counts: ad.AnnData) -> ad.AnnData:
    sc.pp.normalize_total(adata_raw_counts, target_sum=1e4)
    sc.pp.log1p(adata_raw_counts)

    # remove rows where key metadata columns are all NaN
    mask_missing = pd.isna(adata_raw_counts.obs["cell_line"])
    adata_raw_counts = adata_raw_counts[~mask_missing].copy()

    for col in ["plate", "cell_line", "dose", "drug"]:
        adata_raw_counts.obs[col] = pd.Categorical(
            adata_raw_counts.obs[col], categories=adata_raw_counts.obs[col].unique()
        )

    # average by batch, cell line, dose, and drug
    pseudobulk_by_batch = utils.average_by(
        adata_raw_counts, by=["plate", "cell_line", "dose", "drug"]
    )

    # split control and perturbed
    adata_control, adata_pert = (
        pseudobulk_by_batch[pseudobulk_by_batch.obs["drug"] == "Vehicle"].copy(),
        pseudobulk_by_batch[pseudobulk_by_batch.obs["drug"] != "Vehicle"].copy(),
    )

    assert all(
        (
            (adata_control.obs["cell_line"] == cl) & (adata_control.obs["plate"] == p)
        ).sum()
        == 1
        for cl, p in zip(adata_pert.obs["cell_line"], adata_pert.obs["plate"])
    ), (
        "Error: Each perturbed sample must have exactly one matching control (same cell_line and plate)."
    )

    # subtract control from perturbed
    for i in range(adata_pert.n_obs):
        idx = adata_control.obs["cell_line"] == adata_pert.obs["cell_line"].values[i]
        idx &= adata_control.obs["plate"] == adata_pert.obs["plate"].values[i]
        adata_pert.X[i] -= adata_control.X[idx, :].squeeze()

    # average across batches
    adata_pert = utils.average_by(adata_pert, by=["cell_line", "dose", "drug"])

    return adata_pert


def load_sciplex_matrix(
    matrix_path: os.PathLike, obs: pd.DataFrame, var: pd.DataFrame
) -> ad.AnnData:
    df = pd.read_csv(matrix_path, sep="\t", header=None)
    df.columns = ["gene", "cell", "count"]

    # convert indices from 1-based to 0-based
    df["gene"] -= 1
    df["cell"] -= 1

    n_cells = len(obs)
    n_genes = len(var)

    # build cells × genes matrix
    X = sp.csr_matrix(
        (df["count"].values, (df["cell"].values, df["gene"].values)),
        shape=(n_cells, n_genes),
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def load_metadata(
    cell_path: os.PathLike, gene_path: os.PathLike
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = pd.read_csv(cell_path, sep=" ")
    gene = pd.read_csv(
        gene_path, sep=" ", header=0, names=["ensembl_gene_id", "gene_name"]
    )
    gene["symbol"] = gene["gene_name"].copy()
    return cell, gene


def load_raw_data(
    matrix_path: os.PathLike = paths.SCIPLEX_MATRIX,
    cell_path: os.PathLike = paths.SCIPLEX_CELL_METADATA,
    gene_path: os.PathLike = paths.SCIPLEX_GENE_METADATA,
) -> ad.AnnData:
    """Load raw sciPlex3 data and return as AnnData object."""
    # load metadata
    cell_meta, gene_meta = load_metadata(cell_path, gene_path)

    # construct the full AnnData
    adata = load_sciplex_matrix(matrix_path, cell_meta, gene_meta)

    adata.obs = adata.obs.rename(
        columns={
            "cell_type": "cell_line",
            "product_name": "drug",
            "plate_oligo": "plate",
        }
    )

    return adata


def main(
    recompute_pseudobulk: bool = False,
    matrix_path: os.PathLike = paths.SCIPLEX_MATRIX,
    cell_path: os.PathLike = paths.SCIPLEX_CELL_METADATA,
    gene_path: os.PathLike = paths.SCIPLEX_GENE_METADATA,
    coding_genes_path: os.PathLike = paths.SCIPLEX_CODING_GENES,
    doses_to_keep: set[str] = DOSES_TO_KEEP,
    pseudobulked_path: os.PathLike = paths.SCIPLEX_PSEUDOBULK,
    write_filtered: os.PathLike = paths.SCIPLEX_PSEUDOBULK_FILTERED,
):
    """This function reads raw sciPlex3 data, pseudobulks and computes perturbation deltas,
    and applies some standard filtering.

    Args:
        recompute_pseudobulk (bool, optional): Whether to recompute pseudobulk deltas or use values
        saved at the path pseudobulked_path. Defaults to False.
        matrix_path (os.PathLike, optional): Path to UMI count matrix. Defaults to MATRIX_PATH.
        cell_path (os.PathLike, optional): Path to cell annotations. Defaults to CELL_PATH.
        gene_path (os.PathLike, optional): Path to gene annotations. Defaults to GENE_PATH.
        coding_genes_path (os.PathLike, optional): text file containing genes to keep for analysis. Defaults
        to CODING_GENES.
        doses_to_keep (set[str], optional): doses to keep in final dataset. Defaults to DOSES_TO_KEEP.
        pseudobulked_path (os.PathLike, optional): path to write (or read) pseudobulked data. Defaults
        to PSEUDOBULKED_DATA.
        write_filtered (os.PathLike, optional): path to write final filtered dataset to. Defaults to WRITE_FILTERED.
    """

    # Get pseudobulked data (can take a while if recompute_pseudobulk==True)
    if recompute_pseudobulk:
        adata_raw = load_raw_data(matrix_path, cell_path, gene_path)
        adata_pert = pseudobulk(adata_raw)
        adata_pert.write_h5ad(pseudobulked_path)

    else:
        adata_pert = ad.read_h5ad(pseudobulked_path)

    LOGGER.info("read pseudobulked")

    # Example filtering
    idx = adata_pert.obs["dose"].isin(doses_to_keep)
    adata_pert = adata_pert[idx]

    coding_genes = pd.read_csv(coding_genes_path, sep=",")
    adata_pert = adata_pert[
        :, adata_pert.var["gene_name"].isin(coding_genes["Gene name"].values)
    ]

    LOGGER.info("writing results")

    adata_pert.write_h5ad(write_filtered)


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    main(recompute_pseudobulk=True)

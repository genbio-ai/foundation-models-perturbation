import ast
from pathlib import Path
from typing import Collection

import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import os

from benchmark import paths

# Essential
ESSENTIAL_DATA_DIR = paths.ESSENTIAL_DIR
ESSENTIAL_H5AD = paths.ESSENTIAL_RAW

# Norman
NORMAN_DATA_DIR = paths.NORMAN_DIR
NORMAN_SINGLE_CELL_DATA = paths.NORMAN_SINGLE_CELL

# Tahoe
TAHOE_DATA_DIR = paths.TAHOE_DIR
TAHOE_H5AD_BY_CELL_LINE = paths.TAHOE_RAW_DATA
TAHOE_CODING_GENES = paths.TAHOE_CODING_GENES

# Sciplex
SCIPLEX_DATA_DIR = paths.SCIPLEX_DIR
SCIPLEX_CELL_PATH = paths.SCIPLEX_CELL_METADATA
SCIPLEX_GENE_PATH = paths.SCIPLEX_GENE_METADATA
SCIPLEX_MATRIX_PATH = paths.SCIPLEX_MATRIX
SCIPLEX_CODING_GENES = paths.SCIPLEX_CODING_GENES


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
    matrix_path: os.PathLike = SCIPLEX_MATRIX_PATH,
    cell_path: os.PathLike = SCIPLEX_CELL_PATH,
    gene_path: os.PathLike = SCIPLEX_GENE_PATH,
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


def get_data(dataset_name: str, get_raw_counts: bool = False):
    """
    Utility to load, format, and normalize (if needed) perturbation data.
    """
    match dataset_name:
        case dataset_name if dataset_name.lower().startswith("essential"):
            # dataset_name is essential_K-562
            cell_line = dataset_name.split("_")[1]
            essential_cell_lines = ["K-562", "Hep-G2", "hTERT-RPE1", "Jurkat"]
            for cell_line_name in essential_cell_lines:
                if cell_line.lower().replace("-", "") == cell_line_name.lower().replace(
                    "-", ""
                ):
                    cell_line = cell_line_name
                    break
            else:
                # if we don't break out of loop raise an error
                raise ValueError(
                    f"{cell_line=} not recognized for essential, must be one of {essential_cell_lines}"
                )

            adata = ad.read_h5ad(ESSENTIAL_H5AD)
            adata = adata[adata.obs["cell_line"] == cell_line].copy()
            obs_out = adata.obs
            obs_out["batch_id"] = obs_out["gem_group"].astype(str)
            obs_out["pert_id"] = adata.obs["gene_id"].astype(str)
            obs_out["gene_id"] = adata.obs["gene_id"].astype(str)
            obs_out["is_control"] = adata.obs["is_control"].astype(bool)


            adata_out = ad.AnnData(X=adata.X, obs=obs_out, var=adata.var)


        case "norman":
            adata = sc.read_h5ad(NORMAN_SINGLE_CELL_DATA)
            obs_out = pd.DataFrame(
                {
                    "batch_id": adata.obs["gemgroup"].astype(str),
                    "pert_id": adata.obs["gene_id"].astype(str),
                    "is_control": (adata.obs["nperts"] == 0).astype(bool),
                }
            )
            adata_out = ad.AnnData(X=adata.X, obs=obs_out, var=adata.var)


        case dataset_name if dataset_name.lower().startswith("tahoe"):
            # expecting database name to look something like Tahoe100M_CVCL_0023
            cell_line = "_".join(dataset_name.split("_")[-2:])
            path = (
                TAHOE_H5AD_BY_CELL_LINE
                / f"{cell_line}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad.gz"
            )
            adata = ad.read_h5ad(path, backed='r')

            adata.obs["dose"] = adata.obs["drugname_drugconc"].map(
                lambda x: ast.literal_eval(x)[0][1]
            )
            adata.obs["drug"] = adata.obs["drug"].map(lambda x: x.strip())
            adata = adata[
                (adata.obs["pass_filter"] == "full")
                & ((adata.obs["dose"] == 0.0) | (adata.obs["dose"] == 5.0))
            ].to_memory()
            coding_genes = [
                line.strip()
                for line in Path(TAHOE_CODING_GENES).read_text().splitlines()
            ]
            assert adata.var.index.isin(coding_genes).sum() == len(coding_genes), (
                "check index is compatible with coding genes"
            )
            adata = adata[:, adata.var.index.isin(coding_genes)]

            obs_out = pd.DataFrame(
                {
                    "batch_id": adata.obs["plate"].astype(str),
                    "pert_id": adata.obs["drug"].astype(str),
                    "is_control": (adata.obs["dose"] == 0).astype(bool),
                    "plate": adata.obs["plate"],
                    "drug": adata.obs["drug"],
                    "drugname_drugconc": adata.obs["drugname_drugconc"],
                    "cell_line": adata.obs["cell_line"],
                }
            )

            # for all datasets we name the control perturbation 'control'
            obs_out["pert_id"] = obs_out["pert_id"].str.replace("DMSO_TF", "control")

            adata_out = ad.AnnData(X=adata.X, obs=obs_out, var=adata.var)


        case dataset_name if dataset_name.lower().startswith("sciplex"):
            _, cell_line = dataset_name.split("_")

            adata_raw = load_raw_data(
                SCIPLEX_MATRIX_PATH, SCIPLEX_CELL_PATH, SCIPLEX_GENE_PATH
            )

            mask_missing = pd.isna(adata_raw.obs["cell_line"])
            adata_raw = adata_raw[
                (~mask_missing)
                & (
                    (adata_raw.obs["dose"] == 10000.0)
                    | (adata_raw.obs["drug"] == "Vehicle")
                )
            ].copy()
            adata_raw = adata_raw[adata_raw.obs["cell_line"] == cell_line]

            coding_genes = pd.read_csv(SCIPLEX_CODING_GENES, sep=",")
            adata = adata_raw[
                :, adata_raw.var["gene_name"].isin(coding_genes["Gene name"].values)
            ]

            obs_out = pd.DataFrame(
                {
                    "batch_id": adata.obs["plate"].astype(str),
                    "pert_id": adata.obs["drug"].astype(str),
                    "is_control": (adata.obs["drug"] == "Vehicle").astype(bool),
                }
            )

            # for all datasets we name the control perturbation 'control'
            obs_out["pert_id"] = obs_out["pert_id"].str.replace("Vehicle", "control")

            adata_out = ad.AnnData(X=adata.X, obs=obs_out, var=adata.var)


        case _:
            raise NotImplementedError(f"Dataset name {dataset_name} not recognized.")

    assert adata_out.shape[0] >= 1

    if not get_raw_counts:
        sc.pp.normalize_total(adata_out, target_sum=int(1e4))
        sc.pp.log1p(adata_out)

    # check for required fields:

    assert "batch_id" in adata_out.obs.columns
    assert "pert_id" in adata_out.obs.columns
    assert "is_control" in adata_out.obs.columns

    return adata_out


def average_by(adata: ad.AnnData, by: str | Collection[str], **kwargs) -> ad.AnnData:
    # Directly copied from Toby's pseudobulking code.
    adata = sc.get.aggregate(adata, by=by, func="mean", **kwargs)
    adata.X = adata.layers["mean"]
    del adata.layers["mean"]
    return adata


def pseudobulk_delta(adata: ad.AnnData) -> ad.AnnData:
    # Adatped from Toby's pseudobulking code.

    # Assumes we have data from a single cell line.

    # firstly, average by batch
    adata_pb = average_by(
        adata,
        by=["pert_id", "is_control", "batch_id"],
    )

    # then average across batches
    adata_pb = average_by(adata_pb, by=["pert_id", "is_control"])

    # split control and perturbed
    adata_control = adata_pb[adata_pb.obs["is_control"].astype(bool)].copy()
    adata_pert = adata_pb[~adata_pb.obs["is_control"].astype(bool)].copy()

    # subtract control from perturbed
    assert adata_control.shape[0] == 1
    adata_pert.X -= adata_control.X[0, :]

    return adata_pert


def bootstrap_csr_mean(X, n_trials, rng):
    """
    Use bootstrapping to estimate the distribution of the mean (over rows) of X.
    Args:
        X: CSR.
        n_trials: Number of bootstrap trials.
        rng: RNG object.

    Returns:
        X_boot: (n_trials, X.shape[1]) ndarray of estimated row means.
    """
    # Switch to dense for faster indexing:
    X = X.toarray()
    # Estimate mean:
    num_rows = X.shape[0]
    idx = rng.integers(num_rows, size=(n_trials, num_rows))
    X_samp = X[idx]  # (n_trials, num_rows, X.shape[1])
    X_boot = X_samp.mean(axis=1)
    return X_boot

import traceback
import logging
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sp

from benchmark import paths

LOGGER = logging.getLogger(__name__)

DOSES_TO_KEEP = {"10000.0"}


def deg_single_cell_line(
    adata_raw_counts: ad.AnnData,
    cell_line: str,
    alpha: float = 0.05,
    min_cells_per_drug_and_plate: int = 20,
    log_dir: None | Path = None,
    log_normalize: bool = True,
    filter_min_cells_per_drug_plate: bool = True,
) -> ad.AnnData:
    """Process DEG for a single cell line."""
    if log_normalize:
        sc.pp.normalize_total(adata_raw_counts, target_sum=1e4)
        sc.pp.log1p(adata_raw_counts)

    t_test_df = []
    if filter_min_cells_per_drug_plate:
        adata_raw_counts = adata_raw_counts[
            adata_raw_counts.obs.groupby(["plate", "drug"]).transform("count")[
                "cell_line"
            ]
            > min_cells_per_drug_and_plate
        ]

    adata_raw_counts.obs["drug_dose"] = (
        adata_raw_counts.obs["drug"].astype(str)
        + "_"
        + adata_raw_counts.obs["dose"].astype(str)
    )

    for (plate,), group in adata_raw_counts.obs.reset_index().groupby(["plate"]):
        t_tested = sc.tl.rank_genes_groups(
            adata_raw_counts[group.index],
            groupby="drug_dose",
            reference="Vehicle_0.0",
            copy=True,
        )
        t_tests = sc.get.rank_genes_groups_df(t_tested, group=None)
        t_tests["plate"] = plate
        t_tests["vote"] = (t_tests["pvals_adj"] < alpha) * np.sign(t_tests["scores"])

        t_test_df.append(t_tests)

    t_tests = pl.from_pandas(pd.concat(t_test_df))

    if "group" not in t_tests.columns:
        groups = adata_raw_counts.obs["drug_dose"].unique().tolist()
        groups.remove("Vehicle_0.0")
        assert len(groups) == 1
        # scanpy drops the "group" column when there's only one comparison group
        t_tests = t_tests.with_columns(group=pl.lit(groups[0]))

    if log_dir is not None:
        t_tests.with_columns(pl.col("vote").cast(pl.Int8)).write_parquet(
            log_dir / f"{cell_line}_before_voting.parquet"
        )

    aggregated_t_tests = (
        # keep only class with most counts
        t_tests.with_columns(
            num_votes=pl.col("vote").count().over("vote", "group", "names")
        )
        .filter(pl.col("num_votes") == pl.col("num_votes").max().over("group", "names"))
        # In case of tie: (0,1 -> 1) (0,-1 -> -1)  (1,-1 -> 0)
        .group_by(["group", "names"])
        .agg(pl.col("vote").sum().sign().cast(pl.Int8))
    )

    vote_counts = t_tests.group_by(["group", "names"]).agg(
        pl.col("vote").sum().sign().cast(pl.Int8),
        not_de=(pl.col("vote") == 0).sum().cast(pl.UInt8),
        up=(pl.col("vote") == 1).sum().cast(pl.UInt8),
        down=(pl.col("vote") == -1).sum().cast(pl.UInt8),
    )

    if log_dir is not None:
        aggregated_t_tests.write_parquet(log_dir / f"{cell_line}.parquet")

    pivotted = aggregated_t_tests.to_pandas().pivot(
        values="vote", columns="names", index="group"
    )
    X = pivotted.to_numpy(dtype=np.int8)

    # Ensure all layers use the same column order as the main matrix
    gene_order = pivotted.columns

    up = (
        vote_counts.pivot(values="up", index="group", on="names")
        .to_pandas()
        .set_index("group")
        .reindex(columns=gene_order)
        .to_numpy()
    )
    down = (
        vote_counts.pivot(values="down", index="group", on="names")
        .to_pandas()
        .set_index("group")
        .reindex(columns=gene_order)
        .to_numpy()
    )
    not_de = (
        vote_counts.pivot(values="not_de", index="group", on="names")
        .to_pandas()
        .set_index("group")
        .reindex(columns=gene_order)
        .to_numpy()
    )

    assert not np.isnan(up).any(), "Found NaN values in 'up' layer"
    assert not np.isnan(down).any(), "Found NaN values in 'down' layer"
    assert not np.isnan(not_de).any(), "Found NaN values in 'not_de' layer"

    obs = pd.DataFrame(pivotted.index)
    obs["dose"] = obs["group"].map(lambda x: x.rsplit("_", 1)[1])
    obs["drug"] = obs["group"].map(lambda x: x.rsplit("_", 1)[0])
    obs["pert_id"] = obs["drug"].copy()
    obs["cell_line"] = cell_line

    # Ensure var matches the gene order in X
    var_reordered = adata_raw_counts.var.reindex(gene_order)
    result_ad = ad.AnnData(X=X, obs=obs, var=var_reordered)
    result_ad.layers["up"] = up
    result_ad.layers["down"] = down
    result_ad.layers["not_de"] = not_de

    return result_ad


def deg(
    adata_raw_counts: ad.AnnData,
    alpha: float = 0.05,
    min_cells_per_drug_and_plate: int = 20,
    log_dir: None | Path = None,
) -> ad.AnnData:
    """Process DEG for all cell lines and combine results."""
    # remove rows where key metadata columns are all NaN
    mask_missing = pd.isna(adata_raw_counts.obs["cell_line"])
    adata_raw_counts = adata_raw_counts[~mask_missing].copy()

    # process each cell line separately
    cell_line_results = []
    for cell_line in adata_raw_counts.obs["cell_line"].unique():
        cell_line_data = adata_raw_counts[
            adata_raw_counts.obs["cell_line"] == cell_line
        ].copy()

        if len(cell_line_data) == 0:
            continue

        result = deg_single_cell_line(
            cell_line_data,
            cell_line,
            alpha=alpha,
            min_cells_per_drug_and_plate=min_cells_per_drug_and_plate,
            log_dir=log_dir,
        )
        cell_line_results.append(result)

    if not cell_line_results:
        raise ValueError("No valid cell line data found")

    # concatenate all results
    combined_result = ad.concat(cell_line_results, join="inner", merge="same")

    return combined_result


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


def compute_deg(
    adata_raw: ad.AnnData,
    coding_genes_path: os.PathLike = paths.SCIPLEX_CODING_GENES,
    log_dir: None | Path = None,
) -> ad.AnnData:
    try:
        coding_genes = pd.read_csv(coding_genes_path, sep=",")
        adata_raw = adata_raw[
            :, adata_raw.var["gene_name"].isin(coding_genes["Gene name"].values)
        ]
        result = deg(adata_raw, log_dir=log_dir)

    except Exception as e:
        print(e)
        if log_dir is not None:
            (log_dir / "sciplex3.log").write_text(traceback.format_exc())

    return result


def main(
    recompute_pseudobulk: bool = False,
    matrix_path: os.PathLike = paths.SCIPLEX_MATRIX,
    cell_path: os.PathLike = paths.SCIPLEX_CELL_METADATA,
    gene_path: os.PathLike = paths.SCIPLEX_GENE_METADATA,
    coding_genes_path: os.PathLike = paths.SCIPLEX_CODING_GENES,
    doses_to_keep: set[str] = DOSES_TO_KEEP,
    deg_path: os.PathLike = paths.SCIPLEX_DEG,
    write_filtered: os.PathLike = paths.SCIPLEX_DEG_FILTERED,
):
    """This function reads raw sciPlex3 data, performs differential expression analysis
    by comparing drug treatments to controls, and applies standard filtering.

    Args:
        recompute_pseudobulk (bool, optional): Whether to recompute pseudobulk deltas or use values
        saved at the path pseudobulked_path. Defaults to False.
        matrix_path (os.PathLike, optional): Path to UMI count matrix. Defaults to MATRIX_PATH.
        cell_path (os.PathLike, optional): Path to cell annotations. Defaults to CELL_PATH.
        gene_path (os.PathLike, optional): Path to gene annotations. Defaults to GENE_PATH.
        coding_genes_path (os.PathLike, optional): text file containing genes to keep for analysis. Defaults
        to CODING_GENES.
        doses_to_keep (set[str], optional): doses to keep in final dataset. Defaults to DOSES_TO_KEEP.
        deg_path (os.PathLike, optional): Path to save/load differential expression results. Defaults to DEG_PATH.
        write_filtered (os.PathLike, optional): path to write final filtered dataset to. Defaults to WRITE_FILTERED.
    """

    # Get pseudobulked data (can take a while if recompute_pseudobulk==True)
    if recompute_pseudobulk:
        adata_raw = load_raw_data(matrix_path, cell_path, gene_path)
        adata_deg = compute_deg(adata_raw, coding_genes_path)
        adata_deg.write_h5ad(deg_path)

    else:
        adata_deg = ad.read_h5ad(deg_path)

    LOGGER.info("read differential expression results")

    # Example filtering
    idx = adata_deg.obs["dose"].isin(doses_to_keep)
    adata_deg = adata_deg[idx]

    LOGGER.info("writing results")

    adata_deg.write_h5ad(write_filtered)


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    main(recompute_pseudobulk=True)

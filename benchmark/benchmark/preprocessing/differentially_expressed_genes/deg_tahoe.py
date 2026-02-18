import traceback
import ast
import logging
import os
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import polars as pl
import scanpy as sc
import tqdm
import numpy as np


from benchmark import paths

LOGGER = logging.getLogger(__name__)

DRUGS_TO_REMOVE = {
    # duplicate of Trametinib?
    "Trametinib (DMSO_TF solvate)",
    # two drugs
    "Sacubitril/Valsartan",
    # no SMILES given
    "Verteporfin",
}

# see below slide regarding cell line quality
# https://docs.google.com/presentation/d/1U7p-krRWM9tEYGsv6RD45xEa8ptgU6ASoBoL-rUC0pA/edit?slide=id.g3745bf3b1eb_1_0#slide=id.g3745bf3b1eb_1_0
CELL_LINES_TO_REMOVE = {
    "CVCL_1571",
    "CVCL_1531",
    "CVCL_1715",
    "CVCL_1577",
    "CVCL_1716",
}
DOSES_TO_KEEP = {5.0}


def deg(
    adata_raw_counts: ad.AnnData,
    alpha: float = 0.05,
    min_cells_per_drug_and_plate: int = 20,
    log_dir: None | Path = None,
) -> ad.AnnData:
    sc.pp.normalize_total(adata_raw_counts, target_sum=1e4)
    sc.pp.log1p(adata_raw_counts)

    t_test_df = []
    adata_raw_counts = adata_raw_counts[
        adata_raw_counts.obs.groupby(["plate", "drug"]).transform("count")["cell_line"]
        > min_cells_per_drug_and_plate
    ]
    for (plate,), group in adata_raw_counts.obs.reset_index().groupby(["plate"]):
        t_tested = sc.tl.rank_genes_groups(
            adata_raw_counts[group.index],
            groupby="drugname_drugconc",
            reference="[('DMSO_TF', 0.0, 'uM')]",
            copy=True,
        )

        t_tests = sc.get.rank_genes_groups_df(t_tested, group=None)
        t_tests["plate"] = plate
        t_tests["vote"] = (t_tests["pvals_adj"] < alpha) * np.sign(t_tests["scores"])

        t_test_df.append(t_tests)

    t_tests = pl.from_pandas(pd.concat(t_test_df))
    if "group" not in t_tests.columns:
        groups = adata_raw_counts.obs["drugname_drugconc"].unique().to_list()
        groups.remove("[('DMSO_TF', 0.0, 'uM')]")
        # scanpy drops group column if only one element because
        t_tests = t_tests.with_columns(group=pl.lit(groups[0]))

    cell_line = adata_raw_counts.obs["cell_line"][0]
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
    obs["dose"] = obs["group"].map(lambda x: ast.literal_eval(x)[0][1])
    obs["drug"] = obs["group"].map(lambda x: ast.literal_eval(x)[0][0]).str.strip()

    # Ensure var matches the gene order in X
    var_reordered = adata_raw_counts.var.reindex(gene_order)
    result_ad = ad.AnnData(X=X, obs=obs, var=var_reordered)
    result_ad.layers["up"] = up
    result_ad.layers["down"] = down
    result_ad.layers["not_de"] = not_de

    return result_ad


def loop_over_tahoe_data(
    tahoe_dir: os.PathLike = paths.TAHOE_RAW_DATA,
    filter_full: bool = True,
    gene_metadata_path: os.PathLike = paths.TAHOE_GENE_METADATA,
    coding_genes_path: os.PathLike = paths.TAHOE_CODING_GENES,
    log_dir: None | Path = None,
) -> ad.AnnData:
    degs = []

    for path in tqdm.tqdm(sorted(list(Path(tahoe_dir).glob("*.h5ad.gz")))):
        try:
            # loop over cell line data to not blow up RAM
            data = ad.read_h5ad(path, backed="r")

            # filter_full corresponds to stricter quality control filtering
            if filter_full:
                filtered_data = data[data.obs["pass_filter"] == "full"]
            else:
                filtered_data = data

            in_memory = filtered_data.to_memory()

            gene_metadata = pd.read_parquet(gene_metadata_path)
            in_memory.var = (
                in_memory.var.merge(
                    gene_metadata, left_on="gene_name", right_on="gene_symbol"
                )
                .rename(
                    {"gene_symbol": "symbol", "ensembl_id": "ensembl_gene_id"}, axis=1
                )
                .drop("token_id", axis=1)
            )
            coding_genes_path = Path(coding_genes_path)
            coding_genes = {
                line.strip() for line in coding_genes_path.read_text().splitlines()
            }
            in_memory = in_memory[:, in_memory.var["symbol"].isin(coding_genes)]
            in_memory.var = in_memory.var.set_index("symbol")

            result = deg(in_memory, log_dir=log_dir)
            result.obs["cell_line"] = in_memory.obs["cell_line"][0]
            cell_line = in_memory.obs["cell_line"][0]

            if log_dir is not None:
                result.write_h5ad(log_dir / f"{cell_line}.h5ad")
            degs.append(result)

        except Exception as e:
            print(e)
            cell_line = in_memory.obs["cell_line"][0]
            if log_dir is not None:
                (log_dir / f"{cell_line}.log").write_text(traceback.format_exc())

    all_degs = ad.concat(degs, join="inner", merge="same")

    return all_degs


def main(
    recompute_pseudobulk: bool = False,
    tahoe_dir: os.PathLike = paths.TAHOE_RAW_DATA,
    deg_path: os.PathLike = paths.TAHOE_DEG,
    filter_full: bool = True,
    gene_metadata_path: os.PathLike = paths.TAHOE_GENE_METADATA,
    coding_genes_path: os.PathLike = paths.TAHOE_CODING_GENES,
    drug_metadata_path: os.PathLike = paths.TAHOE_DRUG_METADATA,
    cell_lines_to_remove: set[str] = CELL_LINES_TO_REMOVE,
    drugs_to_remove: set[str] = DRUGS_TO_REMOVE,
    doses_to_keep: set[str] = DOSES_TO_KEEP,
    write_filtered: os.PathLike = paths.TAHOE_DEG_FILTERED,
    split_path: os.PathLike = paths.TAHOE_SPLIT_PATH,
):
    """This function reads raw Tahoe Data, pseudobulk and computes perturbation deltas according to
    BA-ATE-Batch, and applies some standard filtering.

    Args:
        recompute_pseudobulk (bool, optional): Whether to recompute pseudobulk deltas or use values
        saved at the path pseudobulked_path. Defaults to False.
        tahoe_dir (os.PathLike, optional): Directory with raw Tahoe data. Defaults to TAHOE_RAW_DATA.
        pseudobulked_path (os.PathLike, optional): path to write (or read) pseudobulked data. Defaults
        to PSEUDOBULKED_DATA.
        filter_full (bool, optional): Whether to use the Tahoe full filter or only minimal filter.
        Defaults to True.
        gene_metadata_path (os.PathLike, optional): path to parquet with gene metadata from Tahoe dataset
        see https://huggingface.co/datasets/tahoebio/Tahoe-100M/tree/main/metadata. Defaults to GENE_METADATA.
        coding_genes_path (os.PathLike, optional): text file containing genes to keep for analysis. Defaults
        to CODING_GENES.
        drug_metadata_path (os.PathLike, optional): path to parquet with drug metadata from Tahoe dataset
        see https://huggingface.co/datasets/tahoebio/Tahoe-100M/tree/main/metadata. Defaults to DRUG_METADATA.
        cell_lines_to_remove (set[str], optional): names of cell lines to remove from final dataset. Defaults
        to CELL_LINES_TO_REMOVE.
        drugs_to_remove (set[str], optional): names of drugs to remove from final dataset. Defaults to
        DRUGS_TO_REMOVE.
        doses_to_keep (set[str], optional): doses to keep in final dataset. Defaults to DOSES_TO_KEEP.
        write_filtered (os.PathLike, optional): path to write final filtered dataset to. Defaults to WRITE_FILTERED.
    """

    # Get pseudobulked data (can take a while if recompute_pseudobulk==True)
    if recompute_pseudobulk:
        adata_deg = loop_over_tahoe_data(
            tahoe_dir, filter_full, gene_metadata_path, coding_genes_path
        )

        tahoe_splits = pd.read_csv(split_path)
        tahoe_splits["drug"] = tahoe_splits["drug"].str.strip()

        adata_deg.obs = adata_deg.obs.merge(tahoe_splits, on="drug", how="left")
        adata_deg.write_h5ad(deg_path)

    else:
        adata_deg = ad.read_h5ad(deg_path)

    LOGGER.info("read pseudobulked")

    # Example filtering
    idx = ~adata_deg.obs["cell_line"].isin(cell_lines_to_remove)
    idx &= ~adata_deg.obs["drug"].isin(drugs_to_remove)
    idx &= adata_deg.obs["dose"].isin(doses_to_keep)
    adata_deg = adata_deg[idx]

    drug_metadata = pd.read_parquet(drug_metadata_path)

    adata_deg.obs = adata_deg.obs.merge(drug_metadata, on="drug", how="left")

    LOGGER.info("writing results")

    adata_deg.write_h5ad(write_filtered)


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    main(recompute_pseudobulk=True)

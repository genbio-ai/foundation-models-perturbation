import logging

import anndata as ad
import scanpy as sc
import numpy as np

from benchmark import paths

LOGGER = logging.getLogger(__name__)
CELL_LINES = ["Hep-G2", "Jurkat", "K-562", "hTERT-RPE1"]
MINIMUM_NUM_CELLS = 30


def deg(
    adata_raw_counts: ad.AnnData,
    alpha: float = 0.05,
    minimum_num_cells: int = MINIMUM_NUM_CELLS,
) -> ad.AnnData:
    # Preprocess
    LOGGER.info("Preprocessing")
    sc.pp.normalize_total(adata_raw_counts, target_sum=1e4)
    sc.pp.log1p(adata_raw_counts)

    LOGGER.info("Calculating average expression")

    # cannot perform t-test if only one cell is present
    adata_raw_counts = adata_raw_counts[
        adata_raw_counts.obs.groupby(["gene_id"]).transform("count")["soma_joinid"]
        > minimum_num_cells
    ]

    adata_raw_counts.obs["gene_id"] = adata_raw_counts.obs["gene_id"].astype(str)
    adata_raw_counts.obs["gene"] = adata_raw_counts.obs["gene"].astype(str)

    t_tested = sc.tl.rank_genes_groups(
        adata_raw_counts,
        groupby="gene_id",
        reference="non-targeting",
        copy=True,
    )

    t_tests = sc.get.rank_genes_groups_df(t_tested, group=None)
    t_tests["deg"] = (t_tests["pvals_adj"] < alpha) * np.sign(t_tests["scores"])

    pivotted = t_tests.pivot(values="deg", columns="names", index="group")

    var = adata_raw_counts.var.loc[pivotted.columns.to_list()]
    obs = pivotted.drop(columns=pivotted.columns).merge(
        adata_raw_counts.obs.drop_duplicates("gene")[
            ["gene", "gene_id", "cell_line", "organism"]
        ],
        right_on="gene_id",
        left_index=True,
        how="left",
    )
    obs.index = obs["gene_id"].astype(str) + "_" + obs["cell_line"].astype(str)

    return ad.AnnData(X=pivotted.to_numpy().astype(np.int8), obs=obs, var=var)


def main():
    # Load Essential dataset
    pseudobulk_deltas_list = []

    raw_data = ad.read_h5ad(paths.ESSENTIAL_RAW)
    for cell_line in CELL_LINES:
        LOGGER.info(f"Starting processing of {cell_line}")
        rna = raw_data[raw_data.obs["cell_line"] == cell_line]

        # pseudobulk
        pseudobulk_deltas_list.append(deg(rna, minimum_num_cells=MINIMUM_NUM_CELLS))

    pseudobulk_deltas = ad.concat(pseudobulk_deltas_list, join="inner", merge="same")

    LOGGER.info("Writing data")
    pseudobulk_deltas.write_h5ad(paths.ESSENTIAL_DEG)


if __name__ == "__main__":
    main()

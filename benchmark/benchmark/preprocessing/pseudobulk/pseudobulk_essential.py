import logging
from pathlib import Path
import anndata as ad
import scanpy as sc
from benchmark import paths
from benchmark.preprocessing.pseudobulk import utils

LOGGER = logging.getLogger(__name__)
CELL_LINES = ["Hep-G2", "Jurkat", "K-562", "hTERT-RPE1"]


def pseudobulk(adata_raw_counts: ad.AnnData) -> ad.AnnData:
    # Preprocess
    LOGGER.info("Preprocessing")
    sc.pp.normalize_total(adata_raw_counts, target_sum=1e4)
    sc.pp.log1p(adata_raw_counts)

    LOGGER.info("Calculating average expression")

    # firstly, average by batch
    pseudobulk_rna = utils.average_by(
        adata_raw_counts,
        by=["gene_id", "cell_line", "is_control", "gem_group"],
    )

    # then average across batches
    pseudobulk_rna = utils.average_by(
        pseudobulk_rna, by=["gene_id", "cell_line", "is_control"]
    )

    # split control and perturbed
    LOGGER.info("Subtracting controls")
    adata_control, adata_pert = (
        pseudobulk_rna[pseudobulk_rna.obs["is_control"].astype(bool)].copy(),
        pseudobulk_rna[~pseudobulk_rna.obs["is_control"].astype(bool)].copy(),
    )

    # subtract control from perturbed
    assert adata_control.obs["cell_line"].nunique() == len(
        adata_control.obs["cell_line"]
    )
    for i, row in enumerate(adata_control.obs[["cell_line"]].itertuples()):
        pert_index = adata_pert.obs["cell_line"] == row.cell_line
        adata_pert.X[pert_index] -= adata_control.X[i, :]

    return adata_pert


def main():
    # Load Essential dataset
    raw_data = ad.read_h5ad(paths.ESSENTIAL_RAW)
    pseudobulk_deltas_list = []
    for cell_line in CELL_LINES:
        LOGGER.info(f"Starting processing of {cell_line}")
        rna = raw_data[raw_data.obs["cell_line"] == cell_line]
        # pseudobulk
        pseudobulk_deltas_list.append(pseudobulk(rna))

    pseudobulk_deltas = ad.concat(pseudobulk_deltas_list, join="inner", merge="same")

    LOGGER.info("Writing data")
    pseudobulk_deltas.write_h5ad(paths.ESSENTIAL_PSEUDOBULK)


if __name__ == "__main__":
    main()

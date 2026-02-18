import ast
import logging
import os
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import tqdm

from benchmark import paths
from benchmark.preprocessing.pseudobulk import utils

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


def pseudobulk(adata_raw_counts: ad.AnnData) -> ad.AnnData:
    sc.pp.normalize_total(adata_raw_counts, target_sum=1e4)
    sc.pp.log1p(adata_raw_counts)

    # drugname_drugconc is a string that looks like '[(drugname, concentration, unit)]'
    # unit is always micromolar.
    adata_raw_counts.obs["dose"] = adata_raw_counts.obs["drugname_drugconc"].map(
        lambda x: ast.literal_eval(x)[0][1]
    )
    adata_raw_counts.obs["drug"] = adata_raw_counts.obs["drug"].map(lambda x: x.strip())

    # firstly, average by batch
    pseudobulk_by_batch = utils.average_by(
        adata_raw_counts, by=["plate", "cell_line", "dose", "drug"]
    )

    # split control and perturbed
    adata_control, adata_pert = (
        pseudobulk_by_batch[pseudobulk_by_batch.obs["dose"] == 0].copy(),
        pseudobulk_by_batch[pseudobulk_by_batch.obs["dose"] != 0].copy(),
    )

    # subtract control from perturbed
    for i in range(adata_pert.n_obs):
        idx = adata_control.obs["cell_line"] == adata_pert.obs["cell_line"].values[i]
        idx &= adata_control.obs["plate"] == adata_pert.obs["plate"].values[i]
        adata_pert.X[i] -= adata_control.X[idx, :].squeeze()

    # then average across batches
    adata_pert = utils.average_by(adata_pert, by=["cell_line", "dose", "drug"])

    return adata_pert


def loop_over_tahoe_data(
    tahoe_dir: os.PathLike = paths.TAHOE_RAW_DATA,
    filter_full: bool = True,
) -> ad.AnnData:
    pseudobulks = []

    for path in tqdm.tqdm(list(Path(tahoe_dir).glob("*.h5ad.gz"))):
        # loop over cell line data to not blow up RAM
        data = ad.read_h5ad(path, backed="r")

        # filter_full corresponds to stricter quality control filtering
        if filter_full:
            filtered_data = data[data.obs["pass_filter"] == "full"]
        else:
            filtered_data = data

        in_memory = filtered_data.to_memory()

        pseudobulks.append(pseudobulk(in_memory))

    all_pseudobulks = ad.concat(pseudobulks, join="inner", merge="same")

    return all_pseudobulks


def main(
    recompute_pseudobulk: bool = False,
    tahoe_dir: os.PathLike = paths.TAHOE_RAW_DATA,
    pseudobulked_path: os.PathLike = paths.TAHOE_PSEUDOBULK_BY_BATCH,
    filter_full: bool = True,
    gene_metadata_path: os.PathLike = paths.TAHOE_GENE_METADATA,
    coding_genes_path: os.PathLike = paths.TAHOE_CODING_GENES,
    drug_metadata_path: os.PathLike = paths.TAHOE_DRUG_METADATA,
    cell_lines_to_remove: set[str] = CELL_LINES_TO_REMOVE,
    drugs_to_remove: set[str] = DRUGS_TO_REMOVE,
    doses_to_keep: set[str] = DOSES_TO_KEEP,
    write_filtered: os.PathLike = paths.TAHOE_PSEUDOBULK,
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
        adata_pert = loop_over_tahoe_data(tahoe_dir, filter_full)
        adata_pert.write_h5ad(pseudobulked_path)

    else:
        adata_pert = ad.read_h5ad(pseudobulked_path)

    LOGGER.info("read pseudobulked")

    # Example filtering
    idx = ~adata_pert.obs["cell_line"].isin(cell_lines_to_remove)
    idx &= ~adata_pert.obs["drug"].isin(drugs_to_remove)
    idx &= adata_pert.obs["dose"].isin(doses_to_keep)
    adata_pert = adata_pert[idx]

    # Adding gene metadata
    gene_metadata = pd.read_parquet(gene_metadata_path)
    adata_pert.var = (
        adata_pert.var.merge(gene_metadata, left_on="gene_name", right_on="gene_symbol")
        .rename({"gene_symbol": "symbol", "ensembl_id": "ensembl_gene_id"}, axis=1)
        .drop("token_id", axis=1)
    )

    coding_genes = {line.strip() for line in coding_genes_path.read_text().splitlines()}
    adata_pert = adata_pert[:, adata_pert.var["symbol"].isin(coding_genes)]

    drug_metadata = pd.read_parquet(drug_metadata_path)

    # https://pubchem.ncbi.nlm.nih.gov/compound/5362420#section=SMILES
    # not sure why this isn't in the Tahoe metadata
    # Unfortunately not canonical in the same way as the others, not sure how they are made canonical
    # but it isn't with rdkit.
    # drug_metadata.loc[drug_metadata["drug"] == "Verteporfin", "canonical_smiles"] = (
    #     "CC1=C(C2=CC3=NC(=CC4=C(C(=C(N4)C=C5[C@@]6([C@@H](C(=CC=C6C(=N5)C=C1N2)C(=O)OC)C(=O)OC)C)C)CCC(=O)OC)C(=C3C)CCC(=O)O)C=C"
    # )

    adata_pert.obs = adata_pert.obs.merge(drug_metadata, on="drug", how="left")

    LOGGER.info("writing results")

    adata_pert.write_h5ad(write_filtered)


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    main(recompute_pseudobulk=True)

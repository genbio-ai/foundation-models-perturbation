import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get("BENCHMARK_DATA_ROOT", Path(__file__).parent.parent / "data")
)


# Tahoe
TAHOE_DIR = DATA_ROOT / "tahoe"
TAHOE_RAW_DATA = TAHOE_DIR / "raw"
TAHOE_GENE_METADATA = TAHOE_DIR / "gene_metadata.parquet"
TAHOE_DRUG_METADATA = TAHOE_DIR / "drug_metadata.parquet"
TAHOE_CODING_GENES = TAHOE_DIR / "tahoe_coding_non_mt_genes.txt"
TAHOE_SPLIT_PATH = TAHOE_DIR / "tahoe_splits_drug.csv"
TAHOE_DEG = TAHOE_DIR / "deg_by_batch.h5ad"
TAHOE_DEG_FILTERED = TAHOE_DIR / "tahoe_deg_filtered.h5ad"
TAHOE_PSEUDOBULK_BY_BATCH = TAHOE_DIR / "pseudobulk_by_batch_platematched.h5ad"
TAHOE_PSEUDOBULK = TAHOE_DIR / "tahoe_pseudobulk_pert_platematched.h5ad"
TAHOE_PSEUDOBULK_WITH_SPLITS = (
    TAHOE_DIR / "tahoe_pseudobulk_pert_platematched_withsplits.h5ad"
)
TAHOE_DRUG_EMBEDDINGS = TAHOE_DIR / "tahoe_drug_embeddings.h5ad"

# Essential
ESSENTIAL_DIR = DATA_ROOT / "essential"
ESSENTIAL_RAW = ESSENTIAL_DIR / "raw_data.h5ad"
ESSENTIAL_PSEUDOBULK = ESSENTIAL_DIR / "pseudobulk_deltas.h5ad"
ESSENTIAL_DEG = ESSENTIAL_DIR / "essential_deg.h5ad"
ESSENTIAL_DEG_WITH_SPLIT = ESSENTIAL_DIR / "essential_deg_with_split.h5ad"
ESSENTIAL_PSEUDOBULK_ALL_PERTS = (
    ESSENTIAL_DIR / "essential_pseudobulk_deltas_all_perts.h5ad"
)
ESSENTIAL_CONTROLS_ADATA_PATH = ESSENTIAL_DIR / "essential_pseudobulk_full_processed.h5ad"
ESSENTIAL_CONTROLS_CACHE_PATH = ESSENTIAL_DIR / "cache/control_expressions_essential_pseudobulk.pt"

# sciPlex3
SCIPLEX_DIR = DATA_ROOT / "sciplex"
SCIPLEX_MATRIX = (
    SCIPLEX_DIR / "GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz"
)
SCIPLEX_CELL_METADATA = SCIPLEX_DIR / "GSM4150378_sciPlex3_pData.txt.gz"
SCIPLEX_GENE_METADATA = (
    SCIPLEX_DIR / "GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz"
)
SCIPLEX_CODING_GENES = SCIPLEX_DIR / "mart_export_protein_coding.txt"
SCIPLEX_PSEUDOBULK = SCIPLEX_DIR / "sciplex3_pseudobulk.h5ad"
SCIPLEX_PSEUDOBULK_FILTERED = SCIPLEX_DIR / "sciplex3_pseudobulk_filtered.h5ad"
SCIPLEX_DEG = SCIPLEX_DIR / "sciplex3_deg_by_batch.h5ad"
SCIPLEX_DEG_FILTERED = SCIPLEX_DIR / "sciplex3_deg_filtered.h5ad"
SCIPLEX_DRUG_EMBEDDINGS = SCIPLEX_DIR / "sciplex_drug_embeddings.h5ad"

# Norman
NORMAN_DIR = DATA_ROOT / "norman"
NORMAN_GEARS = NORMAN_DIR / "GEARS_perturb_processed.h5ad"
NORMAN_SCPERTURB = NORMAN_DIR / "NormanWeissman2019_filtered.h5ad"
NORMAN_SINGLE_CELL = NORMAN_DIR / "norman_singlecell_data.h5ad"
NORMAN_PSEUDOBULK = NORMAN_DIR / "norman_pseudobulk_deltas.h5ad"

# gene_embeddings
GENE_EMBEDDINGS = DATA_ROOT / "gene_embeddings"
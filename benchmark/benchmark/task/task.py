import dataclasses
import anndata as ad
from benchmark import paths
import typing

from benchmark.task import regression_evaluate
from pathlib import Path
from datetime import datetime
import json

HERE = Path(__file__).parent
SUBMISSION_DIR = HERE.parent.parent / "submissions"

ESSENTIAL_DEG = paths.ESSENTIAL_DEG_WITH_SPLIT
ESSENTIAL_PSEUDOBULK = paths.ESSENTIAL_PSEUDOBULK_ALL_PERTS
TAHOE_PSEUDOBULK = paths.TAHOE_PSEUDOBULK_WITH_SPLITS
NORMAN_PSEUDOBULK = paths.NORMAN_PSEUDOBULK

TaskName = typing.Literal[
    "expression/pert-prediction-essential-deg",
    "expression/pert-prediction-essential-regression-all-perturbations",
    "expression/pert-prediction-tahoe-regression",
    "expression/pert-prediction-tahoe-deg",
    "expression/pert-prediction-sciplex3-deg",
    "expression/pert-prediction-sciplex3-regression",
    "expression/pert-prediction-norman-regression",
]


@dataclasses.dataclass
class BenchmarkTask:
    name: TaskName
    fold: str
    test: ad.AnnData | None = None

    def setup(self) -> dict[str, ad.AnnData]:
        fold_id = self.fold
        match self.name:
            case name if "tahoe" in name.lower():
                if "." in fold_id:
                    cell_line, fold_id_actual = fold_id.split(".")
                else:
                    raise ValueError(f"Cannot parse fold_id {fold_id}.")

                if name == "expression/pert-prediction-tahoe-deg":
                    anndata_path = paths.TAHOE_DEG_FILTERED

                elif name == "expression/pert-prediction-tahoe-regression":
                    anndata_path = TAHOE_PSEUDOBULK
                else:
                    raise ValueError("Not Recognized")
                adata = ad.read_h5ad(anndata_path)

                CELL_LINES = sorted(adata.obs["cell_line"].unique())
                if cell_line not in CELL_LINES:
                    raise ValueError(
                        f"Tahoe only supports cell lines {CELL_LINES}, got '{cell_line}'"
                    )
                mask_cell_line = adata.obs["cell_line"] == cell_line
                mask_test_fold = adata.obs["fold_" + fold_id_actual] == "test"
                adata.obs = adata.obs.rename(columns={"pubchem_cid": "pert_id"})
                adata.obs = adata.obs[
                    [
                        "pert_id",
                        "drug",
                        "targets",
                        "moa-broad",
                        "moa-fine",
                        "human-approved",
                        "clinical-trials",
                        "gpt-notes-approval",
                        "canonical_smiles",
                    ]
                ]
                out_dict = {
                    "train": adata[~mask_test_fold & mask_cell_line],
                    "test": adata[mask_test_fold & mask_cell_line],
                }

            case name if "essential" in name.lower():
                cell_line, fold_id_actual = fold_id.split(".")

                if name == "expression/pert-prediction-essential-deg":
                    anndata_path = ESSENTIAL_DEG
                    columns_to_keep = ["pert_id"]
                elif (
                    name
                    == "expression/pert-prediction-essential-regression-all-perturbations"
                ):
                    anndata_path = ESSENTIAL_PSEUDOBULK
                    columns_to_keep = ["pert_id", "symbol"]
                else:
                    raise ValueError(f"{self.name=} Not Recognized")
                adata = ad.read_h5ad(anndata_path)

                adata.obs["test_split"] = (
                    adata.obs["test_split"].astype(int).astype(str)
                )
                mask_test_fold = adata.obs["test_split"] == fold_id_actual
                mask_cell_line = adata.obs["cell_line"] == cell_line
                adata.obs = adata.obs.rename(columns={"gene_id": "pert_id"})
                adata.obs = adata.obs[columns_to_keep]
                out_dict = {
                    "train": adata[~mask_test_fold & mask_cell_line],
                    "test": adata[mask_test_fold & mask_cell_line],
                }

            case name if "sciplex" in name.lower():
                if "." in fold_id:
                    cell_line, fold_id_actual = fold_id.split(".")

                else:
                    raise ValueError(f"Cannot parse fold_id {fold_id}.")

                if name == "expression/pert-prediction-sciplex3-deg":
                    anndata_path = paths.SCIPLEX_DEG_FILTERED

                elif name == "expression/pert-prediction-sciplex3-regression":
                    anndata_path = paths.SCIPLEX_PSEUDOBULK_FILTERED

                else:
                    raise ValueError(f"{self.name=} Not Recognized")
                adata = ad.read_h5ad(anndata_path)

                CELL_LINES = sorted(adata.obs["cell_line"].unique())
                if cell_line not in CELL_LINES:
                    raise ValueError(
                        f"sci-Plex 3 only supports cell lines {CELL_LINES}, got '{cell_line}'"
                    )
                mask_cell_line = adata.obs["cell_line"] == cell_line

                adata.obs["pert_id"] = adata.obs["drug"]

                mask_test_fold = adata.obs["fold_" + fold_id_actual] == "test"
                out_dict = {
                    "train": adata[~mask_test_fold & mask_cell_line],
                    "test": adata[mask_test_fold & mask_cell_line],
                }

            case name if "norman" in name.lower():
                cell_line, fold_id_actual = fold_id.split(".")

                local_path = NORMAN_PSEUDOBULK

                adata = ad.read_h5ad(local_path)
                adata.obs["test_split"] = adata.obs["test_split"].astype(str)
                mask_test_fold = adata.obs["test_split"] == fold_id_actual
                adata.obs = adata.obs.rename(columns={"gene_id": "pert_id"})

                # Add symbol column (same as pert_id for Norman)
                adata.obs["symbol"] = adata.obs["pert_id"]

                adata.obs = adata.obs[["pert_id", "symbol"]]
                out_dict = {
                    "train": adata[~mask_test_fold],
                    "test": adata[mask_test_fold],
                }

            case _:
                raise ValueError(f"{self.name=} not recognized")

        self.test = out_dict["test"]
        return out_dict["train"], out_dict["test"]

    def evaluate(self, preds, labels):
        match self.name:
            case name if "deg" in name:
                # lazy import to avoid torch if unnecessary
                from benchmark.task import deg_evaluate

                return deg_evaluate.evaluate(preds, labels)

            case name if "regression" in name:
                return regression_evaluate.evaluate(preds, labels)
            case _:
                raise ValueError("name not recognized")

    def submit(self, preds, name, description=None, submission_dir=SUBMISSION_DIR):
        results = self.evaluate(preds, self.test)
        dataset_normalized = self.name.replace("-", "_")
        timestamp = datetime.now().isoformat()

        full_submission_dir = submission_dir / dataset_normalized / self.fold
        full_submission_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{timestamp.replace(':', '-')}.json"
        filepath = full_submission_dir / filename

        json_dict = {
            "timestamp": timestamp,
            "dataset": dataset_normalized,
            "fold": self.fold,
            "metrics": results,
            "name": name,
            "description": description,
        }
        with open(filepath, "w") as f:
            json.dump(json_dict, f, indent=2)

        return json_dict

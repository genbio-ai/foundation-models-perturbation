import collections

import anndata as ad
import fire
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from benchmark.experimental_error import utils as ut
from benchmark.preprocessing.differentially_expressed_genes import deg_tahoe
import benchmark


def main(
    dataset_name: str,
    n_trials_outer: int = 3,
    n_trials_inner: int = 3,
    seed: int = 5050,
    min_cells_per_drug_and_plate: int = 20,
) -> pd.DataFrame:
    seed_seq = np.random.SeedSequence(seed)
    cell_line = dataset_name.replace("Tahoe100M_", "")

    # Data:
    cell_line_adata = ut.get_data(dataset_name, get_raw_counts=True)

    # filter here so we don't sample plates with too few cell lines
    cell_line_adata = cell_line_adata[
        cell_line_adata.obs.groupby(["plate", "drug"]).transform("count")["cell_line"]
        > min_cells_per_drug_and_plate
    ]
    pert_id_list = sorted(
        cell_line_adata.obs["pert_id"][
            cell_line_adata.obs["is_control"] == False
        ].unique()
    )

    # sample predictions for all perturbations and runs
    sample_predictions: dict[str, list] = collections.defaultdict(list)
    for pert_id in tqdm(pert_id_list[:]):
        plates = cell_line_adata.obs[cell_line_adata.obs["pert_id"] == pert_id][
            "plate"
        ].unique()
        for outer_seed in seed_seq.generate_state(n_trials_outer):
            rng_outer = np.random.default_rng(outer_seed)
            seed_seq_inner = np.random.SeedSequence(outer_seed)
            sampled_plates = rng_outer.choice(plates, size=len(plates))

            adata_parts = []
            for i, plate in enumerate(sampled_plates):
                adata_plate = cell_line_adata[
                    (
                        cell_line_adata.obs["is_control"]
                        | (cell_line_adata.obs["pert_id"] == pert_id)
                    )
                    & (cell_line_adata.obs["batch_id"] == plate)
                ]
                # rename plate so that the same plate looks different
                # even if it is resampled
                # should copy the view
                adata_plate.obs["plate"] = f"plate_{i}"
                adata_parts.append(adata_plate)
            adata = ad.concat(adata_parts)
            adata.obs = adata.obs.reset_index()
            for inner_trial_seed in seed_seq_inner.generate_state(n_trials_inner):
                obs_ids = (
                    adata.obs.groupby(["plate", "pert_id"])
                    .sample(frac=1, replace=True, random_state=inner_trial_seed)
                    .index
                )
                sample_adata = adata[obs_ids].copy()
                sample_adata.obs["gene_id"] = sample_adata.obs["pert_id"]

                differentially_expressed = deg_tahoe.deg(sample_adata)

                sample_predictions[pert_id].append(differentially_expressed)

    # compute metrics for different folds
    aggregated_results = []
    for fold_index in range(5):
        task = benchmark.BenchmarkTask(
            name="expression/pert-prediction-tahoe-deg",
            fold=f"{cell_line}.{fold_index}",
        )
        train, test = task.setup()
        for sample_index in range(len(list(sample_predictions.values())[0])):
            differentially_expressed = ad.concat(
                sample_predictions[pert_id][sample_index]
                for pert_id in test.obs["drug"].str.strip()
            )

            predictions = np.eye(3)[differentially_expressed.X + 1].transpose(0, 2, 1)

            results = task.evaluate(predictions, test.X)
            results["fold"] = fold_index
            results["cell_line"] = cell_line

            aggregated_results.append(results)

        pl.DataFrame(aggregated_results).write_parquet(
            f"tahoe_bootstrap_{cell_line}.parquet"
        )


if __name__ == "__main__":
    fire.Fire(main)

import collections
import os
import pickle
from pathlib import Path
from typing import Literal

import anndata as ad
import fire
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from tqdm import tqdm

import benchmark
from benchmark import paths
from benchmark.preprocessing.differentially_expressed_genes import deg_sciplex3


def main(
    n_trials_outer: int = 3,
    n_trials_inner: int = 3,
    seed: int = 5050,
    min_cells_per_drug_and_plate=20,
) -> pl.DataFrame:
    seed_seq = np.random.SeedSequence(seed)

    # get data
    adata_raw_counts = deg_sciplex3.load_raw_data()
    adata_raw_counts.obs["batch_id"] = adata_raw_counts.obs["plate"].copy()
    adata_raw_counts.obs["pert_id"] = adata_raw_counts.obs["drug"].copy()
    adata_raw_counts.obs["is_control"] = adata_raw_counts.obs["drug"] == "Vehicle"

    # filtering
    coding_genes = pd.read_csv(paths.SCIPLEX_CODING_GENES, sep=",")
    adata_raw_counts = adata_raw_counts[
        :, adata_raw_counts.var["gene_name"].isin(coding_genes["Gene name"].values)
    ].copy()

    idx = (adata_raw_counts.obs["dose"] == 10000.0) | (
        adata_raw_counts.obs["dose"] == 0.0
    )
    # remove rows where key metadata columns are all NaN
    mask_missing = pd.isna(adata_raw_counts.obs["cell_line"])
    adata_raw_counts = adata_raw_counts[(~mask_missing) & idx].copy()

    # process each cell line separately
    for cell_line in adata_raw_counts.obs["cell_line"].unique():
        cell_line_adata = adata_raw_counts[
            adata_raw_counts.obs["cell_line"] == cell_line
        ].copy()

        if len(cell_line_adata) == 0:
            continue

        # preproc
        sc.pp.normalize_total(cell_line_adata, target_sum=1e4)
        sc.pp.log1p(cell_line_adata)

        # filter plates so we don't sample plates with too few cell lines
        print(
            f"Before filtering: {len(cell_line_adata.obs['drug'].unique())} unique drugs"
        )
        cell_line_adata = cell_line_adata[
            cell_line_adata.obs.groupby(["plate", "drug"]).transform("count")[
                "cell_line"
            ]
            > min_cells_per_drug_and_plate
        ]
        print(
            f"After filtering: {len(cell_line_adata.obs['drug'].unique())} unique drugs"
        )

        pert_id_list = sorted(
            cell_line_adata.obs["pert_id"][
                cell_line_adata.obs["is_control"] == False
            ].unique()
        )
        print(f"Processing {len(pert_id_list)} perturbations for {cell_line}")

        # Check if sample_predictions already exists for this cell line
        pickle_path = f"sample_predictions_{cell_line}.pkl"
        if os.path.exists(pickle_path):
            print(f"Loading existing sample_predictions from {pickle_path}")
            with open(pickle_path, "rb") as f:
                sample_predictions = pickle.load(f)
        else:
            print(f"Generating new sample_predictions for {cell_line}")
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

                    for inner_trial_seed in seed_seq_inner.generate_state(
                        n_trials_inner
                    ):
                        obs_ids = (
                            adata.obs.groupby(["plate", "pert_id"])
                            .sample(frac=1, replace=True, random_state=inner_trial_seed)
                            .index
                        )
                        sample_adata = adata[obs_ids].copy()
                        sample_adata.obs["gene_id"] = sample_adata.obs["pert_id"].copy()

                        differentially_expressed = deg_sciplex3.deg_single_cell_line(
                            sample_adata,
                            cell_line=cell_line,
                            filter_min_cells_per_drug_plate=False,
                            log_normalize=False,
                        )

                        sample_predictions[pert_id].append(differentially_expressed)

            # Save sample_predictions for this cell line
            with open(pickle_path, "wb") as f:
                pickle.dump(sample_predictions, f)
                print(f"Saved sample_predictions to {pickle_path}")

        aggregated_results = []
        for fold_index in range(5):
            task = benchmark.BenchmarkTask(
                name="expression/pert-prediction-sciplex3-deg",
                fold=f"{cell_line}.{fold_index}",
            )
            train, test = task.setup()
            # Filter test drugs to only include those present in sample_predictions
            available_test_drugs = [
                pert_id for pert_id in test.obs["drug"] if pert_id in sample_predictions
            ]

            if not available_test_drugs:
                print(
                    f"Warning: No test drugs available in sample_predictions for fold {fold_index}"
                )
                continue

            print(
                f"Fold {fold_index}: Processing {len(available_test_drugs)}/{len(test.obs['drug'])} test drugs"
            )

            for sample_index in range(len(list(sample_predictions.values())[0])):
                differentially_expressed = ad.concat(
                    sample_predictions[pert_id][sample_index]
                    for pert_id in available_test_drugs
                )

                predictions = np.eye(3)[differentially_expressed.X + 1].transpose(
                    0, 2, 1
                )

                # Filter test data to match available drugs
                test_filtered = test[test.obs["drug"].isin(available_test_drugs)]
                results = task.evaluate(predictions, test_filtered.X)
                results["fold"] = fold_index
                results["cell_line"] = cell_line

                aggregated_results.append(results)

        save_path = Path(f"results/experimental_error_raw_sciplex3_{cell_line}.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(aggregated_results).write_csv(save_path)

    # Combine results from all cell lines and process quantiles
    quantiles = [0.005, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.995]

    # Read all cell line results
    all_results_dfs = []
    for cell_line in adata_raw_counts.obs["cell_line"].unique():
        try:
            df = pl.read_csv(f"results/experimental_error_raw_sciplex3_{cell_line}.csv")
            all_results_dfs.append(df)
        except Exception:
            continue

    if all_results_dfs:
        all_results = pl.concat(all_results_dfs)
        quantile_results = pl.concat(
            [
                all_results.group_by("cell_line", "fold")
                .agg(pl.all().quantile(quantile))
                .with_columns(quantile=pl.lit(quantile))
                for quantile in quantiles
            ]
        )
        final_save_path = Path(f"results/exp_err_sciplex3_deg_n_trials_inner_{n_trials_inner}_outer_{n_trials_outer}.csv")
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        quantile_results.write_csv(final_save_path)
        return quantile_results

    return pl.DataFrame()


if __name__ == "__main__":
    fire.Fire(main)

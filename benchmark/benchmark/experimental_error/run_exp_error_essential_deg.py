import numpy as np
import polars as pl
from tqdm import tqdm
from pathlib import Path
from benchmark.experimental_error import utils as ut
from benchmark.preprocessing.differentially_expressed_genes import deg_essential
import benchmark


def main(
    dataset_name: str,
    n_trials: int = 50,
    seed: int = 5050,
) -> pl.DataFrame:
    seed_seq = np.random.SeedSequence(seed)

    # Data:
    adata = ut.get_data(dataset_name, get_raw_counts = True)

    match dataset_name:
        case "essential_k562":
            cell_line = "K-562"
        case "essential_hepg2":
            cell_line = "Hep-G2"
        case "essential_htertrpe1":
            cell_line = "hTERT-RPE1"
        case "essential_jurkat":
            cell_line = "Jurkat"

    aggregated_results = []
    for seed in tqdm(seed_seq.generate_state(n_trials)):
        obs_ids = (
            adata.obs.reset_index()
            .groupby("pert_id")
            .sample(frac=1, replace=True, random_state=seed)["obs_id"]
        )
        sample_adata = adata[obs_ids].copy()
        sample_adata.obs["gene_id"] = sample_adata.obs["pert_id"]

        differentially_expressed = deg_essential.deg(sample_adata)
        differentially_expressed.obs = differentially_expressed.obs.set_index("gene_id")
        for fold_index in range(5):
            task = benchmark.BenchmarkTask(
                name="expression/pert-prediction-essential-deg",
                fold=f"{cell_line}.{fold_index}",
            )

            train, test = task.setup()

            predictions = np.eye(3)[
                differentially_expressed[test.obs["pert_id"].to_list()].X + 1
            ].transpose(0, 2, 1)

            results = task.evaluate(predictions, test.X)
            results["fold"] = fold_index
            results["cell_line"] = cell_line

            aggregated_results.append(results)

    return pl.DataFrame(aggregated_results)


if __name__ == "__main__":
    datasets = [
        "essential_jurkat",
        "essential_htertrpe1",
        "essential_hepg2",
        "essential_k562",
    ]
    quantiles = [0.005, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.995]
    dfs = []
    n_trials = 20
    for dataset in datasets:
        dfs.append(main(dataset, n_trials=n_trials))

    all_results = pl.concat(dfs)

    save_path = Path(f"results/exp_err_essential_deg_n_trials_{n_trials}.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(
        [
            all_results.group_by("cell_line", "fold")
            .agg(pl.all().quantile(quantile))
            .with_columns(quantile=pl.lit(quantile))
            for quantile in quantiles
        ]
    ).write_csv(save_path)

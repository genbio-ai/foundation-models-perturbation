import time
from typing import Literal
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from jaxtyping import Float
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmark.experimental_error import utils as ut


def split_by_obs_value(adata: ad.AnnData, field: str | list[str]):
    """
    Utility function to split AnnData by some obs value.

    Args:
        adata: AnnData object.
        field (str): Column of obs to use for splitting.

    Returns:
        value_to_adata (dict): Maps field value to AnnData object.
    """
    value_to_adata = {}
    for name, group in adata.obs.groupby(field):
        cur_adata = adata[group.index].copy()
        value_to_adata[name] = cur_adata
    return value_to_adata


def run_pert(
    perturbation: str,
    n_outer_bootstraps: int,
    n_inner_bootstraps: int,
    seed: np.random.SeedSequence,
    plate_adatas: dict[tuple[str, str], ad.AnnData],
    quantile_grid: np.ndarray,
    adata_obs: pd.DataFrame,
) -> pd.DataFrame:
    plates = adata_obs[adata_obs["pert_id"] == perturbation]["batch_id"].unique()

    results: Float[np.ndarray, "n_outer n_inner g "] = bootstrap(
        perturbation,
        n_outer_bootstraps,
        n_inner_bootstraps,
        seed,
        plate_adatas,
        plates=plates,
    )

    ground_truth = np.zeros(results.shape[-1])
    for plate in plates:
        ground_truth += np.asarray(
            plate_adatas[(plate, perturbation)].X.mean(axis=0)
        ).flatten()
        ground_truth -= np.asarray(
            plate_adatas[(plate, "control")].X.mean(axis=0)
        ).flatten()
    ground_truth /= len(plates)

    bootstrap_noise: Float[np.ndarray, "n_outer n_inner g "] = results - ground_truth
    bootstrap_noise: Float[np.ndarray, "n_total g "] = bootstrap_noise.reshape(
        -1, bootstrap_noise.shape[-1]
    )

    bootstrap_l2: Float[np.ndarray, "n_total"] = np.linalg.norm(
        bootstrap_noise, ord=2, axis=1
    )

    quantile_values = np.quantile(bootstrap_l2, q=quantile_grid)

    df = pd.DataFrame(
        {
            "quantile": quantile_grid,
            "value": quantile_values,
            "pert_id": perturbation,
            "n_batches_pert": len(plates),
        }
    )
    return df


def bootstrap(
    perturbation: str,
    n_outer_bootstraps: int,
    n_inner_bootstraps: int,
    seed_sequence: np.random.SeedSequence,
    plate_adatas: dict[tuple[str, str], ad.AnnData],
    plates: list[str],
) -> Float[np.ndarray, "n_outer n_inner g"]:
    results = []

    # I think we only need the seed sequence across processes
    # but maintaining the structure here as I'm not sure.
    outer_sequence = seed_sequence.spawn(n_outer_bootstraps)
    for seed_sequence_outer in outer_sequence:
        rng_outer = np.random.default_rng(seed_sequence_outer)
        batch = rng_outer.choice(plates, size=len(plates))

        batch_results = []

        inner_sequence = seed_sequence_outer.spawn(len(batch))
        for plate, inner_seed in zip(batch, inner_sequence):
            rng_inner = np.random.default_rng(inner_seed)

            batch_results.append(
                ut.bootstrap_csr_mean(
                    plate_adatas[(plate, perturbation)].X,
                    n_trials=n_inner_bootstraps,
                    rng=rng_inner,
                )
                - ut.bootstrap_csr_mean(
                    plate_adatas[(plate, "control")].X,
                    n_trials=n_inner_bootstraps,
                    rng=rng_inner,
                )
            )

        # average across plates
        results.append(np.stack(batch_results).mean(axis=0))

    return np.stack(results)


def main(
    dataset_name: str,
    n_trials_outer: int = 10,
    n_trials_inner: int = 20,
    n_jobs: int = 8,
    seed: int = 5050,
    q_grid: list[float] = [0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.995],
) -> pd.DataFrame:
    t = time.time()
    seed_seq = np.random.SeedSequence(seed)

    # Data:
    adata = ut.get_data(dataset_name)
    pert_id_list = sorted(
        adata.obs["pert_id"][adata.obs["is_control"] == False].unique()
    )

    plate_adatas = split_by_obs_value(adata, ["batch_id", "pert_id"])
    print(f"{(time.time() - t) / 60.0:.1f} min: data loaded and preprocessed")

    # Run in parallel over perturbations:
    pert_seed_seqs = seed_seq.spawn(len(pert_id_list))
    df_results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_pert)(
            pert_id,
            n_trials_outer,
            n_trials_inner,
            pert_seed_seqs[i],
            plate_adatas,  # Unsure exactly how bad passing these across processes is.
            # I hope that it's done by reference but I'm not 100% sure. Same for the
            # adata.obs DataFrame
            q_grid,
            adata.obs,
        )
        for i, pert_id in tqdm(enumerate(pert_id_list[:]), total=len(pert_id_list))
    )
    print(
        f"{len(df_results_list)} perturbations done in {(time.time() - t) / 60.0:.1f} min"
    )

    # Save results:
    df_out = pd.concat(df_results_list, axis=0, ignore_index=True)

    # add cell_line column
    df_out["cell_line"] = dataset_name

    # these should go out of scope anyway
    del plate_adatas
    del adata

    return df_out


if __name__ == "__main__":
    # This plate matched experimental error script is suitable for sciplex3 and Tahoe
    dataset_to_run: Literal["tahoe", "sciplex"] = "sciplex"

    if dataset_to_run == "tahoe":
        dataset_name_list = [
            "Tahoe100M_CVCL_0023",
            "Tahoe100M_CVCL_0028",
            "Tahoe100M_CVCL_0069",
            "Tahoe100M_CVCL_0099",
            "Tahoe100M_CVCL_0131",
            "Tahoe100M_CVCL_0152",
            "Tahoe100M_CVCL_0179",
            "Tahoe100M_CVCL_0218",
            "Tahoe100M_CVCL_0292",
            "Tahoe100M_CVCL_0293",
            "Tahoe100M_CVCL_0320",
            "Tahoe100M_CVCL_0332",
            "Tahoe100M_CVCL_0334",
            "Tahoe100M_CVCL_0359",
            "Tahoe100M_CVCL_0366",
            "Tahoe100M_CVCL_0371",
            "Tahoe100M_CVCL_0397",
            "Tahoe100M_CVCL_0399",
            "Tahoe100M_CVCL_0428",
            "Tahoe100M_CVCL_0459",
            "Tahoe100M_CVCL_0480",
            "Tahoe100M_CVCL_0504",
            "Tahoe100M_CVCL_0546",
            "Tahoe100M_CVCL_1055",
            "Tahoe100M_CVCL_1056",
            "Tahoe100M_CVCL_1094",
            "Tahoe100M_CVCL_1097",
            "Tahoe100M_CVCL_1098",
            "Tahoe100M_CVCL_1119",
            "Tahoe100M_CVCL_1125",
            "Tahoe100M_CVCL_1239",
            "Tahoe100M_CVCL_1285",
            "Tahoe100M_CVCL_1381",
            "Tahoe100M_CVCL_1478",
            "Tahoe100M_CVCL_1495",
            "Tahoe100M_CVCL_1517",
            "Tahoe100M_CVCL_1531",
            "Tahoe100M_CVCL_1547",
            "Tahoe100M_CVCL_1550",
            "Tahoe100M_CVCL_1571",
            "Tahoe100M_CVCL_1577",
            "Tahoe100M_CVCL_1635",
            "Tahoe100M_CVCL_1666",
            "Tahoe100M_CVCL_1693",
            "Tahoe100M_CVCL_1715",
            "Tahoe100M_CVCL_1716",
            "Tahoe100M_CVCL_1717",
            "Tahoe100M_CVCL_1724",
            "Tahoe100M_CVCL_1731",
            "Tahoe100M_CVCL_C466",
        ]  # 45 cell lines
    elif dataset_to_run == "sciplex":
        dataset_name_list = [
            "sciplex_MCF7",
            "sciplex_A549",
            "sciplex_K562",
        ]  # 3 cell lines

    else:
        raise ValueError(f"Unrecognized {dataset_to_run=}")

    n_trials_outer = 10
    n_trials_inner = 20
    seed: int = 5050
    for dataset in dataset_name_list:
        results = main(
            dataset,
            n_trials_outer=n_trials_outer,
            n_trials_inner=n_trials_inner,
            seed=seed,
        )

        # If you want to rerun this script, you may need to delete the csv file you write to,
        # otherwise it will assume that it's just a new cell line and append
        results_path = Path(
            f"./results/exp_err_{dataset_to_run}_outer_{n_trials_outer}_inner_{n_trials_inner}_seed_{seed}.csv"
        )
        results_path.parent.mkdir(parents = True, exist_ok=True)
        results.to_csv(
            results_path, index=False, mode="a", header=not results_path.exists()
        )

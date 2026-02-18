import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmark.experimental_error import utils as ut


def split_by_obs_value(adata, field):
    """
    Utility function to split AnnData by some obs value.

    Args:
        adata: AnnData object.
        field (str): Column of obs to use for splitting.

    Returns:
        value_to_adata (dict): Maps field value to AnnData object.
    """
    value_to_adata = {}
    for value in adata.obs[field].unique():
        idx = np.where(adata.obs[field] == value)[0]
        cur_adata = adata[idx].copy()
        value_to_adata[value] = cur_adata
    return value_to_adata


def run_pert(
    adata_pert,
    pert_id,
    q_grid,
    n_trials_inner,
    n_trials_outer,
    seed,
    ctrl_boot_results,
    adata_point_estimate,
):
    """
    Estimate experimental error bound for one perturbation.
    """
    assert adata_point_estimate.shape[0] == 1
    pert_boot_results = two_stage_bootstrap_global(
        adata_pert, seed, n_trials_inner, n_trials_outer
    )
    delta_boot = (
        pert_boot_results - ctrl_boot_results
    )  # (n_trials_outer, n_trials_inner, n_genes)
    delta_boot = delta_boot.reshape(
        -1, delta_boot.shape[2]
    )  # (n_trials_total, n_genes)
    delta_point_estimate = adata_point_estimate.X[0, :].toarray()  # (1, n_genes)

    # L2
    L2_boot = np.linalg.norm(delta_boot - delta_point_estimate, ord=2, axis=1)
    q_L2 = np.quantile(L2_boot, q=q_grid)

    # MSE
    MSE_boot = np.mean((delta_boot - delta_point_estimate)**2, axis=1)
    q_MSE = np.quantile(MSE_boot, q=q_grid)

    # MAE
    MAE_boot = np.mean(np.abs(delta_boot - delta_point_estimate), axis=1)
    q_MAE = np.quantile(MAE_boot, q=q_grid)

    # Spearman
    spearman_coeffs = [
        scipy.stats.spearmanr(delta_boot[i, :], delta_point_estimate.flatten())[0]
        for i in range(delta_boot.shape[0])
    ]
    q_spearman = np.quantile(spearman_coeffs, q=q_grid)

    # Pearson
    pearson_coeffs = [
        scipy.stats.pearsonr(delta_boot[i, :], delta_point_estimate.flatten())[0]
        for i in range(delta_boot.shape[0])
    ]
    q_pearson = np.quantile(pearson_coeffs, q=q_grid)


    df = pd.DataFrame({
        'quantile': q_grid, 
        'L2': q_L2,
        'MSE': q_MSE,
        'MAE': q_MAE,
        'Spearman': q_spearman,
        'Pearson': q_pearson,
        'pert_id': pert_id,
        'n_batches_pert': len(adata_pert.obs['batch_id'].unique())
        })
    return df


def two_stage_bootstrap_global(adata, root_seq, n_trials_inner, n_trials_outer):
    """
    Use bootstrapping at the batch and cell level to compute an empirical distribution
    for the mean transcriptomic profiles in adata, weighting each batch equally.
    """

    batch_to_adata = split_by_obs_value(adata, "batch_id")
    batch_id_list = sorted(batch_to_adata.keys())
    outer_results = []

    # root_seq = np.random.SeedSequence(seed)
    outer_seqs = root_seq.spawn(n_trials_outer)

    for i, ss_outer in enumerate(outer_seqs):
        # Resample batches:
        if len(batch_id_list) > 1:
            rng_outer = np.random.default_rng(ss_outer)
            boot_batches = rng_outer.choice(
                batch_id_list, size=len(batch_id_list), replace=True
            )
        else:
            boot_batches = batch_id_list
        # Resample cells in each batch:
        batch_results = []
        inner_seqs = ss_outer.spawn(len(boot_batches))
        for b, ss_inner in zip(boot_batches, inner_seqs):
            rng_inner = np.random.default_rng(ss_inner)
            cur_adata = batch_to_adata[b]
            cur_results = ut.bootstrap_csr_mean(
                cur_adata.X, n_trials_inner, rng_inner
            )  # (n_trial_inner, n_genes)
            batch_results.append(cur_results)
        batch_results = np.array(batch_results)  # (n_batches, n_trial_inner, n_genes)
        cur_outer_results = batch_results.mean(
            axis=0
        )  # Key step, averaging over batches with equal weight -> (n_trials_inner, n_genes)
        outer_results.append(cur_outer_results)
    outer_results = np.array(outer_results)  # (n_trials_outer, n_trials_inner, n_genes)
    return outer_results


def main(
    dataset_name: str,
    n_trials_outer: int = 10,
    n_trials_inner: int = 10,
    n_jobs: int = 32,
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

    # Compute point estimates:
    adata_pb_delta = ut.pseudobulk_delta(adata)

    # Pre-bootstrap the control data:
    adata_ctrl = adata[adata.obs["is_control"] == True]
    adata_pert = adata[adata.obs["is_control"] == False]
    pert_to_adata = split_by_obs_value(adata_pert, "pert_id")
    ctrl_seed_seq = seed_seq.spawn(1)[0]
    ctrl_boot_results = two_stage_bootstrap_global(
        adata_ctrl, ctrl_seed_seq, n_trials_inner, n_trials_outer
    )

    del adata, adata_pert, adata_ctrl
    print(f"{(time.time() - t) / 60.0:.1f} min: data loaded and preprocessed")

    # Run in parallel over perturbations:
    pert_seed_seqs = seed_seq.spawn(len(pert_id_list))
    df_results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_pert)(
            pert_to_adata[pert_id],
            pert_id,
            q_grid,
            n_trials_inner,
            n_trials_outer,
            pert_seed_seqs[i],
            ctrl_boot_results,
            adata_pb_delta[adata_pb_delta.obs["pert_id"] == pert_id],
        )
        for i, pert_id in tqdm(enumerate(pert_id_list), total=len(pert_id_list))
    )
    print(
        f"{len(df_results_list)} perturbations done in {(time.time() - t) / 60.0:.1f} min"
    )

    # Save results:
    df_out = pd.concat(df_results_list, axis=0, ignore_index=True)
    save_path = Path(f"./results/exp_err_{dataset_name}_outer_{n_trials_outer}_inner_{n_trials_inner}_seed_{seed}.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False)
    return df_out


if __name__ == "__main__":
    datasets = [
        "essential_jurkat",
        "essential_htertrpe1",
        "essential_hepg2",
        "essential_k562",
    ]
    for dataset in datasets:
        main(dataset)

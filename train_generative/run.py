"""
Entry point for training diffusion / flow / Schrodinger models on perturbation data.

Usage:
    # Flow matching (default)
    python run.py +experiment=flow data_path=path/to/data.h5ad embeddings_path=path/to/embeddings.pkl

    # Diffusion
    python run.py +experiment=diffusion data_path=... embeddings_path=...

    # Schrodinger bridge
    python run.py +experiment=schrodinger data_path=... embeddings_path=...

    # Override any parameter
    python run.py +experiment=flow training.epochs=50 pca_dim=30
"""

import os
import pickle

import anndata as ad
import hydra
import numpy as np
import scanpy as sc
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler

from data import PerturbationDataLoader, PerturbationDataset
from models import DiffusionAutoEncoder, DiT
from trainer import Trainer
from utils import apply_pca, count_parameters, set_seed


def create_train_val_splits(adata, train_genes, val_genes):
    """Split adata into train/val based on gene assignments (controls in both)."""
    train_adata = adata[
        adata.obs["gene_id"].isin(train_genes) | (adata.obs["is_control"] == True)
    ].copy()
    val_adata = adata[
        adata.obs["gene_id"].isin(val_genes) | (adata.obs["is_control"] == True)
    ].copy()
    return train_adata, val_adata


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {cfg.data_path}")
    adata = ad.read_h5ad(cfg.data_path)

    print(f"Loading embeddings from {cfg.embeddings_path}")
    with open(cfg.embeddings_path, "rb") as f:
        gene_embeddings = pickle.load(f)

    # Filter to genes with embeddings
    all_pert_genes = list(adata[adata.obs["is_control"] == False].obs["gene_id"].unique())
    contained_genes = set(all_pert_genes) & set(gene_embeddings.keys())
    adata = adata[
        adata.obs["gene_id"].isin(contained_genes) | (adata.obs["is_control"] == True)
    ].copy()

    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # Train / val split (random gene-level split)
    # ------------------------------------------------------------------
    val_fraction = cfg.get("val_fraction", 0.1)
    all_genes = sorted(contained_genes)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(all_genes)
    n_val = max(1, int(len(all_genes) * val_fraction))
    val_genes = all_genes[:n_val]
    train_genes = all_genes[n_val:]
    print(f"Train genes: {len(train_genes)}, Val genes: {len(val_genes)}")

    train_adata, val_adata = create_train_val_splits(adata, train_genes, val_genes)

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    pca_dim = cfg.pca_dim
    train_adata, val_adata, pca_obj, scaler_obj = apply_pca(train_adata, val_adata, n_components=pca_dim)
    pca_min = float(train_adata.obsm["X_pca"].min())
    pca_max = float(train_adata.obsm["X_pca"].max())
    print(f"PCA dim={pca_dim}, min={pca_min:.2f}, max={pca_max:.2f}")

    # ------------------------------------------------------------------
    # Datasets & dataloaders
    # ------------------------------------------------------------------
    cell_line = cfg.cell_line
    perturbation_map = {cell_line: {"gene_id": train_genes}}
    val_perturbation_map = {cell_line: {"gene_id": val_genes}}
    perturbation_to_embeddings = {"gene_id": gene_embeddings}

    train_dataset = PerturbationDataset(
        adata=train_adata,
        control_column="is_control",
        perturbation_covariates=["gene_id"],
        cell_type_to_perturbation_map=perturbation_map,
        cell_type_column="cell_line",
        cell_data_key="X_pca",
        perturbation_to_embeddings=perturbation_to_embeddings,
    )
    val_dataset = PerturbationDataset(
        adata=val_adata,
        control_column="is_control",
        perturbation_covariates=["gene_id"],
        cell_type_to_perturbation_map=val_perturbation_map,
        cell_type_column="cell_line",
        cell_data_key="X_pca",
        perturbation_to_embeddings=perturbation_to_embeddings,
    )

    batch_size = cfg.training.batch_size
    train_loader = PerturbationDataLoader(train_dataset, batch_size=batch_size)
    val_loader = PerturbationDataLoader(val_dataset, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_type = cfg.model_type
    dit_cfg = cfg.dit

    # Override in_channels / out_channels based on model type
    in_channels = dit_cfg.get("in_channels", 1)
    out_channels = dit_cfg.get("out_channels", None)

    dit = DiT(
        input_size=pca_dim,
        in_channels=in_channels,
        hidden_size=dit_cfg.hidden_size,
        depth=dit_cfg.depth,
        num_heads=dit_cfg.num_heads,
        z_emb_dim=dit_cfg.get("z_emb_dim", dit_cfg.hidden_size),
        z_emb_dropout=dit_cfg.get("z_emb_dropout", 0.0),
        attention_dropout=dit_cfg.get("attention_dropout", 0.0),
        projection_dropout=dit_cfg.get("projection_dropout", 0.0),
        mlp_dropout=dit_cfg.get("mlp_dropout", 0.0),
        out_channels=out_channels,
    )

    ae_model = DiffusionAutoEncoder(dit, unconditional=False)
    print(f"Model parameters: {count_parameters(ae_model):,}")

    # Create the generative model wrapper
    if model_type == "diffusion":
        from diffusion import AEGaussianDiffusion

        diff_cfg = cfg.diffusion
        generative_model = AEGaussianDiffusion(
            ae_model,
            seq_length=pca_dim,
            timesteps=diff_cfg.get("timesteps", 1000),
            sampling_timesteps=diff_cfg.get("sampling_timesteps", 250),
            objective=diff_cfg.get("objective", "pred_noise"),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            is_ddim_sampling=diff_cfg.get("ddim_sampling", False),
            realspace_min=pca_min,
            realspace_max=pca_max,
        )
    elif model_type == "flow":
        from flow import FlowMatching1D

        flow_cfg = cfg.flow
        generative_model = FlowMatching1D(
            ae_model,
            seq_length=pca_dim,
            sampling_timesteps=flow_cfg.get("sampling_timesteps", 250),
            step_size=flow_cfg.get("step_size", 0.05),
            realspace_min=pca_min,
            realspace_max=pca_max,
        )
    elif model_type == "schrodinger":
        from schrodinger import SchrodingerBridge1D

        sb_cfg = cfg.schrodinger
        generative_model = SchrodingerBridge1D(
            ae_model,
            seq_length=pca_dim,
            sampling_timesteps=sb_cfg.get("sampling_timesteps", 250),
            sigma=sb_cfg.get("sigma", 1.0),
            use_sde_sampling=sb_cfg.get("use_sde_sampling", True),
            realspace_min=pca_min,
            realspace_max=pca_max,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'diffusion', 'flow', or 'schrodinger'.")

    print(f"Using model: {model_type}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    train_cfg = cfg.training
    optimizer = torch.optim.AdamW(
        generative_model.parameters(),
        lr=train_cfg.lr,
        betas=tuple(train_cfg.betas),
        weight_decay=train_cfg.weight_decay,
    )

    total_steps = len(train_loader) * train_cfg.epochs
    warmup_steps = int(train_cfg.warmup_ratio * total_steps)
    lr_scheduler = get_scheduler(
        name=train_cfg.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = cfg.wandb.get("enabled", False)
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=cfg.wandb.get("name", f"{cfg.cell_line}-{model_type}"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    results_dir = cfg.results_dir
    trainer = Trainer(
        model=generative_model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=results_dir,
        cfg=cfg,
        use_wandb=use_wandb,
        ema_decay=train_cfg.ema_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        patience=train_cfg.patience,
    )

    trainer.train(
        num_epochs=train_cfg.epochs,
        eval_freq=train_cfg.eval_freq,
        save_freq=train_cfg.save_freq,
    )

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

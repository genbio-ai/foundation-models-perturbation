import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from data import load_graph, split_edges, create_dataloader
from models import GNN
from trainer import Trainer
from utils import set_seed, count_parameters


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load graph
    use_weights = cfg.get('use_weights', False)
    print(f"Loading graph from {cfg.graph_path}...")
    data, node_id_to_name = load_graph(
        cfg.graph_path,
        use_weights=use_weights
    )
    print(f"Loaded graph with {data.num_nodes} nodes and {data.edge_index.shape[1]} edges")
    if use_weights:
        print(f"Using edge weights (only effective with GCN, ignored by GIN)")

    # Split edges into train/val (when not in debug mode)
    if not cfg.get('debug_mode', False):
        print(f"Splitting edges into train/val with {cfg.data.val_ratio:.1%} validation...")
        train_data, val_data = split_edges(data, val_ratio=cfg.data.val_ratio, seed=cfg.seed)
        print(f"Train edges: {train_data.edge_index.shape[1]}, Val edges: {val_data.val_edge_index.shape[1]}")
    else:
        # For debug mode: use full graph
        print("DEBUG MODE: Using full graph without train/test split")
        train_data = data
        val_data = data

    # Create model
    print("Creating model...")
    model = GNN(
        dim=cfg.model.dim,
        num_layers=cfg.model.num_layers,
        num_embs=data.num_nodes,
        dropout=cfg.model.dropout,
        layer_type=cfg.model.layer_type
    ).to(device)

    print(f"Model has {count_parameters(model)} trainable parameters")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    # Create data loaders for link prediction
    print("Creating link prediction data loaders...")

    # In debug mode, disable shuffling for reproducibility
    debug_mode = cfg.get('debug_mode', False)
    train_loader = create_dataloader(
        train_data,
        loss_type='link_prediction',
        batch_size=cfg.link_prediction.batch_size,
        num_negative_samples=cfg.link_prediction.num_negative_samples,
        shuffle=not debug_mode,
        debug_mode=debug_mode,
    )
    # For validation: in debug mode, val_loader won't be used (validation and embeddings skipped)
    # so just create a simple loader; otherwise use val_edge_index
    if cfg.get('debug_mode', False):
        # Val loader not used in debug mode, just create dummy from train_data
        val_loader = train_loader
    else:
        val_data_for_loader = val_data.clone()
        val_data_for_loader.edge_index = val_data.val_edge_index
        val_loader = create_dataloader(
            val_data_for_loader,
            loss_type='link_prediction',
            batch_size=cfg.link_prediction.batch_size,
            num_negative_samples=cfg.link_prediction.num_negative_samples,
            shuffle=False,
            debug_mode=False,
        )

    # Create save directory
    weights_str = "weighted" if use_weights and cfg.model.layer_type == "gcn" else "unweighted"

    # Get batch size from link prediction config
    batch_size = cfg.link_prediction.batch_size

    # Add 'overfit_' prefix if in debug mode
    debug_prefix = "overfit_" if cfg.get('debug_mode', False) else ""
    save_name = f"{debug_prefix}dim{cfg.model.dim}_layers{cfg.model.num_layers}_{cfg.model.layer_type}_{weights_str}_link_prediction_lr{cfg.training.lr}_wd{cfg.training.weight_decay}_dropout{cfg.model.dropout}_bs{batch_size}_thresh{cfg.graph_thresh}"
    save_dir = os.path.join(cfg.save_dir, save_name)

    # Save config to save_dir
    os.makedirs(save_dir, exist_ok=True)
    config_save_path = os.path.join(save_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved config to {config_save_path}")

    # Initialize wandb if enabled
    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name if cfg.wandb.name else save_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # Log model architecture
        wandb.watch(model, log='all', log_freq=100)

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_data=train_data.clone(),  # Clone to avoid affecting dataloader
        val_data=val_data.clone(),      # Clone to avoid affecting dataloader
        device=device,
        save_dir=save_dir,
        node_id_to_name=node_id_to_name,
        patience=cfg.training.patience,
        use_wandb=use_wandb,
        grad_norm_clip=cfg.training.grad_norm_clip,
        loss_type='link_prediction',
        debug_mode=cfg.get('debug_mode', False),
    )

    # Train
    trainer.train(cfg.training.num_epochs)

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

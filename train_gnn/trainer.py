import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        train_data,
        val_data,
        device,
        save_dir,
        node_id_to_name,
        patience=10,
        use_wandb=False,
        grad_norm_clip=None,
        loss_type='link_prediction',
        debug_mode=False,
    ):
        """
        Trainer for GNN with configurable loss.

        Args:
            model: GNN model
            optimizer: Optimizer
            train_loader: DataLoader for training (RandomWalkDataLoader or LinkPredictionDataLoader)
            val_loader: DataLoader for validation
            train_data: PyG Data object for training (for message passing)
            val_data: PyG Data object for validation (for message passing)
            device: Device to train on
            save_dir: Directory to save model and embeddings
            node_id_to_name: Dict mapping node indices to ENSEMBL IDs
            patience: Early stopping patience
            use_wandb: Whether to log to Weights & Biases
            grad_norm_clip: Gradient norm clipping threshold (None to disable)
            loss_type: Type of loss ('random_walk' or 'link_prediction')
            debug_mode: Skip validation for quick overfitting tests
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.node_id_to_name = node_id_to_name
        self.patience = patience
        self.use_wandb = use_wandb
        self.grad_norm_clip = grad_norm_clip
        self.loss_type = loss_type
        self.debug_mode = debug_mode

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0

    def train_epoch(self, epoch, pbar=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for pos_samples, neg_samples in self.train_loader:

            # breakpoint()
            iter_start_time = time.time()

            pos_samples = pos_samples.to(self.device)
            neg_samples = neg_samples.to(self.device)

            # Forward pass - use train_data's edge_index for message passing
            embeddings = self.model(
                x=None,
                edge_index=self.train_data.edge_index,
                edge_weight=getattr(self.train_data, 'edge_weight', None)
            )

            # Compute loss
            loss = self.train_loader.compute_loss(embeddings, pos_samples, neg_samples)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = None
            if self.grad_norm_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_norm_clip
                ).item()

            self.optimizer.step()

            # Track iteration time
            iter_time = time.time() - iter_start_time

            # Get learning rate
            lr = self.optimizer.param_groups[0]['lr']

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log to wandb (per iteration)
            if self.use_wandb:
                import wandb
                log_dict = {
                    'train/loss_step': loss.item(),
                    'train/lr': lr,
                    'train/iter_time': iter_time,
                    'global_step': self.global_step,
                }
                if grad_norm is not None:
                    log_dict['train/grad_norm'] = grad_norm
                wandb.log(log_dict)

            # Update progress bar with running average
            if pbar is not None:
                postfix = {'train_loss': f'{total_loss / num_batches:.4f}'}
                if grad_norm is not None:
                    postfix['grad_norm'] = f'{grad_norm:.3f}'
                pbar.set_postfix(postfix)
                pbar.update(1)

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Create progress bar for validation
        val_pbar = tqdm(
            total=len(self.val_loader),
            desc="Validation",
            ncols=100,
            leave=False
        )

        for pos_samples, neg_samples in self.val_loader:
            pos_samples = pos_samples.to(self.device)
            neg_samples = neg_samples.to(self.device)

            # Forward pass - use val_data's edge_index for message passing
            embeddings = self.model(
                x=None,
                edge_index=self.val_data.edge_index,
                edge_weight=getattr(self.val_data, 'edge_weight', None)
            )

            # Compute loss
            loss = self.val_loader.compute_loss(embeddings, pos_samples, neg_samples)

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            val_pbar.set_postfix({'val_loss': f'{total_loss / num_batches:.4f}'})
            val_pbar.update(1)

        val_pbar.close()
        return total_loss / num_batches

    def train(self, num_epochs):
        """Train the model with early stopping."""
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Create progress bar for this epoch
            pbar = tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                ncols=100
            )

            train_loss = self.train_epoch(epoch, pbar)
            pbar.close()

            # Skip validation in debug mode or for random walk
            if self.debug_mode:
                val_loss = train_loss  # Use train loss as proxy
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} (DEBUG MODE: no validation)")
            elif self.loss_type == 'random_walk':
                val_loss = train_loss  # Use train loss as proxy
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} (no validation for random walk)")
            else:
                val_loss = self.validate()
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Log to wandb (epoch-level metrics)
            if self.use_wandb:
                import wandb
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss_epoch': train_loss,
                    'train/patience_counter': self.patience_counter,
                }
                # Only log validation metrics if not in debug mode or random walk mode
                if not self.debug_mode and self.loss_type != 'random_walk':
                    log_dict['val/loss'] = val_loss
                    log_dict['val/best_loss'] = self.best_loss
                wandb.log(log_dict)

            # Early stopping (disabled in debug mode)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                if self.use_wandb:
                    import wandb
                    if self.debug_mode or self.loss_type == 'random_walk':
                        wandb.run.summary['train/best_loss'] = self.best_loss
                    else:
                        wandb.run.summary['val/best_loss'] = self.best_loss
                    wandb.run.summary['best_epoch'] = epoch + 1
            else:
                self.patience_counter += 1
                # Skip early stopping in debug mode
                if not self.debug_mode and self.patience_counter >= self.patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

        # Load best model
        self.load_checkpoint('best_model.pt')

        # Save embeddings (skip in debug mode)
        if not self.debug_mode:
            self.save_embeddings()

        print("Training complete!")

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']

    @torch.no_grad()
    def save_embeddings(self):
        """Save embeddings at different layers and final layer."""
        self.model.eval()

        # Get embeddings using forward pass with intermediate flag
        edge_index = self.val_data.edge_index
        edge_weight = getattr(self.val_data, 'edge_weight', None)

        final_emb, layer_embeddings = self.model(
            x=None,
            edge_index=edge_index,
            edge_weight=edge_weight,
            return_intermediate=True
        )

        # Convert to numpy
        final_emb = final_emb.cpu().numpy()
        layer_embeddings = [emb.cpu().numpy() for emb in layer_embeddings]

        # Create embeddings directory inside save_dir
        embeddings_dir = self.save_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Save final embeddings
        final_emb_dict = {self.node_id_to_name[i]: final_emb[i] for i in range(len(final_emb))}
        final_emb_path = embeddings_dir / "final_embeddings.npz"
        np.savez(final_emb_path, **final_emb_dict)

        # Save layer-wise embeddings (layer0, layer1, layer2, ...)
        for layer_idx, emb in enumerate(layer_embeddings):
            layer_emb_dict = {self.node_id_to_name[i]: emb[i] for i in range(len(emb))}
            layer_emb_path = embeddings_dir / f"layer{layer_idx}_embeddings.npz"
            np.savez(layer_emb_path, **layer_emb_dict)

        print(f"Saved embeddings to {embeddings_dir}")
        print(f"  - final_embeddings.npz")
        for layer_idx in range(len(layer_embeddings)):
            print(f"  - layer{layer_idx}_embeddings.npz")

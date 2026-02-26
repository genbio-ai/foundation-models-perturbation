"""
Simple PyTorch trainer for diffusion / flow / Schrodinger models.

Plain single-GPU training with EMA, gradient clipping, early stopping,
checkpoint save/load, and optional WandB logging.
"""

import time
from pathlib import Path

import torch
from ema_pytorch import EMA
from omegaconf import OmegaConf
from tqdm.auto import tqdm


class Trainer:
    """
    Parameters
    ----------
    model : nn.Module
        The generative model (AEGaussianDiffusion, FlowMatching1D, or SchrodingerBridge1D).
    optimizer : torch.optim.Optimizer
    scheduler : LR scheduler (from transformers or torch).
    train_loader : PerturbationDataLoader
    val_loader : PerturbationDataLoader or None
    device : torch.device
    save_dir : str or Path
    cfg : OmegaConf config (for saving alongside checkpoint)
    use_wandb : bool
    ema_decay : float
    max_grad_norm : float
    patience : int
        Early stopping patience (in eval cycles).
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        save_dir,
        cfg=None,
        use_wandb=False,
        ema_decay=0.995,
        ema_update_every=10,
        max_grad_norm=1.0,
        patience=10,
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.use_wandb = use_wandb
        self.max_grad_norm = max_grad_norm
        self.patience = patience

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(device)

        self.step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs, eval_freq=1, save_freq=100):
        start_epoch = self._try_resume()

        for epoch in range(start_epoch, num_epochs):
            # Evaluate
            if epoch % eval_freq == 0:
                val_loss = self.validate() if self.val_loader is not None else None
                self._log_val(epoch, val_loss)

                if val_loss is not None and self._early_stop_check(epoch, val_loss):
                    print(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch} (val_loss={self.best_val_loss:.4f})")
                    break

            # Train one epoch
            self.train_epoch(epoch)

            # Checkpoint
            if save_freq and epoch % save_freq == 0 and epoch != 0:
                self.save_checkpoint(f"model-{epoch}.pt")

        # Final checkpoint
        self.save_checkpoint(f"model-{epoch}.pt")
        self._rename_best(epoch)
        print("Training complete.")

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", ncols=120, leave=False)

        for batch in pbar:
            self.step += 1
            start = time.time()

            source_cells = batch["source_cell"].to(self.device)
            target_cells = batch["target_cell"].to(self.device)
            z_emb = batch["perturbation_embedding"]
            if z_emb is not None:
                z_emb = z_emb.to(self.device)

            terms = self.model(source_cells=source_cells, target_cells=target_cells, z_emb=z_emb)
            loss = terms["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.ema.update()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}", gn=f"{grad_norm:.2f}")

            if self.use_wandb:
                import wandb

                log = {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/step": self.step,
                    "train/epoch": epoch,
                    "train/iter_time": elapsed,
                }
                for k, v in terms.items():
                    if k != "loss":
                        log[f"train/{k}"] = v
                wandb.log(log, step=self.step)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, n_steps=30):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            loader_iter = iter(self.val_loader)
            for _ in range(n_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.val_loader)
                    batch = next(loader_iter)

                source_cells = batch["source_cell"].to(self.device)
                target_cells = batch["target_cell"].to(self.device)
                z_emb = batch["perturbation_embedding"]
                if z_emb is not None:
                    z_emb = z_emb.to(self.device)

                terms = self.model(source_cells=source_cells, target_cells=target_cells, z_emb=z_emb)
                total_loss += terms["loss"].item()

        self.model.train()
        return total_loss / n_steps

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename):
        path = self.save_dir / filename
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
        }
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        torch.save(data, str(path))
        print(f"Saved checkpoint: {path}")

        if self.cfg is not None:
            cfg_path = self.save_dir / "config.yaml"
            with open(cfg_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))

    def load_checkpoint(self, path):
        data = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.ema.load_state_dict(data["ema"])
        self.step = data["step"]
        self.best_val_loss = data.get("best_val_loss", float("inf"))
        self.best_epoch = data.get("best_epoch", 0)
        self.patience_counter = data.get("patience_counter", 0)
        if self.scheduler is not None and "scheduler" in data:
            self.scheduler.load_state_dict(data["scheduler"])
        print(f"Loaded checkpoint: {path} (step={self.step})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_resume(self):
        """Auto-resume from milestone.txt if it exists."""
        milestone_file = self.save_dir / "milestone.txt"
        if not milestone_file.exists():
            return 0
        milestone = milestone_file.read_text().strip()
        if not milestone.isdigit():
            return 0
        ckpt = self.save_dir / f"model-{milestone}.pt"
        if ckpt.exists():
            self.load_checkpoint(ckpt)
            epoch = int(milestone)
            print(f"Resuming from epoch {epoch}")
            return epoch + 1
        return 0

    def _early_stop_check(self, epoch, val_loss):
        if val_loss < self.best_val_loss:
            print(f"Val loss improved: {self.best_val_loss:.4f} -> {val_loss:.4f}")
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            self.save_checkpoint("model-best.pt")
            return False

        self.patience_counter += 1
        print(f"No improvement ({self.patience_counter}/{self.patience})")
        return self.patience_counter >= self.patience

    def _log_val(self, epoch, val_loss):
        if val_loss is not None:
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f}  best={self.best_val_loss:.4f}")
            if self.use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/epoch": epoch, "val/best_loss": self.best_val_loss}, step=self.step)

    def _rename_best(self, final_epoch):
        """Copy best checkpoint with epoch info."""
        import shutil
        best_path = self.save_dir / "model-best.pt"
        if best_path.exists():
            final_path = self.save_dir / f"best_model_epoch{self.best_epoch}.pt"
            shutil.copy(best_path, final_path)
            print(f"Best model: {final_path} (val_loss={self.best_val_loss:.4f})")

        # Write milestone
        milestone_file = self.save_dir / "milestone.txt"
        milestone_file.write_text(str(final_epoch))

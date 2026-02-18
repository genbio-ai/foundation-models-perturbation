import lightning as L
from jaxtyping import Int, Bool
import torch
from .input import LinearInputTransform

class FitStatisticsCallback(L.Callback):
    """
    Lightning callback to fit z-score statistics before training.
    Ensures statistics are computed once on training data without data leakage.
    """
    def __init__(self):
        super().__init__()
        self.stats_fitted = False
    
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Fit statistics on training data before training starts."""
        
        if self.stats_fitted:
            return
        
        print("Fitting z-score statistics on training data...")
        
        # Get the train dataloader from the datamodule
        if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
            train_dataloader = trainer.datamodule.train_dataloader()
        else:
            train_dataloader = trainer.train_dataloader
            if train_dataloader is None:
                raise RuntimeError(
                    "Could not access train_dataloader. "
                    "Make sure you're using a datamodule or the dataloader is properly configured."
                )
        
        # Fit statistics for each LinearInputTransform
        fitted_count = 0
        for i, transform in enumerate(pl_module.input_transforms):
            if isinstance(transform, LinearInputTransform) and transform.normalize_type == "zscore":
                print(f"\nFitting statistics for transform {i} ({transform.input_key})")
                transform.fit_statistics(train_dataloader)
                fitted_count += 1
        
        self.stats_fitted = True
        print(f"Statistics fitting complete! Fitted {fitted_count} transform(s).")
        print("Training will now begin with frozen statistics.")

class ClassWeightingCallback(L.Callback):
    """
    Lightning callback to fit class-weighting statistics before training.
    Ensures statistics are computed once on training data without data leakage.
    """
    def __init__(self, target_key: str):
        super().__init__()
        self.stats_fitted = False
        self.target_key = target_key
    
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Fit statistics on training data before training starts."""
        
        print("Fitting class weighting statistics on training data...")
        
        # Get the train dataloader from the datamodule
        if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
            train_dataloader = trainer.datamodule.train_dataloader()
        else:
            train_dataloader = trainer.train_dataloader
            if train_dataloader is None:
                raise RuntimeError(
                    "Could not access train_dataloader. "
                    "Make sure you're using a datamodule or the dataloader is properly configured."
                )
        
        counts: Int[torch.Tensor, "genes classes"] | None = None
        with torch.no_grad():
            for batch in train_dataloader:
                batch_y: Int[torch.Tensor, "perturbations genes"] = 1 + batch[self.target_key]
                onehot_y: Bool[torch.Tensor, "perturbations genes classes"] = (
                    batch_y[:, :, None]
                    == torch.arange(0, 3, device=batch_y.device)[None, None, :]
                )
                if counts is None:
                    counts = onehot_y.sum(dim=0)
                else:
                    counts += onehot_y.sum(dim=0)
        n_pert = counts.sum(dim=1)[0]
        self.weights = (n_pert / (3 * counts)).nan_to_num(posinf=0)
        self.stats_fitted = True
        print(f"Class-weighting statistics fitting complete!")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch["class_weights"] = self.weights
        

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch["class_weights"] = self.weights

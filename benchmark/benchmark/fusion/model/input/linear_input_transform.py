import torch
import torch.nn as nn
import lightning as L

class LinearInputTransform(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        input_key: str,
        normalize_type: str = "l2",  # "l2", "zscore", or "none"
        dropout: float = 0,
        use_layer_norm: bool = False,
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
            input_key: Key to extract input from batch dict
            normalize_type: Type of normalization ("l2", "zscore", or "none")
            dropout: Dropout probability (default: 0)
        """
        super().__init__()
        self.input_key = input_key
        self.normalize_type = normalize_type
        self.use_layer_norm = use_layer_norm

        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)  # Initialize dropout layer
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(output_dim)
        
        # Buffers for z-score statistics (not trainable parameters)
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        self.stats_fitted = False

    def fit_statistics(self, dataloader):
        """
        Compute mean and std from training data.
        
        Args:
            dataloader: Training dataloader to compute statistics from
        """
        if self.normalize_type != "zscore":
            return
        
        all_features = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[self.input_key]
                all_features.append(x.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        self.mean = all_features.mean(dim=0).to(self.linear.weight.device)
        self.std = all_features.std(dim=0).to(self.linear.weight.device)
        
        # Avoid division by zero
        self.std = torch.clamp(self.std, min=1e-8)
        self.stats_fitted = True
        
        print(f"Statistics fitted for {self.input_key}: mean shape {self.mean.shape}, std shape {self.std.shape}")

    def _normalize(self, x):
        """Apply the appropriate normalization."""
        if self.normalize_type == "l2":
            # L2 normalization
            norm_x = torch.linalg.norm(x, dim=1, keepdim=True)
            return x / (1e-8 + norm_x)
        
        elif self.normalize_type == "zscore":
            # Z-score normalization using fitted statistics
            if not self.stats_fitted:
                raise RuntimeError(
                    f"Statistics not fitted for {self.input_key}. "
                    "Call fit_statistics() on training data before using zscore normalization."
                )
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            return (x - mean) / std
        
        else:  # "none"
            return x

    def forward(self, batch: dict, return_normalized_input: bool = False):
        x = batch[self.input_key]
        normalized_x = self._normalize(x)
        
        # Apply linear transform then dropout
        transformed = self.linear(normalized_x)
        if self.use_layer_norm:
            transformed = self.norm(transformed)
        transformed = self.dropout(transformed) 
        
        if return_normalized_input:
            return transformed, normalized_x
        return transformed
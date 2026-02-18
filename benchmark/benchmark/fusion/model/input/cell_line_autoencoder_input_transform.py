import torch
import torch.nn as nn
from typing import Dict

class CellLineAutoencoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.1,
    ):
        """
        Autoencoder for cell line expression profiles.
        
        Args:
            input_dim: Number of genes in expression profile
            latent_dim: Dimension of latent embedding
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256])
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode expression to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to expression."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full autoencoder forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class CellLineAutoencoderInputTransform(nn.Module):
    def __init__(
        self,
        autoencoder: nn.Module,
        cell_line_expressions: Dict[int, torch.Tensor],
        input_key: str = "cell_line_label",
        normalize_input: bool = True,
        normalize_output: bool = True,
    ):
        """
        Input transform that uses autoencoder latent space for cell line embeddings.
        Trains jointly with the main model.
        
        Args:
            autoencoder: CellLineAutoencoder instance
            cell_line_expressions: Dict mapping cell line label to average control expression
            input_key: Key in batch dict containing cell line labels
            normalize_input: Whether to normalize input expressions (log1p + zscore)
            normalize_output: Whether to L2-normalize the latent embeddings
        """
        super().__init__()
        self.input_key = input_key
        self.autoencoder = autoencoder
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        
        # Convert dict to tensor lookup table
        num_cell_lines = len(cell_line_expressions)
        expression_dim = next(iter(cell_line_expressions.values())).shape[0]
        
        expression_lookup = torch.zeros(num_cell_lines, expression_dim)
        for label, expr in cell_line_expressions.items():
            expression_lookup[label] = expr
        
        # Register as buffer (not a trainable parameter, but moves with model)
        self.register_buffer('expression_lookup', expression_lookup)
        
        # Compute normalization statistics if needed
        if normalize_input:
            mean = expression_lookup.mean(dim=0)
            std = expression_lookup.std(dim=0)
            std = torch.clamp(std, min=1e-8)  # Avoid division by zero
            self.register_buffer('expr_mean', mean)
            self.register_buffer('expr_std', std)
    
    def _normalize_expression(self, x):
        """Normalize expression data."""
        if self.normalize_input:
            # Log transform + z-score
            x = torch.log1p(x)
            x = (x - self.expr_mean) / self.expr_std
        return x
    
    def forward(self, batch: dict, return_reconstruction: bool = False):
        """
        Forward pass.
        
        Args:
            batch: Batch dictionary
            return_reconstruction: If True, also return the reconstruction for loss computation
        
        Returns:
            latent: Latent embeddings (batch_size, latent_dim)
            reconstruction (optional): Reconstructed expressions for autoencoder loss
            original (optional): Original expressions for autoencoder loss
        """
        # Get cell line labels from batch
        cell_line_labels = batch[self.input_key].squeeze(-1)  # (batch_size,)
        
        # Lookup average control expression
        expressions = self.expression_lookup[cell_line_labels]  # (batch_size, n_genes)
        
        # Normalize input
        normalized_expr = self._normalize_expression(expressions)
        
        # Encode to latent space
        if return_reconstruction:
            reconstruction, latent = self.autoencoder(normalized_expr)
        else:
            latent = self.autoencoder.encode(normalized_expr)
            reconstruction = None
        
        # Optionally normalize output
        if self.normalize_output:
            norm = torch.linalg.norm(latent, dim=1, keepdim=True)
            latent = latent / (1e-8 + norm)
        
        if return_reconstruction:
            return latent, reconstruction, normalized_expr
        return latent
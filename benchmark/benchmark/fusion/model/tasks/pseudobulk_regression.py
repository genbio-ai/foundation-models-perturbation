import torch
import torch.nn as nn

class PseudobulkRegressionTask(nn.Module):
    def __init__(
        self,
        target_key: str,
        input_dim: int,
        hidden_dim: int,
        n_hidden_layers: int, 
        output_dim: int,
        dropout: float,
        use_gelu: bool,
        xavier_init: bool,
        cls_alone: bool = False,
        **kwargs, 
    ):
        super().__init__()
        self.target_key = target_key
        self.use_gelu = use_gelu
        self.xavier_init = xavier_init
        self.n_hidden_layers = n_hidden_layers
        self.cls_alone = cls_alone
        
        # Determine input dimension for the main MLP.
        mlp_input_dim = input_dim if self.cls_alone else input_dim * 2

        # A single MLP handles the combined input
        layers = []
        if n_hidden_layers == 0:
            layers.append(nn.Linear(mlp_input_dim, output_dim))
        else:
            layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            layers.append(nn.GELU() if self.use_gelu else nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU() if self.use_gelu else nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        if self.xavier_init:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        if self.cls_alone:
            return self.mlp(batch[0, :, :])  # MLP applied to (batch_size, input_dim)
        else:
            combined = torch.cat([batch[0, :, :], batch[-1, :, :]], dim=1) # (batch_size, input_dim*2)
            return self.mlp(combined)
    
    def loss_fn(self, y_hat, batch):
        return torch.norm(y_hat - batch[self.target_key], dim=-1).mean()
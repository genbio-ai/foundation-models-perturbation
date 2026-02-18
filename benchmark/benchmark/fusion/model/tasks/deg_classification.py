import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool


class DEGClassificationTask(nn.Module):
    def __init__(
        self,
        target_key: str,
        input_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        output_dim: int,
        dropout: float,
        use_gelu: bool = True,
        xavier_init: bool = True,
    ):
        super().__init__()
        self.target_key = target_key
        self.output_dim = output_dim
        self.use_gelu = use_gelu
        self.xavier_init = xavier_init
        self.n_hidden_layers = n_hidden_layers

        layers = []
        if n_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim * 3))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU() if self.use_gelu else nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU() if self.use_gelu else nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim * 3))
        self.mlp = nn.Sequential(*layers)

        self.cross_entropy = nn.CrossEntropyLoss()

        if xavier_init:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, batch):
        return self.mlp(batch[0, :, :]).view(-1, self.output_dim, 3)

    def loss_fn(self, y_hat: Float[torch.Tensor, "perturbations genes classes"], batch):
        """Expects labels -1, 0, +1 in batch[self.target_key]"""

        y: Int[torch.Tensor, "perturbations genes "] = 1 + batch[self.target_key]
        assert "class_weights" in batch

        loss = torch.vmap(
            torch.nn.functional.cross_entropy, in_dims=(1, 1, 0), out_dims=-1
        )(y_hat, y, batch["class_weights"]).mean()

        return loss

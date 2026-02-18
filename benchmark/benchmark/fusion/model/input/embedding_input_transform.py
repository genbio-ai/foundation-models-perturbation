import torch
import torch.nn as nn

class EmbeddingInputTransform(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        input_key: str,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_key = input_key

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        indices = batch[self.input_key]

        # Handle shape (B, 1) -> (B,)
        if indices.dim() > 1 and indices.shape[-1] == 1:
            indices = indices.squeeze(-1)

        x = self.embedding(indices)
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x

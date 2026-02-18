import anndata as ad
import numpy as np
from typing import List
import torch
from benchmark.paths import DATA_ROOT

class TensorPrecomputedEmbedding:
    def __init__(
        self,
        key_added: str,
        required_key: str,
        adata_path: str,
        embedding_name: str,
    ):
        self.key_added = key_added
        self.embedding_name = embedding_name
        self.required_key = required_key
        self.adata = ad.read_h5ad(DATA_ROOT / adata_path)

    def set_universe(self, universe: List):
        pass

    def encode(self, x):
        return {
            self.key_added: torch.nan_to_num(
                torch.tensor(
                    self.adata[str(x)].obsm[self.embedding_name].ravel(),
                    dtype=torch.float32,
                )
            )
        }

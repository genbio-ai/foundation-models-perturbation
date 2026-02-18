import anndata as ad
import numpy as np
from typing import List
import torch
from benchmark.paths import DATA_ROOT

class TensorPrecomputedOptionalEmbedding:
    def __init__(
        self,
        key_added: str,
        required_key: str,
        adata_path: str,
    ):
        self.key_added = key_added
        self.required_key = required_key
        self.adata = ad.read_h5ad(DATA_ROOT / adata_path)

    def set_universe(self, universe: List):
        pass

    def encode(self, x):
        if str(x) in self.adata.obs_names:
            values = torch.nan_to_num(
                torch.tensor(
                    self.adata[str(x)].X.toarray().ravel(),
                    dtype=torch.float32,
                )
            )
            attn_mask = torch.zeros(1, dtype=torch.bool)
        else:
            values = torch.zeros(self.adata.n_vars, dtype=torch.float32)
            attn_mask = torch.ones(1, dtype=torch.bool)

        return {
            self.key_added: values,
            self.key_added + "_attn_mask": attn_mask,
        }

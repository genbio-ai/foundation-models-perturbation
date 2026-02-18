from torch.utils.data import Dataset
import torch
from typing import Dict, List
import numpy as np
import anndata as ad

class EssentialDEGDataset(Dataset):
    def __init__(
        self,
        condition_encoders: List,
        adata: ad.AnnData,
    ):
        super().__init__()
        self.adata = adata
        self.n_genes = self.adata.n_vars

        self.condition_encoders = condition_encoders
        unique_values = {}
        for enc in condition_encoders:
            assert enc.key_added != "deg_label"

            k = enc.required_key
            if k not in unique_values:
                unique_values[k] = (
                    self.adata.obs[k].unique().tolist()
                )
            enc.set_universe(unique_values[k])

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        sample = {}
        for enc in self.condition_encoders:
            k = enc.required_key
            try:
                res = enc.encode(self.adata.obs[k].values[idx])
                for k, v in res.items():
                    sample[k] = v
            except KeyError:
                print(enc.required_key)
                print(enc)
                print(self.adata.obs[k].values[idx])

        sample["deg_label"] = torch.tensor(
            self.adata.X[idx, :],
            dtype=torch.int64,
        )

        return sample

from torch.utils.data import Dataset
import torch
from typing import List
import numpy as np
import anndata as ad


class EssentialPseudobulkDataset(Dataset):
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
            assert enc.key_added != "pseudobulk_delta"

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
            res = enc.encode(self.adata.obs[k].values[idx])
            for k, v in res.items():
                sample[k] = v

        sample["pseudobulk_delta"] = torch.tensor(
            self.adata.X[idx, :],
            dtype=torch.float32,
        )

        return sample
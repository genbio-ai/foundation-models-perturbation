"""
Perturbation DataLoader for single-cell perturbation data.

Samples control-perturbation pairs for training generative models.
"""

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PerturbationDataset(Dataset):
    """
    Dataset for sampling control-perturbation pairs.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing single-cell data.
    control_column : str
        Column name indicating control cells (e.g., 'is_control').
    perturbation_covariates : list[str]
        Column names in adata.obs that define perturbations (e.g., ['gene_id']).
    cell_type_to_perturbation_map : dict
        Mapping from cell types to their available perturbation covariates.
        e.g., {'K-562': {'gene_id': ['ENSG00000001', 'ENSG00000002']}}
    cell_type_column : str
        Column name for cell types (default: 'cell_type').
    cell_data_key : str
        Key in adata.obsm for cell representations (default: 'X_pca').
    perturbation_to_embeddings : dict, optional
        Mapping from perturbation covariate names to {value: embedding} dicts.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        control_column: str,
        perturbation_covariates: List[str],
        cell_type_to_perturbation_map: Dict[str, Dict[str, List[str]]],
        cell_type_column: str = "cell_type",
        cell_data_key: str = "X_pca",
        perturbation_to_embeddings: Optional[Dict[str, Dict[str, Union[List, np.ndarray]]]] = None,
    ):
        self.adata = adata.copy()
        self.control_column = control_column
        self.perturbation_covariates = perturbation_covariates
        self.cell_type_to_perturbation_map = cell_type_to_perturbation_map
        self.cell_type_column = cell_type_column
        self.cell_data_key = cell_data_key
        self.perturbation_to_embeddings = perturbation_to_embeddings

        self._prepare_data()
        self._validate_embeddings()
        self._build_mappings()

    def _prepare_data(self):
        """Load cell data and build lookup arrays."""
        if self.cell_data_key == "X":
            cell_data_np = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
        else:
            cell_data_np = self.adata.obsm[self.cell_data_key]

        self.cell_data = torch.tensor(cell_data_np, dtype=torch.float32)
        self.control_mask = self.adata.obs[self.control_column].astype(bool).values
        self.cell_type_array = self.adata.obs[self.cell_type_column].values

        self.perturbation_arrays = {}
        for covar_name in self.perturbation_covariates:
            self.perturbation_arrays[covar_name] = self.adata.obs[covar_name].values

        n_ctrl = self.control_mask.sum()
        n_pert = (~self.control_mask).sum()
        print(f"Dataset: {n_ctrl} control cells, {n_pert} perturbed cells")

    def _validate_embeddings(self):
        """Check that embeddings cover all perturbation values."""
        if self.perturbation_to_embeddings is None:
            return
        for covar_name in self.perturbation_covariates:
            if covar_name not in self.perturbation_to_embeddings:
                raise ValueError(f"Missing embeddings for covariate '{covar_name}'")

            all_values = set()
            for perturbations in self.cell_type_to_perturbation_map.values():
                if covar_name in perturbations:
                    all_values.update(perturbations[covar_name])

            for val in all_values:
                if val not in self.perturbation_to_embeddings[covar_name]:
                    raise ValueError(f"Missing embedding for '{covar_name}':'{val}'")

    def _build_mappings(self):
        """Build source (control) -> perturbation index mappings."""
        self.source_distributions = {}
        self.source_to_perturbations = {}
        self.perturbation_distributions = {}

        source_idx = 0
        perturbation_idx = 0

        for cell_type in self.cell_type_to_perturbation_map:
            control_mask = self.control_mask & (self.cell_type_array == cell_type)
            if control_mask.sum() == 0:
                print(f"Warning: No control cells for {cell_type}")
                continue

            self.source_distributions[source_idx] = np.where(control_mask)[0]
            self.source_to_perturbations[source_idx] = []

            available = self.cell_type_to_perturbation_map[cell_type]
            covar_names = list(available.keys())
            covar_values = [available[name] for name in covar_names]

            for combo in itertools.product(*covar_values):
                perturb_mask = (~self.control_mask) & (self.cell_type_array == cell_type)
                for k, v in zip(covar_names, combo):
                    perturb_mask &= self.perturbation_arrays[k] == v

                if perturb_mask.sum() > 0:
                    self.perturbation_distributions[perturbation_idx] = np.where(perturb_mask)[0]
                    self.source_to_perturbations[source_idx].append(perturbation_idx)
                    perturbation_idx += 1

            if self.source_to_perturbations[source_idx]:
                self.source_to_perturbations[source_idx] = np.array(
                    self.source_to_perturbations[source_idx]
                )
                source_idx += 1
            else:
                del self.source_distributions[source_idx]
                del self.source_to_perturbations[source_idx]

        self.n_sources = len(self.source_distributions)
        self.n_perturbations = len(self.perturbation_distributions)
        print(f"Built {self.n_sources} source distributions, {self.n_perturbations} perturbation distributions")

        if self.n_sources == 0:
            raise ValueError("No valid source distributions. Check cell_type_to_perturbation_map.")

        self.approx_length = sum(len(c) for c in self.source_distributions.values()) + sum(
            len(c) for c in self.perturbation_distributions.values()
        )

    def __len__(self):
        return self.approx_length

    def __getitem__(self, idx):
        source_idx = np.random.randint(0, self.n_sources)
        source_cell_indices = self.source_distributions[source_idx]
        source_cell_idx = np.random.choice(source_cell_indices)

        target_pert_idx = np.random.choice(self.source_to_perturbations[source_idx])
        target_cell_indices = self.perturbation_distributions[target_pert_idx]
        target_cell_idx = np.random.choice(target_cell_indices)

        source_cell = self.cell_data[source_cell_idx]
        target_cell = self.cell_data[target_cell_idx]

        # Get perturbation embedding
        perturbation_embedding = None
        if self.perturbation_to_embeddings is not None:
            if len(self.perturbation_covariates) == 1:
                covar_name = self.perturbation_covariates[0]
                value = self.perturbation_arrays[covar_name][target_cell_idx]
                perturbation_embedding = torch.tensor(
                    self.perturbation_to_embeddings[covar_name][value],
                    dtype=torch.float32,
                )
            else:
                embeddings = []
                for covar_name in self.perturbation_covariates:
                    value = self.perturbation_arrays[covar_name][target_cell_idx]
                    embeddings.append(self.perturbation_to_embeddings[covar_name][value])
                perturbation_embedding = torch.tensor(
                    np.concatenate(embeddings), dtype=torch.float32
                )

        return {
            "source_cell": source_cell,
            "target_cell": target_cell,
            "perturbation_embedding": perturbation_embedding,
        }


class PerturbationDataLoader:
    """
    DataLoader wrapper for PerturbationDataset.

    Parameters
    ----------
    dataset : PerturbationDataset
    batch_size : int
    num_workers : int
    """

    def __init__(self, dataset: PerturbationDataset, batch_size: int = 32, num_workers: int = 0, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = {}
        for key in batch[0]:
            if key == "perturbation_embedding":
                embeddings = [item[key] for item in batch]
                collated[key] = torch.stack(embeddings) if embeddings[0] is not None else None
            elif isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        return collated

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

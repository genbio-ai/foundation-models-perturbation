import random

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_pca(train_adata, val_adata, n_components=50):
    """
    Apply StandardScaler + PCA to train and validation AnnData objects.

    Fits on training data, transforms both train and validation.
    Stores results in adata.obsm['X_pca'].

    Returns:
        (train_adata, val_adata, pca, scaler)
    """
    train_X = train_adata.X.toarray() if sp.issparse(train_adata.X) else train_adata.X
    val_X = val_adata.X.toarray() if sp.issparse(val_adata.X) else val_adata.X

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)

    pca = PCA(n_components=n_components)
    train_adata.obsm["X_pca"] = pca.fit_transform(train_X_scaled)
    val_adata.obsm["X_pca"] = pca.transform(val_X_scaled)

    return train_adata, val_adata, pca, scaler


def scale_to_diffusion_space(x, min, max):
    """Normalize from [min, max] to [-1, 1]."""
    return 2 * (x - min) / (max - min) - 1


def scale_to_real_space(x, min, max):
    """Unnormalize from [-1, 1] back to [min, max]."""
    return 0.5 * (x + 1) * (max - min) + min

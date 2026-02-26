import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graph(graph_path, use_weights=False):
    """
    Load graph from adjacency list format.

    Args:
        graph_path: Path to graph file (tab-separated: protein1, protein2, scores...)
        use_weights: Whether to use combined_score as edge weights

    Returns:
        data: PyG Data object
        node_id_to_name: Dict mapping node indices to ENSEMBL IDs
    """
    edges = []
    weights = []
    node_names = set()

    with open(graph_path, 'r') as f:
        next(f)  # Skip header

        for line in f:
            parts = line.strip().split('\t')
            protein1 = parts[0]
            protein2 = parts[1]

            node_names.add(protein1)
            node_names.add(protein2)
            edges.append((protein1, protein2))

            if use_weights and len(parts) > 9:
                weights.append(float(parts[9]))  # combined_score is at index 9

    # Create mapping from node names to indices
    node_names = sorted(list(node_names))
    node_name_to_id = {name: idx for idx, name in enumerate(node_names)}
    node_id_to_name = {idx: name for name, idx in node_name_to_id.items()}

    # Convert edges to indices
    edge_index = []
    for protein1, protein2 in edges:
        idx1 = node_name_to_id[protein1]
        idx2 = node_name_to_id[protein2]
        # Add both directions for undirected graph
        edge_index.append([idx1, idx2])
        edge_index.append([idx2, idx1])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # Create Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=len(node_names)
    )

    if use_weights:
        edge_weight = torch.tensor(weights + weights, dtype=torch.float)
        # Normalize weights to [0, 1] range (combined_score is typically 0-1000)
        edge_weight = edge_weight / 1000.0
        data.edge_weight = edge_weight

    return data, node_id_to_name


def split_edges(data, val_ratio=0.1, seed=42):
    """
    Split edges into train and validation sets.

    Args:
        data: PyG Data object
        val_ratio: Ratio of edges for validation
        seed: Random seed

    Returns:
        train_data: Data object with train edges
        val_data: Data object with val edges (full graph for message passing)
    """
    torch.manual_seed(seed)

    # Get unique edges (remove duplicates from bidirectional)
    edge_index = data.edge_index
    num_edges = edge_index.shape[1] // 2  # Divide by 2 for undirected

    # Create mask for one direction only
    mask = edge_index[0] < edge_index[1]
    unique_edges = edge_index[:, mask]

    # Get edge weights for unique edges if available
    has_weights = hasattr(data, 'edge_weight') and data.edge_weight is not None
    if has_weights:
        unique_weights = data.edge_weight[mask]

    # Shuffle and split
    perm = torch.randperm(unique_edges.shape[1])
    num_val = int(val_ratio * unique_edges.shape[1])

    val_edges = unique_edges[:, perm[:num_val]]
    train_edges = unique_edges[:, perm[num_val:]]

    # Make bidirectional
    train_edge_index = torch.cat([train_edges, train_edges.flip(0)], dim=1)
    val_edge_index = torch.cat([val_edges, val_edges.flip(0)], dim=1)

    # Create train data (only train edges for message passing)
    train_data = Data(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes
    )

    # Create val data (use full graph for message passing, but only val edges for loss)
    val_data = Data(
        edge_index=data.edge_index,  # Full graph
        val_edge_index=val_edge_index,  # Validation edges only
        num_nodes=data.num_nodes
    )

    # Handle edge weights
    if has_weights:
        train_weights = unique_weights[perm[num_val:]]
        val_weights = unique_weights[perm[:num_val]]

        # Make bidirectional weights
        train_data.edge_weight = torch.cat([train_weights, train_weights], dim=0)
        val_data.val_edge_weight = torch.cat([val_weights, val_weights], dim=0)
        # Keep full graph weights for message passing
        val_data.edge_weight = data.edge_weight

    return train_data, val_data


class LinkPredictionDataset(Dataset):
    """
    PyTorch Dataset for link prediction.
    Returns: (pos_edges, neg_edges)
    """

    def __init__(
        self,
        data: Data,
        num_negative_samples: int = 1,
        debug_mode: bool = False,
    ):
        """
        Args:
            data: PyG Data object
            num_negative_samples: Negative samples per positive edge
            debug_mode: Use fixed seed for reproducible negative samples
        """
        self.data = data
        self.num_negative_samples = num_negative_samples
        self.debug_mode = debug_mode

        # Get unique edges (one direction)
        edge_index = data.edge_index
        mask = edge_index[0] < edge_index[1]
        self.pos_edges = edge_index[:, mask]

    def __len__(self):
        return self.pos_edges.shape[1]

    def __getitem__(self, idx):
        """Return edge index for collate_fn to process."""
        return idx

    def collate_fn(self, batch_indices):
        """
        Custom collate function that creates batches.

        Args:
            batch_indices: List of edge indices

        Returns:
            (pos_edge_index, neg_edge_index)
        """
        # Get batch of positive edges
        pos_edge_index = self.pos_edges[:, batch_indices]

        # In debug mode, use fixed seed for reproducible negative samples
        if self.debug_mode:
            set_seed(42)

        # Sample negative edges
        neg_edge_index = negative_sampling(
            edge_index=self.data.edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.shape[1] * self.num_negative_samples,
        )

        return pos_edge_index, neg_edge_index

    def compute_loss(self, embeddings: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        """
        Compute link prediction BCE loss.

        Args:
            embeddings: Node embeddings [num_nodes, dim]
            pos_edge_index: Positive edges [2, num_pos]
            neg_edge_index: Negative edges [2, num_neg]

        Returns:
            Loss value
        """
        # Positive edge scores
        pos_src = embeddings[pos_edge_index[0]]
        pos_dst = embeddings[pos_edge_index[1]]
        pos_scores = (pos_src * pos_dst).sum(dim=-1)

        # Negative edge scores
        neg_src = embeddings[neg_edge_index[0]]
        neg_dst = embeddings[neg_edge_index[1]]
        neg_scores = (neg_src * neg_dst).sum(dim=-1)

        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        return pos_loss + neg_loss


def create_dataloader(data, loss_type='link_prediction', **kwargs):
    """
    Create dataloader for link prediction.

    Args:
        data: PyG Data object
        loss_type: Only 'link_prediction' is supported
        **kwargs: Additional arguments for the dataloader

    Returns:
        DataLoader instance
    """
    if loss_type == 'link_prediction':
        # Extract parameters
        batch_size = kwargs.pop('batch_size', 1024)
        num_negative_samples = kwargs.pop('num_negative_samples', 1)
        shuffle = kwargs.pop('shuffle', True)
        num_workers = kwargs.pop('num_workers', 0)
        debug_mode = kwargs.pop('debug_mode', False)

        # Create dataset
        dataset = LinkPredictionDataset(
            data=data,
            num_negative_samples=num_negative_samples,
            debug_mode=debug_mode,
        )

        # Create PyTorch DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Attach compute_loss method to loader for consistency
        loader.compute_loss = dataset.compute_loss

        return loader

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Only 'link_prediction' is supported.")

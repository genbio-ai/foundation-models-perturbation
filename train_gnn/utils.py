import os
import torch


def get_num_workers(num_workers):
    """
    Get number of workers for data loading.

    Args:
        num_workers: Number of workers. If -1, use all available CPUs.

    Returns:
        Number of workers
    """
    if num_workers == -1:
        return os.cpu_count()
    return num_workers


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch
import anndata as ad
import numpy as np
from typing import Dict, Optional
import os
from benchmark.paths import DATA_ROOT

def compute_average_control_expressions(
    adata_path: str,
    control_gene: str = "control",  # Gene name that indicates control
    gene_key: str = "drugname",  # Column indicating gene
    cell_line_key: str = "cell_line",
    cell_line_label_mapping: Optional[Dict[str, int]] = None,
    cache_path: Optional[str] = None,
) -> Dict[int, torch.Tensor]:
    """
    Compute average control expression for each cell line.
    
    Args:
        adata_path: Path to AnnData file
        control_gene: Gene name that indicates control samples
        gene_key: Column in obs indicating gene
        cell_line_key: Column in obs with cell line names
        cell_line_label_mapping: Dict mapping cell line names to integer labels
        cache_path: Optional path to cache the computed expressions
    
    Returns:
        Dictionary mapping cell line label (int) to average expression (tensor)
    """

    print("Computing average control expressions for cell lines...")
    
    # Check cache first
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached control expressions from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"Loading data from {adata_path}...")
    adata = ad.read_h5ad(DATA_ROOT / adata_path)

    print(f"Number of genes in file: {adata.n_vars}")
    
    # Filter for control samples
    control_mask = adata.obs[gene_key] == control_gene
    print(f"Found {control_mask.sum()} control samples")
    
    if control_mask.sum() == 0:
        raise ValueError(f"No control samples found with {gene_key}='{control_gene}'")
    
    cell_line_expressions = {}
    unique_cell_lines = adata.obs[cell_line_key].unique()
    
    print(f"Computing average control expressions for {len(unique_cell_lines)} cell lines...")
    
    for cell_line in unique_cell_lines:
        # Get all control samples for this cell line
        cell_line_mask = (adata.obs[cell_line_key] == cell_line) & control_mask
        n_samples = cell_line_mask.sum()
        
        if n_samples == 0:
            print(f"Warning: No control samples found for {cell_line}, skipping...")
            continue
        
        print(f"  {cell_line}: {n_samples} control samples")
        
        # Get indices
        indices = np.where(cell_line_mask)[0]
        
        # Compute average expression
        expressions = []
        for idx in indices:
            expr = adata.X[idx, :]
            if hasattr(expr, 'toarray'):  # Handle sparse matrices
                expr = expr.toarray().flatten()
            elif hasattr(expr, 'A1'):
                expr = expr.A1
            else:
                expr = np.array(expr).flatten()
            expressions.append(expr)
        
        avg_expression = np.mean(expressions, axis=0)
        avg_expression_tensor = torch.tensor(avg_expression, dtype=torch.float32)
        
        # Get label from mapping
        if cell_line_label_mapping is not None:
            if cell_line not in cell_line_label_mapping:
                print(f"Warning: {cell_line} not in label mapping, skipping...")
                continue
            label = cell_line_label_mapping[cell_line]
        else:
            # Fallback: use simple integer mapping
            label = list(unique_cell_lines).index(cell_line)
        
        cell_line_expressions[label] = avg_expression_tensor
    
    print(f"Computed expressions for {len(cell_line_expressions)} cell lines")
    
    # Cache if requested
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cell_line_expressions, cache_path)
        print(f"Cached control expressions to {cache_path}")
    
    return cell_line_expressions
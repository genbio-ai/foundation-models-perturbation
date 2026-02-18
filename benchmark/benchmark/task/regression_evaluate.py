import anndata as ad
import numpy as np
from scipy.stats import pearsonr, spearmanr

def evaluate(preds: ad.AnnData, labels: ad.AnnData) -> dict[str, float]:
    '''Evaluate perturbation response predictions for Essential dataset.

    Note:
        Primary metric is L2.

    Args:
        preds (ad.AnnData): AnnData object with predicted expression changes in
            adata.X and corresponding perturbation identifiers in 
            adata.obs['pert_id'].
        labels (ad.AnnData): AnnData object with true expression changes in
            adata.X and corresponding perturbation identifiers in 
            adata.obs['pert_id].
    
    Returns:
        dict[str, float]: A dictionary containing the following keys:
            - primary_metric: 'L2'
            - L2: L2 error between true and predicted expression values, 
                averaged over perturbations. 
            - MSE: Mean squared error between true and predicted values. 
            - MAE: Mean absolute error between true and predicted values.
            - Spearman: Spearman correlation between true and predicted 
                expression values, averaged over perturbations. 
            - Pearson: Pearson correlation between true and predicted 
                expression values, averaged over perturbations. 
    '''
    y_pred = preds.X
    y_true = labels.X
    assert np.shape(y_pred) == np.shape(y_true), f"Predictions and targets must have the same shape. Got {np.shape(y_pred)} and {np.shape(y_true)}"

    assert (preds.obs['pert_id'] == labels.obs['pert_id']).all(), f"Predictions and targets do not have matching pert_id column."

    # Calculate metrics
    l2 = np.mean(np.linalg.norm(y_pred - y_true, ord=2, axis=1))
    mse = np.mean(np.square(y_pred - y_true))
    mae = np.mean(np.abs(y_pred - y_true))
    spearman = np.mean([spearmanr(y_true[i, :], y_pred[i, :])[0] for i in range(y_pred.shape[0])])
    pearson = np.mean([pearsonr(y_true[i, :], y_pred[i, :])[0] for i in range(y_pred.shape[0])])

    return {
        'primary_metric': 'L2',
        'L2': l2,
        'MSE': mse,
        'MAE': mae,
        'Spearman': spearman,
        'Pearson': pearson
    }

import anndata as ad
import numpy as np
import torch
from jaxtyping import Float, Int
from benchmark.task import metrics

METRICS: dict[str, metrics.VectorizedClassificationMetric] = {
    "f1_score": metrics.f1_score,
    "recall": metrics.recall,
    "precision": metrics.precision,
}

SLOW_METRICS: dict[str, metrics.VectorizedClassificationMetric] = {
    "roc_auc": metrics.roc_auc,
    "average_precision": metrics.average_precision,
}


def evaluate(
    preds: Float[np.ndarray, " perturbations classes genes"]
    | Float[torch.Tensor, " perturbations classes genes"],
    labels: Int[np.ndarray, " perturbations genes"] | ad.AnnData,
    skip_slow: bool = False,
) -> dict[str, float]:
    """Evaluate perturbation response predictions for Tahoe dataset.

    Note:
        Primary metric is F1 score.

    Args:
        preds (Float[np.ndarray, " perturbations classes genes"]): numpy array
        of predictions. The first dimension corresponds to perturbations, the
        second to the 3 classes (down 0, not differentially expressed 1, and up 2).
        The last dimension corresponds to the different genes.
        labels (Int[np.ndarray, " perturbations genes"]): numpy array of labels.
        corresponding to the `anndata_labels.X`. The labels should be in [-1,0,1]
        skip_slow (bool): whether to skip the computation of metrics that take a
        while (~40 seconds per fold per cell line each).

    Returns:
        dict[str, float]: A dictionary containing the following keys:
            - primary_metric: 'f1_score'
            - f1_score: f1 score averaged over classes and genes.
            - recall: recall averaged over classes and genes.
            - precision: precision score averaged over classes and genes.
            - roc_auc: ROC AUC averaged over classes and genes. The thresholds
            adapted to each gene leading to a slight loss in numeric precision.
            - average_precision: Precsion averaged over thresholds, classes and genes.
            The thresholds aren't adapted to each gene leading to a slight loss
            in numeric precision.

    """

    if isinstance(labels, ad.AnnData):
        labels = labels.X

    # preds can stay on device provided by user (avoids unnecessary copies to cpu and parallelization
    # across multiple gpus)
    if isinstance(preds, torch.Tensor):
        device = preds.device
    else:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    y_pred = torch.Tensor(preds).to(device=device)

    # has to be long not int otherwise indexing doesn't work
    # and -1,0,1 must be converted to 0,1,2
    y_true = torch.LongTensor(labels + 1).to(device=device)

    assert y_pred[:, 0, :].shape == y_true.shape, (
        f"Predictions and targets must have the same shape up to dimension 1 of predictions. Got {y_pred.shape} and {y_true.shape}"
    )

    metrics_to_compute = METRICS if skip_slow else {**SLOW_METRICS, **METRICS}

    results_dict = {
        metric_name: metric(y_pred, y_true).mean().item()
        for metric_name, metric in metrics_to_compute.items()
    }

    results_dict["primary_metric"] = "f1_score"
    return results_dict

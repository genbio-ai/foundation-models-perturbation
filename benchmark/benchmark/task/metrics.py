import functools
import typing

import torch
import torchmetrics
import tqdm
from jaxtyping import Float, Int, Bool
from torch import Tensor


# signature expected by the evaluate function
class VectorizedClassificationMetric(typing.Protocol):
    def __call__(
        self,
        predictions: Float[Tensor, " perturbations classes genes"],
        targets: Int[Tensor, " perturbations genes"],
    ) -> Float[Tensor, " genes"]: ...


@torch.no_grad
def f1_score(
    predictions: Float[Tensor, " perturbations classes genes"],
    targets: Int[Tensor, " perturbations genes"],
) -> Float[Tensor, " genes"]:
    result: Float[Tensor, " classes genes"] = torch.vmap(
        functools.partial(
            torchmetrics.functional.f1_score,
            num_classes=3,
            average=None,
            task="multiclass",
        ),
        in_dims=-1,
        out_dims=-1,
    )(predictions, targets)

    # Mean is computed only across classes that are present in the groundtruth
    # for each gene
    boolean_targets: Bool[Tensor, " perturbations genes classes"] = (
        torch.nn.functional.one_hot(targets, num_classes=3)
    )
    class_is_present: Bool[Tensor, " classes genes"] = boolean_targets.any(dim=0).T

    averaged = (result * class_is_present).sum(dim=0) / class_is_present.sum(dim=0)

    return averaged.detach().cpu()


@torch.no_grad
def recall(
    predictions: Float[Tensor, " perturbations classes genes"],
    targets: Int[Tensor, " perturbations genes"],
) -> Float[Tensor, " genes"]:
    result: Float[Tensor, " classes genes"] = torch.vmap(
        functools.partial(
            torchmetrics.functional.recall,
            num_classes=3,
            average=None,
            task="multiclass",
        ),
        in_dims=-1,
        out_dims=-1,
    )(predictions, targets)

    # Mean is computed only across classes that are present in the groundtruth
    # for each gene
    boolean_targets: Bool[Tensor, " perturbations genes classes"] = (
        torch.nn.functional.one_hot(targets, num_classes=3)
    )
    class_is_present: Bool[Tensor, " classes genes"] = boolean_targets.any(dim=0).T

    averaged = (result * class_is_present).sum(dim=0) / class_is_present.sum(dim=0)

    return averaged.detach().cpu()


@torch.no_grad
def precision(
    predictions: Float[Tensor, " perturbations classes genes"],
    targets: Int[Tensor, " perturbations genes"],
) -> Float[Tensor, " genes"]:
    result = torch.vmap(
        functools.partial(
            torchmetrics.functional.precision,
            num_classes=3,
            average=None,
            task="multiclass",
        ),
        in_dims=-1,
        out_dims=-1,
    )(predictions, targets)

    # Mean is computed only across classes that are present in the groundtruth
    # for each gene
    boolean_targets: Bool[Tensor, " perturbations genes classes"] = (
        torch.nn.functional.one_hot(targets, num_classes=3)
    )
    class_is_present: Bool[Tensor, " classes genes"] = boolean_targets.any(dim=0).T

    averaged = (result * class_is_present).sum(dim=0) / class_is_present.sum(dim=0)
    return averaged.detach().cpu()


@torch.no_grad
def roc_auc(
    predictions: Float[Tensor, " perturbations classes genes"],
    targets: Int[Tensor, " perturbations genes"],
) -> Float[Tensor, " genes"]:
    results = torch.stack(
        [
            torchmetrics.functional.auroc(
                predictions[..., i], targets[..., i], task="multiclass", num_classes=3
            )
            for i in tqdm.tqdm(range(targets.shape[-1]))
        ]
    )

    return results.detach().cpu()


@torch.no_grad
def average_precision(
    predictions: Float[Tensor, " perturbations classes genes"],
    targets: Int[Tensor, " perturbations genes"],
) -> Float[Tensor, " genes"]:
    # choices = torch.randint(len(predictions.unique()), size=(300,))
    # thresholds = torch.sort(predictions.unique()[choices])[0]

    results = torch.stack(
        [
            torchmetrics.functional.average_precision(
                predictions[..., i], targets[..., i], task="multiclass", num_classes=3
            )
            for i in tqdm.tqdm(range(targets.shape[-1]))
        ]
    )

    return results.detach().cpu()

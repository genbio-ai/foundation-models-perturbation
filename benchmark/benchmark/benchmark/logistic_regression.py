import numpy as np
import torch
from jaxtyping import Float, Int
from sklearn import base
from torch import nn, optim

from benchmark.task import metrics


class LinearModel(nn.Module):
    """Linear model for logistic regression"""

    def __init__(self, n_features, n_targets):
        super().__init__()
        self.n_features, self.n_targets = n_features, n_targets
        self.linear = nn.Linear(n_features, 3 * n_targets, bias=True)

    def forward(self, x):
        return self.linear(x).view(-1, self.n_targets, 3)  # (n_samples, n_targets, 3)


class LogisticRegression(base.BaseEstimator, base.ClassifierMixin):
    """Fit a vecotrized multiclass logistic regression"""

    def __init__(
        self,
        C: float = 10.0,
        balance_loss: bool = True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = C
        self.balance_loss = balance_loss

    def fit(self, X: Float[np.ndarray, "p d"], y: Int[np.ndarray, "p g"]):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.int64, device=self.device)
        self.model_ = LinearModel(X.shape[1], y.shape[1]).to(device=self.device)

        # Full batch LBFGS optimization
        kwargs = {"max_iter": 100, "line_search_fn": "strong_wolfe"}
        optimizer = optim.LBFGS(self.model_.parameters(), **kwargs)

        counts: Int[torch.Tensor, "g 3"] = (
            y[:, :, None] == torch.arange(0, 3, device=y.device)[None, None, :]
        ).sum(dim=0)

        # if a class isn't present it's weight is +inf. We replace with
        # 0 otherwise its contribution is 0 * inf = nan
        weights: Float[torch.Tensor, "g 3"] = (
            y.shape[0] / (3 * counts)
        ).nan_to_num(posinf=0)

        if not self.balance_loss:
            weights = torch.ones_like(weights)

        def closure():
            optimizer.zero_grad()
            outputs: Float[torch.Tensor, "p 3g"] = self.model_(X)
            outputs = outputs.view(outputs.shape[0], -1, 3)

            loss = (
                torch.vmap(
                    torch.nn.functional.cross_entropy, in_dims=(1, 1, 0), out_dims=-1
                )(outputs, y, weights).mean()
                + self.model_.linear.weight.norm(p="fro") ** 2 / self.C
                # Don't regularize bias because that's a bad idea
                # There may be a factor 2 missing compared to how C is sometimes defined
            )
            loss.backward()
            return loss

        optimizer.step(closure)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            return torch.argmax(self.model_(X), dim=-1).cpu()

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            return torch.softmax(self.model_(X), dim=-1).cpu()

    def score(self, X, y):
        self.model_.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            predictions = torch.softmax(self.model_(X), dim=-1).transpose(1, 2)

            f1_score = metrics.f1_score(
                predictions, torch.Tensor(y).long().to(device=self.device)
            )

            return f1_score.mean()

    def __repr__(self) -> str:
        return (
            f"Logistic Regression with inverse L2 regularization coefficient C={self.C}"
        )
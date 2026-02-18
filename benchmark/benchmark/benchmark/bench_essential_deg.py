import hydra
import torch
import numpy as np
import anndata as ad
from omegaconf import DictConfig

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from benchmark.benchmark import logistic_regression
from benchmark import BenchmarkTask
from benchmark import paths

@hydra.main(version_base=None, config_path="config/essential", config_name="config_essential_deg")
def main(cfg: DictConfig) -> None:

    # Load the training and test data
    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    # Load embedding
    if cfg.emb_name == "random":
        emb_name = "random"
        train_emb = np.random.random((train.X.shape[0], 100))
        test_emb = np.random.random((test.X.shape[0], 100))

        available_train_perts = train.obs["pert_id"].to_list()
        available_test_perts = test.obs["pert_id"].to_list()
    elif cfg.emb_name == "pca":
        emb_name = "pca"
        pca_emb = PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))
        train_emb = pca_emb[:train.shape[0]]
        test_emb = pca_emb[train.shape[0]:]

        available_train_perts = train.obs["pert_id"].to_list()
        available_test_perts = test.obs["pert_id"].to_list()
    else:
        emb_name = cfg.emb_name.replace("CELL_LINE", cfg.cell_line)
        emb = ad.read_h5ad(paths.GENE_EMBEDDINGS / f"{emb_name}.h5ad")
        emb = emb[emb.obs_names.isin(train.obs["pert_id"].tolist() + test.obs["pert_id"].tolist())].copy()

        available_train_perts = [pert for pert in train.obs["pert_id"] if pert in emb.obs_names]
        available_test_perts = [pert for pert in test.obs["pert_id"] if pert in emb.obs_names]

        train_emb = emb[available_train_perts].X.toarray()
        test_emb = emb[available_test_perts].X.toarray()

    # Define estimator pipeline
    assert cfg.estimator_name in ["logistic_regression", "no_change", "prior", "most_frequent"]
    if cfg.estimator_name == "no_change":
        estimator = DummyClassifier(strategy="constant", constant=np.ones(test.n_vars))
        emb_name = cfg.estimator_name
    elif cfg.estimator_name == "prior":
        estimator = DummyClassifier(strategy="prior")
        emb_name = cfg.estimator_name
    elif cfg.estimator_name == "most_frequent":
        estimator = DummyClassifier(strategy="most_frequent")
        emb_name = cfg.estimator_name
    elif cfg.estimator_name == "logistic_regression":
        model = GridSearchCV(
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=100)),
                ("classifier", logistic_regression.LogisticRegression(C=cfg.model.C, balance_loss=cfg.model.balance_loss)),
            ]),
            param_grid={"classifier__C": [1, 1e3, 1e5, 1e7]},
        )
        no_scaling_model = GridSearchCV(
            Pipeline([
                ("classifier", logistic_regression.LogisticRegression(C=cfg.model.C, balance_loss=cfg.model.balance_loss)),
            ]),
            param_grid={"classifier__C": [1, 1e3, 1e5, 1e7]},
        )
        estimator = no_scaling_model if emb_name == "pca" else model

    # Pad the prediction with most frequent,
    # and add one example of each class so predictions always have 3 classes
    backup_estimator = DummyClassifier(strategy="most_frequent")
    dummy_train_examples = train.X + 1
    dummy_train_examples = np.concatenate((
        dummy_train_examples,
        np.repeat(np.arange(3)[:, None], dummy_train_examples.shape[1], axis=1),
    ))
    backup_train_emb = np.random.random((dummy_train_examples.shape[0], 100))
    backup_test_emb = np.random.random((test.X.shape[0], 100))

    if cfg.estimator_name in {"prior", "most_frequent", "no_change"}:
        estimator.fit(backup_train_emb, dummy_train_examples)
        test_pred = torch.Tensor(np.stack(estimator.predict_proba(test_emb), axis=1))
    else:
        # Fit the backup estimator
        backup_estimator.fit(backup_train_emb, dummy_train_examples)
        test_pred = np.stack(backup_estimator.predict_proba(backup_test_emb), axis=1)

        # Fit the real estimator
        estimator.fit(train_emb, train.X[train.obs["pert_id"].isin(available_train_perts)] + 1)
        estimator_predictions = estimator.predict_proba(test_emb)
        test_pred = torch.Tensor(test_pred).to(estimator_predictions.device)
        test_pred[test.obs["pert_id"].isin(available_test_perts)] = estimator_predictions
    
    # Name the model
    if cfg.estimator_name in {"prior", "most_frequent", "no_change"}:
        name = cfg.estimator_name + " baseline"
    else:
        name = cfg.estimator_name + "_baseline_" + emb_name

    description = f"embedding: {emb_name}\nmodel: {estimator}\npca_components=100"

    # Evaluate the predictions
    return task.submit(test_pred.transpose(2, 1), name=name, description=description)


if __name__ == "__main__":
    main()

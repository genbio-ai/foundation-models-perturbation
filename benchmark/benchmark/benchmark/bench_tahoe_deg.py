import hydra
import torch
import numpy as np
import anndata as ad
from omegaconf import DictConfig

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from benchmark.benchmark import logistic_regression
from benchmark import BenchmarkTask
from benchmark import paths

@hydra.main(version_base=None, config_path="config/tahoe", config_name="config_tahoe_deg")
def main(cfg: DictConfig):

    # Load the training and test data
    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    # Load embeddings
    emb_name = cfg.emb_name
    if emb_name == "pca":
        pca_emb = PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))
        train_emb = pca_emb[: train.shape[0]]
        test_emb = pca_emb[train.shape[0]:]
    elif emb_name == "random":
        train_emb = np.random.random((train.X.shape[0], 100))
        test_emb = np.random.random((test.X.shape[0], 100))
    else:
        compound_embeddings = ad.read_h5ad(paths.TAHOE_DRUG_EMBEDDINGS)
        compound_embeddings.obs["pubchem_cid"] = compound_embeddings.obs_names.astype(np.int64)
        train_emb = compound_embeddings[train.obs["pert_id"].astype(int).astype(str).tolist()].obsm[emb_name]
        test_emb = compound_embeddings[test.obs["pert_id"].astype(int).astype(str).tolist()].obsm[emb_name]

    assert train_emb.shape[0] == train.n_obs
    assert test_emb.shape[0] == test.n_obs

    # Define estimator pipeline
    assert cfg.estimator_name in ["logistic_regression", "no_change", "prior", "most_frequent"]
    if cfg.estimator_name == "no_change":
        estimator = DummyClassifier(strategy="constant", constant=np.ones(test.n_vars))
    elif cfg.estimator_name == "prior":
        estimator = DummyClassifier(strategy="prior")
    elif cfg.estimator_name == "most_frequent":
        estimator = DummyClassifier(strategy="most_frequent")
    elif cfg.estimator_name == "logistic_regression":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=100)),
            ("classifier", logistic_regression.LogisticRegression(C=cfg.model.C, balance_loss=cfg.model.balance_loss)),
        ])
        no_scaling_model = Pipeline([
            ("classifier", logistic_regression.LogisticRegression(C=cfg.model.C, balance_loss=cfg.model.balance_loss)),
        ])
        estimator = no_scaling_model if emb_name == "pca" else model
    
    if cfg.estimator_name in {"prior", "most_frequent", "no_change"}:
        # Pad the prediction with most frequent,
        # and add one example of each class so predictions always have 3 classes
        dummy_train_examples = train.X + 1
        dummy_train_examples = np.concatenate((
            dummy_train_examples,
            np.repeat(np.arange(3)[:, None], dummy_train_examples.shape[1], axis=1),
        ))
        backup_train_emb = np.random.random((dummy_train_examples.shape[0], 100))
        backup_test_emb = np.random.random((test.X.shape[0], 100))

        # there is one sample of each class added on to make sure the dummy classifiers
        # predicts the right number of classes. This doesn't effect most_frequent and 
        # no_change but does slightly regularize prior
        estimator.fit(backup_train_emb, dummy_train_examples)
        predictions = torch.Tensor(np.stack(estimator.predict_proba(backup_test_emb), axis=1)).transpose(2, 1)
    else:
        estimator.fit(train_emb, train.X + 1)
        predictions = estimator.predict_proba(test_emb).transpose(2, 1)

    # Name the model
    if cfg.estimator_name in {"prior", "most_frequent", "no_change"}:
        name = cfg.estimator_name + " baseline"
    else:
        name = cfg.estimator_name + "_baseline_" + emb_name

    description = f"embedding: {emb_name}\nmodel: {estimator}\npca_components=100"

    # Evaluate the predictions
    return task.submit(predictions, name=f"{cfg.estimator_name}_{cfg.emb_name}", description=description)


if __name__ == "__main__":
    main()

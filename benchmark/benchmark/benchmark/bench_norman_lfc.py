import hydra
import numpy as np
import anndata as ad
from omegaconf import DictConfig

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold

from benchmark import BenchmarkTask
from benchmark import paths

@hydra.main(version_base=None, config_path="config/norman", config_name="config_norman_lfc")
def main(cfg: DictConfig) -> None:

    # Load the training and test data
    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    # Load embedding
    if cfg.emb_name == "random":
        emb_name = "random"
        train_emb = np.random.random((train.X.shape[0], 50))
        test_emb = np.random.random((test.X.shape[0], 50))

        available_train_perts = train.obs["pert_id"].tolist()
        available_test_perts = test.obs["pert_id"].tolist()
    elif cfg.emb_name == "pca":
        emb_name = "pca"
        pca_emb = PCA(n_components=50).fit_transform(np.concatenate((train.X, test.X)))
        train_emb = pca_emb[:train.shape[0]]
        test_emb = pca_emb[train.shape[0]:]

        available_train_perts = train.obs["pert_id"].tolist()
        available_test_perts = test.obs["pert_id"].tolist()
    else:
        emb_name = cfg.emb_name.replace("CELL_LINE", cfg.cell_line)
        emb = ad.read_h5ad(paths.GENE_EMBEDDINGS / f"{emb_name}.h5ad")
        emb = emb[emb.obs_names.isin(train.obs["pert_id"].tolist() + test.obs["pert_id"].tolist())].copy()

        available_train_perts = [pert for pert in train.obs["pert_id"] if pert in emb.obs_names]
        available_test_perts = [pert for pert in test.obs["pert_id"] if pert in emb.obs_names]

        train_emb = emb[available_train_perts].X.toarray()
        test_emb = emb[available_test_perts].X.toarray()

    # Define estimator pipeline
    assert cfg.estimator_name in ["no change", "context mean", "knn", "lasso"]
    if cfg.estimator_name == "no change":
        pipeline = DummyRegressor(strategy="constant", constant=np.zeros(5045))
        hparam_grid = {}
    elif cfg.estimator_name == "context mean":
        pipeline = DummyRegressor()
        hparam_grid = {}
    elif cfg.estimator_name == "knn":
        if cfg.emb_name == "pca":
            pipeline = Pipeline([("pseudobulk", KNeighborsRegressor())])
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=50)),
                ("pseudobulk", KNeighborsRegressor()),
            ])
        hparam_grid = {"pseudobulk__n_neighbors": [5, 10, 15, 20]}
    elif cfg.estimator_name == "lasso":
        if cfg.emb_name == "pca":
            pipeline = Pipeline([("pseudobulk", Lasso())])
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=50)),
                ("pseudobulk", Lasso()),
            ])
        hparam_grid = {"pseudobulk__alpha": [1e-3, 1e-2, 1e-1, 1]}

    # Build estimator
    def l2(y, y_pred):
        return -np.linalg.norm(y - y_pred, axis=1).mean()

    estimator = GridSearchCV(
        pipeline,
        hparam_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring=make_scorer(l2),
    )

    # Pad the prediction with context mean
    backup_estimator = DummyRegressor()
    backup_train_emb = np.random.random((train.X.shape[0], 50))
    backup_test_emb = np.random.random((test.X.shape[0], 50))
    backup_estimator.fit(backup_train_emb, train.X)
    backup_preds = backup_estimator.predict(backup_test_emb)
    test_pred = ad.AnnData(backup_preds, obs=test.obs.copy(), var=test.var.copy())

    # Fit the model and make actual predictions
    estimator.fit(train_emb, train.X[train.obs["pert_id"].isin(available_train_perts)])
    test_pred.X[test.obs["pert_id"].isin(available_test_perts)] = estimator.predict(test_emb)

    # Name the model
    if cfg.estimator_name in ["context mean", "no change"]:
        name = cfg.estimator_name + " baseline"
    else:
        name = cfg.estimator_name + "_baseline_" + emb_name

    # Describe the model
    if emb_name == "random":
        description = "Embedding: np.random.random((..., 50))\n"
    elif emb_name == "pca":
        description = "Embedding: PCA(n_components=50).fit_transform(np.concatenate((train.X, test.X)))\n"
    else:
        description = "Embedding: " + emb_name + "\n"
    description += "Sklearn pipeline: " + str(estimator) + "\n"
    description += "Best params: " + str(estimator.best_params_) + "\n"
    description += "Missing embeddings are filled with context mean: DummyRegressor()\n"

    # Evaluate the predictions
    return task.submit(test_pred, name=name, description=description)

if __name__ == "__main__":
    main()
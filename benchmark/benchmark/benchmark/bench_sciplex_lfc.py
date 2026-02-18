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

@hydra.main(version_base=None, config_path="config/sciplex", config_name="config_sciplex_lfc")
def main(cfg: DictConfig) -> None:

    # Load the training and test data
    task = BenchmarkTask(cfg.task_name, f"{cfg.cell_line}.{cfg.fold_id}")
    train, test = task.setup()

    # Load embedding
    if cfg.emb_name == "random":
        emb_name = "random"
        train_emb = np.random.random((train.X.shape[0], 100))
        test_emb = np.random.random((test.X.shape[0], 100))
    elif cfg.emb_name == "pca":
        emb_name = "pca"
        pca_emb = PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))
        train_emb = pca_emb[:train.shape[0]]
        test_emb = pca_emb[train.shape[0]:]
    else:
        emb_name = cfg.emb_name
        emb = ad.read_h5ad(paths.SCIPLEX_DRUG_EMBEDDINGS)
        train_emb = emb[train.obs["drug"].astype(str).tolist()].obsm[cfg.emb_name]
        test_emb = emb[test.obs["drug"].astype(str).tolist()].obsm[cfg.emb_name]
        
    assert train_emb.shape[0] == train.n_obs
    assert test_emb.shape[0] == test.n_obs

    # Define estimator pipeline
    assert cfg.estimator_name in ["no change", "context mean", "knn", "lasso"]
    if cfg.estimator_name == "no change":
        pipeline = DummyRegressor(strategy="constant", constant=np.zeros(train.n_vars))
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
                ("pca", PCA(n_components=100)),
                ("pseudobulk", KNeighborsRegressor()),
            ])
        hparam_grid = {"pseudobulk__n_neighbors": [20, 40, 60, 80, 100]}
    elif cfg.estimator_name == "lasso":
        if cfg.emb_name == "pca":
            pipeline = Pipeline([("pseudobulk", Lasso())])
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=100)),
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

    # Fit the model and make predictions
    estimator.fit(train_emb, train.X)
    preds = estimator.predict(test_emb)
    test_pred = ad.AnnData(preds, obs=test.obs.copy(), var=test.var.copy())
    
    # Name the model
    if cfg.estimator_name in ["context mean", "no change"]:
        name = cfg.estimator_name + " baseline"
    else:
        name = cfg.estimator_name + "_baseline_" + emb_name

    # Describe the model
    if emb_name == "random":
        description = "Embedding: np.random.random((..., 100))\n"
    elif emb_name == "pca":
        description = "Embedding: PCA(n_components=100).fit_transform(np.concatenate((train.X, test.X)))\n"
    else:
        description = "Embedding: " + emb_name + " from " + str(paths.SCIPLEX_DRUG_EMBEDDINGS) + "\n"
    description += "Sklearn pipeline: " + str(estimator) + "\n"
    description += "Best params: " + str(estimator.best_params_) + "\n"

    # Evaluate the predictions
    return task.submit(test_pred, name=name, description=description)

if __name__ == "__main__":
    main()
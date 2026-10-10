from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUPPORTED_REGRESSORS = (
    "Huber",
    "Ridge",
    "CatBoost",
    "PLSR",
    "KNN",
    "ElasticNet",
    "ExtraTrees",
    "GradientBoosting",
    "RandomForest",
    "ANN",
)


def make_regressor(
    name: str,
    n_train: int,
    seed: int,
    catboost_parameters: dict | None = None,
):
    """Create one of the ten fixed Stage-3 regressors used for fair comparison."""
    if name not in SUPPORTED_REGRESSORS:
        raise ValueError(f"Unknown regressor {name!r}; choose from {SUPPORTED_REGRESSORS}")
    if name == "Huber":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=50000, tol=1e-4)),
        ])
    if name == "Ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if name == "PLSR":
        components = max(1, min(3, int(n_train) - 1))
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=components, scale=False, max_iter=2000)),
        ])
    if name == "KNN":
        neighbors = max(2, min(7, int(np.sqrt(max(n_train, 4)))))
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=neighbors, weights="distance", p=2)),
        ])
    if name == "ElasticNet":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(
                alpha=0.01, l1_ratio=0.3, max_iter=100000,
                tol=1e-4, selection="random", random_state=seed,
            )),
        ])
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=16, min_samples_leaf=3,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.03, max_depth=2,
            min_samples_leaf=3, loss="huber", random_state=seed,
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
    if name == "ANN":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs",
                alpha=0.01, max_iter=3000, random_state=seed,
            )),
        ])

    from catboost import CatBoostRegressor

    parameters = {
        "iterations": 700,
        "depth": 3,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
    }
    if catboost_parameters:
        parameters.update(catboost_parameters)
    return CatBoostRegressor(
        **parameters,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


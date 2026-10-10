"""Evaluate strict spatial five-fold SSC regression from exported image states."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from msrl4ssc.regressors import SUPPORTED_REGRESSORS, make_regressor


SEED = 74
PCA_COMPONENTS = 4
CATBOOST_PARAMETERS = {
    "iterations": 700,
    "depth": 3,
    "learning_rate": 0.04,
    "l2_leaf_reg": 5.0,
}
PATHS = {
    "S1": {
        "state_key": "S1_FoldMsrlImage",
        "conventional": ["VH", "DEM", "Soil_Clay", "Soil_Sand", "Pb_030"],
    },
    "S2": {
        "state_key": "S2_FoldMsrlImage",
        "conventional": [
            "SI3", "SI5", "SI8", "kNDMI", "TCARI",
            "DEM", "Soil_Clay", "Soil_Sand", "Pb_030",
        ],
    },
    "Both": {
        "state_key": "Both_FoldMsrlImage",
        "conventional": [
            "SI3", "SI5", "SI8", "kNDMI", "TCARI",
            "DEM", "Soil_Clay", "Soil_Sand", "Pb_030", "VH",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3"))
    return parser.parse_args()


def estimator(model_name: str, raw_dim: int, state_dim: int, n_train: int) -> Pipeline:
    transform = ColumnTransformer([
        ("conventional", "passthrough", list(range(raw_dim))),
        (
            "image_pca",
            PCA(n_components=PCA_COMPONENTS, random_state=SEED, svd_solver="randomized"),
            list(range(raw_dim, raw_dim + state_dim)),
        ),
    ])
    regressor = make_regressor(
        model_name,
        n_train=n_train,
        seed=SEED,
        catboost_parameters=CATBOOST_PARAMETERS,
    )
    return Pipeline([("features", transform), ("regressor", regressor)])


def main() -> None:
    args = parse_args()
    feature_root = args.feature_root.resolve()
    sample_index = (
        args.sample_index.resolve()
        if args.sample_index is not None
        else feature_root.parent / "master_index_seed74.csv"
    )
    protocol_file = feature_root / "msrl_oof_features.npz"
    for required in (sample_index, protocol_file):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")

    frame = pd.read_csv(sample_index)
    protocol = np.load(protocol_file, allow_pickle=True)
    sample_ids = np.asarray(protocol["Sample_ID"]).astype(str)
    indexed = frame.set_index(frame["Sample_ID"].astype(str), drop=False)
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in indexed.index]
    if missing_ids:
        raise KeyError(f"Sample index is missing {len(missing_ids)} feature IDs")
    frame = indexed.loc[sample_ids].reset_index(drop=True)
    target = np.asarray(protocol["SSC"], dtype=np.float64)
    folds = np.asarray(protocol["fold_74"], dtype=np.int64)
    if not np.array_equal(np.sort(np.unique(folds)), np.arange(1, 6)):
        raise ValueError("fold_74 must contain the five folds 1..5")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []
    oof_columns: dict[str, np.ndarray] = {}
    for path_name, specification in PATHS.items():
        columns = specification["conventional"]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(f"{path_name} conventional columns are missing: {missing}")
        raw = frame[columns].to_numpy(np.float32)
        for model_name in SUPPORTED_REGRESSORS:
            prediction = np.full(len(frame), np.nan, dtype=np.float64)
            for fold in range(1, 6):
                train = folds != fold
                test = folds == fold
                payload_file = feature_root / f"source_fold_{fold}" / "all_sample_features.npz"
                if not payload_file.is_file():
                    raise FileNotFoundError(f"Missing fold feature file: {payload_file}")
                payload = np.load(payload_file, allow_pickle=True)
                states = np.asarray(payload[specification["state_key"]], dtype=np.float32)
                if len(states) != len(frame):
                    raise ValueError(f"Feature/sample length mismatch in {payload_file}")
                x = np.column_stack([raw, states])
                model = estimator(model_name, raw.shape[1], states.shape[1], int(train.sum()))
                model.fit(x[train], target[train])
                fold_prediction = np.asarray(model.predict(x[test])).reshape(-1)
                prediction[test] = fold_prediction
                fold_rows.append({
                    "fold": fold,
                    "path": path_name,
                    "model": model_name,
                    "n_test": int(test.sum()),
                    "pca_components": PCA_COMPONENTS,
                    "r2": float(r2_score(target[test], fold_prediction)),
                    "rmse": float(mean_squared_error(target[test], fold_prediction) ** 0.5),
                })
            oof_columns[f"prediction_{path_name}_{model_name}"] = prediction

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby(["path", "model"], sort=False)
        .agg(
            mean_fold_r2=("r2", "mean"),
            std_fold_r2=("r2", "std"),
            mean_fold_rmse=("rmse", "mean"),
            std_fold_rmse=("rmse", "std"),
        )
        .reset_index()
    )
    for row in summary.itertuples():
        prediction = oof_columns[f"prediction_{row.path}_{row.model}"]
        mask = np.isfinite(prediction)
        summary.loc[
            (summary.path == row.path) & (summary.model == row.model), "oof_r2"
        ] = r2_score(target[mask], prediction[mask])
        summary.loc[
            (summary.path == row.path) & (summary.model == row.model), "oof_rmse"
        ] = mean_squared_error(target[mask], prediction[mask]) ** 0.5

    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame({
        "Sample_ID": sample_ids,
        "fold": folds,
        "SSC": target,
        **oof_columns,
    }).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


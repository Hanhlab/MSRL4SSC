from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "RPD": float("nan"), "Bias": float("nan"), "n": 0}
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")
    return {
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "RPD": float(sd / rmse) if np.isfinite(sd) and rmse > 0 else float("nan"),
        "Bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }

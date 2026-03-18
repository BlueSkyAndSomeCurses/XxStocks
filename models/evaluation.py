from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def _to_numpy_1d(values) -> np.ndarray:
    arr = np.asarray(values)
    return arr.reshape(-1)


def _binarize_sign(values) -> np.ndarray:
    arr = _to_numpy_1d(values)
    return np.where(arr >= 0, 1, -1)


def evaluate_predictions(y_true, y_pred, task: str = "binary") -> dict[str, float]:
    y_true_arr = _to_numpy_1d(y_true)
    y_pred_arr = _to_numpy_1d(y_pred)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    n = min(y_true_arr.size, y_pred_arr.size)
    y_true_arr = y_true_arr[:n]
    y_pred_arr = y_pred_arr[:n]

    if task == "binary":
        y_true_cls = _binarize_sign(y_true_arr)
        y_pred_cls = _binarize_sign(y_pred_arr)

        return {
            "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true_cls, y_pred_cls)
            ),
            "precision": float(
                precision_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)
            ),
            "f1": float(f1_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)),
        }

    if task in {"continuous", "non-binary", "non_binary"}:
        mae = mean_absolute_error(y_true_arr, y_pred_arr)
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        r2 = r2_score(y_true_arr, y_pred_arr)

        denom = np.where(np.abs(y_true_arr) < 1e-8, np.nan, np.abs(y_true_arr))
        mape = np.abs((y_true_arr - y_pred_arr) / denom)
        mape = float(np.nanmean(mape)) if np.isfinite(np.nanmean(mape)) else float("nan")

        directional_accuracy = float(
            (_binarize_sign(y_true_arr) == _binarize_sign(y_pred_arr)).mean()
        )

        return {
            "mae": float(mae),
            "rmse": rmse,
            "r2": float(r2),
            "mape": mape,
            "directional_accuracy": directional_accuracy,
        }

    raise ValueError("task must be either 'binary' or 'continuous'.")

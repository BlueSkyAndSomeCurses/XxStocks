from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


def _to_numpy_1d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1)


def _binarize_sign(values) -> np.ndarray:
    arr = _to_numpy_1d(values)
    return np.where(arr >= 0, 1, -1)


def directional_accuracy_returns(y_true, y_pred, ignore_flat: bool = True) -> float:
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    n = min(y_true.size, y_pred.size)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)

    if ignore_flat:
        mask = true_dir != 0
        if not np.any(mask):
            return float("nan")
        return float((true_dir[mask] == pred_dir[mask]).mean())

    return float((true_dir == pred_dir).mean())


def weighted_directional_accuracy(y_true, y_pred) -> float:
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    n = min(y_true.size, y_pred.size)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    weights = np.abs(y_true)
    total_weight = weights.sum()
    if total_weight == 0:
        return float("nan")

    correct = (np.sign(y_true) == np.sign(y_pred)).astype(float)
    return float((correct * weights).sum() / total_weight)


def information_coefficient(y_true, y_pred, method: str = "pearson") -> float:
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    n = min(y_true.size, y_pred.size)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    if method == "spearman":
        coef, _ = spearmanr(y_true, y_pred)
    else:
        coef, _ = pearsonr(y_true, y_pred)
    return float(coef)


def strategy_pnl_metrics(
    y_true_returns,
    y_pred,
    bps_cost: float = 0.0,
    annualisation: float = 252 * 13,
) -> dict[str, float]:
    y_true = _to_numpy_1d(y_true_returns)
    y_pred = _to_numpy_1d(y_pred)
    n = min(y_true.size, y_pred.size)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    positions = np.sign(y_pred)
    pnl = positions * y_true

    if bps_cost > 0:
        turnover = np.abs(np.diff(positions, prepend=0.0))
        pnl = pnl - turnover * (bps_cost / 1e4)

    mean_ret = float(pnl.mean())
    std_ret = float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0
    sharpe = (
        float(mean_ret / std_ret * np.sqrt(annualisation))
        if std_ret > 0
        else float("nan")
    )

    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = float(drawdown.min()) if drawdown.size else 0.0

    return {
        "pnl_total": float(cum[-1]) if cum.size else 0.0,
        "pnl_mean_per_bar": mean_ret,
        "sharpe_annualised": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": float((pnl > 0).mean()) if pnl.size else float("nan"),
    }


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
        r2 = float(r2_score(y_true_arr, y_pred_arr))

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
            "r2": r2
        }

    if task in {"continuous", "non-binary", "non_binary"}:
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
        r2 = float(r2_score(y_true_arr, y_pred_arr))

        # Standard MAPE blows up near zero log-returns; mask the offenders.
        denom = np.where(np.abs(y_true_arr) < 1e-8, np.nan, y_true_arr)
        mape = float(np.nanmean(np.abs((y_true_arr - y_pred_arr) / denom)))

        dir_acc = directional_accuracy_returns(y_true_arr, y_pred_arr)
        wda = weighted_directional_accuracy(y_true_arr, y_pred_arr)
        ic_p = information_coefficient(y_true_arr, y_pred_arr, method="pearson")
        ic_s = information_coefficient(y_true_arr, y_pred_arr, method="spearman")
        pnl = strategy_pnl_metrics(y_true_arr, y_pred_arr)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "weighted_directional_accuracy": wda,
            "information_coefficient_pearson": ic_p,
            "information_coefficient_spearman": ic_s,
            **{f"strategy_{k}": v for k, v in pnl.items()},
        }

    raise ValueError("task must be either 'binary' or 'continuous'.")

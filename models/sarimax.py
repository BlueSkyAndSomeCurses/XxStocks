import polars as pl
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


def train_sarimax(
    y: pl.Series,
    X: pl.DataFrame,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    disp: int | None = 5
):
    endog = np.asarray(y, dtype=float)
    exog = np.asarray(X, dtype=float)

    model = SARIMAX(
        endog=endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        # time_varying_regression=True,
        # mle_regression=False,
        # simple_differencing=True,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model.fit(disp=disp)


def forecast_sarimax(fitted_model, X_test: pl.DataFrame, y_test: pl.Series | None = None):
    """Static N-step forecast.

    Kept for backwards compatibility, but for any test horizon longer than the
    AR/MA order the AR component decays to its unconditional mean and the
    forecast collapses onto the exog-only regression line. Prefer
    ``rolling_one_step_forecast`` for evaluation.
    """
    exog_test = np.asarray(X_test, dtype=float)
    forecast = fitted_model.forecast(steps=len(exog_test), exog=exog_test)
    return forecast


def rolling_one_step_forecast(
    fitted_model,
    y_test: pl.Series,
    X_test: pl.DataFrame,
):
    """Walk-forward one-step-ahead predictions.

    For each step we ask the model for ``forecast(steps=1, exog=x_t)`` and then
    extend the model state with the realised ``(y_t, x_t)`` via ``append`` so
    the next prediction conditions on the true history rather than its own
    multi-step trajectory. This is the standard fair evaluation protocol for
    SARIMAX on a held-out test set and avoids the magnitude collapse we saw in
    the previous report (``R² ≈ -5000``).
    """
    exog_test = np.asarray(X_test, dtype=float)
    endog_test = np.asarray(y_test, dtype=float)

    preds = np.empty(len(endog_test), dtype=float)
    state = fitted_model

    for t in range(len(endog_test)):
        x_step = exog_test[t : t + 1]
        step_forecast = state.forecast(steps=1, exog=x_step)
        preds[t] = float(np.asarray(step_forecast)[0])
        state = state.append(endog=endog_test[t : t + 1], exog=x_step, refit=False)

    return preds


def direction_accuracy(actual: pl.Series, predicted_values) -> float:
    actual_diff = actual.diff().drop_nulls().to_numpy()

    if len(predicted_values) < 2:
        return 0.0

    pred_series = pl.Series(predicted_values)
    pred_diff = pred_series.diff().drop_nulls().to_numpy()

    n = min(len(actual_diff), len(pred_diff))
    if n == 0:
        return 0.0

    actual_dir = actual_diff[:n] > 0
    pred_dir = pred_diff[:n] > 0
    return float((actual_dir == pred_dir).mean())

def save_sarimax_model(fitted_model, filepath: str):
    try:
        fitted_model.save(filepath)
    except Exception as e:
        print(f"Failed to save model: {e}")

def load_sarimax_model(filepath: str):
    try:
        model = SARIMAXResults.load(filepath)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
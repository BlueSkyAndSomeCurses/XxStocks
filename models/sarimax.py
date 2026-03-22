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
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model.fit(disp=disp)


def forecast_sarimax(fitted_model, X_test: pl.DataFrame):
    exog_test = np.asarray(X_test, dtype=float)
    forecast = fitted_model.forecast(steps=len(exog_test), exog=exog_test)
    return forecast


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
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.forecasting import regression
from src.forecasting.data_loader import aligned_xy, load_price_monthly
from src.forecasting.schemas import Asset, Method

MIN_TRAIN_MONTHS = 36


class BacktestReport(BaseModel):
    asset: Asset
    method: Method
    horizon_months: int
    n_windows: int
    mape_pct: float
    rmse: float
    coverage_80_pct: float
    coverage_95_pct: float
    per_window: list[dict]


def _sarima_window(train: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=100)
        fc = fit.get_forecast(steps=horizon)
        return (fc.predicted_mean.values,
                fc.conf_int(alpha=0.20).values,
                fc.conf_int(alpha=0.05).values)


def _regression_window(X_train: pd.DataFrame, y_train: pd.Series,
                       X_future: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_clean, _, _ = regression.drop_collinear(X_train)
    X_future_clean = X_future[X_clean.columns]
    model, scaler = regression._fit(X_clean, y_train)
    X_scaled = scaler.transform(X_clean)
    X_future_scaled = scaler.transform(X_future_clean)
    point = model.predict(X_future_scaled)
    sims = regression._bootstrap_residuals(model, X_scaled, y_train.values, X_future_scaled)
    ci80 = np.column_stack([np.percentile(sims, 10, axis=0), np.percentile(sims, 90, axis=0)])
    ci95 = np.column_stack([np.percentile(sims, 2.5, axis=0), np.percentile(sims, 97.5, axis=0)])
    return point, ci80, ci95


def rolling_backtest(asset: Asset, method: Method, *,
                     horizon_months: int = 3, step_months: int = 6,
                     min_train: int = MIN_TRAIN_MONTHS) -> BacktestReport:
    if method == "sarima":
        series = load_price_monthly(asset)
        X = None
    else:
        X, series = aligned_xy(asset)

    n = len(series)
    if n < min_train + horizon_months + step_months:
        raise ValueError(f"Слишком короткий ряд для backtest ({n} точек)")

    windows: list[dict] = []
    actuals: list[float] = []
    preds: list[float] = []
    in_80: list[bool] = []
    in_95: list[bool] = []

    for end in range(min_train, n - horizon_months + 1, step_months):
        train_series = series.iloc[:end]
        if method == "sarima":
            point, ci80, ci95 = _sarima_window(train_series, horizon_months)
        else:
            X_train = X.iloc[:end]
            X_future = X.iloc[end:end + horizon_months]
            point, ci80, ci95 = _regression_window(X_train, train_series, X_future)
        truth = series.iloc[end:end + horizon_months].values
        for i in range(horizon_months):
            actuals.append(float(truth[i]))
            preds.append(float(point[i]))
            in_80.append(bool(ci80[i, 0] <= truth[i] <= ci80[i, 1]))
            in_95.append(bool(ci95[i, 0] <= truth[i] <= ci95[i, 1]))
        windows.append({
            "train_end": str(series.index[end - 1].date()),
            "horizon_start": str(series.index[end].date()),
            "actuals": [float(v) for v in truth],
            "predictions": [float(v) for v in point],
        })

    actuals_arr = np.array(actuals)
    preds_arr = np.array(preds)
    mape = float(np.mean(np.abs((actuals_arr - preds_arr) / actuals_arr)) * 100)
    rmse = float(np.sqrt(np.mean((actuals_arr - preds_arr) ** 2)))
    coverage_80 = float(np.mean(in_80) * 100)
    coverage_95 = float(np.mean(in_95) * 100)

    return BacktestReport(
        asset=asset, method=method, horizon_months=horizon_months,
        n_windows=len(windows), mape_pct=mape, rmse=rmse,
        coverage_80_pct=coverage_80, coverage_95_pct=coverage_95,
        per_window=windows,
    )

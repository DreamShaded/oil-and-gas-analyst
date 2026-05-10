from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.forecasting.schemas import ForecastPoint


def fit_and_forecast(series: pd.Series, horizon_months: int,
                     order: tuple[int, int, int] = (1, 1, 1),
                     seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)
                     ) -> tuple[list[ForecastPoint], dict]:
    if len(series) < 36:
        raise ValueError(f"Слишком короткий ряд для SARIMA (нужно ≥36 точек, есть {len(series)})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=200)

        fc = fit.get_forecast(steps=horizon_months)
        mean = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)

    points: list[ForecastPoint] = []
    for ts, point, (l80, u80), (l95, u95) in zip(
        mean.index, mean.values, ci80.values, ci95.values, strict=True,
    ):
        points.append(ForecastPoint(
            period=pd.Timestamp(ts).date(),
            point=float(point),
            lower_80=float(l80), upper_80=float(u80),
            lower_95=float(l95), upper_95=float(u95),
        ))

    diagnostics = {
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "n_observations": int(len(series)),
        "last_observed_period": str(series.index[-1].date()),
        "last_observed_value": float(series.iloc[-1]),
    }
    return points, diagnostics

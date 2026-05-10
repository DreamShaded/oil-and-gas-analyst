from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.forecasting.backtest import _regression_window, _sarima_window
from src.forecasting.data_loader import aligned_xy, load_price_monthly
from src.forecasting.schemas import Asset, Method


class ConformalEval(BaseModel):
    asset: Asset
    method: Method
    horizon_months: int
    n_calib_windows: int
    n_test_windows: int
    q80_per_step: list[float]
    q95_per_step: list[float]
    coverage_80_pct: float
    coverage_95_pct: float
    mape_pct: float
    rmse: float


def _forecast(asset: Asset, method: Method, series_train: pd.Series,
              X_train: pd.DataFrame | None, X_future: pd.DataFrame | None,
              horizon: int) -> np.ndarray:
    if method == "sarima":
        point, _, _ = _sarima_window(series_train, horizon)
    else:
        point, _, _ = _regression_window(X_train, series_train, X_future)
    return point


def honest_conformal_eval(asset: Asset, method: Method, *,
                          horizon_months: int = 3,
                          step_months: int = 6,
                          calib_frac: float = 0.5) -> ConformalEval:
    """Честная оценка покрытия conformal CI.

    Делим **бэктест-окна** на calibration (первая половина) и test (вторая):
    - на calib собираем |error| → квантили q80/q95 per horizon step,
    - на test проверяем фактическое покрытие интервалом ŷ ± q.
    Никакого data-leakage: квантили считаются по предыдущим окнам, проверка — на будущих.
    """
    if method == "sarima":
        series = load_price_monthly(asset)
        X = None
    else:
        X, series = aligned_xy(asset)

    n = len(series)
    min_train = 36
    starts = list(range(min_train, n - horizon_months + 1, step_months))
    if len(starts) < 6:
        raise ValueError(f"Слишком мало окон ({len(starts)}) для honest-эвала")

    # Adaptive conformal: калибровочные окна — это **последние** окна перед test'ом,
    # а не первые, чтобы quantile отражал текущий режим волатильности.
    cut = int(len(starts) * (1.0 - calib_frac))
    test_starts = starts[cut:]
    calib_starts = starts[max(0, cut - len(test_starts)):cut]

    # Относительные ошибки |y-ŷ|/y; квантили в долях.
    calib_rel: list[list[float]] = [[] for _ in range(horizon_months)]
    test_rel_with_actuals: list[list[tuple[float, float]]] = [[] for _ in range(horizon_months)]
    test_actuals: list[float] = []
    test_preds: list[float] = []

    for end in calib_starts + test_starts:
        train = series.iloc[:end]
        if method == "regression":
            X_tr = X.iloc[:end]
            X_fu = X.iloc[end:end + horizon_months]
            preds = _forecast(asset, method, train, X_tr, X_fu, horizon_months)
        else:
            preds = _forecast(asset, method, train, None, None, horizon_months)
        truth = series.iloc[end:end + horizon_months].values
        for i in range(horizon_months):
            actual = float(truth[i])
            pred = float(preds[i])
            if abs(actual) < 1e-9:
                continue
            rel = abs(actual - pred) / abs(actual)
            if end in calib_starts:
                calib_rel[i].append(rel)
            else:
                test_rel_with_actuals[i].append((rel, pred))
                test_actuals.append(actual)
                test_preds.append(pred)

    # Те же ACI-параметры, что и в `conformal.py` — для согласованности теста и продакшна.
    PCTL_80 = 0.87
    PCTL_95 = 0.97
    INFLATION = 1.15
    q80 = [float(np.quantile(e, PCTL_80)) * INFLATION if e else 0.0 for e in calib_rel]
    q95 = [float(np.quantile(e, PCTL_95)) * INFLATION if e else 0.0 for e in calib_rel]

    # Покрытие: интервал применяется в долях от ŷ → проверка |error|/y vs q*y/y → rel <= q
    in_80: list[bool] = []
    in_95: list[bool] = []
    for i, step_errs in enumerate(test_rel_with_actuals):
        for rel, _pred in step_errs:
            in_80.append(rel <= q80[i])
            in_95.append(rel <= q95[i])

    coverage_80 = float(np.mean(in_80) * 100) if in_80 else 0.0
    coverage_95 = float(np.mean(in_95) * 100) if in_95 else 0.0
    actuals = np.array(test_actuals)
    preds = np.array(test_preds)
    mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100) if len(actuals) else 0.0
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2))) if len(actuals) else 0.0

    return ConformalEval(
        asset=asset, method=method, horizon_months=horizon_months,
        n_calib_windows=len(calib_starts), n_test_windows=len(test_starts),
        q80_per_step=q80, q95_per_step=q95,
        coverage_80_pct=coverage_80, coverage_95_pct=coverage_95,
        mape_pct=mape, rmse=rmse,
    )

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.forecasting.schemas import ForecastPoint, ScenarioShock

BOOTSTRAP_ITERATIONS = 1000
VIF_THRESHOLD = 10.0


def _compute_vif(X: pd.DataFrame) -> dict[str, float]:
    if X.shape[1] < 2:
        return {col: 1.0 for col in X.columns}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return {
        col: float(variance_inflation_factor(X_scaled, i))
        for i, col in enumerate(X.columns)
    }


def drop_collinear(X: pd.DataFrame, threshold: float = VIF_THRESHOLD) -> tuple[pd.DataFrame, dict, list[str]]:
    """Итеративно выкидывает фактор с самым высоким VIF, пока все VIF < threshold."""
    current = X.copy()
    dropped: list[str] = []
    while current.shape[1] > 1:
        vif = _compute_vif(current)
        worst = max(vif, key=lambda k: vif[k])
        if vif[worst] < threshold:
            break
        dropped.append(worst)
        current = current.drop(columns=[worst])
    final_vif = _compute_vif(current)
    return current, final_vif, dropped


def _fit(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    return model, scaler


def _bootstrap_residuals(model: Ridge, X_scaled: np.ndarray, y: np.ndarray,
                         X_future_scaled: np.ndarray) -> np.ndarray:
    residuals = y - model.predict(X_scaled)
    rng = np.random.default_rng(42)
    horizon = X_future_scaled.shape[0]
    sims = np.zeros((BOOTSTRAP_ITERATIONS, horizon))
    for i in range(BOOTSTRAP_ITERATIONS):
        boot = rng.choice(residuals, size=horizon, replace=True)
        sims[i] = model.predict(X_future_scaled) + boot
    return sims


def fit_and_forecast(X: pd.DataFrame, y: pd.Series,
                     horizon_months: int,
                     shocks: list[ScenarioShock] | None = None
                     ) -> tuple[list[ForecastPoint], dict]:
    if len(y) < 24:
        raise ValueError(f"Слишком короткий ряд (нужно ≥24 точек, есть {len(y)})")
    if X.empty:
        raise ValueError("Нет фундаментальных факторов")

    X_clean, vifs, dropped = drop_collinear(X)
    if shocks:
        bad = [s.factor for s in shocks if s.factor not in X_clean.columns]
        if bad:
            raise ValueError(
                f"Шок по фактору {bad}, который выкинут VIF-фильтром "
                f"(коллинеарен другим). Доступны: {list(X_clean.columns)}"
            )

    model, scaler = _fit(X_clean, y)
    X_scaled = scaler.transform(X_clean)

    last_row = X_clean.iloc[-1]
    X_future = pd.DataFrame([last_row.values] * horizon_months, columns=X_clean.columns)
    if shocks:
        for sh in shocks:
            X_future[sh.factor] = X_future[sh.factor] + sh.delta
    X_future_scaled = scaler.transform(X_future)

    point_preds = model.predict(X_future_scaled)
    sims = _bootstrap_residuals(model, X_scaled, y.values, X_future_scaled)
    l80, u80 = np.percentile(sims, 10, axis=0), np.percentile(sims, 90, axis=0)
    l95, u95 = np.percentile(sims, 2.5, axis=0), np.percentile(sims, 97.5, axis=0)

    last_period = y.index[-1]
    future_periods = pd.date_range(
        start=last_period + pd.offsets.MonthBegin(1),
        periods=horizon_months, freq="MS",
    )
    points = [
        ForecastPoint(
            period=ts.date(),
            point=float(point_preds[i]),
            lower_80=float(l80[i]), upper_80=float(u80[i]),
            lower_95=float(l95[i]), upper_95=float(u95[i]),
        )
        for i, ts in enumerate(future_periods)
    ]

    coefs = dict(zip(X_clean.columns, model.coef_, strict=True))
    diagnostics = {
        "r2_in_sample": float(model.score(X_scaled, y)),
        "coefficients_standardized": {k: float(v) for k, v in coefs.items()},
        "intercept": float(model.intercept_),
        "n_observations": int(len(y)),
        "factors": list(X_clean.columns),
        "factors_dropped_by_vif": dropped,
        "vifs_after_filter": {k: float(v) for k, v in vifs.items()},
        "last_observed_period": str(last_period.date()),
        "last_observed_value": float(y.iloc[-1]),
        "shocks_applied": [s.model_dump() for s in (shocks or [])],
    }
    return points, diagnostics

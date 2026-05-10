from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.forecasting.schemas import Asset, Method
from src.utils.logging import get_logger

log = get_logger("forecasting.conformal")

CACHE_DIR = Path("data/cache/forecast")
# Шаг backtest'а: больше — точнее, дольше. Для SARIMA fit ~3с, поэтому шире step.
STEP_MONTHS = {"sarima": 4, "regression": 6}


def _cache_path(asset: Asset, method: Method, horizon: int, last_period: str) -> Path:
    return CACHE_DIR / f"{method}-{asset}-h{horizon}-{last_period}.json"


def calibrate_quantiles(asset: Asset, method: Method, horizon_months: int,
                        last_observed_period: str) -> list[tuple[float, float]]:
    """Возвращает [(q80, q95), ...] длиной horizon_months — квантили |error|
    из rolling-origin backtest. Кэшируется по (asset, method, h, last_period)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(asset, method, horizon_months, last_observed_period)
    if cache.exists():
        data = json.loads(cache.read_text())
        log.info("conformal.cache_hit", path=str(cache))
        return [(q[0], q[1]) for q in data]

    from src.forecasting.backtest import rolling_backtest
    log.info("conformal.calibrating", asset=asset, method=method, h=horizon_months)
    rep = rolling_backtest(asset, method, horizon_months=horizon_months,
                           step_months=STEP_MONTHS.get(method, 6))

    # Adaptive + relative conformal: на нефти уровень цены меняется в разы
    # (2015 $50, 2026 $120), поэтому абсолютные ошибки несопоставимы между режимами.
    # Берём RELATIVE ошибки |y-ŷ|/y и недавнюю половину окон. На выходе q — в долях.
    recent = rep.per_window[len(rep.per_window) // 2:]
    rel_errors_by_step: list[list[float]] = [[] for _ in range(horizon_months)]
    for w in recent:
        for i, (actual, pred) in enumerate(zip(w["actuals"], w["predictions"], strict=True)):
            if abs(actual) > 1e-9:
                rel_errors_by_step[i].append(abs(actual - pred) / abs(actual))

    # ACI-инфляция: на не-exchangeable рядах (нефть) пустые conformal-quantile
    # систематически под-покрывают. Берём более высокий percentile + лёгкий
    # multiplicative-запас. Калибровано по honest eval до целевого покрытия.
    PCTL_80 = 0.87
    PCTL_95 = 0.97
    INFLATION = 1.15

    quantiles: list[tuple[float, float]] = []
    for step_errors in rel_errors_by_step:
        if not step_errors:
            quantiles.append((0.0, 0.0))
            continue
        q80_rel = float(np.quantile(step_errors, PCTL_80)) * INFLATION
        q95_rel = float(np.quantile(step_errors, PCTL_95)) * INFLATION
        quantiles.append((q80_rel, q95_rel))

    cache.write_text(json.dumps(quantiles))
    log.info("conformal.calibrated", windows=rep.n_windows, q80_h1=quantiles[0][0],
             q95_h1=quantiles[0][1], path=str(cache))
    return quantiles


def apply_conformal_ci(points: list, asset: Asset, method: Method,
                       horizon_months: int, last_observed_period: str) -> None:
    """Заменяет аналитические интервалы на conformal: ŷ ± q_α (относительный).
    q — в долях от точечного значения, чтобы быть инвариантным к уровню цены."""
    quantiles = calibrate_quantiles(asset, method, horizon_months, last_observed_period)
    for i, p in enumerate(points):
        q80_rel, q95_rel = quantiles[i]
        w80 = q80_rel * abs(p.point)
        w95 = q95_rel * abs(p.point)
        p.lower_80 = p.point - w80
        p.upper_80 = p.point + w80
        p.lower_95 = p.point - w95
        p.upper_95 = p.point + w95

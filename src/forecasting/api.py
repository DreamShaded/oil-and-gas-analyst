from __future__ import annotations

from src.forecasting import conformal, regression, sarima
from src.forecasting.data_loader import aligned_xy, load_price_monthly
from src.forecasting.schemas import Asset, Forecast, ForecastPoint, Method, ScenarioShock
from src.utils.logging import get_logger

log = get_logger("forecasting.api")


def forecast_price(asset: Asset, *, horizon_months: int = 3,
                   method: Method = "sarima",
                   shocks: list[ScenarioShock] | None = None,
                   conformal_ci: bool = True) -> Forecast:
    if horizon_months <= 0 or horizon_months > 24:
        raise ValueError("horizon_months ∈ [1, 24]")

    if method == "sarima":
        if shocks:
            raise ValueError("Шоки факторов применимы только к method='regression'")
        series = load_price_monthly(asset)
        points, diag = sarima.fit_and_forecast(series, horizon_months)
        factors: list[str] = []
    elif method == "regression":
        X, y = aligned_xy(asset)
        points, diag = regression.fit_and_forecast(X, y, horizon_months, shocks)
        factors = diag["factors"]
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    if conformal_ci:
        try:
            conformal.apply_conformal_ci(
                points, asset, method, horizon_months,
                diag["last_observed_period"],
            )
            diag["ci_method"] = "conformal"
        except Exception as e:
            log.warning("forecast.conformal_failed", error=str(e))
            diag["ci_method"] = "analytic"
    else:
        diag["ci_method"] = "analytic"

    interp = _interpret(asset, method, points, diag, shocks)
    log.info("forecast.done", asset=asset, method=method, h=horizon_months,
             first=points[0].point if points else None)
    return Forecast(
        asset=asset, method=method, horizon_months=horizon_months,
        points=points, interpretation=interp, factors_used=factors,
        fit_diagnostics=diag,
    )


def _interpret(asset: Asset, method: Method, points: list[ForecastPoint],
               diag: dict, shocks: list[ScenarioShock] | None) -> str:
    if not points:
        return "Прогноз не получен."
    last_value = diag.get("last_observed_value")
    last_period = diag.get("last_observed_period")
    first = points[0]
    last = points[-1]
    direction = "ростом" if last.point > first.point else ("снижением" if last.point < first.point else "стабилизацией")
    width80 = (last.upper_80 - last.lower_80) / max(abs(last.point), 1e-9) * 100

    lines = [
        f"{asset.upper()} ({method}): прогноз на {len(points)} мес. вперёд от {last_period} (последнее значение ${last_value:.2f}).",
        f"Точечный прогноз: ${first.point:.2f} → ${last.point:.2f} (горизонт), что соответствует {direction}.",
        f"80% интервал на горизонте: ${last.lower_80:.2f}..${last.upper_80:.2f} (ширина {width80:.1f}% от точечного).",
        f"95% интервал на горизонте: ${last.lower_95:.2f}..${last.upper_95:.2f}.",
    ]
    ci_method = diag.get("ci_method", "analytic")
    lines.append(f"Метод CI: {ci_method} (conformal = квантили |error| из rolling backtest).")
    if method == "sarima":
        lines.append(f"AIC модели: {diag.get('aic', 0):.1f}, BIC: {diag.get('bic', 0):.1f}.")
    if method == "regression":
        coefs = diag.get("coefficients_standardized", {})
        top = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        if top:
            lines.append("Топ-3 фактора (по абс. стандартизованному коэффициенту):")
            for name, val in top:
                sign = "+" if val >= 0 else "−"
                lines.append(f"  • {name}: β={sign}{abs(val):.2f}")
        if shocks:
            lines.append("Применены шоки:")
            for s in shocks:
                lines.append(f"  • {s.factor} {'+' if s.delta >= 0 else ''}{s.delta} ({s.description or 'сценарий'})")
        lines.append(f"R² на обучающей выборке: {diag.get('r2_in_sample', 0):.2f} (только in-sample, не валидация).")
    lines.append("⚠ Прогноз — модельная оценка, не торговая рекомендация.")
    return "\n".join(lines)

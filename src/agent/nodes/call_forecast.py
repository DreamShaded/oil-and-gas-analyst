from __future__ import annotations

import re

from src.agent.nodes.classify_intent import _last_human_text
from src.agent.state import AgentState
from src.forecasting import forecast_price
from src.utils.logging import get_logger

log = get_logger("agent.call_forecast")

ASSET_PATTERNS = {
    "wti":   re.compile(r"\b(wti|west\s+texas)\b", re.IGNORECASE),
    "urals": re.compile(r"\b(urals|юралс|уральск)", re.IGNORECASE),
    "brent": re.compile(r"\b(brent|брент)\b", re.IGNORECASE),
}
HORIZON_RE = re.compile(r"(\d{1,2})\s*(?:мес|month)", re.IGNORECASE)
SCENARIO_RE = re.compile(
    r"(?:OPEC|ОПЕК)[\s\S]{0,30}?(сократ|снизит|уменьш|нарастит|увеличит)[\s\S]{0,20}?(\d+[.,]?\d*)\s*(?:м?млн|mb|mbd)",
    re.IGNORECASE,
)


def _detect_asset(text: str) -> str:
    for asset, pat in ASSET_PATTERNS.items():
        if pat.search(text):
            return asset
    return "brent"


def _detect_horizon(text: str) -> int:
    m = HORIZON_RE.search(text)
    if m:
        try:
            return max(1, min(24, int(m.group(1))))
        except ValueError:
            pass
    return 3


def _detect_scenario(text: str):
    from src.forecasting import ScenarioShock
    m = SCENARIO_RE.search(text)
    if not m:
        return None
    direction = m.group(1).lower()
    delta_str = m.group(2).replace(",", ".")
    try:
        magnitude = float(delta_str)
    except ValueError:
        return None
    sign = -1 if direction.startswith(("сократ", "снизит", "уменьш")) else +1
    return ScenarioShock(
        factor="opec_crude_prod",
        delta=sign * magnitude,
        description=f"OPEC {'-' if sign < 0 else '+'}{magnitude} мб/с",
    )


async def call_forecast_node(state: AgentState) -> dict:
    text = _last_human_text(state.get("messages", []))
    if not text.strip():
        return {"forecast": None}
    asset = _detect_asset(text)
    horizon = _detect_horizon(text)
    scenario = _detect_scenario(text)
    method = "regression" if scenario else "sarima"
    log.info("forecast.dispatching", asset=asset, horizon=horizon, method=method,
             scenario=bool(scenario))
    try:
        fc = forecast_price(asset, horizon_months=horizon, method=method,  # type: ignore[arg-type]
                            shocks=[scenario] if scenario else None)
    except Exception as e:
        log.warning("forecast.failed", error=str(e))
        return {"forecast": {"error": str(e)}}
    return {"forecast": fc.model_dump(mode="json")}

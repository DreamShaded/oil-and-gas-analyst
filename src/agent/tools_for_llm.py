from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from src.self_mod import apply_or_queue
from src.self_mod.tier1_safe import TIER1_ALLOWLIST
from src.utils.logging import get_logger

log = get_logger("agent.tools")


def _coerce(target: str, value: object) -> object:
    """Если LLM прислала число строкой, конвертируем под тип Tier-1."""
    spec = TIER1_ALLOWLIST.get(target)
    if not spec or not isinstance(value, str):
        return value
    expected_type = spec[0]
    try:
        if expected_type is int:
            return int(value)
        if expected_type is float:
            return float(value)
    except (TypeError, ValueError):
        return value
    return value


@tool
def propose_self_improvement(
    target: Annotated[str, "Имя параметра. Пример: 'rag.top_k', 'rag.min_score'. "
                           "Tier-1 whitelist: rag.top_k (3..50), rag.min_score (0..1), "
                           "rag.chunk_size (200..4000), rag.chunk_overlap (0..1000), "
                           "forecast.default_horizon_months (1..24). "
                           "Любое другое имя → Tier-2 (на ревью пользователя)."],
    value: Annotated[float | int | str, "Новое значение параметра."],
    rationale: Annotated[str, "Короткое обоснование на русском: почему это полезно "
                              "пользователю в следующих диалогах."],
) -> str:
    """Предложить улучшение параметра агента. Tier-1 (whitelist + границы) применяется
    автоматически; Tier-2 уходит на страницу «Самомодификация» для одобрения
    пользователем. Использовать ТОЛЬКО когда замечена устойчивая проблема —
    не на каждый запрос."""
    coerced = _coerce(target, value)
    p = apply_or_queue(target, coerced, rationale)
    log.info("tool.proposed", target=target, value=coerced, tier=p.tier.name, status=p.status)
    if p.tier.name == "AUTO":
        return (f"Tier-1 применено мгновенно: `{target}` = `{value!r}`. "
                f"Изменение сохранено в data/runtime_config.json и журнале аудита.")
    return (f"Tier-2: предложение `{p.id}` сохранено на странице «Самомодификация» "
            f"и ждёт явного одобрения пользователем. Причина не-Tier-1: {p.rationale}")


ALL_TOOLS = [propose_self_improvement]

from __future__ import annotations

from typing import Any

# Жёсткий список Tier-1 параметров: target → (тип, min, max).
# Любая правка вне этого списка автоматически становится Tier-2.
TIER1_ALLOWLIST: dict[str, tuple[type, Any, Any]] = {
    "rag.top_k":                       (int,   3,    50),
    "rag.min_score":                   (float, 0.0,  1.0),
    "rag.chunk_size":                  (int,   200,  4000),
    "rag.chunk_overlap":               (int,   0,    1000),
    "forecast.default_horizon_months": (int,   1,    24),
    # Текстовое расширение системного промпта (мягкое уточнение поведения):
    # стилевые, тоновые, доменные подсказки. Длина ограничена 500 символов,
    # ядро BIBLE-инвариантов трогать нельзя.
    "prompt.addendum":                 (str,   0,    500),
}

# Явно запрещено даже на Tier-2 auto (только PR).
FORBIDDEN_TARGETS: frozenset[str] = frozenset({
    "constitution", "bible",
    "prompts.system.analyst",
    "self_mod.tier1_safe",
    "config.source_whitelist", "config.source_blacklist",
})


def is_tier1_eligible(target: str, value: Any) -> tuple[bool, str]:
    """Возвращает (eligible, reason). Tier-1 = в whitelist + тип верный + в границах."""
    if target in FORBIDDEN_TARGETS:
        return False, f"target '{target}' жёстко запрещён"
    spec = TIER1_ALLOWLIST.get(target)
    if spec is None:
        return False, f"target '{target}' не в Tier-1 allowlist"
    expected_type, lo, hi = spec
    if expected_type is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, expected_type):
        return False, f"тип {type(value).__name__} ≠ {expected_type.__name__}"
    if expected_type is str:
        if len(value) < lo or len(value) > hi:
            return False, f"длина строки {len(value)} вне границ [{lo}, {hi}]"
    else:
        if value < lo or value > hi:
            return False, f"значение {value} вне границ [{lo}, {hi}]"
    return True, "ок"

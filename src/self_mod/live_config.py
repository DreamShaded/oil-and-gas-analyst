from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("data/runtime_config.json")
_LOCK = threading.Lock()


def load_overrides() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_overrides(data: dict[str, Any]) -> None:
    with _LOCK:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_override(target: str, value: Any) -> Any | None:
    """Перезаписывает значение в runtime_config.json. Возвращает прежнее (или None)."""
    data = load_overrides()
    before = data.get(target)
    data[target] = value
    save_overrides(data)
    return before


def get_override(target: str, default: Any = None) -> Any:
    return load_overrides().get(target, default)


def get_current_value(target: str) -> Any:
    """Текущее значение: сначала смотрим runtime_config, затем настройки .env."""
    override = get_override(target)
    if override is not None:
        return override
    from src.config import get_settings
    field_map = {
        "rag.top_k": "rag_top_k",
        "rag.min_score": "rag_min_score",
        "rag.chunk_size": "rag_chunk_size",
        "rag.chunk_overlap": "rag_chunk_overlap",
    }
    field = field_map.get(target)
    if not field:
        return None
    try:
        return getattr(get_settings(), field)
    except Exception:
        return None

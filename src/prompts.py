"""Загрузка промптов из каталога `prompts/`.

Все промпты хранятся файлами `.md`, чтобы их можно было редактировать
без правки кода. Поддерживаем простую подстановку через `str.format`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=64)
def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Промпт не найден: {path}")
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **kwargs: object) -> str:
    """Загружает промпт по относительному пути от `prompts/`"""
    path = _PROMPTS_DIR / f"{name}.md"
    text = _read(path)
    return text.format(**kwargs) if kwargs else text

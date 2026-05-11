"""Конфигурация приложения через переменные окружения.

Все настройки читаются из `.env` и валидируются Pydantic ещё на старте"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Корневая модель настроек.
    Универсальные переменные для LLM/embeddings — без привязки к провайдеру.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM (OpenAI-совместимый endpoint)
    llm_api_url: str = Field(..., description="Base URL OpenAI-совместимого API")
    llm_api_token: SecretStr = Field(..., description="Токен LLM-провайдера")
    llm_model: str = Field(..., description="Основная модель для синтеза ответа")
    llm_fast_model: str = Field(..., description="Дешёвая модель для классификации/роутинга")
    llm_timeout_s: int = Field(60, ge=5, le=600)

    # Embeddings
    embeddings_api_url: str
    embeddings_api_token: SecretStr
    embeddings_model: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    rag_collection: str = "oil_gas_reports"

    rag_chunk_size: int = Field(800, ge=200, le=4000)
    rag_chunk_overlap: int = Field(120, ge=0, le=1000)
    rag_top_k: int = Field(8, ge=1, le=50)
    rag_min_score: float = Field(0.30, ge=0.0, le=1.0)

    # Внешние API (опционально на этапе 1)
    tavily_api_key: SecretStr | None = None
    eia_api_key: SecretStr | None = None

    # Приложение
    app_env: Literal["dev", "prod"] = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 7860
    data_dir: Path = Path("./data")

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"


_RUNTIME_OVERRIDE_KEYS: dict[str, str] = {
    "rag.top_k":         "rag_top_k",
    "rag.min_score":     "rag_min_score",
    "rag.chunk_size":    "rag_chunk_size",
    "rag.chunk_overlap": "rag_chunk_overlap",
}


def get_settings() -> Settings:
    """Settings + runtime overrides из data/runtime_config.json (Tier-1 self-mod).
    Не кэшируем: правка через `apply_or_queue` должна сразу влиять на следующий вызов."""
    base = _build_settings()
    from src.self_mod.live_config import load_overrides
    overrides = load_overrides()
    if not overrides:
        return base
    updates: dict[str, object] = {}
    for dotted, settings_field in _RUNTIME_OVERRIDE_KEYS.items():
        if dotted in overrides:
            updates[settings_field] = overrides[dotted]
    return base.model_copy(update=updates) if updates else base


@lru_cache(maxsize=1)
def _build_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

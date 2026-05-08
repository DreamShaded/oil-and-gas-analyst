from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Settings, get_settings


def build_chat_model(*, fast: bool = False, settings: Settings | None = None) -> ChatOpenAI:
    """Возвращает чат-модель.

    Args:
        fast: True — дешёвая модель для классификации/роутинга,
              False — основная модель для синтеза ответа.
        settings: Подмена настроек для тестов; обычно None.
    """
    s = settings or get_settings()
    
    return ChatOpenAI(
        model=s.llm_fast_model if fast else s.llm_model,
        base_url=s.llm_api_url,
        api_key=s.llm_api_token.get_secret_value(),
        timeout=s.llm_timeout_s,
    )


def build_embeddings(*, settings: Settings | None = None) -> OpenAIEmbeddings:
    """Возвращает embeddings-клиент"""
    s = settings or get_settings()

    return OpenAIEmbeddings(
        model=s.embeddings_model,
        base_url=s.embeddings_api_url,
        api_key=s.embeddings_api_token.get_secret_value(),
    )

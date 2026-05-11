from __future__ import annotations

import socket
from typing import Any
from urllib.parse import urlparse, urlunparse

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Settings, get_settings


def _resolve_url(url: str) -> str:
    """host.docker.internal валиден только внутри контейнера. На хосте — fallback на localhost."""
    parsed = urlparse(url)
    if parsed.hostname != "host.docker.internal":
        return url
    try:
        socket.gethostbyname("host.docker.internal")
        return url
    except socket.gaierror:
        new_netloc = parsed.netloc.replace("host.docker.internal", "localhost", 1)
        return urlunparse(parsed._replace(netloc=new_netloc))


def _is_anthropic(url: str) -> bool:
    """Anthropic API не OpenAI-совместим — нужен другой клиент."""
    host = (urlparse(url).hostname or "").lower()
    return "anthropic.com" in host


def build_chat_model(*, fast: bool = False, settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    model = s.llm_fast_model if fast else s.llm_model
    url = _resolve_url(s.llm_api_url)
    token = s.llm_api_token.get_secret_value()

    if _is_anthropic(url):
        kwargs: dict[str, Any] = {"model": model, "api_key": token, "timeout": s.llm_timeout_s}
        if url and not url.endswith("api.anthropic.com"):
            kwargs["base_url"] = url
        return ChatAnthropic(**kwargs)

    return ChatOpenAI(
        model=model,
        base_url=url,
        api_key=token,
        timeout=s.llm_timeout_s,
    )


def build_embeddings(*, settings: Settings | None = None) -> OpenAIEmbeddings:
    """Embeddings — всегда OpenAI-совместимый endpoint (Ollama / vLLM / OpenAI).
    Anthropic embeddings отдельно не поддерживаем — пользователю нужен любой
    OpenAI-совместимый embed-провайдер (локальная Ollama по умолчанию)."""
    s = settings or get_settings()
    return OpenAIEmbeddings(
        model=s.embeddings_model,
        base_url=_resolve_url(s.embeddings_api_url),
        api_key=s.embeddings_api_token.get_secret_value(),
        check_embedding_ctx_length=False,
    )

from __future__ import annotations

import socket
from urllib.parse import urlparse, urlunparse

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


def build_chat_model(*, fast: bool = False, settings: Settings | None = None) -> ChatOpenAI:
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.llm_fast_model if fast else s.llm_model,
        base_url=_resolve_url(s.llm_api_url),
        api_key=s.llm_api_token.get_secret_value(),
        timeout=s.llm_timeout_s,
    )


def build_embeddings(*, settings: Settings | None = None) -> OpenAIEmbeddings:
    s = settings or get_settings()
    return OpenAIEmbeddings(
        model=s.embeddings_model,
        base_url=_resolve_url(s.embeddings_api_url),
        api_key=s.embeddings_api_token.get_secret_value(),
        check_embedding_ctx_length=False,
    )

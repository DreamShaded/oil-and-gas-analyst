from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["forecast", "rag_first", "rag_plus_web", "web_only", "out_of_scope"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: Intent | None
    intent_reason: str | None
    rag_chunks: list[dict[str, Any]]
    rag_sufficient: bool | None
    rag_sufficient_reason: str | None
    web_snippets: list[dict[str, Any]]
    forecast: dict[str, Any] | None
    # Reflect-цикл
    reflections: int
    reflect_verdict: Literal["ok", "fix"] | None
    reflect_reason: str | None
    # Signals для self-mod observer (явные триггеры дополнительно к 5% sampling)
    signals: list[str]

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["forecast", "rag_first", "web_only", "out_of_scope"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: Intent | None
    intent_reason: str | None
    rag_chunks: list[dict[str, Any]]
    rag_sufficient: bool | None
    rag_sufficient_reason: str | None
    web_snippets: list[dict[str, Any]]
    forecast: dict[str, Any] | None

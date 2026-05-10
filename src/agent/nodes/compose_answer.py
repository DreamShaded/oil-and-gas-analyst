from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import build_chat_model
from src.prompts import load_prompt
from src.utils.logging import get_logger

log = get_logger("agent.compose_answer")
MAX_CHUNK_TEXT = 700


def _format_rag_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(нет данных из RAG)"
    lines = ["### Данные RAG"]
    for i, c in enumerate(chunks, 1):
        marker_parts = [c.get("source", "?")]
        if c.get("series_id"):
            marker_parts.append(f"серия {c['series_id']}")
        if c.get("segment"):
            marker_parts.append(c["segment"])
        elif c.get("period_start"):
            marker_parts.append(c["period_start"])
        marker = "[" + ", ".join(str(x) for x in marker_parts) + "]"
        title = c.get("title") or c.get("metric") or "-"
        text = (c.get("text") or "")[:MAX_CHUNK_TEXT]
        lines.append(f"\n**Чанк {i}** {marker} score={c.get('score', 0):.2f}  {title}\n{text}")
    return "\n".join(lines)


def _format_web(snippets: list[dict[str, Any]]) -> str:
    if not snippets:
        return ""
    lines = ["### Веб-сниппеты (свежее)"]
    for i, s in enumerate(snippets, 1):
        marker = f"[Источник: {s['source_domain']}, web"
        if s.get("published_at"):
            marker += f", {s['published_at']}"
        marker += "]"
        lines.append(f"\n**Сниппет {i}** {marker}\n{s.get('title', '')}\n{(s.get('snippet') or '')[:600]}")
    return "\n".join(lines)


def _format_forecast(fc: dict[str, Any] | None) -> str:
    if not fc:
        return ""
    if "error" in fc:
        return f"### Прогноз: ошибка — {fc['error']}"
    lines = [f"### Прогноз ({fc['method']}, h={fc['horizon_months']}m)"]
    lines.append(fc.get("interpretation", ""))
    lines.append("\nТочки прогноза:")
    for p in fc.get("points", []):
        lines.append(f"  {p['period']}  ${p['point']:.2f}  "
                     f"80%[${p['lower_80']:.2f}..${p['upper_80']:.2f}]  "
                     f"95%[${p['lower_95']:.2f}..${p['upper_95']:.2f}]")
    return "\n".join(lines)


def _build_context(state: AgentState) -> str:
    parts: list[str] = []
    if rag := state.get("rag_chunks"):
        parts.append(_format_rag_chunks(rag))
    if web := state.get("web_snippets"):
        parts.append(_format_web(web))
    if fc := state.get("forecast"):
        parts.append(_format_forecast(fc))
    if intent := state.get("intent"):
        reason = state.get("intent_reason") or ""
        parts.append(f"\n_Маршрут: intent={intent}; {reason}_")
    return "\n\n".join(parts) if parts else "(контекст пуст — ответь, что данных нет)"


async def compose_answer_node(state: AgentState) -> dict:
    context_block = _build_context(state)
    system = SystemMessage(content=load_prompt(
        "nodes/compose-answer.system",
        context_block=context_block,
        today=date.today().isoformat(),
    ))
    llm = build_chat_model()
    response = await llm.ainvoke([system, *state["messages"]])
    log.info("compose.done", ctx_len=len(context_block))
    return {"messages": [response]}

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage

from src.agent.state import AgentState
from src.agent.tools_for_llm import ALL_TOOLS, propose_self_improvement
from src.llm.client import build_chat_model
from src.prompts import load_prompt
from src.utils.logging import get_logger

log = get_logger("agent.compose_answer")
MAX_CHUNK_TEXT = 700
MAX_TOOL_HOPS = 3
_TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS}


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
    from src.self_mod import get_override
    context_block = _build_context(state)
    system_text = load_prompt(
        "nodes/compose-answer.system",
        context_block=context_block,
        today=date.today().isoformat(),
    )
    addendum = get_override("prompt.addendum")
    if addendum and isinstance(addendum, str) and addendum.strip():
        system_text += (
            "\n\n## Адаптация поведения (накоплено через самоулучшение)\n\n"
            + addendum.strip()
        )
    system = SystemMessage(content=system_text)
    base_llm = build_chat_model()
    try:
        llm = base_llm.bind_tools(ALL_TOOLS)
    except Exception as e:
        log.warning("compose.bind_tools_failed", error=str(e))
        llm = base_llm

    convo = [system, *state["messages"]]
    new_messages: list = []
    for hop in range(MAX_TOOL_HOPS + 1):
        response = await llm.ainvoke(convo)
        new_messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls or hop == MAX_TOOL_HOPS:
            break
        log.info("compose.tool_calls", n=len(tool_calls), hop=hop)
        convo.append(response)
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else call.name
            args = call.get("args") if isinstance(call, dict) else call.args
            call_id = call.get("id") if isinstance(call, dict) else call.id
            tool_obj = _TOOL_REGISTRY.get(name) or propose_self_improvement
            try:
                result = await tool_obj.ainvoke(args)
            except Exception as e:
                result = f"Ошибка выполнения tool {name}: {e}"
                log.warning("compose.tool_failed", tool=name, error=str(e))
            tool_msg = ToolMessage(content=str(result), tool_call_id=call_id, name=name)
            new_messages.append(tool_msg)
            convo.append(tool_msg)

    log.info("compose.done", ctx_len=len(context_block), final_msgs=len(new_messages))
    return {"messages": new_messages}

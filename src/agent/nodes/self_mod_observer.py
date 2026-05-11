from __future__ import annotations

import random
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.nodes.classify_intent import _last_human_text
from src.agent.state import AgentState
from src.agent.tools_for_llm import ALL_TOOLS, propose_self_improvement
from src.llm.client import build_chat_model
from src.prompts import load_prompt
from src.utils.logging import get_logger

log = get_logger("agent.self_mod_observer")
SAMPLE_RATE = 0.05  # 5% от спокойных запросов + 100% при сигнале


def should_observe(state: AgentState) -> bool:
    """Запускать observer? Триггеры: явный сигнал в state.signals ИЛИ случайный сэмпл."""
    signals = state.get("signals") or []
    if signals:
        return True
    return random.random() < SAMPLE_RATE


def _build_observation(state: AgentState) -> str:
    lines = [f"- intent: `{state.get('intent')}` (reason: {state.get('intent_reason')!s:.120s})"]
    if (rag := state.get("rag_chunks")) is not None:
        n = len(rag)
        top = rag[0]["score"] if rag else None
        lines.append(f"- RAG: {n} чанков, top_score={top}")
    lines.append(f"- RAG sufficient: {state.get('rag_sufficient')} ({state.get('rag_sufficient_reason')})")
    if (web := state.get("web_snippets")) is not None:
        lines.append(f"- web fallback использован: {len(web)} сниппетов")
    if state.get("forecast"):
        lines.append("- forecast вызван")
    if signals := state.get("signals"):
        lines.append(f"- signals: {signals}")
    lines.append(f"- запрос пользователя (превью): {_last_human_text(state.get('messages', []))[:160]}")
    return "\n".join(lines)


async def self_mod_observer_node(state: AgentState) -> dict:
    if not should_observe(state):
        log.debug("observer.skipped_sample")
        return {}

    from src.self_mod import get_override
    current_addendum = (get_override("prompt.addendum") or "").strip() or "(пусто)"
    system = SystemMessage(content=load_prompt(
        "nodes/self-mod-observer.system",
        observation_block=_build_observation(state),
        current_addendum=current_addendum,
        today=date.today().isoformat(),
    ))
    human = HumanMessage(content="Сделай вывод: предложить ли изменение параметра?")

    base_llm = build_chat_model(fast=True)
    try:
        llm = base_llm.bind_tools(ALL_TOOLS)
    except Exception as e:
        log.warning("observer.bind_tools_failed", error=str(e))
        return {}

    try:
        response = await llm.ainvoke([system, human])
    except Exception as e:
        log.warning("observer.llm_failed", error=str(e))
        return {}

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        log.info("observer.no_proposal", text=str(response.content)[:120])
        return {}

    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else call.name
        args = call.get("args") if isinstance(call, dict) else call.args
        if name != "propose_self_improvement":
            continue
        try:
            result = await propose_self_improvement.ainvoke(args)
            log.info("observer.proposed", args=args, result=str(result)[:120])
        except Exception as e:
            log.warning("observer.tool_failed", error=str(e))
    return {}

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from src.agent.nodes.classify_intent import _last_human_text
from src.agent.state import AgentState
from src.llm.client import build_chat_model
from src.prompts import load_prompt
from src.utils.logging import get_logger

log = get_logger("agent.reflect")
MAX_REFLECTIONS = 2


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _parse_verdict(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            verdict = obj.get("verdict")
            reason = obj.get("reason", "")
            if verdict in {"ok", "fix"}:
                return verdict, reason
        except json.JSONDecodeError:
            pass
    low = text.lower()
    if "fix" in low:
        return "fix", "fallback: ключевое слово 'fix' в критике"
    return "ok", "fallback: критик не вернул структурный verdict"


async def reflect_node(state: AgentState) -> dict:
    draft = _last_ai_text(state.get("messages", []))
    query = _last_human_text(state.get("messages", []))
    reflections = state.get("reflections", 0) or 0

    if not draft:
        return {"reflect_verdict": "ok", "reflect_reason": "пустой черновик"}

    if reflections >= MAX_REFLECTIONS:
        log.info("reflect.limit_reached", reflections=reflections)
        return {"reflect_verdict": "ok", "reflect_reason": "достигнут лимит итераций"}

    system = SystemMessage(content=load_prompt("nodes/reflect-critique.system"))
    human = HumanMessage(content=(
        f"## Запрос пользователя\n{query}\n\n"
        f"## Черновик ответа\n{draft[:2500]}\n\n"
        "Оцени черновик и верни JSON по схеме."
    ))
    llm = build_chat_model(fast=True)
    response = await llm.ainvoke([system, human])
    verdict, reason = _parse_verdict(str(response.content))
    log.info("reflect.verdict", verdict=verdict, reason=reason[:120], reflections=reflections)

    signals = list(state.get("signals") or [])
    if verdict == "fix":
        signals.append("reflect_fix")
        # Удаляем trailing AI/ToolMessage из history, чтобы compose повторил с чистого листа.
        to_remove: list = []
        for m in reversed(state.get("messages", [])):
            if isinstance(m, AIMessage | ToolMessage):
                to_remove.append(RemoveMessage(id=m.id))
                continue
            break
        return {
            "reflect_verdict": "fix",
            "reflect_reason": reason,
            "reflections": reflections + 1,
            "signals": signals,
            "messages": to_remove,
        }
    return {
        "reflect_verdict": "ok",
        "reflect_reason": reason,
        "signals": signals,
    }

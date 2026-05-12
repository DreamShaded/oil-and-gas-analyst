from __future__ import annotations

import json
import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState, Intent
from src.llm.client import build_chat_model
from src.prompts import load_prompt
from src.utils.logging import get_logger

log = get_logger("agent.classify_intent")

VALID_INTENTS = {"forecast", "rag_first", "rag_plus_web", "web_only", "out_of_scope"}


def _last_human_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _parse_intent(text: str) -> tuple[Intent, str]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            intent = obj.get("intent")
            reason = obj.get("reason", "")
            if intent in VALID_INTENTS:
                return intent, reason
        except json.JSONDecodeError:
            pass
    low = text.lower()
    for cand in VALID_INTENTS:
        if cand in low:
            return cand, "fallback: ключевое слово в ответе"  # type: ignore[return-value]
    return "rag_first", "fallback: ответ непарсуем, дефолт"


async def classify_intent_node(state: AgentState) -> dict:
    user_text = _last_human_text(state.get("messages", []))
    if not user_text.strip():
        return {"intent": "out_of_scope", "intent_reason": "пустой запрос"}

    system = SystemMessage(content=load_prompt(
        "nodes/classify-intent.system",
        today=date.today().isoformat(),
    ))
    llm = build_chat_model(fast=True)
    response = await llm.ainvoke([system, HumanMessage(content=user_text)])
    intent, reason = _parse_intent(str(response.content))
    log.info("intent.classified", intent=intent, reason=reason[:80])
    return {"intent": intent, "intent_reason": reason}

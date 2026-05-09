"""
Подмешивает системный промпт аналитика и вызывает LLM. Tools, RAG и web-search позже
"""
from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import build_chat_model
from src.prompts import load_prompt


async def answer_node(state: AgentState) -> dict:
    system = SystemMessage(content=load_prompt("system/analyst", date=date.today().isoformat()))
    llm = build_chat_model()
    response = await llm.ainvoke([system, *state["messages"]])

    return {"messages": [response]}

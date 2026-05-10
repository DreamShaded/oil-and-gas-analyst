from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import build_chat_model
from src.prompts import load_prompt


async def refuse_node(state: AgentState) -> dict:
    system = SystemMessage(content=load_prompt("nodes/refuse.system"))
    llm = build_chat_model(fast=True)
    response = await llm.ainvoke([system, *state["messages"]])
    return {"messages": [response]}

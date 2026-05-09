from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.answer import answer_node
from src.agent.state import AgentState


def _build():
    builder = StateGraph(AgentState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()

@lru_cache(maxsize=1)
def get_graph():
    return _build()

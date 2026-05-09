from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.answer import answer_node
from src.agent.state import AgentState


def _build() -> "CompiledStateGraph":  # type: ignore[name-defined]
    builder = StateGraph(AgentState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()

@lru_cache(maxsize=1)
def get_graph():  # noqa: ANN201 — тип из langgraph не экспортируется чисто
    return _build()

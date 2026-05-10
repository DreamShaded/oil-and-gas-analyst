from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.call_forecast import call_forecast_node
from src.agent.nodes.classify_intent import classify_intent_node
from src.agent.nodes.compose_answer import compose_answer_node
from src.agent.nodes.rag_retrieve import rag_retrieve_node
from src.agent.nodes.refuse import refuse_node
from src.agent.nodes.sufficiency_check import sufficiency_check_node
from src.agent.nodes.web_search import web_search_node
from src.agent.state import AgentState


def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "forecast":
        return "call_forecast"
    if intent == "web_only":
        return "web_search"
    if intent == "out_of_scope":
        return "refuse"
    return "rag_retrieve"


def _route_by_sufficiency(state: AgentState) -> str:
    return "compose_answer" if state.get("rag_sufficient") else "web_search"


def _build():
    g = StateGraph(AgentState)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("rag_retrieve", rag_retrieve_node)
    g.add_node("sufficiency_check", sufficiency_check_node)
    g.add_node("web_search", web_search_node)
    g.add_node("call_forecast", call_forecast_node)
    g.add_node("compose_answer", compose_answer_node)
    g.add_node("refuse", refuse_node)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", _route_by_intent, {
        "call_forecast": "call_forecast",
        "web_search": "web_search",
        "rag_retrieve": "rag_retrieve",
        "refuse": "refuse",
    })
    g.add_edge("rag_retrieve", "sufficiency_check")
    g.add_conditional_edges("sufficiency_check", _route_by_sufficiency, {
        "compose_answer": "compose_answer",
        "web_search": "web_search",
    })
    g.add_edge("web_search", "compose_answer")
    g.add_edge("call_forecast", "compose_answer")
    g.add_edge("compose_answer", END)
    g.add_edge("refuse", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return _build()

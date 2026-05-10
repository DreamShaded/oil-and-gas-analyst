from __future__ import annotations

from src.agent.nodes.classify_intent import _last_human_text
from src.agent.state import AgentState
from src.tools.web_search import search_web
from src.utils.logging import get_logger

log = get_logger("agent.web_search")


async def web_search_node(state: AgentState) -> dict:
    query = _last_human_text(state.get("messages", []))
    if not query.strip():
        return {"web_snippets": []}
    try:
        result = search_web(query, max_results=5, topic="news")
        snippets = [s.model_dump() for s in result.snippets]
    except Exception as e:
        log.warning("web.search_failed", error=str(e))
        return {"web_snippets": []}
    log.info("web.retrieved", n=len(snippets))
    return {"web_snippets": snippets}

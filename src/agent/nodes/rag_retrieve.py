from __future__ import annotations

from src.agent.nodes.classify_intent import _last_human_text
from src.agent.state import AgentState
from src.rag import Retriever
from src.utils.logging import get_logger

log = get_logger("agent.rag_retrieve")


async def rag_retrieve_node(state: AgentState) -> dict:
    query = _last_human_text(state.get("messages", []))
    if not query.strip():
        return {"rag_chunks": []}
    try:
        hits = Retriever().search(query, top_k=8)
    except Exception as e:
        log.warning("rag.search_failed", error=str(e))
        return {"rag_chunks": []}
    log.info("rag.retrieved", n=len(hits), top_score=(hits[0]["score"] if hits else None))
    return {"rag_chunks": hits}

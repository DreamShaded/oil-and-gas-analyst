from __future__ import annotations

import re

from src.agent.nodes.classify_intent import _last_human_text
from src.agent.state import AgentState
from src.rag import Retriever
from src.utils.logging import get_logger

log = get_logger("agent.rag_retrieve")

# «Q3 2024», «третьем квартале 2024», «3-й квартал 2024»
QUARTER_NUM_RE = re.compile(r"\bQ([1-4])[^\d]{0,5}(20\d{2})\b", re.IGNORECASE)
QUARTER_RU_RE = re.compile(r"\b([1-4]|перв|втор|треть|четвёрт)[\w-]*\s+кварт\w+[^\d]{1,10}(20\d{2})", re.IGNORECASE)
RU_NUM = {"перв": 1, "втор": 2, "треть": 3, "четвёрт": 4, "четверт": 4}


def _extract_segment(text: str) -> str | None:
    m = QUARTER_NUM_RE.search(text)
    if m:
        return f"{m.group(2)}-Q{m.group(1)}"
    m = QUARTER_RU_RE.search(text)
    if m:
        token, year = m.group(1).lower(), m.group(2)
        q = int(token) if token.isdigit() else next(
            (v for k, v in RU_NUM.items() if token.startswith(k)), None,
        )
        if q:
            return f"{year}-Q{q}"
    return None


async def rag_retrieve_node(state: AgentState) -> dict:
    query = _last_human_text(state.get("messages", []))
    if not query.strip():
        return {"rag_chunks": []}
    r = Retriever()
    segment = _extract_segment(query)
    hits: list = []
    if segment:
        try:
            hits = r.search(query, top_k=8, filters={"segment": segment})
            log.info("rag.segment_match", segment=segment, n=len(hits))
        except Exception as e:
            log.warning("rag.segment_filter_failed", error=str(e))
    if len(hits) < 5:
        try:
            general = r.search(query, top_k=8 - len(hits))
            seen = {h.get("text") for h in hits}
            hits.extend(h for h in general if h.get("text") not in seen)
        except Exception as e:
            log.warning("rag.search_failed", error=str(e))
            if not hits:
                return {"rag_chunks": []}
    log.info("rag.retrieved", n=len(hits), top_score=(hits[0]["score"] if hits else None),
             segment=segment)
    return {"rag_chunks": hits}

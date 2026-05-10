from __future__ import annotations

from src.agent.state import AgentState
from src.config import get_settings
from src.utils.logging import get_logger

log = get_logger("agent.sufficiency_check")


async def sufficiency_check_node(state: AgentState) -> dict:
    """Эвристика: ≥3 чанков выше порога + top score ≥ порог.
    Без LLM: PRD §2.5 не требует судью, а лишние вызовы дороги."""
    hits = state.get("rag_chunks", []) or []
    cfg = get_settings()
    threshold = cfg.rag_min_score
    strong = [h for h in hits if h.get("score", 0) >= threshold]
    enough = len(strong) >= 3 and bool(hits) and hits[0].get("score", 0) >= threshold
    reason = (
        f"найдено {len(strong)} чанков ≥{threshold:.2f}, top={hits[0]['score']:.2f}"
        if hits else "RAG не вернул чанков"
    )
    log.info("sufficiency.checked", enough=enough, reason=reason)
    return {"rag_sufficient": enough, "rag_sufficient_reason": reason}

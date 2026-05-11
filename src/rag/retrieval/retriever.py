from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.rag.retrieval.embedder import Embedder
from src.rag.retrieval.qdrant_store import QdrantStore


class Retriever:
    def __init__(self, store: QdrantStore | None = None, embedder: Embedder | None = None):
        self.store = store or QdrantStore()
        self.embedder = embedder or Embedder()
        self._cfg = get_settings()

    def search(self, query: str, *, top_k: int | None = None,
               filters: dict[str, Any] | None = None,
               min_score: float | None = None,
               diversify: bool = True) -> list[dict[str, Any]]:
        """Если diversify=True (по умолчанию) — round-robin по source: одна метрика
        не может затопить топ. Это критично, когда один источник (Минфин) даёт
        18 метрик × 30+ кварталов, а EIA — 14 метрик × ~150."""
        vector = self.embedder.embed_query(query)
        k = top_k or self._cfg.rag_top_k
        # Глубокий пул кандидатов чтобы редкие, но релевантные source/series_id
        # успели попасть в перевыборку. На ~3к чанков 120 кандидатов — дёшево.
        candidates_n = max(50, k * 15) if diversify else k
        hits = self.store.search(vector, top_k=candidates_n, filters=filters)
        threshold = self._cfg.rag_min_score if min_score is None else min_score
        hits = [h for h in hits if h["score"] >= threshold]
        if not diversify:
            return hits[:k]
        return _round_robin_by_source(hits, k)

    def search_by_source(self, query: str, sources: list[str], *,
                         per_source: int = 4) -> dict[str, list[dict[str, Any]]]:
        vector = self.embedder.embed_query(query)
        out: dict[str, list[dict[str, Any]]] = {}
        for src in sources:
            out[src] = self.store.search(vector, top_k=per_source, filters={"source": src})
        return out

    def is_sufficient(self, hits: list[dict[str, Any]], *,
                      min_hits: int = 3, min_top_score: float | None = None) -> bool:
        if not hits:
            return False
        floor = min_top_score if min_top_score is not None else self._cfg.rag_min_score
        return len(hits) >= min_hits and hits[0]["score"] >= floor


def _round_robin_by_source(hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Round-robin по source: каждый раунд берём ОДИН лучший чанк из каждого
    источника, при этом не повторяя series_id, который уже взяли раньше из того
    же источника. Это даёт top-K с максимальным разнообразием и source, и series_id."""
    if not hits:
        return []
    by_source: dict[str, list[dict[str, Any]]] = {}
    src_order: list[str] = []
    for h in hits:
        src = h.get("source") or "_unknown"
        if src not in by_source:
            by_source[src] = []
            src_order.append(src)
        by_source[src].append(h)

    used_series: dict[str, set] = {s: set() for s in src_order}
    out: list[dict[str, Any]] = []
    while len(out) < k:
        added = False
        for src in src_order:
            bucket = by_source[src]
            picked_idx = None
            for i, h in enumerate(bucket):
                sid = h.get("series_id") or "_default"
                if sid not in used_series[src]:
                    picked_idx = i
                    used_series[src].add(sid)
                    break
            if picked_idx is None and bucket:
                # все series_id из этого source уже представлены — берём следующий
                picked_idx = 0
            if picked_idx is not None:
                out.append(bucket.pop(picked_idx))
                added = True
                if len(out) >= k:
                    return out
        if not added:
            break
    return out

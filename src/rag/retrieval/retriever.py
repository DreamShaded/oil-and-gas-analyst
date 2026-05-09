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
               min_score: float | None = None) -> list[dict[str, Any]]:
        vector = self.embedder.embed_query(query)
        k = top_k or self._cfg.rag_top_k
        hits = self.store.search(vector, top_k=k, filters=filters)
        threshold = self._cfg.rag_min_score if min_score is None else min_score
        return [h for h in hits if h["score"] >= threshold]

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

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.llm.client import build_embeddings
from src.rag.retrieval.embedding_cache import EmbeddingCache
from src.utils.logging import get_logger

log = get_logger("rag.embedder")
DEFAULT_BATCH = 64


class Embedder:
    def __init__(self, batch_size: int = DEFAULT_BATCH, cache: EmbeddingCache | None = None):
        self._client = build_embeddings()
        self._batch = batch_size
        self._model = get_settings().embeddings_model
        self._cache = cache or EmbeddingCache()
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["probe"])[0])
        return self._dim

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        keys = [EmbeddingCache.key(self._model, t) for t in texts]
        cached = self._cache.get_many(keys)
        result: list[list[float] | None] = [cached.get(k) for k in keys]

        missing_idx = [i for i, v in enumerate(result) if v is None]
        if missing_idx:
            log.info("embedder.cache_miss", missing=len(missing_idx), total=len(texts))
            for start in range(0, len(missing_idx), self._batch):
                idx_batch = missing_idx[start:start + self._batch]
                text_batch = [texts[i] for i in idx_batch]
                vec_batch = self._embed_batch(text_batch)
                to_store: list[tuple[str, list[float]]] = []
                for i, v in zip(idx_batch, vec_batch, strict=True):
                    result[i] = v
                    to_store.append((keys[i], v))
                self._cache.put_many(self._model, to_store)
        else:
            log.info("embedder.cache_hit_all", count=len(texts))
        return [v for v in result if v is not None]  # type: ignore[misc]

    def embed_query(self, query: str) -> list[float]:
        keys = [EmbeddingCache.key(self._model, query)]
        cached = self._cache.get_many(keys)
        if keys[0] in cached:
            return cached[keys[0]]
        vec = self._client.embed_query(query)
        self._cache.put_many(self._model, [(keys[0], vec)])
        return vec

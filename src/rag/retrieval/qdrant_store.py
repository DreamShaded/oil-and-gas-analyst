from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import get_settings
from src.rag.schemas import Chunk
from src.utils.logging import get_logger

log = get_logger("rag.qdrant")


def _key_to_uuid(key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class QdrantStore:
    def __init__(self, collection: str | None = None):
        s = get_settings()
        self.client = QdrantClient(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None,
        )
        self.collection = collection or s.rag_collection

    def create_snapshot(self, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        snap = self.client.create_snapshot(collection_name=self.collection)
        url = f"{self._base_http_url()}/collections/{self.collection}/snapshots/{snap.name}"
        target = target_dir / f"{self.collection}.snapshot"
        with httpx.Client(timeout=120.0) as c, c.stream("GET", url) as r:
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
        log.info("qdrant.snapshot_saved", path=str(target), bytes=target.stat().st_size)
        return target

    def restore_snapshot(self, snapshot_path: Path) -> bool:
        if not snapshot_path.exists():
            return False
        url = f"{self._base_http_url()}/collections/{self.collection}/snapshots/upload?priority=snapshot"
        with httpx.Client(timeout=300.0) as c, snapshot_path.open("rb") as f:
            files = {"snapshot": (snapshot_path.name, f, "application/octet-stream")}
            r = c.post(url, files=files)
            r.raise_for_status()
        size = self.collection_size()
        log.info("qdrant.snapshot_restored", points=size, path=str(snapshot_path))
        return size > 0

    def _base_http_url(self) -> str:
        s = get_settings()
        return s.qdrant_url.rstrip("/")

    def collection_size(self) -> int:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            return 0
        info = self.client.get_collection(self.collection)
        return int(info.points_count or 0)

    def ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        log.info("qdrant.collection_created", name=self.collection, dim=dim)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], *,
               batch_size: int = 256) -> None:
        if not chunks:
            return
        points = [
            qm.PointStruct(
                id=_key_to_uuid(_point_key(c)),
                vector=v,
                payload=_payload(c),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch, wait=True)
            log.info("qdrant.upsert_batch", n=len(batch), offset=i, total=len(points))

    def search(self, vector: list[float], *, top_k: int = 8,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        q_filter = _build_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        ).points
        return [{"score": h.score, **(h.payload or {})} for h in hits]


def _point_key(c: Chunk) -> str:
    src = c.source
    return f"{src.source}|{src.series_id or ''}|{src.title or ''}|{src.url or ''}|{c.chunk_index}"


def _payload(c: Chunk) -> dict[str, Any]:
    return {
        "text": c.text,
        "type": c.type,
        "lang": c.lang,
        "source": c.source.source,
        "series_id": c.source.series_id,
        "title": c.source.title,
        "url": c.source.url,
        "page": c.source.page,
        "period_start": c.period_start.isoformat() if c.period_start else None,
        "period_end": c.period_end.isoformat() if c.period_end else None,
        "chunk_index": c.chunk_index,
        **c.extra,
    }


def _build_filter(filters: dict[str, Any]) -> qm.Filter:
    must = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in filters.items()]
    return qm.Filter(must=must)

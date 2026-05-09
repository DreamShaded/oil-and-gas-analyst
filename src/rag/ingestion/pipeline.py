from __future__ import annotations

from pathlib import Path

from src.data.sources.eia_api import load_local_series
from src.data.sources.registry import EIA_DEFAULT_SERIES
from src.rag.ingestion.chunker import split_text
from src.rag.ingestion.data_card_builder import Segmentation, build_cards
from src.rag.ingestion.text_loader import load_text_documents
from src.rag.retrieval.embedder import Embedder
from src.rag.retrieval.qdrant_store import QdrantStore
from src.rag.schemas import Chunk, SourceRef
from src.utils.logging import get_logger

log = get_logger("rag.pipeline")


def _cards_to_chunks(cards) -> list[Chunk]:
    return [
        Chunk(
            text=c.text,
            type="data_card",
            lang="ru",
            source=SourceRef(source=c.source, series_id=c.series_id, title=c.metric),
            period_start=c.period_start,
            period_end=c.period_end,
            extra={"segment": c.segment, "metric": c.metric},
        )
        for c in cards
    ]


def ingest_eia_series(eia_dir: Path, *, segmentation: Segmentation = "Q") -> list[Chunk]:
    chunks: list[Chunk] = []
    for s in EIA_DEFAULT_SERIES:
        try:
            df = load_local_series(s.series_id, eia_dir)
        except FileNotFoundError:
            log.warning("ingest.eia.missing", series=s.series_id)
            continue
        cards = build_cards(df, source="EIA", segmentation=segmentation)
        chunks.extend(_cards_to_chunks(cards))
    return chunks


def ingest_text_dir(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for text, source in load_text_documents(root):
        chunks.extend(split_text(text, source))
    return chunks


def ingest_all(data_dir: Path, *, segmentation: Segmentation = "Q") -> int:
    chunks: list[Chunk] = []
    eia_dir = data_dir / "eia"
    if eia_dir.exists():
        chunks.extend(ingest_eia_series(eia_dir, segmentation=segmentation))
    series_dir = data_dir / "series"
    if series_dir.exists():
        chunks.extend(ingest_text_dir(series_dir))

    if not chunks:
        log.warning("ingest.nothing_to_index", data_dir=str(data_dir))
        return 0

    embedder = Embedder()
    vectors = embedder.embed([c.text for c in chunks])
    store = QdrantStore()
    store.ensure_collection(dim=embedder.dimension)
    store.upsert(chunks, vectors)
    log.info("ingest.done", indexed=len(chunks))
    return len(chunks)

from __future__ import annotations

from pathlib import Path

from src.data.sources import (
    eia_api,
    eia_bulk,
    iea_omr,
    minfin_oil,
    opec_downloader,
    opec_momr,
    opec_woo,
)
from src.rag.ingestion.data_card_builder import Segmentation, build_cards
from src.rag.retrieval.embedder import Embedder
from src.rag.retrieval.qdrant_store import QdrantStore
from src.rag.schemas import Chunk, SourceRef
from src.utils.logging import get_logger

log = get_logger("data.bootstrap")


def collect_eia(out_dir: Path) -> dict:
    try:
        return eia_bulk.fetch_all(out_dir)
    except Exception as e:
        log.warning("bootstrap.eia_bulk_failed", error=str(e))
        return eia_api.fetch_all_default(start=None, end=None, out_dir=out_dir)


def collect_opec_momr(reports_dir: Path, *, limit: int = 6, skip_download: bool = False) -> list[Chunk]:
    if not skip_download:
        try:
            opec_downloader.download_latest_momr(reports_dir, limit=limit)
        except Exception as e:
            log.warning("bootstrap.opec_download_failed", error=str(e))
    return opec_momr.ingest_reports_dir(reports_dir)


def collect_opec_woo(reports_dir: Path, *, limit: int = 3, skip_download: bool = False) -> list[Chunk]:
    if not skip_download:
        try:
            opec_woo.download_latest_woo(reports_dir, limit=limit)
        except Exception as e:
            log.warning("bootstrap.woo_download_failed", error=str(e))
    chunks: list[Chunk] = []
    for pdf in sorted(reports_dir.glob("woo-*.pdf")):
        try:
            chunks.extend(opec_momr.pdf_to_chunks(pdf))
        except Exception as e:
            log.warning("bootstrap.woo_parse_failed", pdf=str(pdf), error=str(e))
    for c in chunks:
        c.source.title = c.source.title.replace("OPEC MOMR", "OPEC WOO")
        c.extra["report"] = "OPEC WOO"
    return chunks


def collect_iea(out_dir: Path, *, max_back: int = 6, skip_download: bool = False) -> list[Chunk]:
    if not skip_download:
        try:
            iea_omr.fetch_latest(out_dir, max_back=max_back)
        except Exception as e:
            log.warning("bootstrap.iea_fetch_failed", error=str(e))
    return iea_omr.ingest_iea_dir(out_dir)


def collect_minfin(out_dir: Path, *, segmentation: Segmentation, skip_download: bool = False) -> list[Chunk]:
    import pandas as pd
    csv = out_dir / "oil_revenue.csv"
    if not skip_download:
        try:
            minfin_oil.fetch_minfin_oil(out_dir)
        except Exception as e:
            log.warning("bootstrap.minfin_fetch_failed", error=str(e))
    if not csv.exists():
        return []
    df = pd.read_csv(csv)
    if df.empty:
        return []
    df["period"] = pd.to_datetime(df["period"])
    chunks: list[Chunk] = []
    for metric, grp in df.groupby("metric"):
        series_df = grp[["period", "value"]].copy()
        series_df["series_id"] = f"minfin/{_slug(metric)}"
        series_df["metric"] = metric
        series_df["unit"] = "млрд руб"
        cards = build_cards(series_df, source="Minfin", segmentation=segmentation)
        chunks.extend(_cards_to_chunks(cards, "Minfin"))
    return chunks


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:60]


def _cards_to_chunks(cards, source: str) -> list[Chunk]:
    return [
        Chunk(
            text=c.text,
            type="data_card",
            lang="ru",
            source=SourceRef(source=source, series_id=c.series_id, title=c.metric),
            period_start=c.period_start,
            period_end=c.period_end,
            extra={"segment": c.segment, "metric": c.metric},
        )
        for c in cards
    ]


def _eia_dataframes_to_chunks(dataframes: dict, segmentation: Segmentation) -> list[Chunk]:
    chunks: list[Chunk] = []
    for df in dataframes.values():
        cards = build_cards(df, source="EIA", segmentation=segmentation)
        chunks.extend(_cards_to_chunks(cards, "EIA"))
    return chunks


def _index(chunks: list[Chunk]) -> int:
    if not chunks:
        log.warning("bootstrap.no_chunks")
        return 0
    embedder = Embedder()
    vectors = embedder.embed([c.text for c in chunks])
    store = QdrantStore()
    store.ensure_collection(dim=embedder.dimension)
    store.upsert(chunks, vectors)
    log.info("bootstrap.indexed", chunks=len(chunks))
    try:
        store.create_snapshot(Path("data/snapshots"))
    except Exception as e:
        log.warning("bootstrap.snapshot_failed", error=str(e))
    return len(chunks)


def run(data_dir: Path = Path("data"),
        *, segmentation: Segmentation = "Q",
        skip_eia: bool = False,
        skip_opec_momr: bool = False, opec_momr_limit: int = 6,
        skip_opec_woo: bool = False, opec_woo_limit: int = 3,
        skip_iea: bool = False, iea_months: int = 6,
        skip_minfin: bool = False,
        offline: bool = False) -> dict:
    eia_chunks: list[Chunk] = []
    if not skip_eia:
        dataframes = collect_eia(data_dir / "eia") if not offline else {}
        eia_chunks = _eia_dataframes_to_chunks(dataframes, segmentation)
    momr_chunks: list[Chunk] = []
    if not skip_opec_momr:
        momr_chunks = collect_opec_momr(data_dir / "reports",
                                        limit=opec_momr_limit, skip_download=offline)
    woo_chunks: list[Chunk] = []
    if not skip_opec_woo:
        woo_chunks = collect_opec_woo(data_dir / "reports",
                                      limit=opec_woo_limit, skip_download=offline)
    iea_chunks: list[Chunk] = []
    if not skip_iea:
        iea_chunks = collect_iea(data_dir / "iea", max_back=iea_months, skip_download=offline)
    minfin_chunks: list[Chunk] = []
    if not skip_minfin:
        minfin_chunks = collect_minfin(data_dir / "minfin",
                                       segmentation=segmentation, skip_download=offline)

    all_chunks = eia_chunks + momr_chunks + woo_chunks + iea_chunks + minfin_chunks
    indexed = _index(all_chunks)
    return {
        "eia": len(eia_chunks),
        "opec_momr": len(momr_chunks),
        "opec_woo": len(woo_chunks),
        "iea": len(iea_chunks),
        "minfin": len(minfin_chunks),
        "indexed": indexed,
    }

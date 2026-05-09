from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pymupdf4llm

from src.rag.ingestion.chunker import split_text
from src.rag.schemas import Chunk, SourceRef
from src.utils.logging import get_logger

log = get_logger("data.opec_momr")

MONTH_RE = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")[\s_]+(\d{4})\b",
    re.IGNORECASE,
)

MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _extract_report_period(text_head: str, fallback_name: str) -> date | None:
    for candidate in (text_head[:2000], fallback_name):
        m = MONTH_RE.search(candidate)
        if m:
            month = MONTH_NUM[m.group(1).lower()]
            year = int(m.group(2))
            return date(year, month, 1)
    return None


def pdf_to_chunks(pdf_path: Path) -> list[Chunk]:
    log.info("opec.parse_pdf", path=str(pdf_path))
    md = pymupdf4llm.to_markdown(str(pdf_path))
    if not md.strip():
        return []

    period = _extract_report_period(md, pdf_path.stem)
    base_source = SourceRef(
        source="OPEC",
        title=f"OPEC MOMR {pdf_path.stem}",
        url=str(pdf_path),
    )
    raw_chunks = split_text(md, base_source)
    for c in raw_chunks:
        if period:
            c.period_start = period
            c.period_end = period
        c.extra["report"] = "OPEC MOMR"
        c.extra["pdf_name"] = pdf_path.name
    log.info("opec.parsed", path=str(pdf_path), chunks=len(raw_chunks), period=str(period))
    return raw_chunks


def ingest_reports_dir(reports_dir: Path) -> list[Chunk]:
    if not reports_dir.exists():
        return []
    chunks: list[Chunk] = []
    for pdf in sorted(reports_dir.glob("*.pdf")) + sorted(reports_dir.glob("*.PDF")):
        try:
            chunks.extend(pdf_to_chunks(pdf))
        except Exception as e:
            log.warning("opec.parse_failed", path=str(pdf), error=str(e))
    return chunks

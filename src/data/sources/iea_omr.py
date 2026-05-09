from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

from src.data.cache import FileCache
from src.rag.ingestion.chunker import split_text
from src.rag.schemas import Chunk, SourceRef
from src.utils.logging import get_logger

log = get_logger("data.iea_omr")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TEXT_TTL = timedelta(days=30)
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def _report_url(month_idx: int, year: int) -> str:
    return f"https://www.iea.org/reports/oil-market-report-{MONTHS[month_idx]}-{year}"


def _candidate_months(today: date, max_back: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = today.year, today.month
    for _ in range(max_back):
        out.append((m - 1, y))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return out


async def _fetch_text(url: str) -> str | None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            if not resp or resp.status != 200:
                return None
            await page.wait_for_timeout(1500)
            main_text = await page.eval_on_selector(
                "main, article, body",
                "el => el.innerText",
            )
            return main_text
        except Exception:
            return None
        finally:
            await browser.close()


def fetch_latest(reports_dir: Path, *, max_back: int = 6,
                 cache_root: Path | None = None) -> list[Path]:
    cache = FileCache(cache_root or Path("data/cache"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    misses = 0
    for month_idx, year in _candidate_months(date.today(), max_back + 4):
        if len(files) >= max_back or misses >= 3:
            break
        url = _report_url(month_idx, year)
        slug = f"iea-omr-{MONTHS[month_idx]}-{year}"
        target = reports_dir / f"{slug}.txt"
        cache_key = f"iea_omr/{slug}.txt"

        if target.exists() and cache.is_fresh(cache_key, TEXT_TTL):
            log.info("iea.skip_fresh", slug=slug)
            files.append(target)
            misses = 0
            continue

        text = asyncio.run(_fetch_text(url))
        if not text or len(text) < 1000:
            misses += 1
            log.info("iea.miss", url=url, misses=misses)
            continue

        payload = {"url": url, "month_idx": month_idx, "year": year, "text": text}
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        cache.write_bytes(cache_key, b"ok")
        log.info("iea.fetched", slug=slug, chars=len(text))
        files.append(target)
        misses = 0
    return files


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def text_to_chunks(path: Path) -> list[Chunk]:
    data = _load(path)
    month_idx = data["month_idx"]
    year = data["year"]
    period = date(year, month_idx + 1, 1)
    source = SourceRef(
        source="IEA",
        title=f"IEA OMR {MONTHS[month_idx].capitalize()} {year}",
        url=data["url"],
    )
    chunks = split_text(data["text"], source)
    for c in chunks:
        c.period_start = period
        c.period_end = period
        c.extra["report"] = "IEA OMR"
    return chunks


def ingest_iea_dir(reports_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(reports_dir.glob("iea-omr-*.txt")):
        try:
            chunks.extend(text_to_chunks(path))
        except Exception as e:
            log.warning("iea.parse_failed", path=str(path), error=str(e))
    return chunks

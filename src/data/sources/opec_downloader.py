from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from playwright.async_api import async_playwright

from src.data.cache import FileCache
from src.utils.logging import get_logger

log = get_logger("data.opec_downloader")

MOMR_INDEX_URL = "https://www.opec.org/monthly-oil-market-report.html"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
INDEX_TTL = timedelta(hours=6)
PDF_TTL = timedelta(days=30)


async def _discover_pdf_urls(limit: int) -> list[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()
            await page.goto(MOMR_INDEX_URL, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(1500)
            urls: list[str] = await page.eval_on_selector_all(
                "a[href*='/assets/assetdb/momr-'][href$='.pdf']",
                "els => Array.from(new Set(els.map(e => e.href)))",
            )
        finally:
            await browser.close()
    log.info("opec.discovered", count=len(urls), sample=urls[:3])
    return urls[:limit]


async def _download_pdf(url: str, target: Path) -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            await ctx.new_page()  # прогрев CF-cookies
            resp = await ctx.request.get(url, timeout=60_000)
            if resp.status != 200:
                raise RuntimeError(f"OPEC {url}: HTTP {resp.status}")
            data = await resp.body()
        finally:
            await browser.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


def download_latest_momr(reports_dir: Path, *, limit: int = 6,
                         cache_root: Path | None = None) -> list[Path]:
    cache = FileCache(cache_root or Path("data/cache"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    urls = asyncio.run(_discover_pdf_urls(limit))
    if not urls:
        log.warning("opec.no_urls_found")
        return []

    downloaded: list[Path] = []
    for url in urls:
        filename = url.rsplit("/", 1)[-1]
        target = reports_dir / filename
        cache_key = f"opec/{filename}"

        if target.exists() and cache.is_fresh(cache_key, PDF_TTL):
            log.info("opec.skip_fresh", file=filename)
            downloaded.append(target)
            continue

        try:
            size = asyncio.run(_download_pdf(url, target))
            cache.write_bytes(cache_key, b"ok")
            log.info("opec.downloaded", file=filename, bytes=size)
            downloaded.append(target)
        except Exception as e:
            log.warning("opec.download_failed", url=url, error=str(e))
    return downloaded

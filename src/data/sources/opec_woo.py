from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from playwright.async_api import async_playwright

from src.data.cache import FileCache
from src.utils.logging import get_logger

log = get_logger("data.opec_woo")

WOO_INDEX_URL = "https://www.opec.org/world-oil-outlook.html"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
PDF_TTL = timedelta(days=180)


async def _discover() -> list[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()
            await page.goto(WOO_INDEX_URL, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(1500)
            urls: list[str] = await page.eval_on_selector_all(
                "a[href*='/assets/assetdb/woo-'][href$='.pdf']",
                "els => Array.from(new Set(els.map(e => e.href)))",
            )
        finally:
            await browser.close()
    urls.sort(reverse=True)
    log.info("woo.discovered", count=len(urls), sample=urls[:3])
    return urls


async def _download(url: str, target: Path) -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            await ctx.new_page()
            resp = await ctx.request.get(url, timeout=120_000)
            if resp.status != 200:
                raise RuntimeError(f"WOO {url}: HTTP {resp.status}")
            data = await resp.body()
        finally:
            await browser.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


def download_latest_woo(reports_dir: Path, *, limit: int = 3,
                        cache_root: Path | None = None) -> list[Path]:
    cache = FileCache(cache_root or Path("data/cache"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    urls = asyncio.run(_discover())[:limit]
    if not urls:
        log.warning("woo.no_urls_found")
        return []

    out: list[Path] = []
    for url in urls:
        filename = url.rsplit("/", 1)[-1]
        target = reports_dir / filename
        cache_key = f"opec_woo/{filename}"
        if target.exists() and cache.is_fresh(cache_key, PDF_TTL):
            log.info("woo.skip_fresh", file=filename)
            out.append(target)
            continue
        try:
            size = asyncio.run(_download(url, target))
            cache.write_bytes(cache_key, b"ok")
            log.info("woo.downloaded", file=filename, bytes=size)
            out.append(target)
        except Exception as e:
            log.warning("woo.download_failed", url=url, error=str(e))
    return out

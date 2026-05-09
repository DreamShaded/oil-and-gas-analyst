from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.tools.source_filter import extract_domain, is_blacklisted, trust_score
from src.utils.logging import get_logger

log = get_logger("tools.web_search")

CACHE_DIR = Path("data/cache/web_search")
CACHE_TTL_SEC = 15 * 60
Topic = Literal["news", "general"]


class WebSnippet(BaseModel):
    title: str
    url: str
    snippet: str
    published_at: str | None = None
    source_domain: str
    trust: int  # +1 whitelist, 0 neutral, -1 blacklisted


class WebSearchResult(BaseModel):
    query: str
    topic: Topic
    snippets: list[WebSnippet]
    urls: list[str]


def _cache_key(query: str, topic: str, max_results: int) -> str:
    return hashlib.sha1(f"{topic}|{max_results}|{query}".encode()).hexdigest()


def _read_cache(key: str) -> dict | None:
    f = CACHE_DIR / f"{key}.json"
    if not f.exists():
        return None
    if time.time() - f.stat().st_mtime > CACHE_TTL_SEC:
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _write_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def _tavily_call(client: TavilyClient, query: str, topic: Topic, max_results: int) -> dict[str, Any]:
    return client.search(query=query, search_depth="advanced", topic=topic, max_results=max_results)


def search_web(query: str, *, max_results: int = 5, topic: Topic = "news") -> WebSearchResult:
    if not query.strip():
        raise ValueError("query пуст")

    cache_key = _cache_key(query, topic, max_results)
    cached = _read_cache(cache_key)
    if cached:
        log.info("web.cache_hit", query=query)
        return WebSearchResult.model_validate(cached)

    s = get_settings()
    if not s.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY не задан в .env")

    client = TavilyClient(api_key=s.tavily_api_key.get_secret_value())
    log.info("web.query", q=query, topic=topic, max=max_results)
    raw = _tavily_call(client, query, topic, max_results * 2)  # с запасом под фильтр

    snippets: list[WebSnippet] = []
    for item in raw.get("results", []):
        url = item.get("url") or ""
        if not url or is_blacklisted(url):
            continue
        snippets.append(WebSnippet(
            title=item.get("title", "").strip(),
            url=url,
            snippet=(item.get("content") or "").strip()[:1000],
            published_at=item.get("published_date"),
            source_domain=extract_domain(url),
            trust=trust_score(url),
        ))
    snippets.sort(key=lambda s: (-s.trust, s.published_at or ""), reverse=False)
    snippets = snippets[:max_results]

    result = WebSearchResult(
        query=query,
        topic=topic,
        snippets=snippets,
        urls=[s.url for s in snippets],
    )
    _write_cache(cache_key, result.model_dump())
    log.info("web.done", returned=len(snippets), filtered=len(raw.get("results", [])) - len(snippets))
    return result

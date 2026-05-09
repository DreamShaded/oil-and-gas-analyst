from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml

CONFIG_DIR = Path("config")


def _flatten(d: dict | None) -> set[str]:
    if not d:
        return set()
    out: set[str] = set()
    for v in d.values():
        if isinstance(v, list):
            out.update(item.strip().lower() for item in v if isinstance(item, str))
    return out


@lru_cache(maxsize=1)
def whitelist() -> set[str]:
    f = CONFIG_DIR / "source-whitelist.yaml"
    if not f.exists():
        return set()
    return _flatten(yaml.safe_load(f.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def blacklist() -> set[str]:
    f = CONFIG_DIR / "source-blacklist.yaml"
    if not f.exists():
        return set()
    return _flatten(yaml.safe_load(f.read_text(encoding="utf-8")))


def extract_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_blacklisted(url: str) -> bool:
    host = extract_domain(url)
    return any(host == bad or host.endswith("." + bad) for bad in blacklist())


def trust_score(url: str) -> int:
    """0 — обычный, +1 — в whitelist, -1 — в blacklist (используется для сортировки)."""
    host = extract_domain(url)
    if any(host == bad or host.endswith("." + bad) for bad in blacklist()):
        return -1
    if any(host == good or host.endswith("." + good) for good in whitelist()):
        return 1
    return 0

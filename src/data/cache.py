from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger("data.cache")


class FileCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key

    def is_fresh(self, key: str, ttl: timedelta) -> bool:
        p = self.path(key)
        if not p.exists():
            return False
        age = datetime.utcnow() - datetime.utcfromtimestamp(p.stat().st_mtime)
        return age < ttl

    def write_bytes(self, key: str, data: bytes) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        log.info("cache.write", key=key, bytes=len(data))
        return p

    def read_bytes(self, key: str) -> bytes:
        return self.path(key).read_bytes()

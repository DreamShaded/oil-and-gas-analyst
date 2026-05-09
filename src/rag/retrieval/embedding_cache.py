from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger("rag.embedding_cache")

DEFAULT_PATH = Path("data/cache/embeddings.sqlite")


class EmbeddingCache:
    def __init__(self, db_path: Path = DEFAULT_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            " key TEXT PRIMARY KEY,"
            " model TEXT NOT NULL,"
            " dim INTEGER NOT NULL,"
            " vector BLOB NOT NULL,"
            " created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha1(f"{model}|{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        cur = self._conn.execute(
            f"SELECT key, dim, vector FROM embeddings WHERE key IN ({placeholders})", keys,
        )
        out: dict[str, list[float]] = {}
        for k, dim, blob in cur.fetchall():
            out[k] = list(struct.unpack(f"{dim}f", blob))
        return out

    def put_many(self, model: str, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        rows = [
            (k, model, len(v), struct.pack(f"{len(v)}f", *v))
            for k, v in items
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings(key, model, dim, vector) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(dim*4),0) FROM embeddings")
        n, bytes_used = cur.fetchone()
        return {"count": int(n), "bytes": int(bytes_used)}

    def close(self) -> None:
        self._conn.close()

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from langdetect import LangDetectException, detect

from src.rag.schemas import Chunk, SourceRef


def load_text_documents(root: Path) -> Iterator[tuple[str, SourceRef]]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".md":
            yield path.read_text(encoding="utf-8"), SourceRef(source=root.name, title=path.stem, url=str(path))
        elif path.suffix.lower() == ".json":
            yield from _flatten_json(path, source=root.name)


def _flatten_json(path: Path, *, source: str) -> Iterator[tuple[str, SourceRef]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(data, list):
        for i, item in enumerate(data):
            text = _stringify(item)
            if text:
                yield text, SourceRef(source=source, title=f"{path.stem}#{i}", url=str(path))
    elif isinstance(data, dict):
        text = _stringify(data)
        if text:
            yield text, SourceRef(source=source, title=path.stem, url=str(path))


def _stringify(obj) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        parts = [f"{k}: {v}" for k, v in obj.items() if v is not None]
        return "\n".join(parts)
    return str(obj)


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def text_to_chunk(text: str, source: SourceRef, chunk_index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        type="text",
        lang=detect_lang(text),
        source=source,
        chunk_index=chunk_index,
    )

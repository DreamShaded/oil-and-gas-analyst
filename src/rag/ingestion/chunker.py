from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.rag.ingestion.text_loader import detect_lang
from src.rag.schemas import Chunk, SourceRef


def split_text(text: str, source: SourceRef, *,
               chunk_size: int | None = None,
               chunk_overlap: int | None = None) -> list[Chunk]:
    if not text.strip():
        return []
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.rag_chunk_size,
        chunk_overlap=chunk_overlap or s.rag_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    lang = detect_lang(text)
    return [
        Chunk(text=piece, type="text", lang=lang, source=source, chunk_index=i)
        for i, piece in enumerate(splitter.split_text(text))
    ]

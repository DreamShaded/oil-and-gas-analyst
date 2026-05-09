from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

ChunkType = Literal["data_card", "text"]


class SourceRef(BaseModel):
    source: str
    series_id: str | None = None
    title: str | None = None
    url: str | None = None
    page: int | None = None


class DataCard(BaseModel):
    source: str
    series_id: str
    segment: str
    metric: str
    text: str
    period_start: date
    period_end: date
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    def deterministic_key(self) -> str:
        return f"{self.source}|{self.series_id}|{self.segment}|{self.metric}"


class Chunk(BaseModel):
    text: str
    type: ChunkType
    lang: str
    source: SourceRef
    period_start: date | None = None
    period_end: date | None = None
    chunk_index: int = 0
    extra: dict = Field(default_factory=dict)

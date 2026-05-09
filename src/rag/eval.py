from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.rag.retrieval.retriever import Retriever


class GoldenCase(BaseModel):
    query: str
    expected_series_ids: list[str] = []
    expected_sources: list[str] = []


class CaseResult(BaseModel):
    query: str
    hit_at_k: bool
    reciprocal_rank: float
    matched_at: int | None
    top_hit: dict[str, Any] | None


class EvalReport(BaseModel):
    n: int
    hit_at_k: float
    mrr: float
    per_case: list[CaseResult]


def load_golden(path: Path) -> list[GoldenCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenCase(**item) for item in raw]


def _match(hit: dict[str, Any], case: GoldenCase) -> bool:
    if case.expected_series_ids and hit.get("series_id") in case.expected_series_ids:
        return True
    return bool(case.expected_sources and hit.get("source") in case.expected_sources)


def evaluate(cases: list[GoldenCase], *, top_k: int = 8,
             retriever: Retriever | None = None) -> EvalReport:
    r = retriever or Retriever()
    results: list[CaseResult] = []
    hits_at_k = 0
    mrr_sum = 0.0
    for case in cases:
        hits = r.search(case.query, top_k=top_k, min_score=0.0)
        matched_at: int | None = None
        for i, h in enumerate(hits, 1):
            if _match(h, case):
                matched_at = i
                break
        rr = (1.0 / matched_at) if matched_at else 0.0
        results.append(CaseResult(
            query=case.query,
            hit_at_k=matched_at is not None,
            reciprocal_rank=rr,
            matched_at=matched_at,
            top_hit=hits[0] if hits else None,
        ))
        hits_at_k += 1 if matched_at else 0
        mrr_sum += rr
    n = len(cases) or 1
    return EvalReport(
        n=len(cases),
        hit_at_k=hits_at_k / n,
        mrr=mrr_sum / n,
        per_case=results,
    )

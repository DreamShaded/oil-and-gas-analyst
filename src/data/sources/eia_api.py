from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.data.sources.registry import EIA_DEFAULT_SERIES, EIASeries, find_series
from src.utils.logging import get_logger

log = get_logger("data.eia_api")

EIA_V2_BASE = "https://api.eia.gov/v2"
PAGE_SIZE = 5000


class EIAError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True)
def _fetch_page(client: httpx.Client, series_id: str, api_key: str,
                start: date | None, end: date | None, offset: int) -> dict[str, Any]:
    params: dict[str, Any] = {"api_key": api_key, "offset": offset, "length": PAGE_SIZE}
    if start:
        params["start"] = start.isoformat()
    if end:
        params["end"] = end.isoformat()
    resp = client.get(f"{EIA_V2_BASE}/seriesid/{series_id}", params=params, timeout=30.0)
    if resp.status_code != 200:
        raise EIAError(f"EIA {series_id}: HTTP {resp.status_code} — {resp.text[:200]}")
    body = resp.json()
    if "error" in body:
        raise EIAError(f"EIA {series_id}: {body['error']}")
    return body


def fetch_series(series: EIASeries, *, start: date | None, end: date | None,
                 out_dir: Path) -> pd.DataFrame:
    s = get_settings()
    key = s.eia_api_key.get_secret_value() if s.eia_api_key else None
    if not key:
        raise EIAError("EIA_API_KEY не задан в .env — используйте eia_bulk.fetch_all без ключа")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    with httpx.Client() as client:
        while True:
            body = _fetch_page(client, series.series_id, key, start, end, offset)
            data = body.get("response", {}).get("data", []) or []
            rows.extend(data)
            if len(data) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    (out_dir / f"{series.series_id}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    df = _to_dataframe(rows, series)
    df.to_csv(out_dir / f"{series.series_id}.csv", index=False)
    log.info("eia_api.fetched", series=series.series_id, rows=len(df))
    return df


def _to_dataframe(rows: list[dict[str, Any]], series: EIASeries) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["period", "value", "series_id", "metric", "unit"])
    df = pd.DataFrame(rows)
    value_col = "value" if "value" in df.columns else next(
        (c for c in df.columns if c not in {"period", "duoarea", "area-name", "product", "product-name"}),
        None,
    )
    if value_col is None:
        raise EIAError(f"{series.series_id}: не нашёл колонку значения. Колонки: {list(df.columns)}")
    df = df[["period", value_col]].rename(columns={value_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
    df["series_id"] = series.series_id
    df["metric"] = series.metric
    df["unit"] = series.unit
    df["frequency"] = series.frequency
    df["source"] = "EIA"
    return df


def fetch_all_default(*, start: date | None, end: date | None, out_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for s in EIA_DEFAULT_SERIES:
        try:
            result[s.series_id] = fetch_series(s, start=start, end=end, out_dir=out_dir)
        except EIAError as e:
            log.warning("eia_api.fetch_failed", series=s.series_id, error=str(e))
    return result


def load_local_series(series_id: str, data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / f"{series_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df["period"] = pd.to_datetime(df["period"]).dt.date
    if "series_id" not in df.columns:
        meta = find_series(series_id)
        df["series_id"] = series_id
        df["metric"] = meta.metric if meta else series_id
        df["unit"] = meta.unit if meta else ""
        df["frequency"] = meta.frequency if meta else "monthly"
        df["source"] = "EIA"
    return df

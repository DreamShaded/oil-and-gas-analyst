from __future__ import annotations

import io
import json
import zipfile
from datetime import timedelta
from pathlib import Path

import httpx
import pandas as pd

from src.data.cache import FileCache
from src.data.sources.registry import (
    EIA_DEFAULT_SERIES,
    EIASeries,
    bulk_categories,
    bulk_url,
)
from src.data.time import parse_eia_period
from src.utils.logging import get_logger

log = get_logger("data.eia_bulk")

BULK_TTL = timedelta(days=1)


def download_bulk(category: str, cache: FileCache) -> Path:
    key = f"eia/bulk/{category}.zip"
    if cache.is_fresh(key, BULK_TTL):
        log.info("bulk.cache_hit", category=category)
        return cache.path(key)

    url = bulk_url(category)
    log.info("bulk.download", category=category, url=url)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return cache.write_bytes(key, resp.content)


def _iter_jsonl(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".txt") or m.endswith(".jsonl")]
        if not members:
            raise RuntimeError(f"В {zip_path.name} нет JSONL-файлов")
        for member in members:
            with zf.open(member) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8"):
                    raw = raw.strip()
                    if raw:
                        yield json.loads(raw)


def _record_to_dataframe(record: dict, series: EIASeries) -> pd.DataFrame:
    rows = record.get("data") or []
    if not rows:
        return pd.DataFrame(columns=["period", "value"])
    df = pd.DataFrame(rows, columns=["period", "value"])
    df["period"] = df["period"].astype(str).map(parse_eia_period)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
    df["series_id"] = series.series_id
    df["metric"] = series.metric
    df["unit"] = series.unit
    df["frequency"] = series.frequency
    df["source"] = "EIA"
    df["source_published_at"] = record.get("last_updated")
    return df


def extract_series(zip_path: Path, wanted: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in _iter_jsonl(zip_path):
        sid = record.get("series_id")
        if sid in wanted:
            out[sid] = record
            if len(out) == len(wanted):
                break
    return out


def fetch_all(out_dir: Path, *, cache_root: Path | None = None) -> dict[str, pd.DataFrame]:
    cache = FileCache(cache_root or Path("data/cache"))
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_by_cat: dict[str, dict[str, EIASeries]] = {}
    for s in EIA_DEFAULT_SERIES:
        wanted_by_cat.setdefault(s.bulk_category, {})[s.series_id] = s

    result: dict[str, pd.DataFrame] = {}
    for category in bulk_categories():
        zip_path = download_bulk(category, cache)
        records = extract_series(zip_path, set(wanted_by_cat[category].keys()))
        for sid, rec in records.items():
            df = _record_to_dataframe(rec, wanted_by_cat[category][sid])
            csv_path = out_dir / f"{sid}.csv"
            df.to_csv(csv_path, index=False)
            result[sid] = df
            log.info("bulk.series_extracted", series=sid, rows=len(df), csv=str(csv_path))
        missing = set(wanted_by_cat[category]) - set(records)
        if missing:
            log.warning("bulk.series_missing", category=category, missing=sorted(missing))
    return result

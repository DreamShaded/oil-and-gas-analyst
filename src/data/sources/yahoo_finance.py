from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.cache import FileCache
from src.utils.logging import get_logger

log = get_logger("data.yahoo_finance")

TICKERS = {
    "dxy":   "DX-Y.NYB",   # ICE U.S. Dollar Index
    "brent_fut": "BZ=F",   # Brent futures (резерв, если EIA отстаёт)
    "wti_fut":   "CL=F",   # WTI futures
}
CACHE_TTL = timedelta(days=1)


def fetch_ticker(name: str, out_dir: Path,
                 cache_root: Path | None = None,
                 period: str = "max", interval: str = "1d") -> pd.DataFrame:
    if name not in TICKERS:
        raise ValueError(f"Unknown ticker name: {name}. Доступны: {list(TICKERS)}")
    cache = FileCache(cache_root or Path("data/cache"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    cache_key = f"yahoo/{name}.csv"

    if csv_path.exists() and cache.is_fresh(cache_key, CACHE_TTL):
        log.info("yahoo.skip_fresh", name=name)
        return pd.read_csv(csv_path, parse_dates=["period"])

    ticker = TICKERS[name]
    log.info("yahoo.download", ticker=ticker, period=period)
    raw = yf.download(ticker, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance вернул пусто для {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    period_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={period_col: "period", "Close": "value"})
    df = df[["period", "value"]].dropna()
    df["period"] = pd.to_datetime(df["period"]).dt.date
    df["series_id"] = ticker
    df["metric"] = name
    df["unit"] = "index" if name == "dxy" else "USD/barrel"
    df["frequency"] = "daily"
    df["source"] = "Yahoo"
    df.to_csv(csv_path, index=False)
    cache.write_bytes(cache_key, b"ok")
    log.info("yahoo.saved", name=name, rows=len(df), csv=str(csv_path))
    return df


def fetch_dxy(out_dir: Path = Path("data/yahoo")) -> pd.DataFrame:
    return fetch_ticker("dxy", out_dir)

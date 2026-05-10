from __future__ import annotations

import contextlib
from pathlib import Path

import pandas as pd

from src.data.sources.eia_api import load_local_series

PRICES_DIR = Path("data/prices")

# Активы: EIA series ID для Brent/WTI, локальный CSV для Urals (нет публичного API).
ASSET_EIA_SERIES = {
    "brent": "PET.RBRTE.D",
    "wti":   "PET.RWTC.D",
}
ASSET_MANUAL_CSV = {
    "urals": PRICES_DIR / "urals.csv",
}

FACTOR_SERIES: dict[str, str] = {
    "us_crude_stocks":   "PET.WCESTUS1.W",
    "us_crude_prod":     "PET.WCRFPUS2.W",
    "world_oil_supply":  "STEO.PAPR_WORLD.M",
    "world_oil_demand":  "STEO.PATC_WORLD.M",
    "opec_crude_prod":   "STEO.PAPR_OPEC.M",
    "nonopec_supply":    "STEO.PAPR_NONOPEC.M",
}
YAHOO_FACTOR_CSV = Path("data/yahoo/dxy.csv")


def _resample_monthly(df: pd.DataFrame) -> pd.Series:
    s = df.set_index("period")["value"].astype(float).sort_index()
    return s.resample("MS").mean().dropna()


def _load_eia_monthly(series_id: str, eia_dir: Path) -> pd.Series:
    df = load_local_series(series_id, eia_dir)
    df = df[["period", "value"]].copy()
    df["period"] = pd.to_datetime(df["period"])
    return _resample_monthly(df)


def _load_csv_monthly(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "period" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{csv_path} должен содержать колонки period,value")
    df["period"] = pd.to_datetime(df["period"])
    return _resample_monthly(df)


def load_price_monthly(asset: str, eia_dir: Path = Path("data/eia")) -> pd.Series:
    if asset in ASSET_EIA_SERIES:
        return _load_eia_monthly(ASSET_EIA_SERIES[asset], eia_dir)
    if asset in ASSET_MANUAL_CSV:
        path = ASSET_MANUAL_CSV[asset]
        if not path.exists():
            raise FileNotFoundError(
                f"Для {asset} нет публичного API; положите ручной CSV в {path} "
                "(колонки: period,value)."
            )
        return _load_csv_monthly(path)
    raise ValueError(f"Неизвестный актив: {asset}. Доступны: {available_assets()}")


def available_assets() -> list[str]:
    out = list(ASSET_EIA_SERIES)
    for a, p in ASSET_MANUAL_CSV.items():
        if p.exists():
            out.append(a)
    return out


def load_factors_monthly(eia_dir: Path = Path("data/eia")) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for name, sid in FACTOR_SERIES.items():
        try:
            frames[name] = _load_eia_monthly(sid, eia_dir)
        except FileNotFoundError:
            continue
    if YAHOO_FACTOR_CSV.exists():
        with contextlib.suppress(ValueError, FileNotFoundError):
            frames["dxy"] = _load_csv_monthly(YAHOO_FACTOR_CSV)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def aligned_xy(asset: str, eia_dir: Path = Path("data/eia")) -> tuple[pd.DataFrame, pd.Series]:
    y = load_price_monthly(asset, eia_dir)
    X = load_factors_monthly(eia_dir)
    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    return df.drop(columns=["y"]), df["y"]

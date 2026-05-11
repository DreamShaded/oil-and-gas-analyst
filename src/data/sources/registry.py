from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Frequency = Literal["daily", "weekly", "monthly", "quarterly", "annual"]


@dataclass(frozen=True)
class EIASeries:
    series_id: str
    metric: str          # snake_case ключ (для API/программного доступа)
    title: str           # английское название EIA
    frequency: Frequency
    unit: str
    bulk_category: str
    label_ru: str = ""   # человекочитаемое русское название для data-card-текста и retrieval


EIA_DEFAULT_SERIES: tuple[EIASeries, ...] = (
    EIASeries("PET.RBRTE.D",         "brent_spot",        "Europe Brent Spot Price FOB",            "daily",   "USD/barrel",            "PET",  "Цена нефти Brent (спот, EIA)"),
    EIASeries("PET.RWTC.D",          "wti_spot",          "Cushing OK WTI Spot Price FOB",          "daily",   "USD/barrel",            "PET",  "Цена нефти WTI (спот, EIA)"),
    EIASeries("PET.WCESTUS1.W",      "us_crude_stocks",   "U.S. Ending Stocks of Crude Oil",        "weekly",  "thousand barrels",      "PET",  "Запасы сырой нефти США (коммерческие)"),
    EIASeries("PET.WCRFPUS2.W",      "us_crude_prod",     "U.S. Field Production of Crude Oil",     "weekly",  "thousand barrels/day",  "PET",  "Добыча сырой нефти в США"),
    EIASeries("PET.WCSSTUS1.W",      "spr_stocks",        "U.S. SPR Crude Oil Stocks",              "weekly",  "thousand barrels",      "PET",  "Стратегический нефтяной резерв США (SPR)"),
    EIASeries("PET.WCRRIUS2.W",      "us_refinery_input", "U.S. Refinery Net Input of Crude Oil",   "weekly",  "thousand barrels/day",  "PET",  "Загрузка НПЗ США (вход сырой нефти)"),
    EIASeries("PET.WCRIMUS2.W",      "us_crude_imports",  "U.S. Crude Oil Imports",                 "weekly",  "thousand barrels/day",  "PET",  "Импорт сырой нефти в США"),
    EIASeries("STEO.PAPR_WORLD.M",   "world_oil_supply",  "World Oil Supply (STEO)",                "monthly", "million barrels/day",   "STEO", "Мировое предложение нефти (STEO)"),
    EIASeries("STEO.PATC_WORLD.M",   "world_oil_demand",  "World Oil Demand (STEO)",                "monthly", "million barrels/day",   "STEO", "Мировой спрос на нефть (STEO)"),
    EIASeries("STEO.PAPR_OPEC.M",    "opec_crude_prod",   "OPEC Crude Oil Production (STEO)",       "monthly", "million barrels/day",   "STEO", "Добыча нефти ОПЕК (STEO)"),
    EIASeries("STEO.PAPR_NONOPEC.M", "nonopec_supply",    "Non-OPEC Liquids Supply (STEO)",         "monthly", "million barrels/day",   "STEO", "Предложение нефти не-ОПЕК (STEO)"),
    EIASeries("STEO.COPRPUS.M",      "us_crude_prod_fc",  "U.S. Crude Oil Production Forecast",     "monthly", "million barrels/day",   "STEO", "Прогноз добычи нефти в США (STEO)"),
    EIASeries("STEO.BREPUUS.M",      "brent_forecast",    "Brent Spot Price Forecast (STEO)",       "monthly", "USD/barrel",            "STEO", "Прогноз цены Brent (STEO)"),
    EIASeries("STEO.WTIPUUS.M",      "wti_forecast",      "WTI Spot Price Forecast (STEO)",         "monthly", "USD/barrel",            "STEO", "Прогноз цены WTI (STEO)"),
)


def find_series(series_id: str) -> EIASeries | None:
    for s in EIA_DEFAULT_SERIES:
        if s.series_id == series_id:
            return s
    return None


def bulk_categories() -> set[str]:
    return {s.bulk_category for s in EIA_DEFAULT_SERIES}


EIA_BULK_BASE = "https://www.eia.gov/opendata/bulk"


def bulk_url(category: str) -> str:
    return f"{EIA_BULK_BASE}/{category}.zip"

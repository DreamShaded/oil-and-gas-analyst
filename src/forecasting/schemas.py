from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Method = Literal["sarima", "regression"]
Asset = Literal["brent", "wti", "urals"]


class ForecastPoint(BaseModel):
    period: date
    point: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class Forecast(BaseModel):
    asset: Asset
    method: Method
    horizon_months: int
    points: list[ForecastPoint]
    interpretation: str
    factors_used: list[str] = Field(default_factory=list)
    fit_diagnostics: dict = Field(default_factory=dict)


class ScenarioShock(BaseModel):
    factor: str
    delta: float
    description: str = ""

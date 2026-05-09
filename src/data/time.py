from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

Frequency = Literal["daily", "weekly", "monthly", "quarterly", "annual"]


TTL_BY_FREQUENCY: dict[Frequency, timedelta] = {
    "daily":     timedelta(hours=12),
    "weekly":    timedelta(days=3),
    "monthly":   timedelta(days=14),
    "quarterly": timedelta(days=45),
    "annual":    timedelta(days=180),
}


def is_fresh(modified_at: datetime, frequency: Frequency, *, now: datetime | None = None) -> bool:
    now = now or datetime.utcnow()
    return now - modified_at < TTL_BY_FREQUENCY[frequency]


def parse_eia_period(token: str) -> date:
    if len(token) == 8 and token.isdigit():
        return date(int(token[:4]), int(token[4:6]), int(token[6:]))
    if len(token) == 6 and token.isdigit():
        return date(int(token[:4]), int(token[4:]), 1)
    if len(token) == 4 and token.isdigit():
        return date(int(token), 1, 1)
    try:
        return date.fromisoformat(token)
    except ValueError as e:
        raise ValueError(f"Не распарсил EIA period: {token}") from e

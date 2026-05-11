from __future__ import annotations

from typing import Literal

import pandas as pd

from src.rag.schemas import DataCard

Segmentation = Literal["Q", "Y"]


def build_cards(df: pd.DataFrame, *, source: str, segmentation: Segmentation = "Q") -> list[DataCard]:
    if df.empty:
        return []
    df = df.copy()
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values("period").reset_index(drop=True)

    freq = "QE" if segmentation == "Q" else "YE"
    grouped = df.set_index("period").groupby(pd.Grouper(freq=freq))

    series_id = str(df["series_id"].iloc[0])
    metric = str(df["metric"].iloc[0])
    unit = str(df["unit"].iloc[0]) if "unit" in df.columns else ""

    cards: list[DataCard] = []
    prev_mean: float | None = None
    yoy_history: dict[str, float] = {}

    is_flow = _is_flow_metric(unit)
    for end_ts, sub in grouped:
        if sub.empty:
            continue
        seg_label = _segment_label(end_ts, segmentation)
        mean = float(sub["value"].mean())
        total = float(sub["value"].sum())
        vmin = float(sub["value"].min())
        vmax = float(sub["value"].max())
        period_start = sub.index.min().date()
        period_end = sub.index.max().date()

        delta_prev = _delta(mean, prev_mean)
        yoy_key = _yoy_key(end_ts, segmentation)
        delta_yoy = _delta(mean, yoy_history.get(yoy_key))
        prev_mean = mean
        yoy_history[_current_yoy_key(end_ts, segmentation)] = mean

        cards.append(DataCard(
            source=source,
            series_id=series_id,
            segment=seg_label,
            metric=metric,
            text=_render_card_text(metric, unit, seg_label, mean, total, vmin, vmax,
                                   delta_prev, delta_yoy, is_flow=is_flow),
            period_start=period_start,
            period_end=period_end,
        ))
    return cards


def _is_flow_metric(unit: str) -> bool:
    """Flow-метрика — то, что суммируется за период (деньги, объёмы за период).
    Stock/price метрики — то, что усредняется (цены, остатки)."""
    u = unit.lower()
    flow_markers = ["руб", "rub", "usd", "$", "млрд", "трлн", "thousand barrels", "тыс барр"]
    # Цены и индексы — не flow, даже если unit содержит USD/barrel.
    if "/barrel" in u or "/баррель" in u or "/day" in u or "index" in u:
        return False
    return any(m in u for m in flow_markers)


def _segment_label(ts: pd.Timestamp, seg: Segmentation) -> str:
    return f"{ts.year}-Q{((ts.month - 1) // 3) + 1}" if seg == "Q" else str(ts.year)


def _current_yoy_key(ts: pd.Timestamp, seg: Segmentation) -> str:
    return f"Q{((ts.month - 1) // 3) + 1}" if seg == "Q" else "Y"


def _yoy_key(ts: pd.Timestamp, seg: Segmentation) -> str:
    return _current_yoy_key(ts, seg)


def _delta(curr: float, prev: float | None) -> tuple[float, float] | None:
    if prev is None or prev == 0:
        return None
    abs_d = curr - prev
    rel_d = (abs_d / abs(prev)) * 100.0
    return abs_d, rel_d


def _fmt_delta(d: tuple[float, float] | None) -> str:
    if d is None:
        return "—"
    abs_d, rel_d = d
    sign = "+" if abs_d >= 0 else "−"
    return f"{sign}{abs(abs_d):.2f} ({sign}{abs(rel_d):.1f}%)"


def _segment_human(segment: str) -> str:
    if "Q" in segment:
        year, q = segment.split("-Q")
        return f"{q}-й квартал {year} года"
    return f"{segment} год"


def _render_card_text(metric: str, unit: str, segment: str, mean: float, total: float,
                      vmin: float, vmax: float,
                      delta_prev: tuple[float, float] | None,
                      delta_yoy: tuple[float, float] | None,
                      *, is_flow: bool) -> str:
    head = f"{metric} — {_segment_human(segment)} ({segment}). Единица: {unit}.\n"
    if is_flow:
        body = (
            f"Итог за период (сумма): {total:.2f}. "
            f"Среднемесячное: {mean:.2f}. Мин/макс месяца: {vmin:.2f} / {vmax:.2f}.\n"
        )
    else:
        body = f"Среднее: {mean:.2f}. Мин/макс: {vmin:.2f} / {vmax:.2f}.\n"
    tail = (
        f"Δ среднего к предыдущему сегменту: {_fmt_delta(delta_prev)}; "
        f"Δ среднего год-к-году: {_fmt_delta(delta_yoy)}."
    )
    return head + body + tail

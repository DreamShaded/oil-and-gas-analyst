from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

app = typer.Typer(help="Нефтегазовый аналитик — CLI", no_args_is_help=True)


@app.callback()
def _root() -> None: ...


@app.command()
def serve() -> None:
    from src.app import run
    run()


@app.command()
def bootstrap(
    data_dir: Path = typer.Option(Path("data")),  # noqa: B008
    segmentation: str = typer.Option("Q", help="Q=квартал, Y=год"),  # noqa: B008
    skip_eia: bool = typer.Option(False),  # noqa: B008
    skip_opec_momr: bool = typer.Option(False),  # noqa: B008
    opec_momr_limit: int = typer.Option(6),  # noqa: B008
    skip_opec_woo: bool = typer.Option(False),  # noqa: B008
    opec_woo_limit: int = typer.Option(3),  # noqa: B008
    skip_iea: bool = typer.Option(False),  # noqa: B008
    iea_months: int = typer.Option(6),  # noqa: B008
    skip_minfin: bool = typer.Option(False),  # noqa: B008
    offline: bool = typer.Option(False, help="Только индекс из уже выгруженного"),  # noqa: B008
) -> None:
    """Полный сбор + индекс: EIA bulk + OPEC MOMR + OPEC WOO + IEA OMR + Минфин Urals."""
    from src.data import bootstrap as bs
    if segmentation not in {"Q", "Y"}:
        raise typer.BadParameter("segmentation должен быть Q или Y")
    s = bs.run(
        data_dir, segmentation=segmentation,
        skip_eia=skip_eia,
        skip_opec_momr=skip_opec_momr, opec_momr_limit=opec_momr_limit,
        skip_opec_woo=skip_opec_woo, opec_woo_limit=opec_woo_limit,
        skip_iea=skip_iea, iea_months=iea_months,
        skip_minfin=skip_minfin, offline=offline,
    )
    typer.echo("\n" + "─" * 40)
    typer.echo(f"EIA chunks:       {s['eia']}")
    typer.echo(f"OPEC MOMR chunks: {s['opec_momr']}")
    typer.echo(f"OPEC WOO chunks:  {s['opec_woo']}")
    typer.echo(f"IEA OMR chunks:   {s['iea']}")
    typer.echo(f"Minfin chunks:    {s['minfin']}")
    typer.echo(f"Indexed total:    {s['indexed']}")


@app.command("download-opec-momr")
def download_opec_momr(
    reports_dir: Path = typer.Option(Path("data/reports")),  # noqa: B008
    limit: int = 6,
) -> None:
    from src.data.sources.opec_downloader import download_latest_momr
    files = download_latest_momr(reports_dir, limit=limit)
    typer.echo(f"Скачано: {len(files)}")
    for f in files:
        typer.echo(f"  - {f}")


@app.command("download-opec-woo")
def download_opec_woo(
    reports_dir: Path = typer.Option(Path("data/reports")),  # noqa: B008
    limit: int = 3,
) -> None:
    from src.data.sources.opec_woo import download_latest_woo
    files = download_latest_woo(reports_dir, limit=limit)
    typer.echo(f"Скачано: {len(files)}")
    for f in files:
        typer.echo(f"  - {f}")


@app.command("download-iea")
def download_iea(
    out_dir: Path = typer.Option(Path("data/iea")),  # noqa: B008
    months: int = 6,
) -> None:
    from src.data.sources.iea_omr import fetch_latest
    files = fetch_latest(out_dir, max_back=months)
    typer.echo(f"Получено: {len(files)}")
    for f in files:
        typer.echo(f"  - {f}")


@app.command("download-minfin")
def download_minfin(
    out_dir: Path = typer.Option(Path("data/minfin")),  # noqa: B008
) -> None:
    """Месячные нефтегазовые доходы РФ (XLSX из minfin.gov.ru/ru/statistics/fedbud/oil)."""
    from src.data.sources.minfin_oil import fetch_minfin_oil
    df = fetch_minfin_oil(out_dir)
    typer.echo(f"Строк: {len(df)}  метрик: {df['metric'].nunique() if not df.empty else 0}")
    if not df.empty:
        typer.echo(f"Период: {df['period'].min()} … {df['period'].max()}")


@app.command("collect-eia-api")
def collect_eia_api(
    start: str = typer.Option(None, help="ISO дата начала"),  # noqa: B008
    end: str = typer.Option(None, help="ISO дата конца"),  # noqa: B008
    out_dir: Path = typer.Option(Path("data/eia")),  # noqa: B008
    series_id: str | None = typer.Option(None, help="Если задан — выкачать только эту серию"),  # noqa: B008
) -> None:
    """Fallback: точечная выгрузка через EIA v2 API с ключом (если bulk недоступен)."""
    from src.data.sources.eia_api import fetch_all_default, fetch_series
    from src.data.sources.registry import find_series

    start_d = date.fromisoformat(start) if start else None
    end_d = date.fromisoformat(end) if end else None
    if series_id:
        meta = find_series(series_id)
        if not meta:
            raise typer.BadParameter(f"Неизвестная серия: {series_id}")
        fetch_series(meta, start=start_d, end=end_d, out_dir=out_dir)
    else:
        fetch_all_default(start=start_d, end=end_d, out_dir=out_dir)


@app.command()
def ingest(
    data_dir: Path = typer.Option(Path("data")),  # noqa: B008
    segmentation: str = typer.Option("Q", help="Q=квартал, Y=год"),  # noqa: B008
) -> None:
    """Идемпотентная переиндексация уже выгруженных CSV/PDF/JSON в Qdrant."""
    from src.rag.ingestion.pipeline import ingest_all
    if segmentation not in {"Q", "Y"}:
        raise typer.BadParameter("segmentation должен быть Q или Y")
    indexed = ingest_all(data_dir, segmentation=segmentation)
    typer.echo(f"Проиндексировано чанков: {indexed}")


@app.command()
def search(query: str, top_k: int = 5) -> None:
    from src.rag import Retriever
    hits = Retriever().search(query, top_k=top_k)
    for i, h in enumerate(hits, 1):
        typer.echo(f"\n[{i}] score={h['score']:.3f} src={h.get('source')} sid={h.get('series_id')}"
                   f" seg={h.get('segment') or '-'}\n{h['text'][:400]}")


@app.command("download-yahoo")
def download_yahoo(
    out_dir: Path = typer.Option(Path("data/yahoo")),  # noqa: B008
) -> None:
    """Скачать DXY (индекс доллара) через yfinance."""
    from src.data.sources.yahoo_finance import fetch_dxy
    df = fetch_dxy(out_dir)
    typer.echo(f"DXY: {len(df)} строк, {df['period'].min()} … {df['period'].max()}")


@app.command()
def backtest(
    asset: str = "brent",
    horizon_months: int = 3,
    method: str = "sarima",
    step_months: int = 6,
) -> None:
    """Rolling-window out-of-sample backtest. MAPE / RMSE / coverage 80% и 95% CI."""
    from src.forecasting.backtest import rolling_backtest
    if asset not in {"brent", "wti", "urals"}:
        raise typer.BadParameter("asset ∈ {brent, wti, urals}")
    if method not in {"sarima", "regression"}:
        raise typer.BadParameter("method ∈ {sarima, regression}")
    rep = rolling_backtest(asset, method, horizon_months=horizon_months,  # type: ignore[arg-type]
                           step_months=step_months)
    typer.echo(f"\n{asset.upper()} / {method} / h={horizon_months}m / windows={rep.n_windows}")
    typer.echo(f"  MAPE        = {rep.mape_pct:.2f}%")
    typer.echo(f"  RMSE        = ${rep.rmse:.2f}")
    typer.echo(f"  Coverage 80% CI = {rep.coverage_80_pct:.1f}% (target 80)")
    typer.echo(f"  Coverage 95% CI = {rep.coverage_95_pct:.1f}% (target 95)")


@app.command("backtest-conformal")
def cli_backtest_conformal(
    asset: str = "brent",
    horizon_months: int = 3,
    method: str = "sarima",
    step_months: int = 6,
) -> None:
    """Честная оценка conformal-CI: половина окон калибрует, половина — проверка покрытия."""
    from src.forecasting.conformal_eval import honest_conformal_eval
    if asset not in {"brent", "wti", "urals"}:
        raise typer.BadParameter("asset ∈ {brent, wti, urals}")
    if method not in {"sarima", "regression"}:
        raise typer.BadParameter("method ∈ {sarima, regression}")
    rep = honest_conformal_eval(asset, method,  # type: ignore[arg-type]
                                horizon_months=horizon_months, step_months=step_months)
    typer.echo(f"\n{asset.upper()} / {method} / h={horizon_months}m — conformal honest eval")
    typer.echo(f"  Калибровочные окна: {rep.n_calib_windows}, тестовые: {rep.n_test_windows}")
    typer.echo(f"  q80 per step: {[f'{q:.2f}' for q in rep.q80_per_step]}")
    typer.echo(f"  q95 per step: {[f'{q:.2f}' for q in rep.q95_per_step]}")
    typer.echo(f"  Coverage 80% CI: {rep.coverage_80_pct:.1f}% (target 80)")
    typer.echo(f"  Coverage 95% CI: {rep.coverage_95_pct:.1f}% (target 95)")
    typer.echo(f"  MAPE: {rep.mape_pct:.2f}%   RMSE: ${rep.rmse:.2f}")


@app.command()
def forecast(
    asset: str = "brent",
    horizon_months: int = 3,
    method: str = "sarima",
    shock_factor: str | None = typer.Option(None, help="Имя фактора для шока (regression only)"),  # noqa: B008
    shock_delta: float = typer.Option(0.0, help="Размер шока, в исходных единицах фактора"),  # noqa: B008
) -> None:
    """Прогноз цены нефти: SARIMA (univariate) или regression (по фундаменталам)."""
    from src.forecasting import ScenarioShock, forecast_price
    if asset not in {"brent", "wti", "urals"}:
        raise typer.BadParameter("asset ∈ {brent, wti, urals}")
    if method not in {"sarima", "regression"}:
        raise typer.BadParameter("method ∈ {sarima, regression}")
    shocks = None
    if shock_factor:
        shocks = [ScenarioShock(factor=shock_factor, delta=shock_delta,
                                description=f"{shock_factor}{shock_delta:+}")]
    fc = forecast_price(asset, horizon_months=horizon_months,  # type: ignore[arg-type]
                        method=method, shocks=shocks)  # type: ignore[arg-type]
    typer.echo("\n" + fc.interpretation + "\n")
    typer.echo("Точки прогноза:")
    for p in fc.points:
        typer.echo(f"  {p.period}  ${p.point:7.2f}   80%[{p.lower_80:.2f}..{p.upper_80:.2f}]"
                   f"   95%[{p.lower_95:.2f}..{p.upper_95:.2f}]")


@app.command("web-search")
def cli_web_search(
    query: str,
    max_results: int = 5,
    topic: str = "news",
) -> None:
    """Поиск через Tavily с фильтром по whitelist/blacklist (config/source-*.yaml)."""
    from src.tools.web_search import search_web
    if topic not in {"news", "general"}:
        raise typer.BadParameter("topic должен быть news или general")
    res = search_web(query, max_results=max_results, topic=topic)  # type: ignore[arg-type]
    typer.echo(f"\nЗапрос: {res.query}  ({len(res.snippets)} результатов)\n")
    for i, s in enumerate(res.snippets, 1):
        marker = "★" if s.trust > 0 else " "
        typer.echo(f"{marker} [{i}] {s.source_domain}  {s.published_at or ''}")
        typer.echo(f"    {s.title}")
        typer.echo(f"    {s.url}\n")


@app.command("eval-retrieval")
def eval_retrieval(
    golden: Path = typer.Option(Path("data/golden/retrieval-golden.json")),  # noqa: B008
    top_k: int = 8,
) -> None:
    from src.rag.eval import evaluate, load_golden
    cases = load_golden(golden)
    report = evaluate(cases, top_k=top_k)
    typer.echo(f"\nN={report.n}  hit@{top_k}={report.hit_at_k:.2%}  MRR={report.mrr:.3f}\n")
    for r in report.per_case:
        mark = "✓" if r.hit_at_k else "✗"
        at = f"@{r.matched_at}" if r.matched_at else "—"
        typer.echo(f"  {mark} {at:>3}  {r.query}")


if __name__ == "__main__":
    app()

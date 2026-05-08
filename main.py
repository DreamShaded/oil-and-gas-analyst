"""CLI-точка входа"""
from __future__ import annotations

import typer

app = typer.Typer(help="Нефтегазовый аналитик — CLI", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Корневой колбэк нужен, чтобы Typer не схлопывал единственную команду"""


@app.command()
def serve() -> None:
    """Запустить Gradio-чат."""
    from src.app import run

    run()


if __name__ == "__main__":
    app()

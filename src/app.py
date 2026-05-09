from __future__ import annotations

from src.config import get_settings
from src.ui.chat import build_chat
from src.utils.logging import configure_logging, get_logger


def run() -> None:
    settings = get_settings()
    configure_logging(env=settings.app_env, level="INFO")
    log = get_logger("app")
    log.info("app.start", host=settings.app_host, port=settings.app_port, env=settings.app_env)
    build_chat().launch(server_name=settings.app_host, server_port=settings.app_port)


if __name__ == "__main__":
    run()

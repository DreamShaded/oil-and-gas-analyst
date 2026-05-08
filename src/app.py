from __future__ import annotations

import gradio as gr

from src.config import get_settings
from src.llm.client import build_chat_model
from src.utils.logging import configure_logging, get_logger


def _build_ui() -> gr.Blocks:
    settings = get_settings()
    log = get_logger("app")

    try:
        build_chat_model(settings=settings)
        config_ok = True
        config_note = f"✅ Конфигурация загружена. Модель: `{settings.llm_model}`"
    except Exception as exc:  # noqa: BLE001 — диагностический показ в UI
        config_ok = False
        config_note = f"❌ Ошибка конфигурации: `{exc}`"
        log.error("config.invalid", error=str(exc))

    def respond(message: str, history: list[dict[str, str]]) -> str:
        if not config_ok:
            return "Конфигурация не загружена — проверьте `.env`."
        return f"[заглушка этапа 1] Получил сообщение: {message!r}"

    with gr.Blocks(title="Нефтегазовый аналитик") as ui:
        gr.Markdown("# Нефтегазовый аналитик\n_Этап 1 — инфраструктура. Реальные ответы появятся на этапе 2._")
        gr.Markdown(config_note)
        gr.ChatInterface(fn=respond, type="messages", autofocus=True)
    return ui


def run() -> None:
    settings = get_settings()
    configure_logging(env=settings.app_env, level="INFO")
    log = get_logger("app")
    log.info("app.start", host=settings.app_host, port=settings.app_port, env=settings.app_env)
    _build_ui().launch(server_name=settings.app_host, server_port=settings.app_port, show_api=False)


if __name__ == "__main__":
    run()

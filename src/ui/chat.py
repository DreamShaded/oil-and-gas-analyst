from __future__ import annotations

from collections.abc import AsyncGenerator

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agent.graph import get_graph
from src.utils.logging import get_logger

_log = get_logger("ui.chat")


def _to_lc_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


async def _respond(message: str, history: list[dict[str, str]]) -> AsyncGenerator[str, None]:
    graph = get_graph()
    state = {"messages": [*_to_lc_messages(history), HumanMessage(content=message)]}

    buffer = ""
    try:
        async for event in graph.astream_events(state, version="v2"):
            if event.get("event") == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                token = getattr(chunk, "content", "") or ""
                if token:
                    buffer += token
                    yield buffer
    except Exception as exc:  # noqa: BLE001 — показываем причину пользователю
        _log.error("chat.stream_error", error=str(exc))
        yield (buffer + "\n\n_⚠ Ошибка обращения к LLM: " + str(exc) + "_") if buffer else (
            "⚠ Не удалось получить ответ от LLM. Проверьте `.env` и доступность провайдера.\n\n"
            f"Детали: `{exc}`"
        )


def build_chat() -> gr.Blocks:
    with gr.Blocks(title="Нефтегазовый аналитик") as ui:
        gr.Markdown(
            "# Нефтегазовый аналитик\n"
        )
        chatbot = gr.Chatbot(
            height="72vh",
            min_height=650,
            resizable=True,
            autoscroll=True,
        )
        gr.ChatInterface(
            fn=_respond,
            chatbot=chatbot,
            autofocus=True,
            examples=[
                "Объясни, как формируется спред Brent–Urals.",
                "Какие ключевые факторы влияют на цену нефти в краткосрочной перспективе?",
            ],
        )
    return ui

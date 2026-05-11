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


# В чат стримим только токены пользовательских узлов. Служебные узлы
# (classify_intent → JSON, reflect → критика, self_mod_observer → tool calls)
# не должны попадать в UI.
_USER_FACING_NODES = {"compose_answer", "refuse"}


def _normalize_content(content) -> str:
    """LLM-провайдеры возвращают content по-разному:
       OpenAI → str, Anthropic → list[dict] с блоками {'type': 'text', 'text': '...'}."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "text_delta"):
                # text / text_delta — варианты Anthropic stream
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")


async def _respond(message: str, history: list[dict[str, str]]) -> AsyncGenerator[str, None]:
    graph = get_graph()
    state = {"messages": [*_to_lc_messages(history), HumanMessage(content=message)]}

    buffer = ""
    compose_attempt = 0
    try:
        async for event in graph.astream_events(state, version="v2"):
            ev = event.get("event")
            node = (event.get("metadata") or {}).get("langgraph_node")
            # Новый запуск compose (reflect-retry) — сообщаем пользователю что
            # переписываем, и обнуляем буфер. Без этого UI «склеит» два разных
            # ответа подряд.
            if ev == "on_chain_start" and event.get("name") == "compose_answer":
                compose_attempt += 1
                if compose_attempt > 1:
                    buffer = "_(переписываю ответ — обнаружен недочёт)_\n\n"
                    yield buffer
                else:
                    buffer = ""
                continue
            if ev != "on_chat_model_stream":
                continue
            if node not in _USER_FACING_NODES:
                continue
            chunk = event["data"].get("chunk")
            token = _normalize_content(getattr(chunk, "content", None))
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
    """Чат-страница (монтируется на корень `/`). Самомодификация — отдельный URL `/self-mod`."""
    with gr.Blocks(title="Нефтегазовый аналитик") as ui:
        gr.Markdown(
            "# Нефтегазовый аналитик\n\n"
            "[🧠 Самомодификация →](/self-mod/)"
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
                "Спрогнозируй цену Brent на 3 месяца",
            ],
        )
    return ui

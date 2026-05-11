from __future__ import annotations

import html

import gradio as gr

from src.self_mod import (
    TIER1_ALLOWLIST,
    apply_or_queue,
    approve_proposal,
    get_current_value,
    get_override,
    list_audit,
    list_proposals,
    reject_proposal,
)

PROMPT_TARGET = "prompt.addendum"

_DISABLED_CSS = """<style>
button[disabled], button[disabled]:hover {
  opacity: 0.45 !important;
  cursor: not-allowed !important;
}
</style>"""


def _h(text) -> str:
    return html.escape(str(text) if text is not None else "")


def _prompt_diff(before: str | None, after: str | None) -> str:
    b = (before or "").strip() or "_(пусто)_"
    a = (after or "").strip() or "_(пусто)_"
    return (
        f"**До (текущий):**\n\n```\n{b}\n```\n\n"
        f"**После (предлагается):**\n\n```\n{a}\n```"
    )


def _scalar_diff(target: str, before, after) -> str:
    same = before == after
    arrow = "═══" if same else "→"
    note = " _(без изменений)_" if same else ""
    return (
        f"#### `{target}`{note}\n\n"
        f"| До | {arrow} | После |\n"
        f"|---|---|---|\n"
        f"| `{_h(before)}` | {arrow} | **`{_h(after)}`** |"
    )


def _human_label(target: str) -> str:
    if target == PROMPT_TARGET:
        return "🧠 Поправка поведения (addendum к системному промпту)"
    if target.startswith("rag."):
        return f"⚙️ Параметр поиска: `{target}`"
    if target.startswith("forecast."):
        return f"📈 Параметр прогноза: `{target}`"
    return f"🔧 `{target}`"


def _render_audit() -> str:
    items = list_audit(limit=20)
    if not items:
        return "_Журнал автоприменения пуст._"
    lines = ["| Когда | Параметр | До | После | Зачем |",
             "|---|---|---|---|---|"]
    for it in items:
        target = it.get("target", "")
        before = it.get("before")
        after = it.get("after")
        if target == PROMPT_TARGET:
            before = (str(before)[:60] + "…") if before and len(str(before)) > 60 else (str(before) if before else "—")
            after = (str(after)[:60] + "…") if after and len(str(after)) > 60 else (str(after) if after else "—")
        lines.append(
            f"| {_h(it.get('ts', '')[:19])} | {_h(target)} "
            f"| {_h(before)} | {_h(after)} "
            f"| {_h((it.get('rationale') or '')[:80])} |"
        )
    return "\n".join(lines)


def _render_history() -> str:
    items = list_proposals()
    decided = [p for p in items if p.status != "pending"]
    if not decided:
        return "_Решений пока нет._"
    lines = ["| Решение | id | Параметр | Значение | Зачем |",
             "|---|---|---|---|---|"]
    for p in reversed(decided[-30:]):
        val_str = str(p.value)
        if len(val_str) > 60:
            val_str = val_str[:60] + "…"
        lines.append(
            f"| {p.status} | `{p.id}` | `{_h(p.target)}` "
            f"| `{_h(val_str)}` | {_h(p.rationale[:80])} |"
        )
    return "\n".join(lines)


def _current_addendum_md() -> str:
    cur = get_override(PROMPT_TARGET) or ""
    return f"```\n{cur or '(пусто — агент работает на базовом промпте)'}\n```"


def _refresh_all():
    return (
        list_proposals(status="pending"),
        _render_audit(),
        _render_history(),
        _current_addendum_md(),
    )


def _on_decide(decision: str, pid: str):
    fn = approve_proposal if decision == "approve" else reject_proposal
    fn(pid)
    return _refresh_all()


def _on_manual_addendum(text: str):
    text = (text or "").strip()
    if not text:
        return ("⚠ Поле пустое", *_refresh_all())
    p = apply_or_queue(PROMPT_TARGET, text, "ручная правка из UI")
    msg = (f"✅ Tier-1 применено (id `{p.id}`)" if p.tier.name == "AUTO"
           else f"📝 Tier-2 на ревью (id `{p.id}`) — {p.rationale}")
    return (msg, *_refresh_all())


def build_self_mod_panel() -> gr.Blocks:
    initial_pending = list_proposals(status="pending")

    with gr.Blocks(title="Самомодификация · Нефтегазовый аналитик") as panel:
        gr.HTML(_DISABLED_CSS)
        gr.Markdown(
            "# Самомодификация\n\n"
            "Агент сам предлагает себе изменения, чтобы **лучше служить пользователю**. "
            "Простые поправки применяются автоматически (Tier-1), существенные ждут "
            "вашего одобрения здесь.\n\n"
            "Главный канал — **поправка поведения** (`prompt.addendum`): короткий текст, "
            "добавляющийся к системному промпту."
        )

        # Объявляем все целевые компоненты, чтобы события могли на них ссылаться.
        pending_state = gr.State(initial_pending)
        gr.Markdown("## Ожидают одобрения")
        cards_placeholder = gr.Markdown(visible=False)  # для @gr.render

        with gr.Accordion("Текущая активная поправка поведения", open=True):
            current_md = gr.Markdown(_current_addendum_md())
            manual_in = gr.Textbox(
                label="Перезаписать вручную (≤500 символов — Tier-1)",
                placeholder="например: «В прогнозах всегда показывай оба метода — SARIMA и регрессию.»",
                lines=4, max_lines=10,
            )
            manual_btn = gr.Button("Применить вручную", variant="secondary",
                                   interactive=False)
            manual_status = gr.Markdown("")

        gr.Markdown("## Журнал автоприменения (Tier-1)")
        audit_md = gr.Markdown(_render_audit())

        gr.Markdown("## История решений (Tier-2)")
        history_md = gr.Markdown(_render_history())

        refresh_btn = gr.Button("🔄 Обновить страницу", size="sm")

        with gr.Accordion("Что агент умеет менять автоматически (Tier-1 allowlist)", open=False):
            wl_lines = []
            for k, t in TIER1_ALLOWLIST.items():
                label = _human_label(k)
                bounds = f"{t[0].__name__} в [{t[1]}, {t[2]}]"
                if t[0] is str:
                    bounds = f"строка длиной [{t[1]}, {t[2]}] символов"
                wl_lines.append(f"- {label} — {bounds}")
            gr.Markdown("\n".join(wl_lines))

        # Карточки рендерятся динамически по pending_state с собственными кнопками.
        @gr.render(inputs=pending_state)
        def render_cards(pending):
            cards_placeholder.update()
            if not pending:
                gr.Markdown("_Нет ожидающих предложений._")
                return
            for p in pending:
                with gr.Group():
                    current = get_current_value(p.target)
                    gr.Markdown(f"### {_human_label(p.target)} · id `{p.id}`")
                    if p.target == PROMPT_TARGET:
                        gr.Markdown(_prompt_diff(current, p.value))
                    else:
                        gr.Markdown(_scalar_diff(p.target, current, p.value))
                    gr.Markdown(f"**Зачем:** {p.rationale}\n\n**Создано:** `{p.created_at}`")
                    with gr.Row():
                        a_btn = gr.Button("Одобрить ✅", variant="primary", size="sm")
                        r_btn = gr.Button("Отклонить ❌", variant="stop", size="sm")
                    a_btn.click(
                        lambda pid=p.id: _on_decide("approve", pid),
                        outputs=[pending_state, audit_md, history_md, current_md],
                    )
                    r_btn.click(
                        lambda pid=p.id: _on_decide("reject", pid),
                        outputs=[pending_state, audit_md, history_md, current_md],
                    )

        # disable manual button если текстарея пуста
        manual_in.change(
            lambda t: gr.update(interactive=bool((t or "").strip())),
            inputs=manual_in, outputs=manual_btn,
        )
        manual_btn.click(
            _on_manual_addendum, inputs=manual_in,
            outputs=[manual_status, pending_state, audit_md, history_md, current_md],
        )
        refresh_btn.click(
            _refresh_all,
            outputs=[pending_state, audit_md, history_md, current_md],
        )

    return panel

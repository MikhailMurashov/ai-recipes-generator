import hashlib
import logging

import streamlit as st
from agent import Agent
from config import MODELS
from storage import (
    delete_context,
    list_contexts,
    new_session_id,
    save_context,
    save_working_memory,
)
from strategies import SlidingWindowSummaryStrategy, StrategyType

logging.basicConfig(level=logging.INFO)


def init_session_state():
    if "session_id" not in st.session_state:
        contexts = list_contexts()
        if contexts:
            ctx = contexts[0]
            st.session_state.session_id = ctx["session_id"]
            st.session_state.system_prompt = ctx.get("system_prompt", "")
            st.session_state.selected_model_key = ctx.get("model_key")

            strategy_type = ctx.get("strategy_type", "sliding_window")
            strategy_state = ctx.get("strategy_state") or {
                "summary": ctx.get("summary", ""),
                "summarized_count": ctx.get("summarized_count", 0),
                "history": ctx.get("history", []),
            }
            agent = Agent(
                system_prompt=st.session_state.system_prompt,
                strategy_type=StrategyType(strategy_type),
                strategy_state=strategy_state,
                working_memory_state=ctx.get("working_memory", {}),
            )
            st.session_state.agent = agent
            st.session_state.message_stats = ctx.get("message_stats", [])
            for s in st.session_state.message_stats:
                agent.total_prompt_tokens += s.get("prompt_tokens") or 0
                agent.total_completion_tokens += s.get("completion_tokens") or 0
                agent.total_tokens += s.get("total_tokens") or 0
        else:
            st.session_state.session_id = new_session_id()
            st.session_state.system_prompt = ""
            st.session_state.agent = Agent()
            st.session_state.message_stats = []
    if "agent" not in st.session_state:
        st.session_state.agent = Agent(
            system_prompt=st.session_state.get("system_prompt", "")
        )
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "message_stats" not in st.session_state:
        st.session_state.message_stats = []


def render_branch_panel(agent: Agent):
    """Render branch management controls for BranchingStrategy."""
    st.subheader("Ветки")

    bid = agent.active_branch_id
    branches = agent.branches

    # Trunk button
    trunk_label = "Ствол" + (" (активен)" if bid is None else "")
    if st.button(trunk_label, key="branch_trunk", use_container_width=True):
        agent.switch_branch(None)
        st.rerun()

    # Branch buttons
    for branch_id, branch in branches.items():
        is_active = branch_id == bid
        col1, col2 = st.columns([4, 1])
        with col1:
            label = branch.name + (" (активна)" if is_active else "")
            if st.button(
                label, key=f"branch_switch_{branch_id}", use_container_width=True
            ):
                agent.switch_branch(branch_id)
                st.rerun()
        with col2:
            if st.button("x", key=f"branch_del_{branch_id}"):
                agent.delete_branch(branch_id)
                st.rerun()

    # Create new branch
    st.divider()
    new_name = st.text_input("Название новой ветки", key="new_branch_name")
    if st.button("Создать ветку", use_container_width=True):
        if new_name.strip():
            agent.create_branch(new_name.strip())
            st.rerun()
        else:
            st.warning("Введите название ветки")

    # Show active label
    if bid and bid in branches:
        st.caption(f"Активная ветка: {branches[bid].name}")
    else:
        st.caption("Активная ветка: Ствол")


def render_sidebar() -> tuple[dict, str]:
    with st.sidebar:
        st.title("Параметры модели")

        agent = st.session_state.agent
        n = len(agent.history)

        # ------------------------------------------------------------------
        # Model selector
        # ------------------------------------------------------------------
        model_keys = list(MODELS.keys())
        saved_key = st.session_state.get("selected_model_key")
        model_index = model_keys.index(saved_key) if saved_key in model_keys else 0
        selected_model_key = st.selectbox("Модель", model_keys, index=model_index)
        st.session_state.selected_model_key = selected_model_key
        selected_model = MODELS[selected_model_key]

        st.divider()
        st.subheader("Системный промпт")
        st.text_area(
            label="system_prompt",
            key="system_prompt",
            height=120,
            label_visibility="collapsed",
        )

        # ------------------------------------------------------------------
        # Memory panel
        # ------------------------------------------------------------------
        st.divider()
        wm_entries = agent.working_memory.entries
        ltm_entries = agent.long_term_memory.entries
        mem_label = f"Память · 📋{len(wm_entries)} · 💾{len(ltm_entries)}"
        mem_open = (
            st.session_state.pop("mem_panel_open", False)
            if "mem_panel_open" in st.session_state
            else False
        )
        with st.expander(mem_label, expanded=mem_open):
            st.caption("Сохраняйте сообщения кнопками 📋/💾 в чате.")

            # Краткосрочная
            st.markdown("**Краткосрочная**")
            window_info = ""
            if agent.strategy_type == StrategyType.SLIDING_WINDOW_SUMMARY:
                window_info = f"окно {SlidingWindowSummaryStrategy.CONTEXT_WINDOW}"
                if agent.summarized_count:
                    window_info += f", сжато {agent.summarized_count}"
            elif hasattr(agent._strategy, "_window_size"):
                window_info = f"окно {agent._strategy._window_size}"
            st.caption(
                f"{n} сообщ."
                + (f" · {window_info}" if window_info else "")
                + " · управляется автоматически"
            )

            st.divider()

            # Рабочая
            st.markdown("**Рабочая** (текущая задача)")
            if wm_entries:
                for entry in wm_entries.values():
                    st.caption(
                        entry.value[:120] + ("…" if len(entry.value) > 120 else "")
                    )
            else:
                st.caption("Пусто — кнопка 📋 у сообщений.")

            st.divider()

            # Долговременная
            st.markdown("**Долговременная** (профиль, знания)")
            if ltm_entries:
                for entry in ltm_entries.values():
                    st.caption(
                        entry.value[:120] + ("…" if len(entry.value) > 120 else "")
                    )
                if st.button(
                    "Очистить долговременную",
                    type="secondary",
                    use_container_width=True,
                ):
                    agent.long_term_memory.clear()
                    st.rerun()
            else:
                st.caption("Пусто — кнопка 💾 у сообщений.")

        st.divider()
        params = {}
        with st.expander("Параметры", expanded=False):
            use_temp = st.checkbox("Температура", value=False)
            if use_temp:
                params["temperature"] = st.slider(
                    "temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05
                )
            else:
                params["temperature"] = None

            use_top_p = st.checkbox("Top P", value=False)
            if use_top_p:
                params["top_p"] = st.slider(
                    "top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.05
                )
            else:
                params["top_p"] = None

            use_max_tokens = st.checkbox("Макс. токенов", value=False)
            if use_max_tokens:
                params["max_tokens"] = st.number_input(
                    "max_tokens", min_value=1, max_value=8192, value=1000, step=100
                )
            else:
                params["max_tokens"] = None

            use_seed = st.checkbox("Сид", value=False)
            if use_seed:
                params["seed"] = st.number_input(
                    "seed", min_value=0, max_value=2**31 - 1, value=42, step=1
                )
            else:
                params["seed"] = None

            use_presence = st.checkbox("Штраф темы (presence)", value=False)
            if use_presence:
                params["presence_penalty"] = st.slider(
                    "presence_penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1,
                )
            else:
                params["presence_penalty"] = None

            use_frequency = st.checkbox("Штраф слов (frequency)", value=False)
            if use_frequency:
                params["frequency_penalty"] = st.slider(
                    "frequency_penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1,
                )
            else:
                params["frequency_penalty"] = None

        st.divider()

        # ------------------------------------------------------------------
        # Strategy picker
        # ------------------------------------------------------------------
        strategy_labels = {
            StrategyType.SLIDING_WINDOW: "Скользящее окно",
            StrategyType.SLIDING_WINDOW_SUMMARY: "Скользящее окно + Саммари",
            StrategyType.STICKY_FACTS: "Sticky Facts",
            StrategyType.BRANCHING: "Ветвление",
        }
        with st.expander("Стратегия контекста", expanded=False):
            options = list(strategy_labels.keys())
            current_index = options.index(agent.strategy_type)
            selected = st.radio(
                "Стратегия",
                options=options,
                index=current_index,
                format_func=lambda t: strategy_labels[t],
                label_visibility="collapsed",
            )
            if selected != agent.strategy_type:
                agent.switch_strategy(selected)
                st.session_state.message_stats = []
                st.rerun()

            # Strategy-specific controls
            if agent.strategy_type == StrategyType.SLIDING_WINDOW_SUMMARY:
                st.checkbox("Использовать саммари", value=True, key="use_summary")
                if agent.summary:
                    with st.expander("Саммари контекста", expanded=False):
                        st.write(agent.summary)

            elif agent.strategy_type == StrategyType.SLIDING_WINDOW:
                win = st.number_input(
                    "Размер окна",
                    min_value=1,
                    max_value=999,
                    value=agent._strategy._window_size,
                    step=1,
                    key="sw_window_size",
                )
                agent._strategy._window_size = win

            elif agent.strategy_type == StrategyType.STICKY_FACTS:
                win = st.number_input(
                    "Размер окна",
                    min_value=1,
                    max_value=999,
                    value=agent._strategy._window_size,
                    step=1,
                    key="sf_window_size",
                )
                agent._strategy._window_size = win

            elif agent.strategy_type == StrategyType.BRANCHING:
                render_branch_panel(agent)

        st.divider()

        if n > 0:
            contexts = list_contexts()
            ctx_map = {c["session_id"]: c for c in contexts}
            ctx = ctx_map.get(st.session_state.session_id)
            if ctx:
                created = ctx.get("created_at", "")[:16].replace("T", " ")
                st.caption(f"Контекст от {created}")
            st.caption(f"Сообщений: {n}")
            if agent.total_tokens > 0:
                st.caption(
                    f"Токены (всего): {agent.total_prompt_tokens} in · "
                    f"{agent.total_completion_tokens} out · Σ {agent.total_tokens}"
                )
        else:
            st.caption("Новый чат")

        if st.button("Сбросить контекст", type="secondary", use_container_width=True):
            delete_context(st.session_state.session_id)
            st.session_state.session_id = new_session_id()
            new_agent = Agent(system_prompt=st.session_state.system_prompt)
            new_agent.long_term_memory = agent.long_term_memory
            st.session_state.agent = new_agent
            st.session_state.message_stats = []
            st.rerun()

    return params, selected_model


def _stats_caption(
    elapsed_s: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> str:
    parts = [f"{elapsed_s:.2f} с"]
    if prompt_tokens is not None:
        parts += [
            f"{prompt_tokens} in",
            f"{completion_tokens} out",
            f"Σ {total_tokens}",
        ]
    return " · ".join(parts)


def _memory_key(content: str) -> str:
    """Auto-generate a memory key from message content."""
    cleaned = content.strip().replace("\n", " ")
    return cleaned[:40].rstrip()


def _render_msg(
    msg: dict,
    stats_list: list,
    assistant_idx: int,
    dimmed: bool = False,
    agent: Agent | None = None,
) -> int:
    """Render a single chat message. Returns updated assistant_idx."""
    content = msg["content"]
    key_suffix = hashlib.md5(content.encode()).hexdigest()[:8]

    with st.chat_message(msg["role"]):
        if dimmed:
            st.markdown(
                f'<div style="opacity:0.4">{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)
        if msg["role"] == "assistant" and assistant_idx < len(stats_list):
            s = stats_list[assistant_idx]
            delta_in = s.get("delta_prompt_tokens", s["prompt_tokens"])
            comp = s["completion_tokens"]
            delta_total = (
                (delta_in or 0) + (comp or 0) if delta_in is not None else None
            )
            st.caption(_stats_caption(s["elapsed_s"], delta_in, comp, delta_total))
            assistant_idx += 1

        if agent is not None and not dimmed:
            btn_col, _ = st.columns([1, 6])
            with btn_col:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "📋",
                        key=f"wm_{key_suffix}",
                        help="Сохранить в рабочую память",
                        use_container_width=True,
                    ):
                        agent.working_memory.set(_memory_key(content), content)
                        save_working_memory(
                            st.session_state.session_id,
                            agent.working_memory.to_state(),
                        )
                        st.toast("Сохранено в рабочую память")
                        st.session_state["mem_panel_open"] = True
                        st.rerun()
                with c2:
                    if st.button(
                        "💾",
                        key=f"ltm_{key_suffix}",
                        help="Сохранить в долговременную память",
                        use_container_width=True,
                    ):
                        agent.long_term_memory.set(_memory_key(content), content)
                        st.toast("Сохранено в долговременную память")
                        st.session_state["mem_panel_open"] = True
                        st.rerun()

    return assistant_idx


def render_chat_history():
    agent = st.session_state.agent
    history = agent.history
    stats_list = st.session_state.message_stats
    assistant_idx = 0

    if agent.strategy_type == StrategyType.SLIDING_WINDOW_SUMMARY:
        summarized_count = agent.summarized_count
        for msg in history[:summarized_count]:
            assistant_idx = _render_msg(msg, stats_list, assistant_idx, dimmed=True)
        if agent.summary:
            st.info(f"**Краткое содержание:**\n\n{agent.summary}")
        for msg in history[summarized_count:]:
            assistant_idx = _render_msg(msg, stats_list, assistant_idx, agent=agent)

    elif agent.strategy_type == StrategyType.SLIDING_WINDOW:
        window_size = agent._strategy._window_size
        cutoff = max(0, len(history) - window_size)
        for msg in history[:cutoff]:
            assistant_idx = _render_msg(msg, stats_list, assistant_idx, dimmed=True)
        for msg in history[cutoff:]:
            assistant_idx = _render_msg(msg, stats_list, assistant_idx, agent=agent)

    elif agent.strategy_type == StrategyType.STICKY_FACTS:
        window_size = agent._strategy._window_size
        cutoff = max(0, len(history) - window_size)
        for msg in history[:cutoff]:
            assistant_idx = _render_msg(msg, stats_list, assistant_idx, dimmed=True)
        facts = agent.facts
        if facts:
            st.info(f"**Известные факты:**\n\n{facts}")
        for msg in history[cutoff:]:
            assistant_idx = _render_msg(msg, stats_list, assistant_idx, agent=agent)

    elif agent.strategy_type == StrategyType.BRANCHING:
        bid = agent.active_branch_id
        branches = agent.branches
        if bid and bid in branches:
            branch = branches[bid]
            checkpoint = branch.checkpoint_index
            for msg in history[:checkpoint]:
                assistant_idx = _render_msg(msg, stats_list, assistant_idx, dimmed=True)
            st.divider()
            st.caption(f"↳ Ветка: {branch.name}")
            for msg in history[checkpoint:]:
                assistant_idx = _render_msg(msg, stats_list, assistant_idx, agent=agent)
        else:
            st.caption("Ветка: Ствол")
            for msg in history:
                assistant_idx = _render_msg(msg, stats_list, assistant_idx, agent=agent)

    else:
        for msg in history:
            assistant_idx = _render_msg(msg, stats_list, assistant_idx, agent=agent)


def handle_input(params: dict, model: str):
    user_input = st.chat_input("Введите сообщение...")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    agent = st.session_state.agent
    agent.system_prompt = st.session_state.system_prompt

    # Route summarization toggle through proxy setter
    if agent.strategy_type == StrategyType.SLIDING_WINDOW_SUMMARY:
        agent.summarization_enabled = st.session_state.get("use_summary", True)

    active_params = {k: v for k, v in params.items() if v is not None}
    logging.info(f"[agent] model={model}, params={active_params}")

    prev_prompt = (
        st.session_state.message_stats[-1]["prompt_tokens"]
        if st.session_state.message_stats
        else 0
    )

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            result = agent.run(user_input, model=model, **active_params)
        st.markdown(result.content)
        delta_prompt = max(0, (result.prompt_tokens or 0) - (prev_prompt or 0))
        delta_total = delta_prompt + (result.completion_tokens or 0)
        stats = {
            "elapsed_s": result.elapsed_s,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "delta_prompt_tokens": delta_prompt,
        }
        st.session_state.message_stats.append(stats)
        st.caption(
            _stats_caption(
                result.elapsed_s, delta_prompt, result.completion_tokens, delta_total
            )
        )
        save_context(
            st.session_state.session_id,
            st.session_state.system_prompt,
            agent.history,
            st.session_state.message_stats,
            model_key=st.session_state.get("selected_model_key"),
            summary=agent.summary,
            summarized_count=getattr(agent._strategy, "_summarized_count", 0),
            strategy_type=agent.strategy_type.value,
            strategy_state=agent.get_strategy_state(),
            working_memory=agent.working_memory.to_state(),
        )
    st.rerun()


def main():
    st.set_page_config(
        page_title="Чат с ассистентом",
        page_icon="🍳",
        layout="wide",
    )
    st.title("Чат с ассистентом")

    init_session_state()
    params, model = render_sidebar()
    render_chat_history()
    handle_input(params, model)


if __name__ == "__main__":
    main()

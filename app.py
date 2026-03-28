import streamlit as st
from agent import Agent
from config import MODELS
from storage import delete_context, list_contexts, new_session_id, save_context


def init_session_state():
    if "session_id" not in st.session_state:
        contexts = list_contexts()
        if contexts:
            ctx = contexts[0]
            st.session_state.session_id = ctx["session_id"]
            st.session_state.system_prompt = ctx.get("system_prompt", "")
            st.session_state.selected_model_key = ctx.get("model_key")
            st.session_state.agent = Agent(system_prompt=st.session_state.system_prompt)
            st.session_state.agent._history = ctx.get("history", [])
            st.session_state.message_stats = ctx.get("message_stats", [])
            for s in st.session_state.message_stats:
                st.session_state.agent.total_prompt_tokens += s.get("prompt_tokens") or 0
                st.session_state.agent.total_completion_tokens += s.get("completion_tokens") or 0
                st.session_state.agent.total_tokens += s.get("total_tokens") or 0
        else:
            st.session_state.session_id = new_session_id()
            st.session_state.system_prompt = ""
            st.session_state.agent = Agent()
            st.session_state.message_stats = []
    if "agent" not in st.session_state:
        st.session_state.agent = Agent(system_prompt=st.session_state.get("system_prompt", ""))
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "message_stats" not in st.session_state:
        st.session_state.message_stats = []


def render_sidebar() -> tuple[dict, str]:
    with st.sidebar:
        st.title("Параметры модели")
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

        st.divider()
        st.subheader("Параметры")

        params = {}

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
                "presence_penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1
            )
        else:
            params["presence_penalty"] = None

        use_frequency = st.checkbox("Штраф слов (frequency)", value=False)
        if use_frequency:
            params["frequency_penalty"] = st.slider(
                "frequency_penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1
            )
        else:
            params["frequency_penalty"] = None

        st.divider()

        n = len(st.session_state.agent.history)
        if n > 0:
            contexts = list_contexts()
            ctx_map = {c["session_id"]: c for c in contexts}
            ctx = ctx_map.get(st.session_state.session_id)
            if ctx:
                created = ctx.get("created_at", "")[:16].replace("T", " ")
                st.caption(f"Контекст от {created}")
            st.caption(f"Сообщений: {n}")
            agent = st.session_state.agent
            if agent.total_tokens > 0:
                st.caption(
                    f"Токены (всего): 📥 {agent.total_prompt_tokens} in · "
                    f"📤 {agent.total_completion_tokens} out · Σ {agent.total_tokens}"
                )
        else:
            st.caption("Новый чат")

        if st.button("Сбросить контекст", type="secondary", use_container_width=True):
            delete_context(st.session_state.session_id)
            st.session_state.session_id = new_session_id()
            st.session_state.agent = Agent(system_prompt=st.session_state.system_prompt)
            st.session_state.message_stats = []
            st.rerun()

    return params, selected_model


def _stats_caption(
    elapsed_s: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> str:
    parts = [f"⏱ {elapsed_s:.2f} с"]
    if prompt_tokens is not None:
        parts += [
            f"📥 {prompt_tokens} in",
            f"📤 {completion_tokens} out",
            f"Σ {total_tokens}",
        ]
    return " · ".join(parts)


def render_chat_history():
    stats_list = st.session_state.message_stats
    assistant_idx = 0
    for msg in st.session_state.agent.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and assistant_idx < len(stats_list):
                s = stats_list[assistant_idx]
                delta_in = s.get("delta_prompt_tokens", s["prompt_tokens"])
                comp = s["completion_tokens"]
                delta_total = (delta_in or 0) + (comp or 0) if delta_in is not None else None
                st.caption(_stats_caption(s["elapsed_s"], delta_in, comp, delta_total))
                assistant_idx += 1


def handle_input(params: dict, model: str):
    user_input = st.chat_input("Введите сообщение...")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.agent.system_prompt = st.session_state.system_prompt

    active_params = {k: v for k, v in params.items() if v is not None}
    print(f"[agent] model={model}, params={active_params}")

    prev_prompt = st.session_state.message_stats[-1]["prompt_tokens"] if st.session_state.message_stats else 0

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            result = st.session_state.agent.run(user_input, model=model, **active_params)
        st.markdown(result.content)
        delta_prompt = (result.prompt_tokens or 0) - (prev_prompt or 0)
        delta_total = delta_prompt + (result.completion_tokens or 0)
        stats = {
            "elapsed_s": result.elapsed_s,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "delta_prompt_tokens": delta_prompt,
        }
        st.session_state.message_stats.append(stats)
        st.caption(_stats_caption(result.elapsed_s, delta_prompt, result.completion_tokens, delta_total))
        save_context(
            st.session_state.session_id,
            st.session_state.system_prompt,
            st.session_state.agent.history,
            st.session_state.message_stats,
            model_key=st.session_state.get("selected_model_key"),
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

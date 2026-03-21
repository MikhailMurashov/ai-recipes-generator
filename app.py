import streamlit as st
from llm_client import chat
from config import LLM_MODEL


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""


def render_sidebar() -> dict:
    with st.sidebar:
        st.title("Параметры модели")
        st.caption(f"Модель: `{LLM_MODEL}`")

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

        if st.button("Сбросить контекст", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        n = len(st.session_state.messages)
        st.caption(f"Сообщений: {n}")

    return params


def render_chat_history():
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])


def handle_input(params: dict):
    user_input = st.chat_input("Введите сообщение...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    call_messages = [
        {"role": "system", "content": st.session_state.system_prompt}
    ] + st.session_state.messages

    active_params = {k: v for k, v in params.items() if v is not None}
    print(f"[chat] params={active_params}, messages={len(call_messages)}")

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            response = chat(call_messages, **params)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(
        page_title="Чат с ассистентом",
        page_icon="🍳",
        layout="wide",
    )
    st.title("Чат с ассистентом")

    init_session_state()
    params = render_sidebar()
    render_chat_history()
    handle_input(params)


if __name__ == "__main__":
    main()

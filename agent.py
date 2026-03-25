from llm_client import chat, ChatResponse


class Agent:
    """Encapsulates conversation state and LLM interaction."""

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self._history: list[dict[str, str]] = []

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def run(self, user_input: str, **llm_params) -> ChatResponse:
        self._history.append({"role": "user", "content": user_input})
        response = chat(self._build_messages(), **llm_params)
        self._history.append({"role": "assistant", "content": response.content})
        return response

    def _build_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history)
        return messages

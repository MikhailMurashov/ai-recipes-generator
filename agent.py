from llm_client import chat, ChatResponse


class Agent:
    """Encapsulates conversation state and LLM interaction."""

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self._history: list[dict[str, str]] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def run(self, user_input: str, **llm_params) -> ChatResponse:
        self._history.append({"role": "user", "content": user_input})
        response = chat(self._build_messages(), **llm_params)
        self._history.append({"role": "assistant", "content": response.content})
        if response.prompt_tokens:
            self.total_prompt_tokens += response.prompt_tokens
        if response.completion_tokens:
            self.total_completion_tokens += response.completion_tokens
        if response.total_tokens:
            self.total_tokens += response.total_tokens
        return response

    def _build_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history)
        return messages

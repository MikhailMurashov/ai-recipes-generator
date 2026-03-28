from llm_client import chat, ChatResponse


class Agent:
    """Encapsulates conversation state and LLM interaction."""

    CONTEXT_WINDOW = 5

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self._history: list[dict[str, str]] = []
        self._summary: str = ""
        self._summarized_count: int = 0
        self.summarization_enabled: bool = True
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    @property
    def summary(self) -> str:
        return self._summary

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
        if len(self._history) > self.CONTEXT_WINDOW and self.summarization_enabled:
            self._update_summary(llm_params.get("model"))
        return response

    def _update_summary(self, model: str | None) -> None:
        new_msgs = self._history[self._summarized_count : -self.CONTEXT_WINDOW]
        if not new_msgs:
            return
        convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in new_msgs)
        if self._summary:
            prompt = (
                f"Обнови саммари беседы, добавив новые сообщения.\n\n"
                f"Текущий саммари:\n{self._summary}\n\n"
                f"Новые сообщения:\n{convo_text}\n\n"
                f"Дай очень краткий обновлённый саммари, сохранив только самые важные факты."
            )
        else:
            prompt = f"Очень кратко изложи суть следующей беседы, сохранив только самые ключевые факты:\n\n{convo_text}"
        response = chat([{"role": "user", "content": prompt}], model=model)
        self._summary = response.content
        self._summarized_count = len(self._history) - self.CONTEXT_WINDOW

    def _build_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self._summary and self.summarization_enabled:
            messages.append(
                {
                    "role": "system",
                    "content": f"Саммари предыдущей беседы:\n{self._summary}",
                }
            )
        messages.extend(self._history[-self.CONTEXT_WINDOW :])
        return messages

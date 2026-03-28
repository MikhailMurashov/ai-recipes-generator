import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum

from llm_client import chat


class StrategyType(str, Enum):
    SLIDING_WINDOW_SUMMARY = "sliding_window_summary"
    SLIDING_WINDOW = "sliding_window"
    STICKY_FACTS = "sticky_facts"
    BRANCHING = "branching"


class BaseStrategy(ABC):
    """Abstract base for all context management strategies."""

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt: str = system_prompt
        self._history: list[dict[str, str]] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def _accumulate_tokens(self, response) -> None:
        if response.prompt_tokens is not None:
            self.total_prompt_tokens += response.prompt_tokens
        if response.completion_tokens is not None:
            self.total_completion_tokens += response.completion_tokens
        if response.total_tokens is not None:
            self.total_tokens += response.total_tokens

    @abstractmethod
    def run(self, user_input: str, **llm_params):
        """Send user_input, return ChatResponse."""

    @abstractmethod
    def _build_messages(self) -> list[dict[str, str]]:
        """Construct the messages list to send to the LLM."""

    @abstractmethod
    def to_state(self) -> dict:
        """Serialize strategy-specific state to a plain dict."""

    @classmethod
    @abstractmethod
    def from_state(cls, system_prompt: str, state: dict) -> "BaseStrategy":
        """Restore a strategy instance from a previously serialized state dict."""


class SlidingWindowSummaryStrategy(BaseStrategy):
    """Original strategy: sliding window with auto-summarization."""

    CONTEXT_WINDOW = 5

    def __init__(
        self, system_prompt: str = "", summarization_enabled: bool = True
    ) -> None:
        super().__init__(system_prompt)
        self._summary: str = ""
        self._summarized_count: int = 0
        self.summarization_enabled: bool = summarization_enabled

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def summarized_count(self) -> int:
        return self._summarized_count

    def run(self, user_input: str, **llm_params):
        self._history.append({"role": "user", "content": user_input})
        response = chat(self._build_messages(), **llm_params)
        self._history.append({"role": "assistant", "content": response.content})
        self._accumulate_tokens(response)
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
        try:
            response = chat([{"role": "user", "content": prompt}], model=model)
            self._summary = response.content
            self._summarized_count = len(self._history) - self.CONTEXT_WINDOW
        except Exception as e:
            logging.warning(f"strategies: summarization failed, skipping: {e}")

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

    def to_state(self) -> dict:
        return {
            "summary": self._summary,
            "summarized_count": self._summarized_count,
            "history": list(self._history),
        }

    @classmethod
    def from_state(
        cls, system_prompt: str, state: dict
    ) -> "SlidingWindowSummaryStrategy":
        instance = cls(system_prompt=system_prompt)
        instance._summary = state.get("summary", "")
        instance._summarized_count = state.get("summarized_count", 0)
        instance._history = list(state.get("history", []))
        return instance


class SlidingWindowStrategy(BaseStrategy):
    """Simple sliding window without summarization."""

    def __init__(self, system_prompt: str = "", window_size: int = 10) -> None:
        super().__init__(system_prompt)
        self._window_size: int = window_size

    def run(self, user_input: str, **llm_params):
        self._history.append({"role": "user", "content": user_input})
        response = chat(self._build_messages(), **llm_params)
        self._history.append({"role": "assistant", "content": response.content})
        self._accumulate_tokens(response)
        # Trim history to last window_size*2 messages
        if len(self._history) > self._window_size * 2:
            self._history = self._history[-self._window_size * 2 :]
        return response

    def _build_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history[-self._window_size :])
        return messages

    def to_state(self) -> dict:
        return {
            "window_size": self._window_size,
            "history": list(self._history),
        }

    @classmethod
    def from_state(cls, system_prompt: str, state: dict) -> "SlidingWindowStrategy":
        instance = cls(
            system_prompt=system_prompt, window_size=state.get("window_size", 10)
        )
        instance._history = list(state.get("history", []))
        return instance


class StickyFactsStrategy(BaseStrategy):
    """Sliding window with LLM-extracted persistent facts."""

    def __init__(self, system_prompt: str = "", window_size: int = 6) -> None:
        super().__init__(system_prompt)
        self._window_size: int = window_size
        self._facts: str = ""

    @property
    def facts(self) -> str:
        return self._facts

    def run(self, user_input: str, **llm_params):
        self._history.append({"role": "user", "content": user_input})
        response = chat(self._build_messages(), **llm_params)
        self._history.append({"role": "assistant", "content": response.content})
        self._accumulate_tokens(response)
        if len(self._history) > self._window_size:
            self._merge_facts(llm_params.get("model"))
        return response

    def _merge_facts(self, model: str | None) -> None:
        new_msgs = self._history[: -self._window_size]
        if not new_msgs:
            return
        convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in new_msgs)
        if self._facts:
            prompt = (
                f"Текущие факты:\n{self._facts}\n\n"
                f"Новые сообщения:\n{convo_text}\n\n"
                f"Обнови список фактов: добавь новое, исправь устаревшее, убери дубли. "
                f"Отвечай ТОЛЬКО списком фактов, каждый с новой строки, начиная с '- '. "
                f"Включай только долгосрочно важное: имя, цели, предпочтения, ключевые договорённости."
            )
        else:
            prompt = (
                f"Сообщения:\n{convo_text}\n\n"
                f"Извлеки ключевые факты о пользователе и контексте задачи. "
                f"Отвечай ТОЛЬКО списком фактов, каждый с новой строки, начиная с '- '. "
                f"Включай только долгосрочно важное: имя, цели, предпочтения, ключевые договорённости."
            )
        try:
            logging.info("merge_facts prompt: %s", prompt)
            response = chat(
                [
                    {
                        "role": "system",
                        "content": "Ты агент по ведению базы фактов о пользователе. Отвечай строго в указанном формате.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model,
            )
            logging.info("merge_facts response: %s", response.content)
            self._facts = response.content
        except Exception as e:
            logging.warning(f"strategies: fact merge failed, skipping: {e}")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self._facts:
            facts_text = "\n".join(self._facts)
            messages.append(
                {
                    "role": "system",
                    "content": f"Известные факты о пользователе и беседе:\n{facts_text}",
                }
            )
        messages.extend(self._history[-self._window_size :])
        return messages

    def to_state(self) -> dict:
        return {
            "window_size": self._window_size,
            "facts": self._facts,
            "history": list(self._history),
        }

    @classmethod
    def from_state(cls, system_prompt: str, state: dict) -> "StickyFactsStrategy":
        instance = cls(
            system_prompt=system_prompt, window_size=state.get("window_size", 6)
        )
        instance._facts = state.get("facts", "")
        instance._history = list(state.get("history", []))
        return instance


@dataclass
class Branch:
    branch_id: str
    name: str
    checkpoint_index: int  # len(trunk) at fork time
    history: list  # branch-local messages after checkpoint


class BranchingStrategy(BaseStrategy):
    """Conversation branching: trunk + named branches."""

    def __init__(self, system_prompt: str = "", window_size: int = 10) -> None:
        super().__init__(system_prompt)
        self._trunk: list[dict[str, str]] = []
        self._branches: dict[str, Branch] = {}
        self._active_branch_id: str | None = None
        self._window_size: int = window_size

    @property
    def history(self) -> list[dict[str, str]]:
        """Return combined trunk+branch view."""
        if self._active_branch_id and self._active_branch_id in self._branches:
            branch = self._branches[self._active_branch_id]
            return list(self._trunk[: branch.checkpoint_index] + branch.history)
        return list(self._trunk)

    @property
    def branches(self) -> dict[str, Branch]:
        return dict(self._branches)

    @property
    def active_branch_id(self) -> str | None:
        return self._active_branch_id

    def _active_history_list(self) -> list[dict[str, str]]:
        """Return a reference to the mutable list we should append to."""
        if self._active_branch_id and self._active_branch_id in self._branches:
            return self._branches[self._active_branch_id].history
        return self._trunk

    def run(self, user_input: str, **llm_params):
        active = self._active_history_list()
        active.append({"role": "user", "content": user_input})
        response = chat(self._build_messages(), **llm_params)
        active.append({"role": "assistant", "content": response.content})
        self._accumulate_tokens(response)
        return response

    def _build_messages(self) -> list[dict[str, str]]:
        combined = self.history
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(combined[-self._window_size :])
        return messages

    def create_branch(self, name: str) -> str:
        """Create a new branch from the current history position. Returns branch_id."""
        branch_id = str(uuid.uuid4())
        checkpoint_index = len(self.history)
        branch = Branch(
            branch_id=branch_id,
            name=name,
            checkpoint_index=checkpoint_index,
            history=[],
        )
        self._branches[branch_id] = branch
        self._active_branch_id = branch_id
        return branch_id

    def switch_branch(self, branch_id: str | None) -> None:
        """Switch to a branch by id, or None to switch to trunk."""
        if branch_id is None or branch_id in self._branches:
            self._active_branch_id = branch_id
        else:
            logging.warning(f"strategies: branch {branch_id!r} not found")

    def delete_branch(self, branch_id: str) -> None:
        """Delete a branch. If it was active, switch to trunk."""
        self._branches.pop(branch_id, None)
        if self._active_branch_id == branch_id:
            self._active_branch_id = None

    def to_state(self) -> dict:
        return {
            "window_size": self._window_size,
            "trunk": list(self._trunk),
            "branches": {bid: asdict(b) for bid, b in self._branches.items()},
            "active_branch_id": self._active_branch_id,
        }

    @classmethod
    def from_state(cls, system_prompt: str, state: dict) -> "BranchingStrategy":
        instance = cls(
            system_prompt=system_prompt, window_size=state.get("window_size", 10)
        )
        instance._trunk = list(state.get("trunk", []))
        instance._active_branch_id = state.get("active_branch_id")
        raw_branches = state.get("branches", {})
        for bid, bdata in raw_branches.items():
            instance._branches[bid] = Branch(
                branch_id=bdata["branch_id"],
                name=bdata["name"],
                checkpoint_index=bdata["checkpoint_index"],
                history=list(bdata.get("history", [])),
            )
        return instance


def make_strategy(
    strategy_type: StrategyType,
    system_prompt: str = "",
    state: dict | None = None,
) -> BaseStrategy:
    """Factory: create or restore a strategy instance."""
    if state is not None:
        if strategy_type == StrategyType.SLIDING_WINDOW_SUMMARY:
            return SlidingWindowSummaryStrategy.from_state(system_prompt, state)
        if strategy_type == StrategyType.SLIDING_WINDOW:
            return SlidingWindowStrategy.from_state(system_prompt, state)
        if strategy_type == StrategyType.STICKY_FACTS:
            return StickyFactsStrategy.from_state(system_prompt, state)
        if strategy_type == StrategyType.BRANCHING:
            return BranchingStrategy.from_state(system_prompt, state)

    # Fresh instances
    if strategy_type == StrategyType.SLIDING_WINDOW_SUMMARY:
        return SlidingWindowSummaryStrategy(system_prompt=system_prompt)
    if strategy_type == StrategyType.SLIDING_WINDOW:
        return SlidingWindowStrategy(system_prompt=system_prompt)
    if strategy_type == StrategyType.STICKY_FACTS:
        return StickyFactsStrategy(system_prompt=system_prompt)
    if strategy_type == StrategyType.BRANCHING:
        return BranchingStrategy(system_prompt=system_prompt)

    raise ValueError(f"Unknown strategy type: {strategy_type}")

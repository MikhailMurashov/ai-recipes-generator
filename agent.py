import logging

from memory import LongTermMemory, WorkingMemory
from strategies import (
    BaseStrategy,
    BranchingStrategy,
    SlidingWindowSummaryStrategy,
    StickyFactsStrategy,
    StrategyType,
    make_strategy,
)

logger = logging.getLogger(__name__)


class Agent:
    """Thin facade over BaseStrategy implementations."""

    def __init__(
        self,
        system_prompt: str = "",
        strategy_type: StrategyType = StrategyType.SLIDING_WINDOW_SUMMARY,
        strategy_state: dict | None = None,
        working_memory_state: dict | None = None,
    ) -> None:
        self._base_system_prompt: str = system_prompt
        self.strategy_type: StrategyType = strategy_type
        self._strategy: BaseStrategy = make_strategy(
            strategy_type, system_prompt, strategy_state
        )
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        if working_memory_state:
            self.working_memory.from_state(working_memory_state)
        logger.info(
            "agent: init strategy=%s restored=%s",
            strategy_type.value,
            strategy_state is not None,
        )

    # ------------------------------------------------------------------
    # Memory injection
    # ------------------------------------------------------------------

    def _build_full_system_prompt(self) -> str:
        parts = []
        if self._base_system_prompt:
            parts.append(self._base_system_prompt)
        wm = self.working_memory.to_context_string()
        if wm:
            parts.append(f"## Рабочая память (текущая задача):\n{wm}")
        ltm = self.long_term_memory.to_context_string()
        if ltm:
            parts.append(f"## Долговременная память:\n{ltm}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Strategy switching
    # ------------------------------------------------------------------

    def switch_strategy(self, new_type: StrategyType) -> None:
        """Switch to a new strategy, carrying over history and system prompt."""
        logger.info(
            "agent: switch_strategy %s -> %s", self.strategy_type.value, new_type.value
        )
        old_history = self._strategy.history
        self._strategy = make_strategy(new_type, self._base_system_prompt, None)
        self._strategy._history = list(old_history)
        self.strategy_type = new_type

    def get_strategy_state(self) -> dict:
        return self._strategy.to_state()

    # ------------------------------------------------------------------
    # Proxy: core properties
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        return self._base_system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._base_system_prompt = value

    @property
    def history(self) -> list[dict[str, str]]:
        return self._strategy.history

    # ------------------------------------------------------------------
    # Proxy: token counters
    # ------------------------------------------------------------------

    @property
    def total_prompt_tokens(self) -> int:
        return self._strategy.total_prompt_tokens

    @total_prompt_tokens.setter
    def total_prompt_tokens(self, value: int) -> None:
        self._strategy.total_prompt_tokens = value

    @property
    def total_completion_tokens(self) -> int:
        return self._strategy.total_completion_tokens

    @total_completion_tokens.setter
    def total_completion_tokens(self, value: int) -> None:
        self._strategy.total_completion_tokens = value

    @property
    def total_tokens(self) -> int:
        return self._strategy.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        self._strategy.total_tokens = value

    # ------------------------------------------------------------------
    # Proxy: SlidingWindowSummaryStrategy-specific
    # ------------------------------------------------------------------

    @property
    def summary(self) -> str:
        if isinstance(self._strategy, SlidingWindowSummaryStrategy):
            return self._strategy.summary
        return ""

    @property
    def summarized_count(self) -> int:
        if isinstance(self._strategy, SlidingWindowSummaryStrategy):
            return self._strategy.summarized_count
        return 0

    @property
    def summarization_enabled(self) -> bool:
        if isinstance(self._strategy, SlidingWindowSummaryStrategy):
            return self._strategy.summarization_enabled
        return False

    @summarization_enabled.setter
    def summarization_enabled(self, value: bool) -> None:
        if isinstance(self._strategy, SlidingWindowSummaryStrategy):
            self._strategy.summarization_enabled = value

    # ------------------------------------------------------------------
    # Proxy: StickyFactsStrategy-specific
    # ------------------------------------------------------------------

    @property
    def facts(self) -> dict[str, str]:
        if isinstance(self._strategy, StickyFactsStrategy):
            return self._strategy.facts
        return {}

    # ------------------------------------------------------------------
    # Proxy: BranchingStrategy-specific
    # ------------------------------------------------------------------

    @property
    def branches(self):
        if isinstance(self._strategy, BranchingStrategy):
            return self._strategy.branches
        return {}

    @property
    def active_branch_id(self) -> str | None:
        if isinstance(self._strategy, BranchingStrategy):
            return self._strategy.active_branch_id
        return None

    def create_branch(self, name: str) -> str:
        if isinstance(self._strategy, BranchingStrategy):
            return self._strategy.create_branch(name)
        raise TypeError("Current strategy does not support branching")

    def switch_branch(self, branch_id: str | None) -> None:
        if isinstance(self._strategy, BranchingStrategy):
            self._strategy.switch_branch(branch_id)

    def delete_branch(self, branch_id: str) -> None:
        if isinstance(self._strategy, BranchingStrategy):
            self._strategy.delete_branch(branch_id)

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self, user_input: str, **llm_params):
        self._strategy.system_prompt = self._build_full_system_prompt()
        logger.info(
            "agent: run strategy=%s history_len=%d input=%r",
            self.strategy_type.value,
            len(self._strategy._history),
            user_input[:80],
        )
        response = self._strategy.run(user_input, **llm_params)
        logger.info(
            "agent: response tokens prompt=%s completion=%s total=%s elapsed=%.2fs",
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
            response.elapsed_s,
        )
        return response

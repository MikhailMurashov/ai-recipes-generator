from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

ADVANCE_SIGNAL = "[→СЛЕДУЮЩИЙ_ЭТАП]"
STEP_SIGNAL = "[→СЛЕДУЮЩИЙ_ШАГ]"


class TaskStage(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


STAGE_ORDER: list[TaskStage] = [
    TaskStage.PLANNING,
    TaskStage.EXECUTION,
    TaskStage.VALIDATION,
    TaskStage.DONE,
]

STAGE_CONTRACTS: dict[TaskStage, str] = {
    TaskStage.PLANNING: (
        "Вход: описание задачи или материалы для работы.\n"
        "Задача: проанализировать код, декомпозировать на конкретные шаги, "
        "сформулировать критерии готовности.\n"
        "Выход: нумерованный список шагов, согласованный с пользователем.\n"
        "Не начинай реализацию — только план."
    ),
    TaskStage.EXECUTION: (
        "Вход: один шаг из согласованного плана (номер шага = current_step).\n"
        "Задача: реализовать ровно один шаг — показать код, объяснить решение.\n"
        "Выход: готовый код для этого шага + краткое объяснение изменений."
    ),
    TaskStage.VALIDATION: (
        "Вход: результат всех шагов выполнения.\n"
        "Задача: проверить соответствие критериям готовности из planning.\n"
        "Выход: чеклист ✅/❌ по каждому критерию + edge cases для проверки.\n"
        "Не считай задачу завершённой, пока пользователь не подтвердил все пункты.\n"
        "Если хотя бы один критерий ❌ — не переходи к следующему этапу.\n"
        "Предложи вернуться к EXECUTION с конкретным списком исправлений."
    ),
    TaskStage.DONE: (
        "Задача завершена.\n"
        "Задача: подвести итог — что было сделано, какие решения приняты, "
        "что стоит учесть в будущем."
    ),
}

STAGE_EXPECTED_ACTIONS: dict[TaskStage, str] = {
    TaskStage.PLANNING: "Составь нумерованный план шагов и критерии готовности",
    TaskStage.EXECUTION: "Реализуй текущий шаг из согласованного плана",
    TaskStage.VALIDATION: "Проверь каждый критерий готовности (✅/❌) и перечисли edge cases",
    TaskStage.DONE: "Подведи краткий итог: что сделано, какие решения приняты",
}


@dataclass
class TaskState:
    stage: TaskStage = TaskStage.PLANNING
    current_step: int = 0

    @property
    def expected_action(self) -> str:
        if self.stage == TaskStage.EXECUTION and self.current_step > 0:
            return f"Реализуй шаг {self.current_step} из согласованного плана"
        return STAGE_EXPECTED_ACTIONS.get(self.stage, "")

    def check_and_advance(self, text: str) -> bool:
        if ADVANCE_SIGNAL in text and self.stage != TaskStage.DONE:
            self.advance()
            return True
        return False

    def check_and_step(self, text: str) -> bool:
        if STEP_SIGNAL in text and self.stage == TaskStage.EXECUTION:
            self.current_step += 1
            return True
        return False

    def advance(self) -> None:
        if self.stage == TaskStage.DONE:
            return
        idx = STAGE_ORDER.index(self.stage)
        if idx < len(STAGE_ORDER) - 1:
            self.stage = STAGE_ORDER[idx + 1]
            self.current_step = 0

    def go_back(self) -> None:
        """Вернуться с VALIDATION обратно в EXECUTION."""
        if self.stage != TaskStage.VALIDATION:
            return
        self.stage = TaskStage.EXECUTION
        self.current_step = 0

    def to_context_string(self) -> str:
        lines = [
            f"Текущий этап задачи: {self.stage.value.upper()}",
            f"Текущий шаг: {self.current_step}",
        ]
        contract = STAGE_CONTRACTS.get(self.stage, "")
        if contract:
            lines.append(f"Контракт этапа:\n{contract}")

        if self.stage != TaskStage.DONE:
            lines.append(
                f"Сигналы перехода (добавь в конец ответа при завершении):\n"
                f"  {ADVANCE_SIGNAL} — завершить текущий этап и перейти к следующему\n"
                f"  {STEP_SIGNAL} — завершить текущий шаг (только в EXECUTION)"
            )

        return "\n".join(lines)

    def to_state(self) -> dict:
        return {
            "stage": self.stage.value,
            "current_step": self.current_step,
        }

    @classmethod
    def from_state(cls, data: dict) -> "TaskState":
        stage_value = data.get("stage", "planning")
        # Migrate old "paused" state: restore the stage it was paused at
        if stage_value == "paused":
            stage_value = data.get("paused_at_stage", "planning") or "planning"
            step = data.get("paused_at_step", 0)
        else:
            step = data.get("current_step", 0)
        return cls(stage=TaskStage(stage_value), current_step=step)

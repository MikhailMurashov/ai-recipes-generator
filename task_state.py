from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
        "Вход: согласованный план. Последовательность шагов.\n"
        "Задача: реализовать весь план пошагово, объяснить решение.\n"
        "Выход: готовое решение для плана + очень краткое объяснение действий."
    ),
    TaskStage.VALIDATION: (
        "Вход: результат всех шагов выполнения.\n"
        "Задача: проверить соответствие критериям готовности из плана.\n"
        "Выход: чеклист ✅/❌ по каждому критерию + edge cases для проверки.\n"
        "Не считай задачу завершённой, пока пользователь не подтвердил все пункты.\n"
        "Если хотя бы один критерий ❌ — Предложи вернуться к EXECUTION с конкретным списком исправлений.\n"
    ),
    TaskStage.DONE: (
        "Задача завершена.\n"
        "Задача: подвести итог — что было сделано, какие решения приняты, что стоит учесть в будущем."
    ),
}

@dataclass
class TaskState:
    stage: TaskStage = TaskStage.PLANNING

    def advance(self) -> None:
        if self.stage == TaskStage.DONE:
            return
        idx = STAGE_ORDER.index(self.stage)
        if idx < len(STAGE_ORDER) - 1:
            self.stage = STAGE_ORDER[idx + 1]

    def go_back(self) -> None:
        """Вернуться с VALIDATION обратно в EXECUTION."""
        if self.stage != TaskStage.VALIDATION:
            return
        self.stage = TaskStage.EXECUTION

    def to_context_string(self) -> str:
        lines = [
            f"Текущий этап задачи: {self.stage.value.upper()}",
        ]
        contract = STAGE_CONTRACTS.get(self.stage, "")
        if contract:
            lines.append(f"Контракт этапа:\n{contract}")

        return "\n".join(lines)

    def to_state(self) -> dict:
        return {"stage": self.stage.value}

    @classmethod
    def from_state(cls, data: dict) -> "TaskState":
        stage_value = data.get("stage", "planning")
        return cls(stage=TaskStage(stage_value))

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskStage(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"
    PAUSED = "paused"


STAGE_ORDER: list[TaskStage] = [
    TaskStage.PLANNING,
    TaskStage.EXECUTION,
    TaskStage.VALIDATION,
    TaskStage.DONE,
]

STAGE_CONTRACTS: dict[TaskStage, str] = {
    TaskStage.PLANNING: (
        "Вход: код или описание задачи для рефакторинга.\n"
        "Задача: проанализировать код, декомпозировать на конкретные шаги, "
        "сформулировать критерии готовности.\n"
        "Выход: нумерованный список шагов, согласованный с пользователем.\n"
        "Не начинай реализацию — только план."
    ),
    TaskStage.EXECUTION: (
        "Вход: один шаг из согласованного плана (номер шага = current_step).\n"
        "Задача: реализовать ровно один шаг — показать код, объяснить решение.\n"
        "Выход: готовый код для этого шага + краткое объяснение изменений.\n"
        "Не переходи к следующему шагу без явного подтверждения пользователя."
    ),
    TaskStage.VALIDATION: (
        "Вход: результат всех шагов выполнения.\n"
        "Задача: проверить соответствие критериям готовности из planning.\n"
        "Выход: чеклист ✅/❌ по каждому критерию + edge cases для проверки.\n"
        "Не считай задачу завершённой, пока пользователь не подтвердил все пункты."
    ),
    TaskStage.DONE: (
        "Задача завершена.\n"
        "Задача: подвести итог — что было сделано, какие решения приняты, "
        "что стоит учесть в будущем."
    ),
}


@dataclass
class TaskState:
    stage: TaskStage = TaskStage.PLANNING
    current_step: int = 0
    paused_at_stage: TaskStage | None = None
    paused_at_step: int = 0

    def pause(self) -> None:
        if self.stage == TaskStage.PAUSED:
            return
        self.paused_at_stage = self.stage
        self.paused_at_step = self.current_step
        self.stage = TaskStage.PAUSED

    def resume(self) -> None:
        if self.stage != TaskStage.PAUSED or self.paused_at_stage is None:
            return
        self.stage = self.paused_at_stage
        self.current_step = self.paused_at_step
        self.paused_at_stage = None
        self.paused_at_step = 0

    def advance(self) -> None:
        if self.stage in (TaskStage.PAUSED, TaskStage.DONE):
            return
        idx = STAGE_ORDER.index(self.stage)
        if idx < len(STAGE_ORDER) - 1:
            self.stage = STAGE_ORDER[idx + 1]
            self.current_step = 0

    def to_context_string(self) -> str:
        if self.stage == TaskStage.PAUSED:
            lines = [
                f"Текущий этап задачи: ПАУЗА "
                f"(возобновить с «{self.paused_at_stage.value}», шаг {self.paused_at_step})"
            ]
            contract = STAGE_CONTRACTS.get(self.paused_at_stage, "")
        else:
            lines = [
                f"Текущий этап задачи: {self.stage.value.upper()}",
                f"Текущий шаг: {self.current_step}",
            ]
            contract = STAGE_CONTRACTS.get(self.stage, "")

        if contract:
            lines.append(f"Контракт этапа:\n{contract}")
        return "\n".join(lines)

    def to_state(self) -> dict:
        return {
            "stage": self.stage.value,
            "current_step": self.current_step,
            "paused_at_stage": (
                self.paused_at_stage.value if self.paused_at_stage else None
            ),
            "paused_at_step": self.paused_at_step,
        }

    @classmethod
    def from_state(cls, data: dict) -> "TaskState":
        raw = data.get("paused_at_stage")
        return cls(
            stage=TaskStage(data.get("stage", "planning")),
            current_step=data.get("current_step", 0),
            paused_at_stage=TaskStage(raw) if raw else None,
            paused_at_step=data.get("paused_at_step", 0),
        )

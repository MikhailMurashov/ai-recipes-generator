import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

STORAGE_DIR = Path(".storage")


@dataclass
class MemoryEntry:
    key: str
    value: str
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


class WorkingMemory:
    """Session-scoped notes for the current task."""

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    def set(self, key: str, value: str) -> None:
        if key in self._entries:
            self._entries[key].value = value
        else:
            self._entries[key] = MemoryEntry(key=key, value=value)

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> dict[str, MemoryEntry]:
        return dict(self._entries)

    def to_context_string(self) -> str:
        if not self._entries:
            return ""
        return "\n".join(f"- {v.value}" for v in self._entries.values())

    def to_state(self) -> dict:
        return {k: asdict(v) for k, v in self._entries.items()}

    def from_state(self, data: dict) -> None:
        self._entries = {k: MemoryEntry(**v) for k, v in data.items()}


class LongTermMemory:
    """Persistent cross-session memory: user profile, decisions, knowledge."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for k, v in raw.items():
                self._entries[k] = MemoryEntry(**v)
        except Exception as e:
            logging.warning("long_term_memory: load failed: %s", e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(
                    {k: asdict(v) for k, v in self._entries.items()},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except Exception as e:
            logging.warning("long_term_memory: save failed: %s", e)

    def set(self, key: str, value: str) -> None:
        if key in self._entries:
            self._entries[key].value = value
        else:
            self._entries[key] = MemoryEntry(key=key, value=value)
        self._save()

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)
        self._save()

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    @property
    def entries(self) -> dict[str, MemoryEntry]:
        return dict(self._entries)

    def to_context_string(self) -> str:
        if not self._entries:
            return ""
        return "\n".join(f"- {v.value}" for v in self._entries.values())

    def to_state(self) -> dict:
        return {k: asdict(v) for k, v in self._entries.items()}

    def from_state(self, data: dict) -> None:
        self._entries = {k: MemoryEntry(**v) for k, v in data.items()}


class Personalization:
    """Persistent free-form user profile text, stored separately from conversations."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.text: str = ""
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            self.text = raw.get("text", "")
        except Exception as e:
            logging.warning("personalization: load failed: %s", e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps({"text": self.text}, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logging.warning("personalization: save failed: %s", e)

    def set_text(self, text: str) -> None:
        self.text = text
        self._save()

    def to_context_string(self) -> str:
        return self.text

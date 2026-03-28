import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

STORAGE_DIR = Path(".storage")


def new_session_id() -> str:
    return str(uuid.uuid4())


def save_context(
    session_id: str,
    system_prompt: str,
    history: list,
    message_stats: list,
    model_key: str | None = None,
    summary: str = "",
    summarized_count: int = 0,
    strategy_type: str = "sliding_window_summary",
    strategy_state: dict | None = None,
):
    STORAGE_DIR.mkdir(exist_ok=True)
    path = STORAGE_DIR / f"{session_id}.json"
    now = datetime.now().isoformat(timespec="seconds")
    created_at = now
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            created_at = existing.get("created_at", now)
        except Exception as e:
            logging.warning(f"storage: {e}")
    data = {
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": now,
        "system_prompt": system_prompt,
        "model_key": model_key,
        "history": history,
        "message_stats": message_stats,
        "summary": summary,
        "summarized_count": summarized_count,
        "strategy_type": strategy_type,
        "strategy_state": strategy_state,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def list_contexts() -> list[dict]:
    """Возвращает список контекстов, отсортированных по updated_at (новые первые)."""
    STORAGE_DIR.mkdir(exist_ok=True)
    result = []
    for f in STORAGE_DIR.glob("*.json"):
        try:
            result.append(json.loads(f.read_text()))
        except Exception as e:
            logging.warning(f"storage: {e}")
    return sorted(result, key=lambda d: d.get("updated_at", ""), reverse=True)


def delete_context(session_id: str):
    path = STORAGE_DIR / f"{session_id}.json"
    path.unlink(missing_ok=True)

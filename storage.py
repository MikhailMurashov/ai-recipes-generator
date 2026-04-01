import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

STORAGE_DIR = Path(".storage")
USERS_FILE = STORAGE_DIR / "users.json"


def new_session_id() -> str:
    return str(uuid.uuid4())


def get_user_dir(username: str) -> Path:
    path = STORAGE_DIR / username
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_personalization_path(username: str) -> Path:
    return get_user_dir(username) / "personalization.json"


def list_users() -> list[str]:
    STORAGE_DIR.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        return []
    try:
        data = json.loads(USERS_FILE.read_text())
        return data.get("users", [])
    except Exception as e:
        logging.warning(f"storage: list_users: {e}")
        return []


def user_exists(username: str) -> bool:
    return username in list_users()


def register_user(username: str) -> bool:
    """Register a new user. Returns False if username is already taken."""
    users = list_users()
    if username in users:
        return False
    users.append(username)
    STORAGE_DIR.mkdir(exist_ok=True)
    try:
        USERS_FILE.write_text(
            json.dumps({"users": users}, ensure_ascii=False, indent=2)
        )
        get_user_dir(username)  # create user directory
        return True
    except Exception as e:
        logging.warning(f"storage: register_user: {e}")
        return False


def save_working_memory(session_id: str, working_memory: dict, username: str) -> None:
    """Update only the working_memory field in an existing session file."""
    path = get_user_dir(username) / f"{session_id}.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
        data["working_memory"] = working_memory
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        logging.warning(f"storage: save_working_memory: {e}")


def save_context(
    session_id: str,
    system_prompt: str,
    history: list,
    message_stats: list,
    username: str,
    model_key: str | None = None,
    summary: str = "",
    summarized_count: int = 0,
    strategy_type: str = "sliding_window_summary",
    strategy_state: dict | None = None,
    working_memory: dict | None = None,
):
    path = get_user_dir(username) / f"{session_id}.json"
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
        "working_memory": working_memory or {},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def list_contexts(username: str) -> list[dict]:
    """Returns list of contexts for a user, sorted by updated_at (newest first)."""
    user_dir = get_user_dir(username)
    result = []
    for f in user_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if "session_id" not in data:
                continue
            result.append(data)
        except Exception as e:
            logging.warning(f"storage: {e}")
    return sorted(result, key=lambda d: d.get("updated_at", ""), reverse=True)


def delete_context(session_id: str, username: str) -> None:
    path = get_user_dir(username) / f"{session_id}.json"
    path.unlink(missing_ok=True)

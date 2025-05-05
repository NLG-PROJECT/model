import os
import json
from datetime import datetime
from typing import Dict, Any

# Constants
FILES_LIST_PATH = 'files_list.json'
USER_SESSION_FILE = 'user_session.json'
USER_LOG_FILE = 'user_logs.jsonl'

def update_files_list(new_entry: Dict[str, Any]):
    """Update the files list with a new entry."""
    if os.path.exists(FILES_LIST_PATH):
        with open(FILES_LIST_PATH, "r") as f:
            files_list = json.load(f)
    else:
        files_list = []

    files_list.append(new_entry)

    with open(FILES_LIST_PATH, "w") as f:
        json.dump(files_list, f, indent=2)

def reset_user_session(new_entry: Dict[str, Any]):
    """Reset the user session with a new entry."""
    session_entry = {
        **new_entry,
        "chat_history": [],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(USER_SESSION_FILE, "w") as f:
        json.dump(session_entry, f, indent=2)

def clear_user_logs():
    """Clear the user logs file."""
    with open(USER_LOG_FILE, "w") as log_file:
        log_file.write("")

def log_user_event(step: str, status: str, detail: str = ""):
    """Log a user event with the given step, status, and detail."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "status": status,
        "detail": detail
    }
    with open(USER_LOG_FILE, "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")



import os
import json
from datetime import datetime
from typing import Dict, Any
import aiofiles
import logging

# Constants
FILES_LIST_PATH = 'files_list.json'
USER_SESSION_FILE = 'user_session.json'
USER_LOG_FILE = 'user_logs.jsonl'

# Logger setup
logger = logging.getLogger(__name__)

async def update_files_list(new_entry: Dict[str, Any]):
    """Update the files list with a new entry."""
    if os.path.exists(FILES_LIST_PATH):
        async with aiofiles.open(FILES_LIST_PATH, "r") as f:
            files_list = json.loads(await f.read())
    else:
        files_list = []

    files_list.append(new_entry)

    async with aiofiles.open(FILES_LIST_PATH, "w") as f:
        await f.write(json.dumps(files_list, indent=2))

async def reset_user_session(new_entry: Dict[str, Any]):
    """Reset the user session with a new entry."""
    session_entry = {
        **new_entry,
        "chat_history": [],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    async with aiofiles.open(USER_SESSION_FILE, "w") as f:
        await f.write(json.dumps(session_entry, indent=2))

async def clear_user_logs():
    """Clear the user logs file."""
    try:
        # First check if file exists
        if os.path.exists(USER_LOG_FILE):
            logger.info(f"Clearing existing log file: {USER_LOG_FILE}")
            # Open in write mode to truncate
            async with aiofiles.open(USER_LOG_FILE, "w") as log_file:
                # Write a newline to ensure file is created
                await log_file.write("\n")
            logger.info("Log file cleared successfully")
        else:
            logger.info(f"Log file does not exist, creating: {USER_LOG_FILE}")
            # Create new file with a newline
            async with aiofiles.open(USER_LOG_FILE, "w") as log_file:
                await log_file.write("\n")
            logger.info("New log file created")
    except Exception as e:
        logger.error(f"Error clearing log file: {str(e)}")
        raise

async def log_user_event(step: str, status: str, detail: str = ""):
    """Log a user event with the given step, status, and detail."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "status": status,
        "detail": detail
    }
    async with aiofiles.open(USER_LOG_FILE, "a") as log_file:
        await log_file.write(json.dumps(log_entry) + "\n")



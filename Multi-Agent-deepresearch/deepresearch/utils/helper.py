
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import uuid


def get_today_str() -> str:
    """Current date in human readable format"""
    return datetime.now().strftime("%a %b %-d, %Y")

def generate_session_id()->str:
    """Generate UUID for each session"""
    return str(uuid.uuid4())


def get_current_dir() -> Path:
    """Get the current directory of the module."""
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

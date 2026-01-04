from .agents import (
    Action,
    Agent,
    DoubaoAgent,
    DoubaoScaleWoBAgent,
    Language,
    RGMAgent,
)
from .api import APIClient
from .devices import AndroidDevice, Device
from .logging_config import setup_logging

# Initialize logging on package import
setup_logging(
    console_level="INFO",  # Clean terminal output
    file_level="DEBUG",  # Detailed file logs
    enable_rich_tracebacks=True,
    log_to_files=True,
)

__all__ = [
    "Device",
    "AndroidDevice",
    "APIClient",
    "Agent",
    "DoubaoAgent",
    "DoubaoScaleWoBAgent",
    "RGMAgent",
    "Action",
    "Language",
    "setup_logging",
]

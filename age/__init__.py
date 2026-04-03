from .agents import (
    Action,
    Agent,
    BrownieScaleWoBAgent,
    DoubaoAgent,
    DoubaoScaleWoBAgent,
    GeneralScaleWoBAgent,
    Language,
    MAIAgent,
    MAIScaleWoBAgent,
    MobileAgent35ScaleWoBAgent,
    OpenAutoGLMScaleWoBAgent,
    RGMAgent,
    RGMScaleWoBAgent,
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
    "APIClient",
    "Action",
    "Agent",
    "AndroidDevice",
    "BrownieScaleWoBAgent",
    "Device",
    "DoubaoAgent",
    "DoubaoScaleWoBAgent",
    "GeneralScaleWoBAgent",
    "Language",
    "MAIAgent",
    "MAIScaleWoBAgent",
    "MobileAgent35ScaleWoBAgent",
    "OpenAutoGLMScaleWoBAgent",
    "RGMAgent",
    "RGMScaleWoBAgent",
    "setup_logging",
]

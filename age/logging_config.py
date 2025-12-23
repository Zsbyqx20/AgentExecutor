"""Centralized logging configuration for the age package.

Provides dual-output logging:
- Console: Clean, focused INFO+ output with Rich formatting
- File: Detailed DEBUG+ logs with full context for debugging
"""

import logging.config
from pathlib import Path


def get_log_dir() -> Path:
    """Get or create the log directory."""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    enable_rich_tracebacks: bool = True,
    log_to_files: bool = True,
) -> None:
    """Configure logging for the entire application.

    Args:
        console_level: Minimum level for terminal output (INFO recommended).
        file_level: Minimum level for file output (DEBUG recommended).
        enable_rich_tracebacks: Use Rich's beautiful exception formatting.
        log_to_files: Enable file logging (disable for testing).
    """
    log_dir = get_log_dir()

    handlers = {
        "console": {
            "class": "rich.logging.RichHandler",
            "level": console_level,
            "formatter": "console",
            "rich_tracebacks": enable_rich_tracebacks,
            "tracebacks_show_locals": True,
            "tracebacks_extra_lines": 3,
            "show_time": True,
            "show_level": True,
            "show_path": False,  # Cleaner console output
            "markup": True,  # Allow Rich markup in log messages
        },
    }

    formatters = {
        "console": {"format": "%(message)s", "datefmt": "[%X]"},
        "file": {
            "format": (
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "%(filename)s:%(funcName)s:%(lineno)d | %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }

    if log_to_files:
        handlers["debug_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": file_level,
            "formatter": "file",
            "filename": str(log_dir / "agent_debug.log"),
            "maxBytes": 10_000_000,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        handlers["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "file",
            "filename": str(log_dir / "agent_errors.log"),
            "maxBytes": 10_000_000,
            "backupCount": 10,  # Keep more error logs
            "encoding": "utf-8",
        }

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": (
                    ["console", "debug_file", "error_file"]
                    if log_to_files
                    else ["console"]
                ),
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)

"""Logger factory for the age package.

Provides a consistent way to get module-level loggers with proper configuration.
"""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name. Typically __name__ for automatic module namespacing.
              If None, uses the caller's module name.

    Returns:
        A configured logger instance.

    Example:
        >>> from age.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello, world!")
    """
    if name is None:
        # Get caller's module name
        import inspect

        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            name = frame.f_back.f_globals.get("__name__", "age")
        else:
            name = "age"

    return logging.getLogger(name)

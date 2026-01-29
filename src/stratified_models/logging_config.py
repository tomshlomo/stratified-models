"""Logging configuration utilities for structlog."""

import logging
from collections.abc import Mapping

import structlog

# Custom level for silent mode (above CRITICAL)
SILENT = logging.CRITICAL + 10


def configure_logging(
    *,
    level: int | str = logging.INFO,
    module_levels: Mapping[str, int | str] | None = None,
) -> None:
    """Configure structlog with stdlib integration.

    This should only be called from application entry points (scripts, CLIs),
    not from library code.

    Args:
        level: Root log level. Use SILENT or "SILENT" for silent mode.
        module_levels: Optional mapping of module names to their log levels.
            Example: {"stratified_models.tuning": logging.WARNING}

    """
    if isinstance(level, str):
        level = _parse_level(level)

    # For structlog filtering, cap at CRITICAL (structlog doesn't support higher)
    structlog_level = min(level, logging.CRITICAL)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=level,
        force=True,
    )

    # Apply module-specific levels
    if module_levels:
        for module_name, module_level in module_levels.items():
            parsed_level = (
                _parse_level(module_level)
                if isinstance(module_level, str)
                else module_level
            )
            logging.getLogger(module_name).setLevel(parsed_level)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(structlog_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _parse_level(level_str: str) -> int:
    """Parse a log level string to an integer."""
    level_upper = level_str.upper()
    if level_upper == "SILENT":
        return SILENT
    result = logging.getLevelName(level_upper)
    if isinstance(result, int):
        return result
    msg = f"Unknown log level: {level_str}"
    raise ValueError(msg)

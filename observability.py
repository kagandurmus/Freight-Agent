"""Structured logging.

`structlog` is configured so that *every* log line in the process -- ours, uvicorn's, the
OpenAI SDK's -- travels through one processor chain and one renderer. Without the stdlib
integration below, our structured events and the libraries' plain strings interleave in two
different formats, which is precisely the situation where nobody reads the logs during an
incident.

`TRIAGE_LOG_JSON=true` switches to JSON lines for a log shipper; the console renderer is for
humans and colours only when stderr is a TTY.
"""

from __future__ import annotations

import logging
import sys

import structlog

__all__ = ["configure_logging", "get_logger"]

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Install one processor chain for the whole process. Safe to call more than once."""
    level_value = _LEVELS.get(level.upper(), logging.INFO)

    shared_processors: list[object] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: object = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level_value),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level_value)

    # Route library loggers through our formatter instead of uvicorn's own.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True

    # These log one line per HTTP request. Useful when debugging the transport, ruinous when
    # you are trying to watch a triage decision go by, so they stay quiet unless asked.
    for name in ("httpx", "httpcore", "openai", "instructor"):
        quiet_logger = logging.getLogger(name)
        quiet_logger.handlers.clear()
        quiet_logger.propagate = True
        quiet_logger.setLevel(max(level_value, logging.WARNING))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

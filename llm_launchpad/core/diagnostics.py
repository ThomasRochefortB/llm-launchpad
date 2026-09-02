"""Local debug logging for Launchpad operations.

Launchpad writes a small rotating debug log under ``~/.llm_launchpad/logs/``
so silent failure paths (cache persistence, auth probes, subprocess cleanup)
leave a trace for troubleshooting without touching the user-facing output.
Secrets such as bearer keys and tokens must never be written to the log.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3

LOG_DIR = Path.home() / ".llm_launchpad" / "logs"
LOG_FILE = LOG_DIR / "llm_launchpad.log"

_LOGGER_NAME = "llm_launchpad"
_configured = False


def setup_logging(log_dir: Path | None = None) -> Path | None:
    """Attach a rotating debug file handler to the Launchpad logger.

    Idempotent: repeated calls are no-ops. The file is created lazily on the
    first record (``delay=True``), and failures fall back to a null handler so
    a read-only home directory never breaks the CLI or TUI. Returns the log
    file path when a handler is active, otherwise ``None``.
    """

    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        for handler in logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                return Path(handler.baseFilename)
        return None

    target_dir = Path(log_dir) if log_dir is not None else LOG_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            target_dir / "llm_launchpad.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            delay=True,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    except OSError:
        _configured = True
        logger.addHandler(logging.NullHandler())
        return None

    handler.name = "launchpad_debug_file"
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True
    return Path(handler.baseFilename)


def log_file_path() -> Path | None:
    """Return the active debug log path, or ``None`` when logging is inactive."""

    for handler in logging.getLogger(_LOGGER_NAME).handlers:
        if isinstance(handler, RotatingFileHandler):
            return Path(handler.baseFilename)
    return None


def log_debug(message: str) -> None:
    """Record a debug note on the Launchpad logger."""

    logging.getLogger(_LOGGER_NAME).debug(message)


def log_exception(context: str) -> None:
    """Record the in-flight exception with traceback under a human context."""

    logging.getLogger(_LOGGER_NAME).debug(context, exc_info=True)

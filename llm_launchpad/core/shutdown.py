"""Process-wide cooperative shutdown signal."""

from __future__ import annotations

import threading

_shutdown_event = threading.Event()


def is_shutting_down() -> bool:
    return _shutdown_event.is_set()


def request_shutdown() -> None:
    _shutdown_event.set()


def shutdown_event() -> threading.Event:
    return _shutdown_event

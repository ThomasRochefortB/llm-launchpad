"""Emit the standard event pair that ends a failed operation.

Every operation in ``Orchestrator`` and ``WarmupRunner`` reports failure the
same way: an ``ErrorEvent`` carrying the message, immediately followed by an
``OperationCompleteEvent`` repeating it as the completion detail. Writing that
pair by hand at two dozen call sites made it easy to drift -- to forget the
completion event, to mark it ``success=True``, or to let the two disagree about
which operation failed. Yielding from one helper makes the pairing structural.
"""

from __future__ import annotations

from collections.abc import Generator

from ..protocol.enums import OperationType
from ..protocol.events import BaseEvent, ErrorEvent, OperationCompleteEvent


def fail_operation(
    operation: OperationType,
    message: str,
    *,
    exit_code: int = 1,
    recoverable: bool = True,
    detail: str | None = None,
    error_exit_code: int | None = None,
) -> Generator[BaseEvent, None, None]:
    """Yield the error and completion events that end ``operation`` in failure.

    ``detail`` defaults to ``message``; pass it explicitly when the completion
    detail differs, or pass ``""`` to leave it empty.
    """
    yield ErrorEvent(
        message=message,
        operation=operation,
        exit_code=error_exit_code,
        recoverable=recoverable,
    )
    yield OperationCompleteEvent(
        operation=operation,
        success=False,
        exit_code=exit_code,
        detail=message if detail is None else detail,
    )

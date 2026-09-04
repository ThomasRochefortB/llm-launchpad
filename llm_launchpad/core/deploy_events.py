"""Read an endpoint URL out of a deploy event stream.

A provider announces its endpoint on several channels at once, and they do not
agree. Modal prints candidate URLs in ordinary log output, emits the real one
on an event, and offers an ephemeral "-dev" address alongside the deployed one.
Picking the wrong candidate means certifying against an endpoint that is not
the one being published.

The TUI worked this out once; every other consumer of ``Orchestrator`` has had
to rediscover it, and got it wrong -- a headless harness certified against a
``-dev`` URL and hung waiting for readiness. Resolving it in one place makes
that precedence a property of the codebase rather than of whoever wrote the
caller.
"""

from __future__ import annotations

from ..protocol.enums import OperationType
from ..protocol.events import (
    EndpointAvailableEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
)

# Highest wins. Events carry the provider's own answer; log lines are scraped,
# and among those the announced function beats incidental guidance URLs while
# the ephemeral -dev address loses to the deployed one.
_PRIORITY_EVENT = 3
_PRIORITY_ANNOUNCED = 2
_PRIORITY_ANNOUNCED_DEV = 1
_PRIORITY_INCIDENTAL = 0


class EndpointUrlResolver:
    """Accumulate the best endpoint URL seen so far in a deploy stream."""

    def __init__(self) -> None:
        self._url = ""
        self._priority = -1

    @property
    def url(self) -> str:
        """The best candidate seen, or an empty string if none has appeared."""

        return self._url

    def observe(self, event: object) -> None:
        """Fold one deploy event into the current best candidate."""

        if isinstance(event, LogEvent):
            self._observe_line(event.line or "")
        elif isinstance(event, EndpointAvailableEvent):
            self._offer(getattr(event.endpoint, "web_url", "") or "", _PRIORITY_EVENT)
        elif isinstance(event, OperationCompleteEvent):
            self._offer(getattr(event.data, "web_url", "") or "", _PRIORITY_EVENT)

    def _observe_line(self, line: str) -> None:
        from .backend import ModalBackend

        found = ModalBackend.extract_modal_web_url(line)
        if not found:
            return
        if "Created web function" in line:
            priority = (
                _PRIORITY_ANNOUNCED_DEV
                if found.endswith("-dev.modal.run")
                else _PRIORITY_ANNOUNCED
            )
        else:
            priority = _PRIORITY_INCIDENTAL
        self._offer(found, priority)

    def _offer(self, url: str, priority: int) -> None:
        if url and priority >= self._priority:
            self._url = url
            self._priority = priority


def failure_detail(
    event: OperationCompleteEvent,
    errors: list[str],
) -> str:
    """Explain a failed operation, falling back to the errors that preceded it.

    Several failure paths complete without a ``detail``; each emits an
    ``ErrorEvent`` first carrying the real reason. A consumer that watches only
    completions reports a failure with nothing after the colon.
    """

    if event.detail:
        return event.detail
    joined = "; ".join(message for message in errors if message)
    return joined or "no reason reported"


def collect_error(event: object, errors: list[str]) -> bool:
    """Record an ErrorEvent's message. Returns whether the event was one."""

    if isinstance(event, ErrorEvent) and event.message:
        errors.append(event.message)
        return True
    return False


def is_deploy_completion(event: object) -> bool:
    """Whether an event terminates the deploy operation."""

    return (
        isinstance(event, OperationCompleteEvent)
        and event.operation == OperationType.DEPLOY
    )

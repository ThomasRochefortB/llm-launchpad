"""Detached deployment worker: runs one durable job to completion.

Launched as ``python -m llm_launchpad.core.job_runner <job-id>`` with a new
session so closing the TUI does not kill it or its provider subprocesses
(modal CLI, SSH tunnels). All progress goes to the job store; the TUI and CLI
tail it. Cancellation is cooperative: the worker observes the persisted flag
between provider calls, captures any allocated resource id, and cleans up.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

from typing import TYPE_CHECKING, Any

from ..protocol.enums import OperationType
from ..protocol.events import (
    BaseEvent,
    EndpointAvailableEvent,
    LogEvent,
    OperationCompleteEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo
from .diagnostics import log_exception
from .resource_targeting import NAME_ADDRESSABLE_PROVIDERS as _NAME_ADDRESSABLE_PROVIDERS_UNUSED  # noqa: F401  (compat re-export)

if TYPE_CHECKING:
    from .job_store import JobStore
    from .orchestrator import Orchestrator


def spawn_job_worker(job_id: str, store_path: str | None = None) -> int:
    """Launch a detached worker for one job; returns its pid."""
    import subprocess

    cmd = [sys.executable, "-m", "llm_launchpad.core.job_runner", job_id]
    if store_path:
        cmd += ["--store", store_path]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    return proc.pid


def _heartbeat_loop(store: JobStore, job_id: str, pid: int, stop: threading.Event) -> None:
    while not stop.wait(5.0):
        try:
            store.heartbeat_job(job_id, pid)
        except Exception:
            pass


def _endpoint_from_event(event: BaseEvent) -> EndpointInfo | None:
    if isinstance(event, EndpointAvailableEvent):
        return event.endpoint
    if isinstance(event, OperationCompleteEvent) and isinstance(event.data, EndpointInfo):
        return event.data
    return None


def run_job(job_id: str, store: JobStore | None = None) -> int:
    """Run one job; returns a process exit code (0 on terminal bookkeeping)."""
    from .job_store import JobStore

    store = store or JobStore()
    record = store.get_job(job_id)
    if record is None:
        print(f"Unknown job {job_id}", file=sys.stderr)
        return 2
    if record.status not in ("pending", "running"):
        return 0
    pid = os.getpid()
    if record.status == "pending" and not store.claim_job(job_id, pid):
        # Cancelled before start, or claimed by another worker: no duplicate rental.
        return 0
    if (
        record.status == "running"
        and record.worker_pid not in (None, pid)
        and store.pid_alive(record.worker_pid)
    ):
        # Another worker owns it; never run allocation twice.
        return 0
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop, args=(store, job_id, pid), kwargs={"stop": stop_heartbeat}, daemon=True
    )
    heartbeat.start()
    try:
        return _run_claimed_job(job_id, store, pid)
    finally:
        stop_heartbeat.set()


class _WorkerCancelled(Exception):
    """Cooperative cancellation observed between provider calls."""


def _run_claimed_job(job_id: str, store: JobStore, pid: int) -> int:
    from .deploy_journal import record_in_flight
    from .deploy_journal import InFlightDeployment
    from .job_store import JobStatus

    record = store.get_job(job_id)
    if record is None:
        return 2
    config = record.config
    orchestrator = _orchestrator()
    # Legacy journal for previous-session recovery tools; the job store is authoritative.
    try:
        record_in_flight(
            InFlightDeployment(
                app_name=config.app_name or "",
                provider=config.provider.value,
                backend=config.backend.value,
                instance_name=config.instance_name,
                gpu_type=config.gpu_type,
                gpu_count=max(1, int(config.gpu_count or 1)),
                started_at_epoch=time.time(),
            )
        )
    except Exception:
        pass
    initial_resource: str | None = record.resource_app_id
    # The approved ladder is snapshotted once; the lifecycle owns progression
    # through explicit retry_allowed outcomes. Resource identity resets per
    # attempt: the next placement must allocate before it can be cancelled.
    attempt_configs: list[DeploymentConfig] = [config, *list(config.fallback_configs)]
    from .deployment_lifecycle import (
        LifecycleAttempt,
        LifecycleCallbacks,
        LifecycleOptions,
        run_lifecycle,
    )
    from ..protocol.enums import AttemptDisposition

    worker_resource: str | None = initial_resource
    worker_url: str | None = None
    worker_endpoint: EndpointInfo | None = None
    worker_index: int = 0

    def _on_event(event: BaseEvent) -> None:
        try:
            store.append_event(job_id, event)
        except Exception:
            pass

    def _on_resource(new_resource: str | None) -> None:
        nonlocal worker_resource
        if new_resource:
            worker_resource = new_resource
            try:
                store.set_resource(job_id, new_resource)
            except Exception:
                pass

    def _on_connection(
        saved_config: DeploymentConfig,
        url: str | None,
        endpoint: EndpointInfo | None,
    ) -> None:
        nonlocal worker_url, worker_endpoint
        if endpoint is not None:
            worker_endpoint = endpoint
            if endpoint.web_url:
                worker_url = endpoint.web_url
        elif url:
            worker_url = url
        try:
            _save_connection(saved_config, worker_url, worker_endpoint)
        except Exception:
            pass

    def _on_attempt_start(index: int, attempt_config: DeploymentConfig) -> None:
        nonlocal worker_index, worker_resource, worker_url, worker_endpoint
        worker_index = index
        # A previous attempt's resource must not leak into the next one.
        if index > 0:
            worker_resource = None
            worker_url = None
            worker_endpoint = None
            try:
                store.update_config(job_id, attempt_config)
                store.append_event(
                    job_id,
                    LogEvent(
                        line=f"Trying the next approved placement ({index + 1}/{len(attempt_configs)}).",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                    ),
                )
            except Exception:
                pass

    def _on_attempt_finish(index: int, outcome: object) -> None:
        from ..protocol.models import DeploymentAttemptOutcome

        nonlocal worker_resource, worker_url, worker_endpoint
        if isinstance(outcome, DeploymentAttemptOutcome):
            if outcome.resource_app_id:
                worker_resource = outcome.resource_app_id
            if outcome.endpoint_url:
                worker_url = outcome.endpoint_url
            if outcome.endpoint is not None:
                worker_endpoint = outcome.endpoint

    from .deployment_preflight import lifecycle_attempts_for_configs

    attempts = [
        LifecycleAttempt(
            config=spec.config,
            retry_allowed=spec.retry_allowed,
            plan=spec.plan,
        )
        for spec in lifecycle_attempts_for_configs(attempt_configs)
    ]
    try:
        if store.is_cancel_requested(job_id):
            _cancel_with_cleanup(
                job_id, store, attempt_configs[0], initial_resource,
                "Cancelled before starting placement.", allocated=False,
            )
            _clear_journal(attempt_configs[0])
            return 0
        result = run_lifecycle(
            orchestrator,
            attempts,
            options=LifecycleOptions(warmup_timeout_seconds=1800, tail_logs=True),
            callbacks=LifecycleCallbacks(
                on_event=_on_event,
                is_cancelled=lambda: store.is_cancel_requested(job_id),
                on_resource=_on_resource,
                on_connection=_on_connection,
                on_credentials=_on_connection,
                on_attempt_start=_on_attempt_start,
                on_attempt_finish=_on_attempt_finish,
            ),
        )
    except Exception as exc:
        log_exception("Deployment worker failed")
        try:
            store.append_event(
                job_id,
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=False, detail=str(exc)),
            )
        except Exception:
            pass
        store.finish_job(job_id, JobStatus.FAILED, f"Failed — check resource in Manage. {exc}".strip())
        return 0
    resource_app_id: str | None = worker_resource
    observed_url: str | None = worker_url
    current_index: int = worker_index
    current = attempt_configs[min(current_index, len(attempt_configs) - 1)]
    outcome = result.attempts[-1] if result.attempts else None
    if outcome is not None and outcome.disposition == AttemptDisposition.CANCELLED:
        _finish_cancelled_from_lifecycle(job_id, store, current, outcome)
        _clear_journal(current)
        return 0
    if result.succeeded:
        try:
            store.update_config(job_id, current)
        except Exception:
            pass
        if observed_url:
            try:
                _emit_connection_summary(store, job_id, current, observed_url)  # type: ignore[arg-type]
            except Exception:
                pass
        store.finish_job(
            job_id, JobStatus.SUCCEEDED, "Finished — open result",
            result={"url": observed_url, "app_name": current.app_name},
        )
        _clear_journal(current)
        return 0
    failure_detail = (
        outcome.failure_detail if outcome and outcome.failure_detail
        else "Deployment failed."
    )
    if outcome is not None and outcome.disposition == AttemptDisposition.RETAINED:
        current.retry_allowed = False
        try:
            store.update_config(job_id, current)
        except Exception:
            pass
        store.finish_job(
            job_id, JobStatus.FAILED, f"Failed — check resource in Manage. {failure_detail}".strip(),
        )
        return 0
    if outcome is not None and not outcome.retry_allowed:
        current.retry_allowed = False
        try:
            store.update_config(job_id, current)
        except Exception:
            pass
    store.finish_job(
        job_id, JobStatus.FAILED, f"Failed — check resource in Manage. {failure_detail}".strip(),
    )
    if not resource_app_id:
        _clear_journal(current)
    return 0


def _run_attempt_phases(
    job_id: str,
    store: JobStore,
    orchestrator: Orchestrator,
    config: DeploymentConfig,
    resource_app_id: str | None,
) -> tuple[str | None, str | None]:
    """Run deploy (+warmup) for one placement through the shared lifecycle.

    Returns (resource_app_id, failure_detail); failure None means the job
    finished SUCCEEDED. Raises _WorkerCancelled after persisting cleanup.
    """

    from .deployment_lifecycle import (
        LifecycleAttempt,
        LifecycleCallbacks,
        LifecycleOptions,
        run_lifecycle,
    )
    from .job_store import JobStatus
    from ..protocol.enums import AttemptDisposition

    observed_url: str | None = None
    observed_endpoint: EndpointInfo | None = None

    def _on_event(event: BaseEvent) -> None:
        store.append_event(job_id, event)

    def _on_resource(new_resource: str | None) -> None:
        nonlocal resource_app_id
        if new_resource:
            resource_app_id = new_resource
            store.set_resource(job_id, new_resource)

    def _on_connection(
        saved_config: DeploymentConfig,
        url: str | None,
        endpoint: EndpointInfo | None,
    ) -> None:
        nonlocal observed_url, observed_endpoint
        # The authoritative endpoint wins over incidental URLs: prefer the
        # last endpoint object with a URL, and persist warmup replacements.
        if endpoint is not None:
            observed_endpoint = endpoint
            if endpoint.web_url:
                observed_url = endpoint.web_url
        elif url:
            observed_url = url
        _save_connection(saved_config, observed_url, observed_endpoint)

    result = run_lifecycle(
        orchestrator,
        [LifecycleAttempt(config=config, retry_allowed=config.retry_allowed)],
        options=LifecycleOptions(warmup_timeout_seconds=1800, tail_logs=True),
        callbacks=LifecycleCallbacks(
            on_event=_on_event,
            is_cancelled=lambda: store.is_cancel_requested(job_id),
            on_resource=_on_resource,
            on_connection=_on_connection,
            # Pre-certification credential save uses the same persistence;
            # publication (summaries, OpenCode sync) happens once in the job
            # finalization below from the verified outcome.
            on_credentials=_on_connection,
        ),
    )
    outcome = result.attempts[0] if result.attempts else None
    if outcome is not None and outcome.resource_app_id:
        resource_app_id = outcome.resource_app_id
    if outcome is not None and outcome.endpoint_url:
        observed_url = outcome.endpoint_url
    if outcome is not None and outcome.endpoint is not None:
        observed_endpoint = outcome.endpoint
    if outcome is not None and outcome.disposition == AttemptDisposition.CANCELLED:
        _finish_cancelled_from_lifecycle(job_id, store, config, outcome)
        raise _WorkerCancelled
    if store.is_cancel_requested(job_id):
        raise _WorkerCancelled
    if result.succeeded:
        try:
            store.update_config(job_id, config)
        except Exception:
            pass
        if observed_url:
            _emit_connection_summary(store, job_id, config, observed_url)
        store.finish_job(
            job_id, JobStatus.SUCCEEDED, "Finished — open result",
            result={"url": observed_url, "app_name": config.app_name},
        )
        return resource_app_id, None
    if outcome is not None and outcome.disposition == AttemptDisposition.RETAINED:
        # Kept for inspection: no retry, but the resource stays reachable.
        config.retry_allowed = False
        try:
            store.update_config(job_id, config)
        except Exception:
            pass
        return resource_app_id, outcome.failure_detail or "Certification failed."
    if outcome is not None and not outcome.retry_allowed:
        config.retry_allowed = False
        try:
            store.update_config(job_id, config)
        except Exception:
            pass
    return resource_app_id, (
        outcome.failure_detail if outcome and outcome.failure_detail
        else "Deployment failed."
    )


# Providers whose teardown is addressed by name. `modal app stop` takes the app
# name, and a Vast destroy resolves its rental from the record it persisted
# under that name -- so for these the name is the handle and no allocation
# event has to arrive first. Prime is the exception: termination genuinely
# needs a pod id, and it emits one the moment the pod exists.
# Canonical addressing lives in core.resource_targeting; the name below stays
# importable for backwards compatibility while callers migrate.
_NAME_ADDRESSABLE_PROVIDERS = _NAME_ADDRESSABLE_PROVIDERS_UNUSED


def _stop_target(
    store: JobStore | Any,
    job_id: str,
    config: DeploymentConfig,
    resource_app_id: str | None,
    *,
    allocated: bool,
) -> tuple[str | None, bool]:
    """Resolve what a cancellation can stop: (resource id, anything to stop).

    The id is re-read from the job when the caller does not have one to hand:
    the warmup path only sees whatever endpoint it was given, while the deploy
    phase already banked the id, and throwing that away would strand a live
    pod. Addressing policy lives in :mod:`core.resource_targeting`.
    """
    from .resource_targeting import resolve_stop_target

    identifier = (resource_app_id or "").strip()
    if not identifier:
        try:
            record = store.get_job(job_id)
        except Exception:
            record = None
        identifier = ((record.resource_app_id if record else "") or "").strip()
    target = resolve_stop_target(
        provider=config.provider,
        app_name=config.app_name,
        resource_id=identifier or None,
        allocated=allocated,
    )
    return target.resource_id, target.stoppable


def _cancel_with_cleanup(
    job_id: str,
    store: JobStore,
    config: DeploymentConfig,
    resource_app_id: str | None,
    prefix: str,
    *,
    allocated: bool = True,
) -> None:
    from .job_store import JobStatus

    def _terminal(detail: str) -> None:
        try:
            store.append_event(
                job_id,
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=False, detail=detail),
            )
        except Exception:
            pass

    resource_app_id, stoppable = _stop_target(
        store, job_id, config, resource_app_id, allocated=allocated
    )
    # A Modal deploy never emits an allocation event -- its app is addressed by
    # name -- so requiring an id here skipped teardown entirely and told the
    # reader nothing had been allocated while the app stayed published.
    if not config.do_deploy or not stoppable:
        detail = "Cancelled before allocating a resource."
        _terminal(detail)
        store.finish_job(job_id, JobStatus.CANCELLED, detail)
        return
    # The caller says which wait was interrupted; saying so is the difference
    # between a log that shows the cleanup happening and one that jumps
    # straight to its verdict.
    try:
        store.append_event(job_id, LogEvent(line=prefix, operation=OperationType.STOP))
    except Exception:
        pass
    try:
        orchestrator = _orchestrator()
        confirmed = False
        for event in orchestrator.stop_app(
            config.backend, app_name=config.app_name, app_id=resource_app_id, provider=config.provider
        ):
            store.append_event(job_id, event)
            if isinstance(event, OperationCompleteEvent) and event.operation == OperationType.STOP:
                confirmed = bool(event.success)
                if not event.success:
                    store.finish_job(
                        job_id, JobStatus.CANCELLED,
                        "Cancellation cleanup failed; check Manage. Recovery record retained.",
                        cleanup_error=event.detail,
                    )
                    return
        if confirmed:
            _terminal("Cancelled; resource stopped.")
            store.finish_job(job_id, JobStatus.CANCELLED, "Cancelled; resource stopped.")
        else:
            _terminal("Cancellation cleanup failed; check Manage. Recovery record retained.")
            store.finish_job(
                job_id, JobStatus.CANCELLED,
                "Cancellation cleanup failed; check Manage. Recovery record retained.",
                cleanup_error="Stop did not confirm.",
            )
    except Exception as exc:
        _terminal("Cancellation cleanup failed; check Manage. Recovery record retained.")
        store.finish_job(
            job_id, JobStatus.CANCELLED,
            "Cancellation cleanup failed; check Manage. Recovery record retained.",
            cleanup_error=str(exc),
        )


def _finish_cancelled_from_lifecycle(
    job_id: str,
    store: JobStore,
    config: DeploymentConfig,
    outcome: object,
) -> None:
    """Persist a lifecycle-observed cancellation with its cleanup outcome."""
    from .job_store import JobStatus
    from ..protocol.enums import CleanupDisposition

    cleanup = getattr(outcome, "cleanup", CleanupDisposition.UNKNOWN)
    cleanup_error = getattr(outcome, "cleanup_error", None)
    failure_detail = getattr(outcome, "failure_detail", None) or "Cancelled."
    if cleanup == CleanupDisposition.CONFIRMED or (
        isinstance(cleanup, str) and cleanup == CleanupDisposition.CONFIRMED.value
    ):
        try:
            store.append_event(
                job_id,
                OperationCompleteEvent(
                    operation=OperationType.DEPLOY, success=False,
                    detail="Cancelled; resource stopped.",
                ),
            )
        except Exception:
            pass
        store.finish_job(job_id, JobStatus.CANCELLED, "Cancelled; resource stopped.")
        return
    if not config.do_deploy:
        detail = "Cancelled before allocating a resource."
        try:
            store.append_event(
                job_id,
                OperationCompleteEvent(
                    operation=OperationType.DEPLOY, success=False, detail=detail
                ),
            )
        except Exception:
            pass
        store.finish_job(job_id, JobStatus.CANCELLED, detail)
        return
    try:
        store.append_event(
            job_id,
            OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=False,
                detail="Cancellation cleanup failed; check Manage. Recovery record retained.",
            ),
        )
    except Exception:
        pass
    store.finish_job(
        job_id, JobStatus.CANCELLED,
        "Cancellation cleanup failed; check Manage. Recovery record retained.",
        cleanup_error=cleanup_error or failure_detail,
    )


def _save_connection(config: DeploymentConfig, url: str | None, endpoint: EndpointInfo | None) -> None:
    if not url:
        return
    try:
        from .connection_store import save_connection

        save_connection(config, endpoint or EndpointInfo(name=config.app_name or "", web_url=url))
    except Exception:
        log_exception("Could not cache deploy connection summary")


def _emit_connection_summary(store: JobStore, job_id: str, config: DeploymentConfig, url: str) -> None:
    """Append the OpenAI-compatible summary block to the durable log.

    Reopened monitors and headless `jobs follow` then show the same block the
    foreground flow always printed.
    """
    try:
        from .opencode import format_connection_summary_lines

        for line in format_connection_summary_lines(config, url):
            store.append_event(job_id, LogEvent(line=line))
    except Exception:
        log_exception("Could not record deploy connection summary")


def _clear_journal(config: DeploymentConfig) -> None:
    try:
        from .deploy_journal import clear_in_flight

        if (config.app_name or "").strip():
            clear_in_flight(config.app_name or "", provider=config.provider.value, backend=config.backend.value)
    except Exception:
        pass


def _orchestrator() -> Orchestrator:
    from .orchestrator import Orchestrator

    return Orchestrator()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one durable deployment job.")
    parser.add_argument("job_id", help="Stable job id from the job store.")
    parser.add_argument("--store", default=None, help="Override job database path.")
    args = parser.parse_args(argv)
    from pathlib import Path

    from .job_store import JobStore

    store = JobStore(Path(args.store) if args.store else None)
    return run_job(args.job_id, store)


if __name__ == "__main__":
    raise SystemExit(main())

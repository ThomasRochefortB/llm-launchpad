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
from pathlib import Path

from typing import TYPE_CHECKING, Any

from ..protocol.enums import OperationType
from ..protocol.events import (
    BaseEvent,
    EndpointAvailableEvent,
    LogEvent,
    OperationCompleteEvent,
    ResourceAllocatedEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo
from .diagnostics import log_exception

if TYPE_CHECKING:
    from .job_store import JobStore
    from .orchestrator import Orchestrator


def _store_for(path: str | None) -> JobStore:
    from .job_store import JobStore

    return JobStore(Path(path) if path else None)


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
    resource_app_id: str | None = record.resource_app_id
    attempt_configs: list[DeploymentConfig] = [config, *list(config.fallback_configs)]
    attempt_index = 0
    try:
        while attempt_index < len(attempt_configs):
            current = attempt_configs[attempt_index]
            if store.is_cancel_requested(job_id):
                _cancel_with_cleanup(job_id, store, current, resource_app_id, "Cancelled before starting placement.")
                _clear_journal(current)
                return 0
            if attempt_index > 0:
                store.update_config(job_id, current)
                store.append_event(
                    job_id,
                    LogEvent(
                        line=f"Trying the next approved placement ({attempt_index + 1}/{len(attempt_configs)}).",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                    ),
                )
            try:
                resource_app_id, failure_detail = _run_attempt_phases(
                    job_id, store, orchestrator, current, resource_app_id
                )
            except _WorkerCancelled:
                _clear_journal(current)
                return 0
            if failure_detail is None:
                _clear_journal(current)
                return 0
            attempt_index += 1
            if attempt_index >= len(attempt_configs):
                store.finish_job(
                    job_id, JobStatus.FAILED, f"Failed — check resource in Manage. {failure_detail}".strip(),
                )
                # Failed deploys keep their journal entry when a resource may exist;
                # a clean failure with no resource clears it.
                if not resource_app_id:
                    _clear_journal(current)
                return 0
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
    return 0


def _run_attempt_phases(
    job_id: str,
    store: JobStore,
    orchestrator: Orchestrator,
    config: DeploymentConfig,
    resource_app_id: str | None,
) -> tuple[str | None, str | None]:
    """Run deploy (+warmup) for one placement.

    Returns (resource_app_id, failure_detail); failure None means the job
    finished SUCCEEDED. Raises _WorkerCancelled after persisting cleanup.
    """

    from .job_store import JobStatus

    observed_url: str | None = None
    observed_endpoint: EndpointInfo | None = None
    deploy_succeeded = False
    failure_detail = ""
    for event in orchestrator.deploy(config):
        endpoint = _endpoint_from_event(event)
        if endpoint is not None:
            if endpoint.web_url:
                observed_url = endpoint.web_url
            observed_endpoint = endpoint
            if endpoint.app_id:
                resource_app_id = endpoint.app_id
                store.set_resource(job_id, resource_app_id)
        if isinstance(event, ResourceAllocatedEvent) and event.app_id:
            resource_app_id = event.app_id
            store.set_resource(job_id, resource_app_id)
        store.append_event(job_id, event)
        if store.is_cancel_requested(job_id):
            _cancel_with_cleanup(job_id, store, config, resource_app_id, "Cancelled; cleaning up allocated resource.")
            raise _WorkerCancelled
        if isinstance(event, OperationCompleteEvent) and event.operation == OperationType.DEPLOY:
            if event.success:
                deploy_succeeded = True
            else:
                failure_detail = event.detail or "Deployment failed."
    if not deploy_succeeded:
        return resource_app_id, failure_detail
    # Persist credentials immediately: a later warmup failure must not strand
    # a live, billable resource without its generated bearer key.
    _save_connection(config, observed_url, observed_endpoint)
    if config.do_warmup and config.do_deploy:
        completed_url = _run_warmup_phase(job_id, store, orchestrator, config, observed_url, observed_endpoint)
        if completed_url is None:
            # Warmup failed or was cancelled; _run_warmup_phase already
            # recorded the outcome (or raised _WorkerCancelled).
            if store.is_cancel_requested(job_id):
                raise _WorkerCancelled
            return resource_app_id, "Certification failed."
        observed_url = completed_url
        _save_connection(config, observed_url, observed_endpoint)
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


def _run_warmup_phase(
    job_id: str,
    store: JobStore,
    orchestrator: Orchestrator,
    config: DeploymentConfig,
    observed_url: str | None,
    observed_endpoint: EndpointInfo | None,
) -> str | None:
    """Warm one deployed placement; returns the verified URL or None.

    None covers warmup failure (after best-effort cleanup of the failed
    resource, mirroring the TUI) and cancellation (via _WorkerCancelled).
    """

    from ..protocol.enums import ComputeProvider
    from ..protocol.events import ErrorEvent, StateChangeEvent
    from ..protocol.enums import DeploymentState

    url = observed_url
    if not url and config.provider == ComputeProvider.MODAL:
        try:
            from .backend import ModalBackend

            username = ModalBackend.get_username()
        except Exception:
            username = None
        if username:
            try:
                from .backend import ModalBackend as _ModalBackend

                url = _ModalBackend.default_server_url(
                    username,
                    app_name=config.app_name,
                    function_slug=config.function_slug,
                )
            except Exception:
                url = None
    if not url:
        detail = "Provider returned no endpoint URL."
        store.append_event(job_id, ErrorEvent(message=detail, operation=OperationType.WARMUP))
        return None
    certification_kwargs: dict[str, Any] = {}
    if config.serving_requirements is not None:
        certification_kwargs = {
            "serving_requirements": config.serving_requirements,
            "placement_assessment": config.placement_assessment,
            "runtime_id": config.llamacpp_runtime_id,
        }
    if config.provider != ComputeProvider.MODAL or config.endpoint_api_key:
        certification_kwargs.update(
            provider=config.provider,
            api_key=config.endpoint_api_key,
            pod_id=observed_endpoint.app_id if observed_endpoint else None,
        )
    if config.vision is not None:
        certification_kwargs["vision"] = config.vision
    completed_url = url
    warmup_ok = False
    for event in orchestrator.warmup(
        config.backend,
        url,
        1800,
        True,
        app_name=config.app_name,
        served_model_name=config.served_model_name,
        **certification_kwargs,  # type: ignore[arg-type]
    ):
        if (
            isinstance(event, OperationCompleteEvent)
            and event.success
            and event.operation == OperationType.WARMUP
        ):
            warmup_ok = True
            if isinstance(event.data, dict):
                maybe_url = event.data.get("url")
                if isinstance(maybe_url, str) and maybe_url.strip():
                    completed_url = maybe_url.strip()
                attestation = event.data.get("attestation")
                if attestation is not None:
                    config.runtime_attestation = attestation
                    if observed_endpoint is not None:
                        observed_endpoint.runtime_attestation = attestation
        store.append_event(job_id, event)
        if store.is_cancel_requested(job_id):
            _cancel_with_cleanup(job_id, store, config, observed_endpoint.app_id if observed_endpoint else None, "Cancelled during warmup.")
            raise _WorkerCancelled
    if warmup_ok:
        store.append_event(
            job_id,
            StateChangeEvent(
                current=DeploymentState.PUBLISHING,
                operation=OperationType.WARMUP,
                detail="Publishing verified endpoint",
            ),
        )
        return completed_url
    _cleanup_failed_certification(job_id, store, orchestrator, config, observed_endpoint)
    return None


def _cleanup_failed_certification(
    job_id: str,
    store: JobStore,
    orchestrator: Orchestrator,
    config: DeploymentConfig,
    observed_endpoint: EndpointInfo | None,
) -> None:
    """Stop a placement that failed certification, mirroring the TUI.

    A server that answered readiness but failed only the image probe stays up
    for inspection; an explicitly kept Prime resource does too.
    """

    from ..protocol.enums import ComputeProvider
    from .provider_options import prime_provider_options
    from .vision_probe import is_vision_probe_failure

    keep_failed = (
        config.provider == ComputeProvider.PRIME
        and prime_provider_options(config).keep_failed_resource
    )
    # The triggering event is the failed WARMUP completion already stored;
    # vision-probe-only failures keep the endpoint for inspection.
    probe_only = False
    try:
        for stored in reversed(store.get_events(job_id)):
            if isinstance(stored.event, OperationCompleteEvent) and stored.event.operation == OperationType.WARMUP:
                probe_only = is_vision_probe_failure(stored.event)
                break
    except Exception:
        probe_only = False
    if keep_failed or probe_only:
        store.append_event(
            job_id,
            LogEvent(line="Certification failed; keeping the endpoint for inspection.", operation=OperationType.WARMUP),
        )
        return
    resource_label = (
        f"Prime pod {observed_endpoint.app_id}"
        if config.provider == ComputeProvider.PRIME and observed_endpoint is not None
        else f"failed {config.provider.display_name} deployment"
    )
    store.append_event(
        job_id,
        LogEvent(line=f"Certification failed; cleaning up {resource_label}.", operation=OperationType.WARMUP),
    )
    try:
        for cleanup_event in orchestrator.stop_app(
            config.backend,
            app_name=config.app_name,
            app_id=(observed_endpoint.app_id if observed_endpoint else None),
            provider=config.provider,
        ):
            store.append_event(job_id, cleanup_event)
            if (
                config.provider == ComputeProvider.VAST
                and isinstance(cleanup_event, OperationCompleteEvent)
                and not cleanup_event.success
            ):
                # A rental whose destruction failed must not be followed by
                # another rental: stop retrying the ladder.
                config.fallback_configs = ()
                try:
                    store.update_config(job_id, config)
                except Exception:
                    pass
    except Exception as exc:
        store.append_event(
            job_id,
            LogEvent(line=f"Cleanup after failed certification failed: {exc}", operation=OperationType.WARMUP),
        )


def _cancel_with_cleanup(
    job_id: str, store: JobStore, config: DeploymentConfig, resource_app_id: str | None, prefix: str
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

    if not config.do_deploy or not resource_app_id:
        detail = "Cancelled before allocating a resource."
        _terminal(detail)
        store.finish_job(job_id, JobStatus.CANCELLED, detail)
        return
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

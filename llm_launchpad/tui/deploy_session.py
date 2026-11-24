"""TUI deployment session helpers: lifecycle attempt wiring and persistent job tailing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..core.deploy_events import EndpointUrlResolver
from ..core.naming import legacy_app_name
from ..protocol.enums import (
    AttemptDisposition,
    CleanupDisposition,
    ComputeProvider,
    OperationType,
)
from ..protocol.events import (
    BaseEvent,
    EndpointAvailableEvent,
    LogEvent,
    OperationCompleteEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo

if TYPE_CHECKING:
    from .app import TuiApp
    from .deployment_jobs import DeploymentJob
    from .screens.monitor import MonitorScreen


def run_tui_deploy_lifecycle(
    app: TuiApp,
    config: DeploymentConfig,
    monitor: MonitorScreen,
    *,
    dispatch: Callable[..., None],
    connection_summary_lines: Callable[[DeploymentConfig, str], list[str]],
    connection_card_payload: Callable[[DeploymentConfig, str], dict[str, str]],
    connection_summary_message: Callable[[dict[str, str]], Any],
) -> str | None:
    """Execute one TUI deployment through ``deployment_lifecycle.run_lifecycle``."""
    from ..core.deployment_lifecycle import (
        LifecycleAttempt,
        LifecycleCallbacks,
        LifecycleOptions,
        run_lifecycle,
    )
    from ..core.deployment_preflight import lifecycle_attempts_for_configs

    deployed_endpoint: EndpointInfo | None = None
    url_resolver = EndpointUrlResolver()
    will_run_warmup = bool(config.do_warmup and config.do_deploy)

    def _emit_connection_summary(target_config: DeploymentConfig, url: str) -> None:
        app._cache_deploy_connection_summary(target_config, url, deployed_endpoint)
        for line in connection_summary_lines(target_config, url):
            dispatch(monitor, LogEvent(line=line))
        poster = getattr(monitor, "post_message", None)
        if callable(poster):
            poster(connection_summary_message(connection_card_payload(target_config, url)))

    def _sync_now(target_config: DeploymentConfig, url: str) -> None:
        target_app_name = target_config.app_name or legacy_app_name(target_config.backend)
        live_rows, prune_providers = app._visible_rows_and_prune_scope()
        app._sync_opencode(
            target_app_name=target_app_name,
            target_url=url,
            target_config=target_config,
            current_rows=live_rows,
            prune_providers=prune_providers,
            monitor=monitor,
            emit_skipped=True,
        )

    def _on_event(event: BaseEvent) -> None:
        nonlocal deployed_endpoint
        if isinstance(event, LogEvent):
            url_resolver.observe(event)
        elif isinstance(event, EndpointAvailableEvent):
            deployed_endpoint = event.endpoint
            app._invalidate_endpoint_snapshot()
            url_resolver.observe(event)
        elif isinstance(event, OperationCompleteEvent):
            if event.operation == OperationType.DEPLOY:
                app._invalidate_endpoint_snapshot()
            if event.success and isinstance(event.data, EndpointInfo):
                deployed_endpoint = event.data
        dispatch(monitor, event, phase_only=True)

    def _on_credentials(
        saved: DeploymentConfig, url: str | None, endpoint: EndpointInfo | None
    ) -> None:
        nonlocal deployed_endpoint
        if endpoint is not None:
            deployed_endpoint = endpoint
            app._invalidate_endpoint_snapshot()
        if url:
            try:
                app._cache_deploy_connection_summary(saved, url, endpoint)
            except Exception:
                pass

    def _on_connection(
        saved: DeploymentConfig, url: str | None, endpoint: EndpointInfo | None
    ) -> None:
        nonlocal deployed_endpoint
        if endpoint is not None:
            deployed_endpoint = endpoint
            app._invalidate_endpoint_snapshot()
        if url:
            if (
                endpoint is not None
                and endpoint.web_url
                and config.provider == ComputeProvider.MODAL
            ):
                try:
                    from ..core.runtime_health import record_explicit_health

                    record_explicit_health(endpoint, "healthy", None)
                except Exception:
                    pass
            _emit_connection_summary(saved, url)
            _sync_now(saved, url)

    def _attempt_with_plan(target: DeploymentConfig) -> LifecycleAttempt:
        spec = lifecycle_attempts_for_configs([target])[0]
        return LifecycleAttempt(
            config=target,
            retry_allowed=target.retry_allowed,
            plan=spec.plan,
        )

    attempts = [_attempt_with_plan(config)]
    for fallback in config.fallback_configs:
        if app._deployment_key(fallback) == app._deployment_key(config):
            attempts.append(_attempt_with_plan(fallback))
            continue
        try:
            app._begin_in_flight(fallback)
        except RuntimeError:
            dispatch(
                monitor,
                LogEvent(
                    line=(
                        f"Skipping fallback placement {fallback.app_name}: "
                        "another deployment owns that target."
                    ),
                    operation=OperationType.DEPLOY,
                ),
            )
            continue
        app._finish_in_flight(fallback, resolved=True)
        attempts.append(_attempt_with_plan(fallback))

    def _observe_and_forward(event: BaseEvent) -> None:
        app._observe_lifecycle_event(config, monitor, event)
        _on_event(event)

    result = run_lifecycle(
        app._orchestrator,
        attempts,
        options=LifecycleOptions(
            warmup_timeout_seconds=1800,
            tail_logs=True,
            warmup_enabled=will_run_warmup,
            modal_username=app._username,
        ),
        callbacks=LifecycleCallbacks(
            on_event=_observe_and_forward,
            is_cancelled=lambda: app._is_deploy_cancelled(monitor),
            on_resource=lambda resource_id: app._record_lifecycle_resource(
                config, monitor, resource_id
            ),
            on_connection=_on_connection,
            on_credentials=_on_credentials,
        ),
    )
    last_outcome = result.attempts[-1] if result.attempts else None
    failure_detail = (
        last_outcome.failure_detail
        if last_outcome and last_outcome.failure_detail
        else result.outcome
    )
    dispatch(
        monitor,
        OperationCompleteEvent(
            operation=OperationType.DEPLOY,
            success=result.succeeded,
            exit_code=(
                0
                if result.succeeded
                else ((last_outcome.failure_exit_code or 1) if last_outcome else 1)
            ),
            detail="" if result.succeeded else failure_detail,
        ),
    )
    if not result.succeeded and result.attempts:
        last = result.attempts[-1]
        if last.retry_allowed and len(result.attempts) < len(attempts):
            dispatch(
                monitor,
                LogEvent(
                    line="Trying an equivalent certified placement at or below "
                    "the approved hourly price.",
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                ),
            )

    job = app._job_for_monitor(monitor)
    if job is not None:
        job.outcome = result.outcome or (
            "Finished — open result"
            if result.succeeded
            else "Failed — check resource in Manage"
        )
        was_cancelled = any(
            attempt.disposition == AttemptDisposition.CANCELLED
            for attempt in result.attempts
        )
        if not was_cancelled:
            app.notify_deployment_finished(
                config,
                success=result.succeeded,
                detail="" if result.succeeded else failure_detail,
                monitor=monitor,
            )
        if result.succeeded:
            app._finish_in_flight(config, resolved=True)
        elif was_cancelled and all(
            attempt.cleanup == CleanupDisposition.FAILED or attempt.cleanup_error
            for attempt in result.attempts
            if attempt.disposition == AttemptDisposition.CANCELLED
        ):
            app._finish_in_flight(config, resolved=False)
        else:
            app._finish_in_flight(config, resolved=True)
    return result.outcome


def tail_persistent_job(
    app: TuiApp,
    job: DeploymentJob,
    *,
    dispatch: Callable[..., None],
) -> None:
    """Forward durable job events to ``job.monitor`` until completion or TUI quit."""
    if not job.persistent_id:
        return
    try:
        store = app._get_job_store()
    except Exception:
        return
    while True:
        if app._quitting:
            return
        try:
            record = store.get_job(job.persistent_id)
        except Exception:
            break
        if record is None:
            break
        try:
            record = store.get_job(job.persistent_id)
            if record is None:
                return
            events = store.get_events(job.persistent_id, after_seq=job.last_seen_seq)
        except Exception:
            break
        for stored in events:
            job.last_seen_seq = max(job.last_seen_seq, stored.seq)
            if (
                isinstance(stored.event, OperationCompleteEvent)
                and not stored.event.success
                and stored.event.exit_code
            ):
                job.last_failure_exit_code = stored.event.exit_code
            try:
                dispatch(job.monitor, stored.event, phase_only=True)
            except Exception:
                pass
            if isinstance(stored.event, EndpointAvailableEvent):
                try:
                    app._invalidate_endpoint_snapshot()
                except Exception:
                    pass
            if (
                isinstance(stored.event, OperationCompleteEvent)
                and stored.event.operation == OperationType.WARMUP
                and stored.event.success
                and job.config.provider == ComputeProvider.MODAL
            ):
                try:
                    from ..core.runtime_health import record_explicit_health

                    record_explicit_health(
                        EndpointInfo(
                            name=job.config.app_name or "",
                            provider=ComputeProvider.MODAL,
                            backend=job.config.backend,
                        ),
                        "healthy",
                        None,
                    )
                except Exception:
                    pass
        job.config = record.config
        job.outcome = record.outcome or job.outcome
        if record.cancel_requested:
            job.cancel_requested.set()
        if record.terminal:
            if events and job.last_seen_seq < record.event_count:
                continue
            with app._deployment_lock:
                job.accepting_cancellation = False
                job.finished.set()
            app._finish_persistent_monitor(job, record)
            if record.status == "succeeded":
                app._post_connection_card_for_record(job, record, store)
                app._sync_finished_job(job, record)
            if record.status != "cancelled":
                app.notify_deployment_finished(
                    record.config,
                    success=record.status == "succeeded",
                    detail="" if record.status == "succeeded" else (record.outcome or ""),
                    monitor=job.monitor,
                )
            return
        if job.cancel_requested.is_set():
            try:
                store.request_cancel(job.persistent_id)
            except Exception:
                pass
        try:
            import time as _time

            _time.sleep(app._PERSISTENT_JOB_POLL_INTERVAL_SECONDS)
        except Exception:
            return

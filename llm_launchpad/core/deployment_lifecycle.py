"""One owner for deploy through readiness, cleanup, and retry decisions.

The CLI, the detached worker, and the in-session TUI fallback each
implemented deploy → save credentials → warmup → cleanup → publication with
slightly different policy. This runner owns the sequence; frontends render
events and supply execution options (timeouts, log following, cancellation).
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from ..protocol.enums import (
    AttemptDisposition,
    CleanupDisposition,
    ComputeProvider,
    DeploymentState,
    OperationIntent,
    OperationType,
    WarmupStatus,
)
from ..protocol.events import (
    BaseEvent,
    EndpointAvailableEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    ResourceAllocatedEvent,
    StateChangeEvent,
)
from ..protocol.models import (
    DeploymentAttemptOutcome,
    DeploymentConfig,
    DeploymentResult,
    EndpointInfo,
    ResolvedDeploymentPlan,
)
from .resource_targeting import resolve_stop_target
from .warmup import StartupPhaseTimer

EventStream = Generator[BaseEvent, None, None]


@dataclass
class LifecycleOptions:
    """Frontend-controlled execution settings; never lifecycle policy."""

    warmup_timeout_seconds: int = 1800
    tail_logs: bool = True
    server_url_override: str | None = None
    warmup_enabled: bool = True
    modal_username: str | None = None
    # Identity of the current attempt for persistence callbacks. The worker
    # sets this per attempt so stored config/credentials belong to the
    # placement that just ran, not the first config captured in a closure.
    attempt_index: int = 0
    attempt_id: str | None = None
    # Ask the certified endpoint for one tool call before publishing it.
    verify_tool_calls: bool = True


@dataclass
class LifecycleAttempt:
    """One approved placement plus its retry eligibility.

    ``config`` is the legacy mutable execution shape adapters consume.
    ``plan`` is the authoritative resolved plan when the caller went through
    preflight; when present, its intent and quoted price govern execution.
    """

    config: DeploymentConfig
    retry_allowed: bool = True
    plan: ResolvedDeploymentPlan | None = None


@dataclass
class LifecycleCallbacks:
    """Persistence and cancellation hooks; in-memory by default."""

    on_event: Callable[[BaseEvent], None] = lambda event: None
    is_cancelled: Callable[[], bool] = lambda: False
    on_resource: Callable[[str | None], None] = lambda resource_id: None
    on_connection: Callable[[DeploymentConfig, str | None, EndpointInfo | None], None] = (
        lambda config, url, endpoint: None
    )
    # Pre-certification credential save, before warmup runs. Persisting the
    # bearer key must not publish the endpoint as verified: frontends save
    # here and publish only in ``on_connection`` after successful warmup.
    # Defaults to None (no pre-save) so a single on_connection publish keeps
    # its historical single-call shape.
    on_credentials: Callable[[DeploymentConfig, str | None, EndpointInfo | None], None] | None = None
    # Optional per-attempt hooks so the worker can persist the placement that
    # just ran. Defaults keep single-attempt callers unchanged.
    on_attempt_start: Callable[[int, DeploymentConfig], None] = (
        lambda index, config: None
    )
    on_attempt_finish: Callable[[int, DeploymentAttemptOutcome], None] = (
        lambda index, outcome: None
    )


@dataclass(frozen=True)
class WarmupOutcome:
    """Typed result of one warmup/certification pass."""

    status: WarmupStatus
    url: str | None = None
    attestation: Any | None = None
    failure_detail: str | None = None
    failure_exit_code: int | None = None
    failure_event: OperationCompleteEvent | None = None


def _endpoint_from_event(event: BaseEvent) -> EndpointInfo | None:
    if isinstance(event, EndpointAvailableEvent):
        return event.endpoint
    if isinstance(event, OperationCompleteEvent) and isinstance(event.data, EndpointInfo):
        return event.data
    return None


def _intent_for_attempt(attempt: LifecycleAttempt) -> OperationIntent:
    if attempt.plan is not None:
        return attempt.plan.intent
    return _intent(attempt.config)


def _intent(config: DeploymentConfig) -> OperationIntent:
    from .quick_deploy import intent_from_legacy_flags

    return intent_from_legacy_flags(
        do_deploy=config.do_deploy,
        run_smoke=config.run_smoke,
        preload=config.preload,
    )


def _price_for_attempt(attempt: LifecycleAttempt) -> float | None:
    """Quoted execution price, never the budget cap."""
    if attempt.plan is not None and attempt.plan.price_per_hour_usd is not None:
        return attempt.plan.price_per_hour_usd
    return attempt.config.price_per_hour_usd


def _is_smoke_success(operation: OperationType, success: bool, intent: OperationIntent) -> bool:
    if not success:
        return False
    if intent == OperationIntent.SMOKE:
        return operation in (OperationType.SMOKE_TEST, OperationType.DEPLOY)
    return operation == OperationType.DEPLOY


def run_lifecycle(
    orchestrator: Any,
    attempts: list[LifecycleAttempt] | tuple[LifecycleAttempt, ...],
    *,
    options: LifecycleOptions | None = None,
    callbacks: LifecycleCallbacks | None = None,
) -> DeploymentResult:
    """Run approved placements until one succeeds or retries are exhausted."""
    opts = options or LifecycleOptions()
    hooks = callbacks or LifecycleCallbacks()
    outcomes: list[DeploymentAttemptOutcome] = []
    attempt_list = list(attempts)
    index = 0
    while index < len(attempt_list):
        attempt = attempt_list[index]
        if hooks.is_cancelled():
            outcome = DeploymentAttemptOutcome(
                disposition=AttemptDisposition.CANCELLED,
                failure_detail="Cancelled before starting placement.",
                cleanup=CleanupDisposition.NOTHING_TO_CLEAN,
                retry_allowed=False,
            )
            outcomes.append(outcome)
            try:
                hooks.on_attempt_finish(index, outcome)
            except Exception:
                pass
            break
        try:
            hooks.on_attempt_start(index, attempt.config)
        except Exception:
            pass
        outcome, _ = _run_single_attempt(
            orchestrator, attempt, opts=opts, hooks=hooks
        )
        outcomes.append(outcome)
        try:
            hooks.on_attempt_finish(index, outcome)
        except Exception:
            pass
        if outcome.disposition == AttemptDisposition.SUCCEEDED:
            return DeploymentResult(
                succeeded=True,
                url=outcome.endpoint_url,
                app_name=attempt.config.app_name,
                attempts=tuple(outcomes),
                outcome="Finished — open result",
            )
        if outcome.disposition == AttemptDisposition.CANCELLED:
            if outcome.cleanup == CleanupDisposition.FAILED or outcome.cleanup_error:
                detail = "Cancellation cleanup failed; check Manage. Recovery record retained."
            else:
                detail = outcome.failure_detail or "Cancelled."
            return DeploymentResult(
                succeeded=False,
                app_name=attempt.config.app_name,
                attempts=tuple(outcomes),
                outcome=detail,
            )
        if outcome.disposition == AttemptDisposition.RETAINED:
            return DeploymentResult(
                succeeded=False,
                app_name=attempt.config.app_name,
                attempts=tuple(outcomes),
                outcome=f"Retained — check resource in Manage. {outcome.failure_detail or 'Certification failed.'}".strip(),
            )
        # A failed attempt that forbids retry stops the ladder even when more
        # placements are approved: an uncertain allocation or unconfirmed
        # cleanup means the next rental could double-bill.
        if not attempt.retry_allowed or not outcome.retry_allowed:
            break
        index += 1
    last = outcomes[-1] if outcomes else None
    detail = (last.failure_detail if last and last.failure_detail else "Deployment failed.").strip()
    return DeploymentResult(
        succeeded=False,
        app_name=attempt_list[0].config.app_name if attempt_list else None,
        attempts=tuple(outcomes),
        outcome=f"Failed — check resource in Manage. {detail}".strip(),
    )


def effective_config_for_attempt(attempt: LifecycleAttempt) -> DeploymentConfig:
    """Render the authoritative execution config for one attempt.

    When a resolved plan is present it governs: intent, quoted price, and any
    resolved tuning/placement/identity flow to the adapters. Legacy callers
    without a plan execute their config directly.
    """
    if attempt.plan is None:
        return attempt.config
    from .quick_deploy import config_from_request

    config = config_from_request(attempt.plan.request)
    # Resolved effective values win over the raw request rendering.
    plan = attempt.plan
    config.serving_requirements = plan.serving_requirements
    config.runtime_tuning = plan.runtime_tuning
    config.placement_assessment = plan.placement_assessment
    config.reasoning = plan.reasoning
    config.vision = plan.vision
    config.price_per_hour_usd = plan.price_per_hour_usd
    config.instance_name = plan.instance_name or config.instance_name
    config.app_name = plan.app_name or config.app_name
    config.function_slug = plan.function_slug or config.function_slug
    config.gpu_type = plan.gpu_type or config.gpu_type
    config.gpu_count = plan.gpu_count if plan.gpu_count is not None else config.gpu_count
    # Preserve caller-supplied secrets and ladder metadata that never live in
    # the immutable request.
    config.endpoint_api_key = attempt.config.endpoint_api_key
    config.provider_options = plan.provider_options or attempt.config.provider_options
    config.fallback_configs = attempt.config.fallback_configs
    config.retry_allowed = attempt.config.retry_allowed
    config.do_warmup = plan.do_warmup
    config.show_debug_logs = plan.show_debug_logs
    return config


def _run_single_attempt(
    orchestrator: Any,
    attempt: LifecycleAttempt,
    *,
    opts: LifecycleOptions,
    hooks: LifecycleCallbacks,
) -> tuple[DeploymentAttemptOutcome, DeploymentConfig]:
    from .provider_options import prime_provider_options
    from .vision_probe import is_vision_probe_failure

    config = effective_config_for_attempt(attempt)
    intent = _intent_for_attempt(attempt)
    phase_timer = StartupPhaseTimer()
    phase_timer.deploy_started()
    observed_url: str | None = None
    observed_endpoint: EndpointInfo | None = None
    resource_app_id: str | None = None
    deploy_ok = False
    failure_detail = ""
    failure_exit_code: int | None = None
    failure_event: OperationCompleteEvent | None = None
    deploy_events = orchestrator.deploy(config)
    from .deploy_events import EndpointUrlResolver

    url_resolver = EndpointUrlResolver()
    try:
        for event in deploy_events:
            # The endpoint key is minted during the deploy, on the config the
            # lifecycle executes -- which is rebuilt from the plan and is not
            # the object the caller kept a handle to. Callers sync OpenCode
            # from their own config, so the key never reached it and every
            # synced provider was written without one: OpenCode then got
            # `401 Invalid API Key` from an endpoint that had deployed,
            # certified and was serving perfectly well.
            if config is not attempt.config and config.endpoint_api_key:
                attempt.config.endpoint_api_key = config.endpoint_api_key
            endpoint = _endpoint_from_event(event)
            if endpoint is not None:
                if endpoint.web_url:
                    url_resolver.observe(event)
                    observed_url = url_resolver.url or endpoint.web_url
                observed_endpoint = endpoint
                if endpoint.app_id:
                    resource_app_id = endpoint.app_id
                    hooks.on_resource(resource_app_id)
            if isinstance(event, ResourceAllocatedEvent) and event.app_id:
                resource_app_id = event.app_id
                hooks.on_resource(resource_app_id)
            if isinstance(event, LogEvent):
                # Modal's deploy stream carries the serving URL as a log line
                # before any endpoint event; capture it with the same
                # precedence the TUI always used (announced beats incidental,
                # deployed beats -dev) so warmup and publication use the
                # deployed address, not a guess.
                url_resolver.observe(event)
                if url_resolver.url:
                    observed_url = url_resolver.url
            if hooks.is_cancelled():
                cleanup = _cancel_cleanup(
                    orchestrator, hooks, config, resource_app_id,
                    deploy_stream=deploy_events,
                )
                if cleanup.disposition == CleanupDisposition.CONFIRMED:
                    failure_message = "Cancelled; resource stopped."
                elif cleanup.disposition == CleanupDisposition.NOTHING_TO_CLEAN:
                    failure_message = "Cancelled before allocating a resource."
                else:
                    failure_message = (
                        cleanup.detail
                        or "Cancellation cleanup failed; check Manage. Recovery record retained."
                    )
                return DeploymentAttemptOutcome(
                    disposition=AttemptDisposition.CANCELLED,
                    resource_app_id=resource_app_id,
                    endpoint_url=observed_url,
                    endpoint=observed_endpoint,
                    failure_detail=failure_message,
                    cleanup=cleanup.disposition,
                    cleanup_error=cleanup.detail or None,
                    retry_allowed=False,
                ), config
            hooks.on_event(event)
            if isinstance(event, OperationCompleteEvent) and event.operation in (
                OperationType.DEPLOY,
                OperationType.SMOKE_TEST,
            ):
                deploy_phase = phase_timer.deploy_event(event.operation)
                if deploy_phase is not None:
                    hooks.on_event(deploy_phase)
                if _is_smoke_success(event.operation, event.success, intent):
                    deploy_ok = True
                elif not event.success:
                    failure_detail = event.detail or "Deployment failed."
                    failure_exit_code = event.exit_code or None
                    failure_event = event
    finally:
        _close_stream(deploy_events)

    if not deploy_ok:
        # A deploy failure with an uncertain billable state must block the
        # next rental. The provider reports its local rollback in the failure
        # data; only a confirmed rollback allows the next approved placement.
        # An explicit unconfirmed rollback blocks even without an allocation
        # event (uncertain create, name conflict): the record may still bill.
        from .provider_adapters import rollback_from_event

        rollback = rollback_from_event(failure_event)
        if rollback is not None and rollback.confirmed:
            retry_allowed = config.retry_allowed
            cleanup = CleanupDisposition.CONFIRMED
        elif rollback is not None:
            retry_allowed = False
            cleanup = CleanupDisposition.FAILED
            if rollback.detail and "Cleanup remains pending" not in failure_detail:
                failure_detail = f"{failure_detail} Cleanup remains pending: {rollback.detail}".strip()
        elif resource_app_id is None:
            retry_allowed = config.retry_allowed
            cleanup = CleanupDisposition.NOTHING_TO_CLEAN
        else:
            retry_allowed = False
            cleanup = CleanupDisposition.UNKNOWN
        return DeploymentAttemptOutcome(
            disposition=AttemptDisposition.FAILED,
            resource_app_id=resource_app_id,
            endpoint_url=observed_url,
            endpoint=observed_endpoint,
            failure_detail=failure_detail or "Deployment failed.",
            cleanup=cleanup,
            retry_allowed=retry_allowed,
            failure_exit_code=failure_exit_code,
        ), config

    # Persist credentials immediately: a later warmup failure must not strand
    # a live, billable resource without its generated bearer key. When warmup
    # follows, this is the pre-certification save; the verified URL publishes
    # again after warmup succeeds.
    warmup_follows = bool(opts.warmup_enabled and config.do_warmup and config.do_deploy)
    if intent == OperationIntent.PRELOAD:
        # Preload-only work has no serving endpoint to certify; deploy success
        # is the terminal state. Do not invoke warmup.
        hooks.on_connection(config, observed_url, observed_endpoint)
        return DeploymentAttemptOutcome(
            disposition=AttemptDisposition.SUCCEEDED,
            resource_app_id=resource_app_id,
            endpoint_url=observed_url,
            endpoint=observed_endpoint,
            runtime_attestation=config.runtime_attestation,
            cleanup=CleanupDisposition.NOTHING_TO_CLEAN,
        ), config
    if not warmup_follows or intent == OperationIntent.SMOKE:
        hooks.on_connection(config, observed_url, observed_endpoint)
    elif hooks.on_credentials is not None and (observed_url or observed_endpoint is not None):
        # Pre-certification credential save; publication happens after warmup.
        hooks.on_credentials(config, observed_url, observed_endpoint)

    if intent == OperationIntent.SMOKE:
        return DeploymentAttemptOutcome(
            disposition=AttemptDisposition.SUCCEEDED,
            resource_app_id=resource_app_id,
            endpoint_url=observed_url,
            endpoint=observed_endpoint,
            cleanup=CleanupDisposition.NOTHING_TO_CLEAN,
        ), config

    if not warmup_follows:
        return DeploymentAttemptOutcome(
            disposition=AttemptDisposition.SUCCEEDED,
            resource_app_id=resource_app_id,
            endpoint_url=observed_url,
            endpoint=observed_endpoint,
            runtime_attestation=config.runtime_attestation,
            cleanup=CleanupDisposition.NOTHING_TO_CLEAN,
        ), config

    warmup_outcome = _run_warmup(
        orchestrator, hooks, config, observed_url, observed_endpoint,
        price_per_hour_usd=_price_for_attempt(attempt),
        resource_app_id=resource_app_id,
        opts=opts, phase_timer=phase_timer,
    )
    warmed_url = warmup_outcome.url
    attestation = warmup_outcome.attestation
    if attestation is not None:
        config.runtime_attestation = attestation
        if observed_endpoint is not None:
            observed_endpoint.runtime_attestation = attestation
    if warmup_outcome.status == WarmupStatus.CANCELLED:
        cleanup = _cancel_cleanup(
            orchestrator, hooks, config,
            resource_app_id,
        )
        if cleanup.disposition == CleanupDisposition.CONFIRMED:
            cancel_message = "Cancelled; resource stopped."
        elif cleanup.disposition == CleanupDisposition.NOTHING_TO_CLEAN:
            cancel_message = "Cancelled before allocating a resource."
        else:
            cancel_message = (
                cleanup.detail
                or "Cancellation cleanup failed; check Manage. Recovery record retained."
            )
        return DeploymentAttemptOutcome(
            disposition=AttemptDisposition.CANCELLED,
            resource_app_id=resource_app_id,
            endpoint_url=observed_url,
            endpoint=observed_endpoint,
            runtime_attestation=attestation,
            failure_detail=cancel_message,
            cleanup=cleanup.disposition,
            cleanup_error=cleanup.detail or None,
            retry_allowed=False,
        ), config
    if warmup_outcome.status != WarmupStatus.SUCCEEDED:
        failure_event = warmup_outcome.failure_event
        keep_failed = (
            config.provider == ComputeProvider.PRIME
            and prime_provider_options(config).keep_failed_resource
        )
        probe_only = (
            failure_event is not None and is_vision_probe_failure(failure_event)
        )
        if keep_failed or probe_only:
            hooks.on_event(
                LogEvent(
                    line="Certification failed; keeping the endpoint for inspection.",
                    operation=OperationType.WARMUP,
                )
            )
            return DeploymentAttemptOutcome(
                disposition=AttemptDisposition.RETAINED,
                resource_app_id=resource_app_id,
                endpoint_url=warmed_url or observed_url,
                endpoint=observed_endpoint,
                runtime_attestation=attestation,
                failure_detail=warmup_outcome.failure_detail or "Certification failed.",
                cleanup=CleanupDisposition.RETAINED,
                retry_allowed=False,
            ), config
        cleanup = _finalize_failed_attempt(
            orchestrator, hooks, config,
            resource_app_id=resource_app_id,
            observed_endpoint=observed_endpoint,
        )
        retry_allowed = config.retry_allowed and cleanup.disposition not in (
            CleanupDisposition.FAILED,
            CleanupDisposition.UNKNOWN,
        )
        return DeploymentAttemptOutcome(
            disposition=AttemptDisposition.FAILED,
            resource_app_id=resource_app_id,
            endpoint_url=warmed_url or observed_url,
            endpoint=observed_endpoint,
            runtime_attestation=attestation,
            failure_detail=warmup_outcome.failure_detail or "Certification failed.",
            cleanup=cleanup.disposition,
            cleanup_error=cleanup.detail or None,
            retry_allowed=retry_allowed,
            failure_exit_code=warmup_outcome.failure_exit_code,
        ), config

    if opts.verify_tool_calls:
        _verify_tool_calling(hooks, config, warmed_url or observed_url, observed_endpoint)

    hooks.on_event(
        StateChangeEvent(
            current=DeploymentState.PUBLISHING,
            operation=OperationType.WARMUP,
            detail="Publishing verified endpoint",
        )
    )
    hooks.on_connection(config, warmed_url, observed_endpoint)
    # Warmup reported HEALTHY before publishing began, so PUBLISHING was the
    # last state anyone saw: a finished deploy kept reading "publishing" in
    # the TUI's context bar and the job record. Close the sequence on the
    # state the endpoint is actually in.
    hooks.on_event(
        StateChangeEvent(
            current=DeploymentState.HEALTHY,
            operation=OperationType.WARMUP,
            detail="Endpoint ready",
        )
    )
    return DeploymentAttemptOutcome(
        disposition=AttemptDisposition.SUCCEEDED,
        resource_app_id=resource_app_id,
        endpoint_url=warmed_url,
        endpoint=observed_endpoint,
        runtime_attestation=attestation,
        cleanup=CleanupDisposition.NOTHING_TO_CLEAN,
    ), config


def _verify_tool_calling(
    hooks: LifecycleCallbacks,
    config: DeploymentConfig,
    url: str | None,
    endpoint: EndpointInfo | None,
) -> None:
    """Record whether coding agents can drive the endpoint.

    A failure is published, not fatal: the endpoint still serves chat, and
    tearing down a certified deployment over a chat-template gap would bill
    the user for nothing. The warning names what will not work instead.
    """
    from .opencode import build_openai_connection_payload
    from .tool_call_probe import verify_tool_calling

    if not url:
        return
    hooks.on_event(
        LogEvent(line="Verifying tool calling for coding agents.", operation=OperationType.WARMUP)
    )
    model_id = str(build_openai_connection_payload(config, url).get("model_id") or "") or None
    result = verify_tool_calling(url, model_id, config.endpoint_api_key)
    if result is None:
        return
    config.tool_calling = result.status
    if endpoint is not None:
        endpoint.tool_calling = result.status
    if result.passed:
        line = "Tool calling verified: coding agents can use this endpoint."
    else:
        line = (
            f"Warning: tool calling failed ({result.detail}). Chat works, but coding "
            "agents such as OpenCode will not be able to edit files or run commands."
        )
    hooks.on_event(LogEvent(line=line, operation=OperationType.WARMUP, is_milestone=True))


def _run_warmup(
    orchestrator: Any,
    hooks: LifecycleCallbacks,
    config: DeploymentConfig,
    observed_url: str | None,
    observed_endpoint: EndpointInfo | None,
    *,
    price_per_hour_usd: float | None = None,
    resource_app_id: str | None = None,
    opts: LifecycleOptions,
    phase_timer: StartupPhaseTimer | None,
) -> WarmupOutcome:
    """Warm one placement; warmup owns observation, the caller owns cleanup.

    Returns a typed :class:`WarmupOutcome`. Missing evidence is failure, never
    success: a missing warmup implementation, an empty event stream, or a
    stream without an explicit successful WARMUP completion all fail when
    warmup was requested. Cancellation is reported as CANCELLED so the caller
    can run its single cleanup decision.

    ``config`` is the effective execution config (the deploy step may have
    mutated it, e.g. assigning a function slug); ``price_per_hour_usd`` is the
    quoted execution price, never the budget cap.
    """
    from ..protocol.enums import ComputeProvider

    url = opts.server_url_override or observed_url
    if not url and observed_endpoint is not None and observed_endpoint.web_url:
        url = observed_endpoint.web_url
    if not url and config.provider == ComputeProvider.MODAL:
        url = _modal_default_url(config, username_override=opts.modal_username)
    if not url:
        # Provisioning reported success but left no usable URL. Report the
        # failure; the caller applies retention policy and cleanup exactly
        # once.
        failure = OperationCompleteEvent(
            operation=OperationType.WARMUP, success=False,
            detail="Provider returned no endpoint URL.",
        )
        hooks.on_event(
            ErrorEvent(
                message="Provider returned no endpoint URL.",
                operation=OperationType.WARMUP,
            )
        )
        hooks.on_event(failure)
        return WarmupOutcome(
            status=WarmupStatus.FAILED,
            url=None,
            failure_detail="Provider returned no endpoint URL.",
            failure_event=failure,
        )
    certification_kwargs: dict[str, Any] = {}
    if config.serving_requirements is not None:
        certification_kwargs = {
            "serving_requirements": config.serving_requirements,
            "placement_assessment": config.placement_assessment,
            "runtime_id": config.llamacpp_runtime_id,
            # Quoted execution price, never the budget cap. Unknown stays
            # unknown so value-per-dollar is not fabricated from a ceiling.
            "price_per_hour_usd": price_per_hour_usd,
        }
    if config.provider != ComputeProvider.MODAL or config.endpoint_api_key:
        certification_kwargs.update(
            provider=config.provider,
            api_key=config.endpoint_api_key,
            pod_id=(observed_endpoint.app_id if observed_endpoint else None)
            or resource_app_id,
        )
    if config.vision is not None:
        certification_kwargs["vision"] = config.vision
    completed_url = url
    attestation: Any | None = None
    failure: OperationCompleteEvent | None = None
    success_count = 0
    failure_count = 0
    saw_events = False
    if phase_timer is not None:
        phase_timer.warmup_started()
    warmup_call = getattr(orchestrator, "warmup", None)
    if warmup_call is None:
        # No warmup implementation cannot verify a serving endpoint. Failing
        # here (instead of treating the URL as verified) keeps "verified"
        # meaning "a warmup explicitly completed successfully".
        failure = OperationCompleteEvent(
            operation=OperationType.WARMUP, success=False,
            detail="Warmup is unavailable: no warmup implementation.",
        )
        hooks.on_event(failure)
        return WarmupOutcome(
            status=WarmupStatus.FAILED,
            url=completed_url,
            failure_detail="Warmup is unavailable: no warmup implementation.",
            failure_event=failure,
        )
    try:
        warmup_events = warmup_call(
            config.backend,
            url,
            opts.warmup_timeout_seconds,
            opts.tail_logs,
            app_name=config.app_name,
            served_model_name=config.served_model_name,
            phase_timer=phase_timer,
            **certification_kwargs,  # type: ignore[arg-type]
        )
    except Exception as exc:
        failure = OperationCompleteEvent(
            operation=OperationType.WARMUP, success=False, detail=str(exc) or "Warmup failed.",
        )
        hooks.on_event(failure)
        return WarmupOutcome(
            status=WarmupStatus.FAILED,
            url=completed_url,
            failure_detail=failure.detail,
            failure_event=failure,
        )
    try:
        for event in warmup_events:
            saw_events = True
            hooks.on_event(event)
            if (
                isinstance(event, OperationCompleteEvent)
                and event.operation == OperationType.WARMUP
            ):
                if event.success:
                    success_count += 1
                    if isinstance(event.data, dict):
                        maybe_url = event.data.get("url")
                        if isinstance(maybe_url, str) and maybe_url.strip():
                            completed_url = maybe_url.strip()
                        attestation = event.data.get("attestation")
                else:
                    failure_count += 1
                    if failure is None:
                        failure = event
            if hooks.is_cancelled():
                close_lifecycle_stream(warmup_events)
                return WarmupOutcome(
                    status=WarmupStatus.CANCELLED,
                    url=completed_url,
                    attestation=attestation,
                    failure_detail="Cancelled during warmup.",
                )
    finally:
        _close_stream(warmup_events)
    if success_count > 0 and failure_count == 0:
        return WarmupOutcome(
            status=WarmupStatus.SUCCEEDED, url=completed_url, attestation=attestation
        )
    if success_count > 0 and failure_count > 0:
        detail = (failure.detail if failure else "") or "Conflicting warmup completions."
        failure = failure or OperationCompleteEvent(
            operation=OperationType.WARMUP, success=False, detail=detail
        )
        return WarmupOutcome(
            status=WarmupStatus.FAILED,
            url=completed_url,
            attestation=attestation,
            failure_detail=f"Conflicting warmup completions: {detail}",
            failure_exit_code=failure.exit_code or None,
            failure_event=failure,
        )
    if failure is None:
        # No completion at all (empty or log-only stream): no opinion is not
        # verification.
        detail = (
            "Warmup produced no completion event."
            if saw_events
            else "Warmup produced no events."
        )
        failure = OperationCompleteEvent(
            operation=OperationType.WARMUP, success=False, detail=detail
        )
        hooks.on_event(failure)
        return WarmupOutcome(
            status=WarmupStatus.FAILED,
            url=completed_url,
            attestation=attestation,
            failure_detail=detail,
            failure_event=failure,
        )
    return WarmupOutcome(
        status=WarmupStatus.FAILED,
        url=completed_url,
        attestation=attestation,
        failure_detail=failure.detail or "Certification failed.",
        failure_exit_code=failure.exit_code or None,
        failure_event=failure,
    )


def warmup_events_emitted(seen: bool) -> bool:
    """Whether a warmup stream yielded any event at all (legacy helper)."""
    return seen


def _finalize_failed_attempt(
    orchestrator: Any,
    hooks: LifecycleCallbacks,
    config: DeploymentConfig,
    *,
    resource_app_id: str | None,
    observed_endpoint: EndpointInfo | None,
) -> Any:
    """Single cleanup decision for a failed-certification attempt."""
    from .provider_adapters import ProviderResource, stop_with_orchestrator

    allocated_id = resource_app_id or (
        observed_endpoint.app_id if observed_endpoint else None
    )
    target = resolve_stop_target(
        provider=config.provider,
        app_name=config.app_name,
        resource_id=allocated_id,
        allocated=True,
    )
    if not target.stoppable:
        return ProviderCleanupShim(
            CleanupDisposition.UNKNOWN,
            "No addressable resource to clean up.",
        )
    resource_label = (
        f"Prime pod {allocated_id}"
        if config.provider == ComputeProvider.PRIME and allocated_id
        else f"failed {config.provider.display_name} deployment"
    )
    hooks.on_event(
        LogEvent(
            line=f"Certification failed; cleaning up {resource_label}.",
            operation=OperationType.WARMUP,
        )
    )
    cleanup = stop_with_orchestrator(
        orchestrator,
        ProviderResource(
            provider=config.provider,
            app_name=config.app_name or "",
            resource_id=allocated_id,
        ),
        backend=config.backend,
    )
    if cleanup.disposition != CleanupDisposition.CONFIRMED:
        hooks.on_event(
            LogEvent(
                line=f"Cleanup after failed certification failed: {cleanup.detail or 'Stop did not confirm.'}",
                operation=OperationType.WARMUP,
            )
        )
    return cleanup


def _cancel_cleanup(
    orchestrator: Any,
    hooks: LifecycleCallbacks,
    config: DeploymentConfig,
    resource_app_id: str | None,
    *,
    deploy_stream: Any | None = None,
    warmup_stream: Any | None = None,
) -> Any:
    from .provider_adapters import ProviderResource, stop_with_orchestrator

    if deploy_stream is not None:
        close_lifecycle_stream(deploy_stream)
    if warmup_stream is not None:
        close_lifecycle_stream(warmup_stream)
    # Cancellation before any allocation is only "before allocating" when the
    # frontend never saw a resource id *and* the provider cannot be addressed
    # by name. A name-addressable Modal/Vast app may already exist server-side
    # even when no id event arrived, so stop by name rather than declaring
    # nothing was allocated.
    target = resolve_stop_target(
        provider=config.provider,
        app_name=config.app_name,
        resource_id=resource_app_id,
        allocated=True,
    )
    if not config.do_deploy:
        return ProviderCleanupShim(
            CleanupDisposition.NOTHING_TO_CLEAN,
            "Cancelled before allocating a resource.",
        )
    if not target.stoppable:
        return ProviderCleanupShim(
            CleanupDisposition.UNKNOWN,
            "Cancelled; no addressable resource to stop. Check Manage.",
        )
    hooks.on_event(LogEvent(line="Cancelled; cleaning up allocated resource.", operation=OperationType.STOP))
    return stop_with_orchestrator(
        orchestrator,
        ProviderResource(
            provider=config.provider,
            app_name=config.app_name or "",
            resource_id=resource_app_id,
        ),
        backend=config.backend,
    )


@dataclass
class ProviderCleanupShim:
    """Lightweight cleanup result without importing the adapter layer."""

    disposition: Any
    detail: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.disposition, str):
            self.disposition = CleanupDisposition(self.disposition)


def _modal_default_url(
    config: DeploymentConfig, *, username_override: str | None = None
) -> str | None:
    try:
        from .backend import ModalBackend

        username = username_override or ModalBackend.get_username()
    except Exception:
        return None
    if not username:
        return None
    try:
        from .backend import ModalBackend as _ModalBackend

        return _ModalBackend.default_server_url(
            username,
            app_name=config.app_name,
            function_slug=config.function_slug,
        )
    except Exception:
        return None


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def close_lifecycle_stream(stream: Any) -> None:
    """Close a provider event stream; cooperative cancellation needs this."""
    _close_stream(stream)


__all__ = [
    "LifecycleAttempt",
    "LifecycleCallbacks",
    "LifecycleOptions",
    "WarmupOutcome",
    "run_lifecycle",
    "warmup_events_emitted",
]

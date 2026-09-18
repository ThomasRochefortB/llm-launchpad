"""Authoritative deployment preflight: validation plus resolution.

Three stages, one entrypoint:

1. Pure request validation: intent, capabilities, required fields, revision
   support, GPU shape, and provider-option type. No network, no mutation.
2. Resolution and readiness: model metadata, architecture compatibility,
   vision, speculative decoding, effective tuning/flags, placement, provider
   readiness. Produces an immutable :class:`ResolvedDeploymentPlan`.
3. Allocation-boundary revalidation: adapters recheck time-sensitive facts
   (offer, price cap, machine identity) immediately before provisioning.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..protocol.enums import (
    BackendType,
    ComputeProvider,
    OperationIntent,
    VisionMode,
)
from ..protocol.models import (
    DeploymentConfig,
    DeploymentRequest,
    ResolvedDeploymentPlan,
)
from .providers import capabilities


@dataclass(frozen=True)
class PreflightFinding:
    """One structured validation outcome with a stable code."""

    code: str
    message: str
    blocking: bool = True
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreflightResult:
    """Validated request plus its resolved plan, or blocking findings."""

    request: DeploymentRequest
    plan: ResolvedDeploymentPlan | None = None
    findings: tuple[PreflightFinding, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None and not any(
            finding.blocking for finding in self.findings
        )


def _legacy_config_for_request(request: DeploymentRequest) -> DeploymentConfig:
    """Render a request to the legacy shape the capability gate consumes."""
    from .quick_deploy import config_from_request

    return config_from_request(request)


def validate_request(request: DeploymentRequest) -> tuple[PreflightFinding, ...]:
    """Pure validation of one request; no I/O and no mutation."""
    findings: list[PreflightFinding] = []
    caps = capabilities(request.provider)
    provider_name = request.provider.display_name

    if request.backend not in caps.backends:
        findings.append(
            PreflightFinding(
                code="backend-unsupported",
                message=(
                    f"{provider_name} does not support "
                    f"{request.backend.display_name} deployments."
                ),
                fields=("backend",),
            )
        )
        return tuple(findings)

    # Prime provisions a serving pod for every request; there is no
    # preload-only or smoke-only execution behind those flags, so refusing is
    # cheaper than silently renting a serving instance.
    if request.intent == OperationIntent.PRELOAD and not caps.supports_preload_only:
        findings.append(
            PreflightFinding(
                code="preload-unsupported",
                message=(
                    f"{provider_name} has no preload-only operation; "
                    "select Deploy to rent an instance."
                ),
                fields=("intent",),
            )
        )
    elif request.intent == OperationIntent.SMOKE and not caps.supports_smoke_test_only:
        findings.append(
            PreflightFinding(
                code="smoke-unsupported",
                message=f"{provider_name} does not support smoke-test-only mode.",
                fields=("intent",),
            )
        )

    if request.backend == BackendType.VLLM:
        model = (request.vllm.model_name or "").strip()
        if not model:
            findings.append(
                PreflightFinding(
                    code="vllm-model-required",
                    message="Prime vLLM deployment requires --model-name."
                    if request.provider == ComputeProvider.PRIME
                    else "vLLM deployment requires a model name (--model-name).",
                    fields=("model_name",),
                )
            )
        if (
            request.vllm.n_gpu is not None
            and (request.gpu_count or 1) > 0
            and request.vllm.n_gpu > (request.gpu_count or 1)
        ):
            findings.append(
                PreflightFinding(
                    code="vllm-tensor-parallel-exceeds-gpus",
                    message=(
                        "vLLM tensor parallelism needs one GPU per shard: "
                        f"{request.vllm.n_gpu} requested across "
                        f"{request.gpu_count or 1} allocated."
                    ),
                    fields=("n_gpu", "gpu_count"),
                )
            )
    else:
        repo = (request.llamacpp.repo_id or "").strip()
        preset = (request.llamacpp.preset or "").strip()
        if not repo and not preset:
            findings.append(
                PreflightFinding(
                    code="llamacpp-repo-required",
                    message="Prime llama.cpp deployment requires --repo-id."
                    if request.provider == ComputeProvider.PRIME
                    else "llama.cpp deployment requires --repo-id.",
                    fields=("repo_id",),
                )
            )

    revision = (
        request.vllm.model_revision
        if request.backend == BackendType.VLLM
        else request.llamacpp.revision
    )
    if revision and request.backend not in caps.pinned_revision_backends:
        if request.provider == ComputeProvider.PRIME and request.backend == BackendType.LLAMACPP:
            message = (
                "Prime llama.cpp currently supports only the default Hugging Face revision."
            )
        else:
            message = (
                f"{provider_name} {request.backend.display_name} currently "
                "supports only the default HF revision."
            )
        findings.append(
            PreflightFinding(
                code="revision-unsupported",
                message=message,
                fields=("revision", "model_revision"),
            )
        )

    gpu_count = request.gpu_count or 1
    if gpu_count < 1:
        findings.append(
            PreflightFinding(
                code="gpu-count-invalid",
                message="GPU count must be at least 1.",
                fields=("gpu_count",),
            )
        )
    elif gpu_count > caps.max_gpu_count:
        if caps.max_gpu_count == 1:
            message = f"{provider_name} deployments currently support a single GPU."
        else:
            message = (
                f"{provider_name} deployments support at most "
                f"{caps.max_gpu_count} GPUs."
            )
        findings.append(
            PreflightFinding(
                code="gpu-count-exceeds-maximum",
                message=message,
                fields=("gpu_count",),
            )
        )

    if request.provider == ComputeProvider.PRIME:
        from ..protocol.models import PrimeProviderOptions

        if request.provider_options is not None and not isinstance(
            request.provider_options, PrimeProviderOptions
        ):
            findings.append(
                PreflightFinding(
                    code="provider-options-mismatch",
                    message="Deployment config contains non-Prime provider options.",
                    fields=("provider_options",),
                )
            )
    if request.provider == ComputeProvider.VAST:
        from ..protocol.models import VastProviderOptions

        options = request.provider_options
        if not isinstance(options, VastProviderOptions) or not options.offer_id.strip():
            findings.append(
                PreflightFinding(
                    code="vast-offer-required",
                    message="Vast.ai deployment requires a selected rental offer.",
                    fields=("provider_options",),
                )
            )

    if request.vision_mode == VisionMode.ON and not caps.supports_vision:
        findings.append(
            PreflightFinding(
                code="vision-unsupported",
                message=f"{provider_name} deployments currently support text models only.",
                fields=("vision_mode",),
            )
        )

    # The legacy capability gate stays the backstop while adapters consume
    # ``DeploymentConfig``: provider-specific checks (Vast runtime refusal)
    # must surface here with the same user-facing sentence.
    from .providers import capabilities as _capabilities

    legacy_refusal: str | None = None
    caps = _capabilities(request.provider)
    if caps.extra_refusal is not None:
        legacy_refusal = caps.extra_refusal(_legacy_config_for_request(request))
    if legacy_refusal and not any(finding.blocking for finding in findings):
        findings.append(
            PreflightFinding(
                code="capability-refused",
                message=legacy_refusal,
            )
        )
    return tuple(findings)


def resolve_request_identity(request: DeploymentRequest) -> DeploymentRequest:
    """Fill default instance/app identity without touching anything else."""
    if request.app_name and request.instance_name:
        return request
    from .naming import (
        build_deployment_name,
        infer_instance_from_app_name,
        slugify_instance_name,
    )

    app_override = (request.app_name or "").strip()
    instance_override = (request.instance_name or "").strip()
    if app_override:
        inferred = infer_instance_from_app_name(app_override, request.backend)
        instance_name = slugify_instance_name(
            instance_override or inferred or app_override
        )
        return replace(request, app_name=app_override, instance_name=instance_name)
    if instance_override:
        instance_name = slugify_instance_name(instance_override)
        return replace(
            request,
            instance_name=instance_name,
            app_name=build_deployment_name(
                request.provider, request.backend, instance_name
            ),
        )
    return request


def plan_from_request(
    request: DeploymentRequest,
    *,
    runtime_tuning: object | None = None,
    placement_assessment: object | None = None,
    reasoning: object | None = None,
    vision: object | None = None,
) -> ResolvedDeploymentPlan:
    """Build the effective plan for one request without allocating anything.

    Identity defaults are filled; deep model/tuning/placement resolution
    reuses the orchestrator's evidence pipeline and records its results here.
    The quoted execution price travels independently of the budget cap.
    """
    from typing import cast

    from ..protocol.models import (
        PlacementAssessment,
        ReasoningCapabilities,
        RuntimeTuning,
        VisionCapabilities,
    )

    resolved_request = resolve_request_identity(request)
    return ResolvedDeploymentPlan(
        request=resolved_request,
        backend=resolved_request.backend,
        provider=resolved_request.provider,
        intent=resolved_request.intent,
        do_warmup=resolved_request.do_warmup,
        show_debug_logs=resolved_request.show_debug_logs,
        vision=cast("VisionCapabilities | None", vision),
        projector_repo=resolved_request.projector_repo,
        projector_revision=resolved_request.projector_revision,
        projector_file=resolved_request.projector_file,
        image_limit=resolved_request.image_limit,
        mm_processor_kwargs=resolved_request.mm_processor_kwargs,
        llamacpp=resolved_request.llamacpp,
        vllm=resolved_request.vllm,
        gpu_type=resolved_request.gpu_type,
        gpu_count=resolved_request.gpu_count,
        required_vram_gb=resolved_request.required_vram_gb,
        serving_requirements=resolved_request.serving_requirements,
        runtime_tuning=cast("RuntimeTuning | None", runtime_tuning),
        placement_assessment=cast("PlacementAssessment | None", placement_assessment),
        reasoning=cast("ReasoningCapabilities | None", reasoning),
        max_context_tokens=resolved_request.max_context_tokens,
        max_output_tokens=resolved_request.max_output_tokens,
        instance_name=resolved_request.instance_name,
        app_name=resolved_request.app_name,
        function_slug=resolved_request.function_slug,
        provider_options=resolved_request.provider_options,
        cost_scenario_id=resolved_request.cost_scenario_id,
        output_tokens_per_month=resolved_request.output_tokens_per_month,
        price_per_hour_usd=resolved_request.price_per_hour_usd,
    )


def revalidate_plan_price(
    plan: ResolvedDeploymentPlan, *, current_price_per_hour_usd: float | None
) -> PreflightFinding | None:
    """Recheck a time-sensitive quote immediately before provisioning.

    A refreshed price above the approved ``max_hourly_cost_usd`` cap refuses
    the allocation like a price increase; an accepted refreshed price is
    recorded explicitly by the caller. Actual invoice charges stay separate.
    """
    cap: float | None = None
    if plan.serving_requirements is not None:
        cap = plan.serving_requirements.max_hourly_cost_usd
    if cap is None or current_price_per_hour_usd is None:
        return None
    try:
        if float(current_price_per_hour_usd) <= float(cap) + 1e-9:
            return None
    except (TypeError, ValueError):
        return None
    return PreflightFinding(
        code="price-cap-exceeded",
        message=(
            f"Current price ${current_price_per_hour_usd:.2f}/h exceeds the "
            f"approved cap of ${cap:.2f}/h. Re-approve before provisioning."
        ),
        fields=("price_per_hour_usd",),
    )


def preflight_request(request: DeploymentRequest) -> PreflightResult:
    """Validate a request and resolve it to an immutable plan.

    Resolution here is deliberately shallow: identity defaults only. Full
    model/tuning/placement resolution happens inside the lifecycle runner so
    the orchestrator's existing evidence pipeline is reused rather than
    forked. The returned plan records exactly what the adapters will execute.
    """
    findings = validate_request(request)
    blocking = [finding for finding in findings if finding.blocking]
    if blocking:
        return PreflightResult(request=request, plan=None, findings=tuple(findings))
    plan = plan_from_request(request)
    return PreflightResult(
        request=plan.request, plan=plan, findings=tuple(findings)
    )


def preflight_config(config: DeploymentConfig) -> PreflightResult:
    """Validate a legacy config through the same authoritative gate."""
    from .quick_deploy import request_from_config

    try:
        request = request_from_config(config)
    except ValueError as exc:
        return PreflightResult(
            request=DeploymentRequest(),
            findings=(
                PreflightFinding(
                    code="intent-unknown", message=str(exc), fields=("intent",)
                ),
            ),
        )
    return preflight_request(request)


@dataclass(frozen=True)
class LifecycleAttemptSpec:
    """Config plus its resolved plan for one lifecycle attempt."""

    config: DeploymentConfig
    plan: ResolvedDeploymentPlan | None
    retry_allowed: bool


def lifecycle_attempts_for_configs(
    configs: list[DeploymentConfig] | tuple[DeploymentConfig, ...],
) -> list[LifecycleAttemptSpec]:
    """Build lifecycle attempts with resolved plans attached when valid.

    Callers construct :class:`LifecycleAttempt` from these so execution
    consumes the resolved plan (intent, quoted price, identity) while adapters
    still receive the legacy config shape. Invalid configs keep ``plan`` None
    so the orchestrator's preflight gate reports the same message.
    """
    attempts: list[LifecycleAttemptSpec] = []
    for config in configs:
        result = preflight_config(config)
        attempts.append(
            LifecycleAttemptSpec(
                config=config,
                plan=result.plan,
                retry_allowed=config.retry_allowed,
            )
        )
    return attempts


__all__ = [
    "LifecycleAttemptSpec",
    "PreflightFinding",
    "PreflightResult",
    "lifecycle_attempts_for_configs",
    "plan_from_request",
    "preflight_config",
    "preflight_request",
    "resolve_request_identity",
    "revalidate_plan_price",
    "validate_request",
]

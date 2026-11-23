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
import math
import shlex
from pathlib import PurePosixPath

from ..gguf_files import is_dspark_gguf, is_target_gguf
from ..protocol.enums import (
    BackendType,
    ComputeProvider,
    OperationIntent,
    VisionMode,
)
from ..protocol.models import (
    DeploymentConfig,
    DeploymentRequest,
    DSparkDraft,
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

    llama = request.llamacpp
    if llama.dspark or llama.dspark_repo or llama.dspark_file:
        if request.provider != ComputeProvider.MODAL or request.backend != BackendType.LLAMACPP:
            findings.append(PreflightFinding("dspark-provider", "DSpark requires Modal with llama.cpp."))
        elif (
            not (llama.repo_id or "").strip()
            or not (llama.quant or "").strip()
            or request.serving_requirements is not None
        ):
            findings.append(PreflightFinding(
                "dspark-target", "DSpark requires a Hugging Face target repo and quantization in Advanced Deploy.",
            ))
        elif llama.dspark_repo and not llama.dspark_file:
            findings.append(PreflightFinding(
                "dspark-file", "Select an exact DSpark draft filename when using a separate draft repository.",
            ))

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


def resolve_dspark_draft(config: DeploymentConfig) -> DSparkDraft:
    """Pin the matching draft and check the joint weight budget before allocation."""
    from huggingface_hub import HfApi

    from .modal_gpu import MODAL_GPU_MEMORY_GB

    # A managed draft cannot share llama.cpp's draft arguments with a manual one.
    draft_flags = {
        "-md", "-hfd", "-hfrd", "--model-draft", "--hf-repo-draft", "--spec-type",
        "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft", "-devd", "--device-draft",
    }
    args = shlex.split(config.server_args or "")
    if any(arg.split("=", 1)[0] in draft_flags or arg.startswith("--spec-draft-") for arg in args):
        raise ValueError("Remove the manual speculative decoding arguments before enabling DSpark.")

    api = HfApi()
    target_repo = (config.repo_id or "").strip()
    target = api.model_info(target_repo, revision=config.revision, files_metadata=True, timeout=10)
    draft_repo = (config.dspark_repo or "").strip() or target_repo
    draft = target if draft_repo.casefold() == target_repo.casefold() else api.model_info(
        draft_repo, files_metadata=True, timeout=10,
    )
    candidates = {
        f.rfilename: f.size for f in draft.siblings or [] if is_dspark_gguf(f.rfilename)
    }
    filename = (config.dspark_file or "").strip()
    if not filename:
        preferred = [name for name in candidates if "q8_0" in name.casefold()]
        choices = preferred or list(candidates)
        if len(choices) != 1:
            detail = ", ".join(sorted(choices)) or "none"
            raise ValueError(f"Select an exact DSpark draft filename. Found: {detail}")
        filename = choices[0]
    size = candidates.get(filename)
    path = PurePosixPath(filename)
    if size is None or size <= 0 or path.is_absolute() or ".." in path.parts or not draft.sha or not target.sha:
        raise ValueError("The DSpark draft must be a named GGUF with a known size in the selected repository.")

    quant = (config.quant or "").casefold()
    weights = [
        f.size for f in target.siblings or []
        if is_target_gguf(f.rfilename) and quant in PurePosixPath(f.rfilename).name.casefold()
    ]
    if not weights or any(weight is None or weight <= 0 for weight in weights):
        raise ValueError("Could not verify the selected main GGUF weight size for DSpark.")
    minimum_gb = max(sum(weight or 0 for weight in weights) / 1e9, config.required_vram_gb or 0) + size / 1e9
    capacity_gb = MODAL_GPU_MEMORY_GB.get((config.gpu_type or "").upper())
    if capacity_gb is not None and minimum_gb > capacity_gb * (config.gpu_count or 1):
        raise ValueError(f"The main model and DSpark draft need at least {minimum_gb:.1f} GB before runtime overhead.")
    config.revision = target.sha
    return DSparkDraft(draft_repo, draft.sha, filename, size)


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
    return price_cap_finding(cap, current_price_per_hour_usd)


def price_cap_finding(cap: float | None, current: float | None) -> PreflightFinding | None:
    """Validate a refreshed provider charge against the user's approved amount."""
    if cap is None:
        return None
    if current is None or not math.isfinite(current) or current <= 0:
        return PreflightFinding("price-unknown", "Current price could not be verified. Refresh offers before provisioning.")
    if math.isfinite(cap) and current <= cap + 1e-9:
        return None
    return PreflightFinding("price-cap-exceeded",
                            f"Current price ${current:.4f} exceeds the approved cap of ${cap:.4f}. Refresh offers.",
                            fields=("price_per_hour_usd",))


def preflight_request(
    request: DeploymentRequest,
    *,
    runtime_tuning: object | None = None,
    placement_assessment: object | None = None,
    reasoning: object | None = None,
    vision: object | None = None,
) -> PreflightResult:
    """Validate a request and resolve it to an immutable plan.

    Resolution here is deliberately shallow: identity defaults only. Full
    model/tuning/placement resolution happens inside the lifecycle runner so
    the orchestrator's existing evidence pipeline is reused rather than
    forked. The returned plan records exactly what the adapters will execute.

    Resolution a caller has already done is passed through rather than
    rediscovered: a ``DeploymentRequest`` carries intent, not evidence, so
    Fast Deploy's certified tuning and placement assessment have no field to
    travel in and would otherwise be dropped here and then assigned -- as
    ``None`` -- onto the config the lifecycle executes.
    """
    findings = validate_request(request)
    blocking = [finding for finding in findings if finding.blocking]
    if blocking:
        return PreflightResult(request=request, plan=None, findings=tuple(findings))
    plan = plan_from_request(
        request,
        runtime_tuning=runtime_tuning,
        placement_assessment=placement_assessment,
        reasoning=reasoning,
        vision=vision,
    )
    return PreflightResult(
        request=plan.request, plan=plan, findings=tuple(findings)
    )


def preflight_config(config: DeploymentConfig) -> PreflightResult:
    """Validate a legacy config through the same authoritative gate.

    Resolution the caller already did travels with the plan. A
    ``DeploymentRequest`` carries intent, not evidence, so tuning and the
    placement assessment cannot survive the round trip through it -- and the
    lifecycle assigns the plan's values onto the config unconditionally, so a
    plan that resolved neither *erased* both. Fast Deploy's certified placement
    then reached the runtime as a tuning re-derived from the objective, and
    certification failed outright with "No placement assessment was supplied".
    """
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
    return preflight_request(
        request,
        runtime_tuning=config.runtime_tuning,
        placement_assessment=config.placement_assessment,
        reasoning=config.reasoning,
        vision=config.vision,
    )


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

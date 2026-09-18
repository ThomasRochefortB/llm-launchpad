"""Deploy-time provider capabilities and the single refusal gate.

``ProviderCapabilities`` in ``protocol/models.py`` describes the *quote*
boundary: which backends a provider can price, and how it bills. These are the
deploy-time facts instead — what a provider can actually run once a
configuration is built.

Every deploy path asks :func:`refuse` before spending anything, so a form never
offers a control the backend will reject after routing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import DeploymentConfig, EndpointInfo
from .vast_runtime import VAST_MAX_GPU_COUNT


@dataclass(frozen=True)
class DeploymentCapabilities:
    """What one provider can deploy, independent of any particular model."""

    provider: ComputeProvider
    backends: frozenset[BackendType]
    max_gpu_count: int
    supports_vision: bool
    # Pinning an HF revision is a runtime property, not a provider one:
    # llama.cpp's `--hf-repo owner/repo:quant` cannot carry a revision, while
    # vLLM's `--revision` can.
    pinned_revision_backends: frozenset[BackendType] = frozenset()
    supports_preload_only: bool = True
    supports_smoke_test_only: bool = True
    public_endpoint: bool = True
    # Prime and Vast rent a specific marketplace offer, so GPU type and count
    # are read from it rather than chosen.
    gpu_shape_from_offer: bool = False
    extra_refusal: Callable[[DeploymentConfig], str | None] | None = field(
        default=None, compare=False, repr=False
    )


def _vast_extra_refusal(config: DeploymentConfig) -> str | None:
    """Defer Vast's runtime-specific detail to the module that owns it."""
    from .vast_runtime import vast_refusal

    return vast_refusal(config)


_CAPABILITIES: dict[ComputeProvider, DeploymentCapabilities] = {
    ComputeProvider.MODAL: DeploymentCapabilities(
        provider=ComputeProvider.MODAL,
        backends=frozenset({BackendType.LLAMACPP, BackendType.VLLM}),
        max_gpu_count=8,
        supports_vision=True,
        pinned_revision_backends=frozenset({BackendType.LLAMACPP, BackendType.VLLM}),
    ),
    ComputeProvider.PRIME: DeploymentCapabilities(
        provider=ComputeProvider.PRIME,
        backends=frozenset({BackendType.LLAMACPP, BackendType.VLLM}),
        max_gpu_count=8,
        supports_vision=True,
        pinned_revision_backends=frozenset({BackendType.VLLM}),
        # Prime provisions a serving pod for every request; there is no
        # preload-only or smoke-only execution behind these flags.
        supports_preload_only=False,
        supports_smoke_test_only=False,
        gpu_shape_from_offer=True,
    ),
    ComputeProvider.VAST: DeploymentCapabilities(
        provider=ComputeProvider.VAST,
        backends=frozenset({BackendType.LLAMACPP, BackendType.VLLM}),
        max_gpu_count=VAST_MAX_GPU_COUNT,
        supports_vision=True,
        # llama.cpp's --hf-repo cannot carry a revision; vLLM's --revision can.
        pinned_revision_backends=frozenset({BackendType.VLLM}),
        supports_preload_only=False,
        supports_smoke_test_only=False,
        public_endpoint=False,
        gpu_shape_from_offer=True,
        extra_refusal=_vast_extra_refusal,
    ),
}


def capabilities(provider: ComputeProvider) -> DeploymentCapabilities:
    """Return what this provider can deploy."""
    try:
        return _CAPABILITIES[provider]
    except KeyError:
        raise ValueError(f"Unknown compute provider: {provider}") from None


def refuse(config: DeploymentConfig) -> str | None:
    """Explain why this configuration cannot be deployed, or return None.

    Compatibility wrapper over the authoritative preflight: new callers should
    use :mod:`llm_launchpad.core.deployment_preflight` for structured findings.
    The user-facing sentence stays identical across forms, CLI, and backends.
    """
    from .deployment_preflight import preflight_config

    findings = preflight_config(config).findings
    blocking = [finding for finding in findings if finding.blocking]
    return blocking[0].message if blocking else None


def revision_refusal(provider: ComputeProvider, backend: BackendType) -> str | None:
    """Explain why this provider cannot pin an HF revision for this runtime."""
    caps = capabilities(provider)
    if backend in caps.pinned_revision_backends:
        return None
    return (
        f"{provider.display_name} {backend.display_name} currently supports only "
        "the default HF revision."
    )


def connected_providers(
    *,
    modal_available: bool,
    prime_authenticated: bool,
    vast_configured: bool,
) -> tuple[ComputeProvider, ...]:
    """List providers with usable credentials, in stable presentation order."""
    connected = []
    if modal_available:
        connected.append(ComputeProvider.MODAL)
    if prime_authenticated:
        connected.append(ComputeProvider.PRIME)
    if vast_configured:
        connected.append(ComputeProvider.VAST)
    return tuple(connected)


def deployment_lister(provider: ComputeProvider) -> Callable[[], list[EndpointInfo] | None]:
    """Return the callable that lists one provider's deployments.

    Normalized by :mod:`llm_launchpad.core.provider_adapters` into a
    :class:`ProviderListing`: Modal's unavailable ``None`` and the
    marketplace providers' exceptions both become error listings there, so
    callers no longer branch on provider to interpret failures.
    """
    if provider == ComputeProvider.VAST:
        from .vast_deployment import VastDeploymentBackend

        return VastDeploymentBackend().list_deployments
    if provider == ComputeProvider.PRIME:
        from .prime_backend import PrimeBackend

        return PrimeBackend().list_deployments
    from .backend import ModalBackend

    return ModalBackend.list_apps

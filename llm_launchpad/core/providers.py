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
        supports_smoke_test_only=False,
        gpu_shape_from_offer=True,
    ),
    ComputeProvider.VAST: DeploymentCapabilities(
        provider=ComputeProvider.VAST,
        backends=frozenset({BackendType.LLAMACPP}),
        max_gpu_count=1,
        supports_vision=False,
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

    Callers must treat a returned string as final and user-facing: it is the
    same sentence whether it surfaces in a form, the CLI, or a backend.
    """

    caps = capabilities(config.provider)
    provider = config.provider.display_name

    if config.backend not in caps.backends:
        return f"{provider} does not support {config.backend.display_name} deployments."
    if config.vision is not None and config.vision.enabled and not caps.supports_vision:
        return f"{provider} deployments currently support text models only."
    if config.revision and config.backend not in caps.pinned_revision_backends:
        return (
            f"{provider} {config.backend.display_name} currently supports only "
            "the default HF revision."
        )
    gpu_count = config.gpu_count or 1
    if gpu_count > caps.max_gpu_count:
        if caps.max_gpu_count == 1:
            return f"{provider} deployments currently support a single GPU."
        return f"{provider} deployments support at most {caps.max_gpu_count} GPUs."
    if not config.do_deploy and not config.run_smoke and not caps.supports_preload_only:
        return f"{provider} has no preload-only operation; select Deploy to rent an instance."
    if config.run_smoke and not caps.supports_smoke_test_only:
        return f"{provider} does not support smoke-test-only mode."
    if caps.extra_refusal is not None:
        return caps.extra_refusal(config)
    return None


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

    Modal reports an unavailable listing as ``None``; the marketplace providers
    raise instead. Callers must handle both.
    """
    if provider == ComputeProvider.VAST:
        from .vast_deployment import VastDeploymentBackend

        return VastDeploymentBackend().list_deployments
    if provider == ComputeProvider.PRIME:
        from .prime_backend import PrimeBackend

        return PrimeBackend().list_deployments
    from .backend import ModalBackend

    return ModalBackend.list_apps

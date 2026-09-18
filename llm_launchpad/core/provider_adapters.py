"""Normalized provider operations behind one adapter contract.

Consumers used to branch on provider to interpret listing failures (Modal
returns ``None`` when unavailable; marketplace providers raise) and to select
implementations. Adapters normalize those differences at the boundary: every
listing is a :class:`ProviderListing`, every stop is a typed cleanup outcome,
and provider-specific deploy/stop execution lives behind
:meth:`ProviderAdapter.deploy` / :meth:`ProviderAdapter.stop` instead of
``if provider == ...`` branches scattered across callers.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..protocol.enums import (
    BackendType,
    CleanupDisposition,
    ComputeProvider,
    OperationType,
)
from ..protocol.events import BaseEvent, OperationCompleteEvent
from ..protocol.models import (
    DeploymentConfig,
    EndpointInfo,
    ProviderListing,
)
from .deployment_states import is_terminal_deployment_state

EventStream = Generator[BaseEvent, None, None]


@dataclass(frozen=True)
class ProviderResource:
    """Normalized handle for an allocated provider resource."""

    provider: ComputeProvider
    app_name: str
    resource_id: str | None = None
    attempt_id: str | None = None


@dataclass(frozen=True)
class ProviderCleanup:
    """Confirmed outcome of stopping one resource."""

    disposition: CleanupDisposition
    detail: str = ""
    storage_consequence: str = ""
    remains_billable: bool = False


@dataclass(frozen=True)
class ProviderDeploymentRollback:
    """Structured provider-local rollback report for a failed deploy."""

    attempted: bool = False
    confirmed: bool = False
    detail: str = ""


def rollback_from_event(event: BaseEvent | None) -> ProviderDeploymentRollback | None:
    """Extract a provider-local rollback report from a deploy completion."""
    if not isinstance(event, OperationCompleteEvent):
        return None
    data = event.data
    if not isinstance(data, dict):
        return None
    marker = data.get("rollback")
    if not isinstance(marker, dict):
        return None
    try:
        return ProviderDeploymentRollback(
            attempted=bool(marker.get("attempted", True)),
            confirmed=bool(marker.get("confirmed", False)),
            detail=str(marker.get("detail") or ""),
        )
    except Exception:
        return None


class ProviderAdapter(Protocol):
    """Normalized deploy/list/stop operations for one provider."""

    @property
    def provider(self) -> ComputeProvider:
        """Which provider this adapter talks to."""

    def list(self) -> ProviderListing:
        """List deployments, normalizing unavailable as an error listing."""

    def deploy(self, config: DeploymentConfig) -> EventStream:
        """Execute provider-specific provisioning, yielding protocol events."""

    def stop(self, resource: ProviderResource) -> ProviderCleanup:
        """Stop one resource and report the confirmed outcome."""


@dataclass
class _ModalAdapter:
    """Modal execution behind the adapter contract."""

    provider: ComputeProvider = ComputeProvider.MODAL

    def list(self) -> ProviderListing:
        from . import providers as _providers

        lister = _providers.deployment_lister(self.provider)
        try:
            rows = lister()
        except Exception as exc:
            return ProviderListing(provider=self.provider, error=str(exc))
        if rows is None:
            return ProviderListing(
                provider=self.provider, error="Provider listing unavailable."
            )
        return ProviderListing(provider=self.provider, rows=tuple(rows))

    def deploy(self, config: DeploymentConfig) -> EventStream:
        # Modal deploy is driven by the orchestrator's shared preparation
        # (preflight, tuning, env). The adapter owns only the Modal-specific
        # execution step; the orchestrator delegates here instead of
        # branching on provider.
        orchestrator = _orchestrator_for_adapter()
        yield from orchestrator.deploy_modal_only(config)

    def stop(self, resource: ProviderResource) -> ProviderCleanup:
        from .backend import ModalBackend
        from ..protocol.events import ErrorEvent

        target = (resource.resource_id or resource.app_name or "").strip()
        if not target:
            return ProviderCleanup(
                disposition=CleanupDisposition.FAILED,
                detail="Modal stop requires an app name.",
            )
        detail = ""
        success = False
        saw_completion = False
        try:
            cmd = ["modal", "app", "stop", "--yes", target]
            for event in ModalBackend.run_streaming(cmd):
                if isinstance(event, OperationCompleteEvent):
                    saw_completion = True
                    success = bool(event.success)
                    detail = event.detail or detail
                elif isinstance(event, ErrorEvent):
                    detail = event.message or detail
        except Exception as exc:
            return ProviderCleanup(
                disposition=CleanupDisposition.FAILED, detail=str(exc)
            )
        if not saw_completion:
            return ProviderCleanup(
                disposition=CleanupDisposition.FAILED,
                detail=detail or "Stop did not confirm.",
            )
        if success:
            storage, billable = _stop_consequence(resource.provider)
            return ProviderCleanup(
                disposition=CleanupDisposition.CONFIRMED,
                detail=detail,
                storage_consequence=storage,
                remains_billable=billable,
            )
        return ProviderCleanup(disposition=CleanupDisposition.FAILED, detail=detail)


@dataclass
class _PrimeAdapter:
    """Prime Intellect execution behind the adapter contract."""

    provider: ComputeProvider = ComputeProvider.PRIME

    def list(self) -> ProviderListing:
        from .prime_backend import PrimeBackend

        try:
            rows = PrimeBackend().list_deployments()
        except Exception as exc:
            return ProviderListing(provider=self.provider, error=str(exc))
        if rows is None:
            return ProviderListing(
                provider=self.provider, error="Provider listing unavailable."
            )
        return ProviderListing(provider=self.provider, rows=tuple(rows))

    def deploy(self, config: DeploymentConfig) -> EventStream:
        orchestrator = _orchestrator_for_adapter()
        yield from orchestrator.deploy_prime_only(config)

    def stop(self, resource: ProviderResource) -> ProviderCleanup:
        from .prime_backend import PrimeBackend

        pod_id = (resource.resource_id or "").strip()
        if not pod_id:
            return ProviderCleanup(
                disposition=CleanupDisposition.FAILED,
                detail="Prime termination requires a pod ID.",
            )
        try:
            PrimeBackend().delete_pod(pod_id)
        except Exception as exc:
            return ProviderCleanup(
                disposition=CleanupDisposition.FAILED, detail=str(exc)
            )
        storage, billable = _stop_consequence(self.provider)
        return ProviderCleanup(
            disposition=CleanupDisposition.CONFIRMED,
            detail=f"Terminated Prime pod: {pod_id}",
            storage_consequence=storage,
            remains_billable=billable,
        )


@dataclass
class _VastAdapter:
    """Vast.ai execution behind the adapter contract."""

    provider: ComputeProvider = ComputeProvider.VAST

    def list(self) -> ProviderListing:
        from .vast_deployment import VastDeploymentBackend

        try:
            rows = VastDeploymentBackend().list_deployments()
        except Exception as exc:
            return ProviderListing(provider=self.provider, error=str(exc))
        if rows is None:
            return ProviderListing(
                provider=self.provider, error="Provider listing unavailable."
            )
        return ProviderListing(provider=self.provider, rows=tuple(rows))

    def deploy(self, config: DeploymentConfig) -> EventStream:
        from .vast_deployment import VastDeploymentBackend

        yield from VastDeploymentBackend().deploy(config)

    def stop(self, resource: ProviderResource) -> ProviderCleanup:
        from .vast_deployment import VastDeploymentBackend

        try:
            VastDeploymentBackend().destroy(
                name=resource.app_name or None,
                instance_id=resource.resource_id,
            )
        except Exception as exc:
            return ProviderCleanup(
                disposition=CleanupDisposition.FAILED, detail=str(exc)
            )
        storage, billable = _stop_consequence(self.provider)
        return ProviderCleanup(
            disposition=CleanupDisposition.CONFIRMED,
            detail="Vast rental and disk destroyed.",
            storage_consequence=storage,
            remains_billable=billable,
        )


def _orchestrator_for_adapter() -> Any:
    from .orchestrator import Orchestrator

    return Orchestrator()


_ADAPTERS: dict[ComputeProvider, ProviderAdapter] = {
    ComputeProvider.MODAL: _ModalAdapter(),
    ComputeProvider.PRIME: _PrimeAdapter(),
    ComputeProvider.VAST: _VastAdapter(),
}


def provider_adapter(provider: ComputeProvider) -> ProviderAdapter:
    """Return the concrete adapter for one provider."""
    try:
        return _ADAPTERS[provider]
    except KeyError:
        raise ValueError(f"Unknown compute provider: {provider}") from None


@dataclass
class _ListingAdapter:
    """Wrap the legacy per-provider listing callables (list-only compat)."""

    provider: ComputeProvider
    lister: Callable[[], Sequence[EndpointInfo] | None]

    def list(self) -> ProviderListing:

        try:
            rows = self.lister()
        except Exception as exc:
            return ProviderListing(provider=self.provider, error=str(exc))
        if rows is None:
            return ProviderListing(
                provider=self.provider, error="Provider listing unavailable."
            )
        return ProviderListing(provider=self.provider, rows=tuple(rows))

    def deploy(self, config: DeploymentConfig) -> EventStream:
        yield from provider_adapter(self.provider).deploy(config)

    def stop(self, resource: ProviderResource) -> ProviderCleanup:
        return provider_adapter(self.provider).stop(resource)


def listing_adapter(provider: ComputeProvider) -> _ListingAdapter:
    """Return a listing adapter using the legacy per-provider callables."""
    from . import providers as _providers

    return _ListingAdapter(provider=provider, lister=_providers.deployment_lister(provider))


def discover_fleet(
    providers: tuple[ComputeProvider, ...] | list[ComputeProvider],
) -> tuple[ProviderListing, ...]:
    """List every provider, normalizing failures to error listings."""
    return tuple(provider_adapter(provider).list() for provider in providers)


def stop_with_orchestrator(
    orchestrator: Any,
    resource: ProviderResource,
    *,
    backend: BackendType = BackendType.LLAMACPP,
) -> ProviderCleanup:
    """Stop one resource through the orchestrator, normalizing its events."""
    detail = ""
    success = False
    saw_completion = False
    try:
        for event in orchestrator.stop_app(
            backend,
            app_name=resource.app_name or None,
            app_id=resource.resource_id,
            provider=resource.provider,
        ):
            if isinstance(event, OperationCompleteEvent) and event.operation == OperationType.STOP:
                saw_completion = True
                success = bool(event.success)
                detail = event.detail or ""
    except Exception as exc:
        return ProviderCleanup(
            disposition=CleanupDisposition.FAILED, detail=str(exc)
        )
    if not saw_completion:
        return ProviderCleanup(
            disposition=CleanupDisposition.FAILED,
            detail=detail or "Stop did not confirm.",
        )
    if success:
        storage, billable = _stop_consequence(resource.provider)
        return ProviderCleanup(
            disposition=CleanupDisposition.CONFIRMED,
            detail=detail,
            storage_consequence=storage,
            remains_billable=billable,
        )
    return ProviderCleanup(disposition=CleanupDisposition.FAILED, detail=detail)


def stop_resource(resource: ProviderResource) -> ProviderCleanup:
    """Stop one resource through its provider adapter (no orchestrator)."""
    return provider_adapter(resource.provider).stop(resource)


def _stop_consequence(provider: ComputeProvider) -> tuple[str, bool]:
    if provider == ComputeProvider.PRIME:
        return "Prime cache disk is kept and remains billable.", True
    if provider == ComputeProvider.VAST:
        return "Vast rental disk is destroyed with the rental.", False
    return "Shared Modal cache is kept for faster redeploys.", False


def is_live_row(row: EndpointInfo) -> bool:
    """Whether a fleet row names a deployment that may still serve."""
    return not is_terminal_deployment_state(row.state or "")


__all__ = [
    "EventStream",
    "ProviderAdapter",
    "ProviderCleanup",
    "ProviderDeploymentRollback",
    "ProviderResource",
    "discover_fleet",
    "is_live_row",
    "listing_adapter",
    "provider_adapter",
    "rollback_from_event",
    "stop_resource",
    "stop_with_orchestrator",
]

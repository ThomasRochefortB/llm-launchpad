"""Shared provider readiness: installed vs credentials vs verified access.

Installation, credential presence, and verified access are different states.
The TUI setup gate must not treat an installed CLI as authenticated, and a
stored API key must not be reported as verified before a live check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

from typing import Any

from ..protocol.enums import ComputeProvider
from .backend import ModalBackend
from .modal_auth import get_modal_auth_status
from .prime_auth import PrimeConfig, load_prime_config
from .vast_auth import VastCredentials, resolve_vast_credentials


class ProviderReadinessStage(str, Enum):
    """Distinct lifecycle states for one provider's local readiness."""

    NOT_INSTALLED = "not_installed"
    MISSING_CREDENTIALS = "missing_credentials"
    CREDENTIALS_PRESENT = "credentials_present"
    READY = "ready"
    AUTH_FAILED = "auth_failed"
    UNREACHABLE = "unreachable"

    @property
    def display_name(self) -> str:
        return {
            ProviderReadinessStage.NOT_INSTALLED: "Not installed",
            ProviderReadinessStage.MISSING_CREDENTIALS: "Not configured",
            ProviderReadinessStage.CREDENTIALS_PRESENT: "Checking",
            ProviderReadinessStage.READY: "Authenticated",
            ProviderReadinessStage.AUTH_FAILED: "Authentication failed",
            ProviderReadinessStage.UNREACHABLE: "Verification unavailable",
        }[self]


@dataclass(frozen=True)
class ProviderReadiness:
    """One provider's readiness with an actionable recovery hint."""

    provider: ComputeProvider
    stage: ProviderReadinessStage
    detail: str = ""
    hint: str | None = None
    checked_at_epoch: float = 0.0

    @property
    def has_credentials(self) -> bool:
        """Whether local state is sufficient to attempt a verified check."""
        return self.stage in (
            ProviderReadinessStage.CREDENTIALS_PRESENT,
            ProviderReadinessStage.READY,
            ProviderReadinessStage.AUTH_FAILED,
            ProviderReadinessStage.UNREACHABLE,
        )

    @property
    def verified(self) -> bool:
        return self.stage == ProviderReadinessStage.READY

    @property
    def display(self) -> str:
        base = self.stage.display_name
        return f"{base}: {self.detail}" if self.detail else base


_READINESS_TTL_SECONDS = 60.0
_readiness_cache: dict[tuple[str, bool], tuple[float, ProviderReadiness]] = {}


def _cached(provider: ComputeProvider, verify: bool) -> ProviderReadiness | None:
    entry = _readiness_cache.get((provider.value, verify))
    if entry is None:
        return None
    checked_at, readiness = entry
    if time.time() - checked_at > _READINESS_TTL_SECONDS:
        return None
    return readiness


def _store(provider: ComputeProvider, verify: bool, readiness: ProviderReadiness) -> ProviderReadiness:
    stamped = ProviderReadiness(
        provider=readiness.provider,
        stage=readiness.stage,
        detail=readiness.detail,
        hint=readiness.hint,
        checked_at_epoch=time.time(),
    )
    _readiness_cache[(provider.value, verify)] = (stamped.checked_at_epoch, stamped)
    return stamped


def clear_provider_readiness_cache() -> None:
    """Forget cached verification results, e.g. after credentials change."""
    _readiness_cache.clear()


def invalidate_provider_readiness(provider: ComputeProvider | None = None) -> None:
    """Drop cached readiness for one provider, or all when omitted."""
    if provider is None:
        clear_provider_readiness_cache()
        return
    for key in [key for key in _readiness_cache if key[0] == provider.value]:
        _readiness_cache.pop(key, None)


def _is_network_failure(message: str) -> bool:
    lowered = (message or "").casefold()
    return any(
        marker in lowered
        for marker in (
            "timed out",
            "timeout",
            "could not reach",
            "check your connection",
            "connection",
            "network",
            "temporarily",
            "rate limit",
            "http 5",
            "http 429",
        )
    )


def check_modal_readiness(*, verify: bool = True, refresh: bool = False) -> ProviderReadiness:
    """Distinguish an installed Modal CLI from verified Modal authentication."""
    if not refresh:
        cached = _cached(ComputeProvider.MODAL, verify)
        if cached is not None:
            return cached
    if not ModalBackend.is_cli_available():
        return _store(
            ComputeProvider.MODAL,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.MODAL,
                stage=ProviderReadinessStage.NOT_INSTALLED,
                detail="Modal CLI not found",
                hint="reinstall llm-launchpad, then run: modal setup",
            ),
        )
    if not verify:
        return _store(
            ComputeProvider.MODAL,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.MODAL,
                stage=ProviderReadinessStage.CREDENTIALS_PRESENT,
                detail="Modal CLI installed; authentication not verified",
                hint="run: modal setup",
            ),
        )
    try:
        status = get_modal_auth_status()
    except Exception as exc:
        return _store(
            ComputeProvider.MODAL,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.MODAL,
                stage=ProviderReadinessStage.UNREACHABLE,
                detail=f"Modal auth check failed: {exc}",
                hint="run: modal setup",
            ),
        )
    if status.authenticated:
        detail = "authenticated"
        if status.profile:
            detail += f" (profile: {status.profile})"
        elif status.detail:
            detail += f" ({status.detail})"
        return _store(
            ComputeProvider.MODAL,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.MODAL,
                stage=ProviderReadinessStage.READY,
                detail=detail,
            ),
        )
    message = status.error or "not authenticated"
    if _is_network_failure(message):
        return _store(
            ComputeProvider.MODAL,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.MODAL,
                stage=ProviderReadinessStage.UNREACHABLE,
                detail=message,
                hint="check your connection, then retry",
            ),
        )
    return _store(
        ComputeProvider.MODAL,
        verify,
        ProviderReadiness(
            provider=ComputeProvider.MODAL,
            stage=ProviderReadinessStage.AUTH_FAILED,
            detail=message,
            hint="run: modal setup",
        ),
    )


def check_prime_readiness(
    *, verify: bool = True, refresh: bool = False, backend: Any | None = None
) -> ProviderReadiness:
    """A stored Prime key is credentials-present, not authenticated."""
    if not refresh:
        cached = _cached(ComputeProvider.PRIME, verify)
        if cached is not None:
            return cached
    try:
        config = load_prime_config()
    except Exception as exc:
        return _store(
            ComputeProvider.PRIME,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.PRIME,
                stage=ProviderReadinessStage.MISSING_CREDENTIALS,
                detail=str(exc),
                hint="run: prime login (or set PRIME_API_KEY)",
            ),
        )
    if not config.api_key:
        return _store(
            ComputeProvider.PRIME,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.PRIME,
                stage=ProviderReadinessStage.MISSING_CREDENTIALS,
                detail="no API key configured",
                hint="run: prime login (or set PRIME_API_KEY)",
            ),
        )
    if not verify:
        return _store(
            ComputeProvider.PRIME,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.PRIME,
                stage=ProviderReadinessStage.CREDENTIALS_PRESENT,
                detail="API key found; not verified over the network",
                hint="verification runs before allocation",
            ),
        )
    try:
        # Imported at module level for test patching; see top of file.
        owner = backend if backend is not None else _prime_backend_for(config)
        preflight = getattr(owner, "preflight", None)
        if callable(preflight):
            ok, message = preflight()
        else:
            ok, message = True, ""
    except Exception as exc:
        message = str(exc)
        ok = False
    if ok:
        return _store(
            ComputeProvider.PRIME,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.PRIME,
                stage=ProviderReadinessStage.READY,
                detail="API key verified",
            ),
        )
    message = message or "Prime verification failed"
    if _is_network_failure(message):
        return _store(
            ComputeProvider.PRIME,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.PRIME,
                stage=ProviderReadinessStage.UNREACHABLE,
                detail=message,
                hint="check your connection, then retry",
            ),
        )
    return _store(
        ComputeProvider.PRIME,
        verify,
        ProviderReadiness(
            provider=ComputeProvider.PRIME,
            stage=ProviderReadinessStage.AUTH_FAILED,
            detail=message,
            hint="run: prime login (or set PRIME_API_KEY)",
        ),
    )


def _prime_backend_for(config: PrimeConfig) -> Any:
    from .prime_backend import PrimeBackend

    return PrimeBackend(config=config)


def check_vast_readiness(
    *, verify: bool = True, refresh: bool = False, backend: Any | None = None
) -> ProviderReadiness:
    """A stored Vast key is credentials-present until the account is read."""
    if not refresh:
        cached = _cached(ComputeProvider.VAST, verify)
        if cached is not None:
            return cached
    try:
        credentials = resolve_vast_credentials()
    except ValueError as exc:
        return _store(
            ComputeProvider.VAST,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.VAST,
                stage=ProviderReadinessStage.MISSING_CREDENTIALS,
                detail=str(exc),
                hint="run: llm-launchpad vast-auth login (or set VAST_API_KEY)",
            ),
        )
    if not credentials.api_key:
        return _store(
            ComputeProvider.VAST,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.VAST,
                stage=ProviderReadinessStage.MISSING_CREDENTIALS,
                detail="no API key configured",
                hint="run: llm-launchpad vast-auth login (or set VAST_API_KEY)",
            ),
        )
    if not verify:
        return _store(
            ComputeProvider.VAST,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.VAST,
                stage=ProviderReadinessStage.CREDENTIALS_PRESENT,
                detail=f"API key found ({credentials.source}); not verified over the network",
                hint="verification runs before allocation",
            ),
        )
    try:
        owner = backend if backend is not None else _vast_backend_for(credentials)
        status = owner.auth_status()
    except Exception as exc:
        return _store(
            ComputeProvider.VAST,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.VAST,
                stage=ProviderReadinessStage.UNREACHABLE,
                detail=str(exc),
                hint="check your connection, then retry",
            ),
        )
    if getattr(status, "authenticated", False):
        account = getattr(status, "account_id", None)
        detail = f"verified ({credentials.source})"
        if account:
            detail += f" (account {account})"
        return _store(
            ComputeProvider.VAST,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.VAST,
                stage=ProviderReadinessStage.READY,
                detail=detail,
            ),
        )
    message = getattr(status, "error", None) or "Vast rejected the API key or its permissions."
    if _is_network_failure(message):
        return _store(
            ComputeProvider.VAST,
            verify,
            ProviderReadiness(
                provider=ComputeProvider.VAST,
                stage=ProviderReadinessStage.UNREACHABLE,
                detail=message,
                hint="check your connection, then retry",
            ),
        )
    return _store(
        ComputeProvider.VAST,
        verify,
        ProviderReadiness(
            provider=ComputeProvider.VAST,
            stage=ProviderReadinessStage.AUTH_FAILED,
            detail=message,
            hint="run: llm-launchpad vast-auth login (or set VAST_API_KEY)",
        ),
    )


def _vast_backend_for(credentials: VastCredentials) -> Any:
    from .vast_backend import VastBackend

    return VastBackend(credentials=credentials)


def check_all_provider_readiness(
    *, verify: bool = True, refresh: bool = False
) -> tuple[ProviderReadiness, ...]:
    """Return Modal, Prime, and Vast readiness in a stable order."""
    return (
        check_modal_readiness(verify=verify, refresh=refresh),
        check_prime_readiness(verify=verify, refresh=refresh),
        check_vast_readiness(verify=verify, refresh=refresh),
    )


def has_provider_credentials(*, refresh: bool = False) -> bool:
    """Whether any provider has local credentials worth verifying.

    This is the TUI setup gate: installation alone is not enough for Modal,
    while a stored Prime or Vast key lets the user in for verification.
    """
    return any(
        readiness.has_credentials
        for readiness in (
            check_modal_readiness(verify=False, refresh=refresh),
            check_prime_readiness(verify=False, refresh=refresh),
            check_vast_readiness(verify=False, refresh=refresh),
        )
    )


def any_provider_verified(*, refresh: bool = False) -> bool:
    """Whether any provider has passed a live authentication check."""
    return any(
        readiness.verified
        for readiness in check_all_provider_readiness(verify=True, refresh=refresh)
    )

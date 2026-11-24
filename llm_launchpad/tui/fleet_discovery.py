"""Multi-provider fleet discovery and cached listing retention for the TUI."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time
from typing import Any

from ..core.diagnostics import log_exception
from ..core.opencode import visible_launchpad_rows
from ..protocol.enums import ComputeProvider
from ..protocol.models import EndpointInfo, FleetDiscovery, ProviderListing


def listing_error_text(provider: ComputeProvider, exc: Exception) -> str:
    """Format and log a provider listing failure."""
    log_exception(f"Could not list {provider.display_name} deployments")
    text = str(exc).strip()
    return text or f"{type(exc).__name__} while listing {provider.display_name}."


def query_configured_providers(
    *,
    modal_enabled: bool,
    prime_enabled: bool,
    vast_enabled: bool,
    modal_lister: Callable[[], Any],
    prime_lister: Callable[[], list[EndpointInfo]],
    vast_lister: Callable[[], list[EndpointInfo]],
) -> tuple[
    tuple[ComputeProvider, ...],
    dict[ComputeProvider, list[EndpointInfo] | None],
    dict[ComputeProvider, str],
]:
    """Query all configured compute providers concurrently."""
    rows_by_provider: dict[ComputeProvider, list[EndpointInfo] | None] = {}
    errors: dict[ComputeProvider, str] = {}
    configured = (
        *((ComputeProvider.MODAL,) if modal_enabled else ()),
        *((ComputeProvider.PRIME,) if prime_enabled else ()),
        *((ComputeProvider.VAST,) if vast_enabled else ()),
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        modal_future = executor.submit(modal_lister) if modal_enabled else None
        prime_future = executor.submit(prime_lister) if prime_enabled else None
        vast_future = executor.submit(vast_lister) if vast_enabled else None

        if modal_future is not None:
            try:
                modal_result = modal_future.result()
            except Exception as exc:
                rows_by_provider[ComputeProvider.MODAL] = None
                errors[ComputeProvider.MODAL] = listing_error_text(
                    ComputeProvider.MODAL, exc
                )
            else:
                rows_by_provider[ComputeProvider.MODAL] = modal_result.rows
                if modal_result.error is not None:
                    errors[ComputeProvider.MODAL] = modal_result.error.message

        for provider, future in (
            (ComputeProvider.PRIME, prime_future),
            (ComputeProvider.VAST, vast_future),
        ):
            if future is None:
                continue
            try:
                rows_by_provider[provider] = future.result()
            except Exception as exc:
                rows_by_provider[provider] = None
                errors[provider] = listing_error_text(provider, exc)

    return configured, rows_by_provider, errors


def record_provider_listings(
    lock: threading.Lock,
    stored_listings: dict[ComputeProvider, ProviderListing],
    configured: tuple[ComputeProvider, ...],
    rows_by_provider: dict[ComputeProvider, list[EndpointInfo] | None],
    errors: dict[ComputeProvider, str],
) -> FleetDiscovery:
    """Turn one discovery pass into listings, retaining rows a failed provider had."""
    now = time.time()
    listings: list[ProviderListing] = []
    with lock:
        for provider in tuple(stored_listings):
            if provider not in configured:
                stored_listings.pop(provider, None)

        for provider in configured:
            rows = rows_by_provider.get(provider)
            if rows is not None:
                listing = ProviderListing(
                    provider=provider,
                    rows=tuple(visible_launchpad_rows(rows)),
                    retrieved_at_epoch=now,
                )
                stored_listings[provider] = listing
            else:
                retained = stored_listings.get(provider)
                listing = ProviderListing(
                    provider=provider,
                    rows=tuple(replace(row) for row in retained.rows) if retained else (),
                    error=errors.get(provider)
                    or f"Could not read the {provider.display_name} deployment list.",
                    retrieved_at_epoch=retained.retrieved_at_epoch if retained else None,
                )
            listings.append(listing)

        return FleetDiscovery(listings=tuple(listings))

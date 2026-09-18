"""Provider-aware storage visibility and stop consequences.

Compute and storage are separate bills. The TUI's Storage screen historically
inventoried Modal caches while Prime disks and Vast rental-local disks lived
elsewhere, and `stop` meant three different things. This module gives every
surface the same vocabulary.
"""

from __future__ import annotations

from ..protocol.enums import ComputeProvider
from ..protocol.models import StopEffect


def stop_effect_for_provider(provider: ComputeProvider) -> StopEffect:
    """Return what stopping compute does on one provider."""
    if provider == ComputeProvider.MODAL:
        return StopEffect(
            provider=provider,
            compute_action="Stop app",
            storage_consequence="Shared Modal volume/cache remains",
            remains_billable=False,
            destructive=False,
            detail="Stopping a Modal app stops compute billing. Cached weights stay in the shared volume for faster redeploys.",
            recovery_hint="Delete cached weights from the Storage screen when they are no longer needed.",
        )
    if provider == ComputeProvider.PRIME:
        return StopEffect(
            provider=provider,
            compute_action="Terminate pod",
            storage_consequence="Persistent cache disk remains and keeps billing",
            remains_billable=True,
            destructive=False,
            detail="Terminating a Prime pod stops pod billing. Its persistent cache disk is kept so the next deploy skips re-downloading weights.",
            recovery_hint="Remove an unused cache disk with: llm-launchpad prime-disks delete <id>",
        )
    if provider == ComputeProvider.VAST:
        return StopEffect(
            provider=provider,
            compute_action="Destroy rental",
            storage_consequence="Rental disk and cached weights are deleted",
            remains_billable=False,
            destructive=True,
            detail="Destroying a Vast rental deletes its disk along with any cached models. Billing stops, but the cache is gone.",
            recovery_hint="There is nothing to clean up after a Vast stop; redeploys re-download weights.",
        )
    return StopEffect(
        provider=provider,
        compute_action="Stop resource",
        storage_consequence="Storage consequence unknown",
        remains_billable=False,
        destructive=False,
        detail="Unknown provider; storage outcome is not modeled.",
    )


def format_stop_preview(provider: ComputeProvider, app_name: str) -> str:
    """Confirmation text that names what survives a stop."""
    effect = stop_effect_for_provider(provider)
    name = (app_name or "").strip() or "(unknown deployment)"
    lines = [
        f"Stop {name} on {provider.display_name}?",
        f"Compute: {effect.compute_action}.",
        f"Storage: {effect.storage_consequence}.",
    ]
    if effect.detail:
        lines.append(effect.detail)
    if effect.recovery_hint:
        lines.append(effect.recovery_hint)
    return "\n".join(lines)


def storage_scope_label(provider: ComputeProvider) -> str:
    """Explicit provider/scope label for the Storage screen."""
    if provider == ComputeProvider.MODAL:
        return "Modal · shared volume cache (file inventory available)"
    if provider == ComputeProvider.PRIME:
        return "Prime · persistent cache disks (disk inventory; files live on the disk)"
    return "Vast · rental-local disks (destroyed with the rental; no file inventory)"

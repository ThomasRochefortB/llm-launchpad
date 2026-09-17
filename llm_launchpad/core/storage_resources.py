"""Provider-aware storage visibility and stop consequences.

Compute and storage are separate bills. The TUI's Storage screen historically
inventoried Modal caches while Prime disks and Vast rental-local disks lived
elsewhere, and `stop` meant three different things. This module gives every
surface the same vocabulary.
"""

from __future__ import annotations

from ..protocol.enums import ComputeProvider
from ..protocol.models import (
    StorageResource,
    StorageSnapshot,
    StopEffect,
)


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


def modal_storage_resources(snapshot: StorageSnapshot) -> list[StorageResource]:
    """Convert a Modal snapshot into provider-scoped storage resources."""
    resources: list[StorageResource] = []
    for row in (*snapshot.llamacpp_models, *snapshot.vllm_models):
        size_gb = row.size_bytes / (1024.0**3) if row.size_bytes else 0.0
        resources.append(
            StorageResource(
                provider=ComputeProvider.MODAL,
                resource_id=f"{row.backend.value}:{row.model_id}:{row.revision or ''}:{row.quant or ''}",
                kind="model-cache",
                display_name=row.model_id,
                size_gb=size_gb,
                location=row.source_volume or "huggingface-cache",
                attached_to=None,
                survives_stop=True,
                billable_after_stop=False,
                price_per_hour_usd=None,
                deletable=True,
                delete_hint="Delete from the Storage screen; stopping an app never deletes cache.",
                managed=True,
            )
        )
    return resources


def prime_storage_resources(
    retained: list[object],
    *,
    attached_disk_id: str | None = None,
) -> list[StorageResource]:
    """Convert retained Prime disks into storage resources.

    `retained` accepts RetainedPrimeDisk rows without importing prime_disks
    here (which would couple storage display to disk provisioning).
    """
    resources: list[StorageResource] = []
    wanted = (attached_disk_id or "").strip()
    for row in retained:
        disk_id = str(getattr(row, "id", "") or "").strip()
        if not disk_id:
            continue
        size = getattr(row, "size_gb", 0) or 0
        try:
            size_gb = float(size)
        except (TypeError, ValueError):
            size_gb = 0.0
        price = getattr(row, "price_per_hour_usd", 0.0) or 0.0
        try:
            price_per_hour = float(price) or None
        except (TypeError, ValueError):
            price_per_hour = None
        resources.append(
            StorageResource(
                provider=ComputeProvider.PRIME,
                resource_id=disk_id,
                kind="persistent-disk",
                display_name=str(getattr(row, "name", "") or disk_id),
                size_gb=size_gb,
                location=str(getattr(row, "location", "") or ""),
                attached_to="this deployment" if wanted and disk_id == wanted else None,
                survives_stop=True,
                billable_after_stop=True,
                price_per_hour_usd=price_per_hour,
                deletable=True,
                delete_hint="llm-launchpad prime-disks delete <id>",
                managed=bool(getattr(row, "managed", True)),
            )
        )
    # The deployment's disk sorts first so stop results name it, not a sibling.
    resources.sort(key=lambda row: (row.resource_id != wanted, row.resource_id))
    return resources


def vast_storage_resources(*, disk_gb: int = 0, instance_label: str = "") -> list[StorageResource]:
    """Describe a Vast rental-local disk, which has no file-level inventory."""
    if disk_gb <= 0 and not instance_label:
        return []
    return [
        StorageResource(
            provider=ComputeProvider.VAST,
            resource_id=instance_label or f"{disk_gb}GB-rental-disk",
            kind="rental-disk",
            display_name=instance_label or f"{disk_gb} GB rental disk",
            size_gb=float(disk_gb) if disk_gb > 0 else None,
            location="rental-local",
            attached_to=instance_label or None,
            survives_stop=False,
            billable_after_stop=False,
            price_per_hour_usd=None,
            deletable=False,
            delete_hint="Destroying the rental deletes this disk; there is no separate delete.",
            managed=False,
        )
    ]


def storage_scope_label(provider: ComputeProvider) -> str:
    """Explicit provider/scope label for the Storage screen."""
    if provider == ComputeProvider.MODAL:
        return "Modal · shared volume cache (file inventory available)"
    if provider == ComputeProvider.PRIME:
        return "Prime · persistent cache disks (disk inventory; files live on the disk)"
    return "Vast · rental-local disks (destroyed with the rental; no file inventory)"

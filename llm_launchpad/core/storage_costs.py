"""Estimate Modal Volume storage costs for cached model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..protocol.models import StorageSnapshot

MODAL_VOLUME_PRICE_PER_GIB_MONTH_USD: Final[float] = 0.09
MODAL_VOLUME_FREE_TIER_GIB_MONTH: Final[float] = 1024.0

_BYTES_PER_GIB: Final[float] = 1024.0**3


@dataclass(frozen=True)
class StorageCostEstimate:
    """Monthly Modal Volume cost estimate for cached storage bytes."""

    total_size_bytes: int
    total_gib_month: float
    gross_monthly_cost_usd: float
    billable_gib_month: float
    estimated_monthly_cost_usd: float


def storage_gib_month(size_bytes: int) -> float:
    """Convert stored bytes to GiB-month for a one-month run-rate estimate."""
    return max(0, size_bytes) / _BYTES_PER_GIB


def gross_monthly_storage_cost_usd(
    size_bytes: int,
    price_per_gib_month_usd: float = MODAL_VOLUME_PRICE_PER_GIB_MONTH_USD,
) -> float:
    """Return monthly list-rate cost before Modal's free storage tier."""
    return storage_gib_month(size_bytes) * price_per_gib_month_usd


def billable_storage_gib_month(
    size_bytes: int,
    free_tier_gib_month: float = MODAL_VOLUME_FREE_TIER_GIB_MONTH,
) -> float:
    """Return GiB-month above Modal's included free Volume tier."""
    return max(0.0, storage_gib_month(size_bytes) - free_tier_gib_month)


def estimated_monthly_storage_cost_usd(
    size_bytes: int,
    price_per_gib_month_usd: float = MODAL_VOLUME_PRICE_PER_GIB_MONTH_USD,
    free_tier_gib_month: float = MODAL_VOLUME_FREE_TIER_GIB_MONTH,
) -> float:
    """Return estimated monthly storage cost after Modal's free tier."""
    return billable_storage_gib_month(size_bytes, free_tier_gib_month) * price_per_gib_month_usd


def estimate_monthly_storage_cost(
    snapshot: StorageSnapshot,
    price_per_gib_month_usd: float = MODAL_VOLUME_PRICE_PER_GIB_MONTH_USD,
    free_tier_gib_month: float = MODAL_VOLUME_FREE_TIER_GIB_MONTH,
) -> StorageCostEstimate:
    """Estimate monthly Volume storage cost for a launchpad storage snapshot."""
    total_size_bytes = max(0, snapshot.total_size_bytes)
    total_gib_month = storage_gib_month(total_size_bytes)
    billable_gib_month = max(0.0, total_gib_month - free_tier_gib_month)
    return StorageCostEstimate(
        total_size_bytes=total_size_bytes,
        total_gib_month=total_gib_month,
        gross_monthly_cost_usd=total_gib_month * price_per_gib_month_usd,
        billable_gib_month=billable_gib_month,
        estimated_monthly_cost_usd=billable_gib_month * price_per_gib_month_usd,
    )

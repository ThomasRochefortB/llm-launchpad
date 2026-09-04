"""Managed persistent Prime disks used as Hugging Face / llama.cpp caches."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from ..protocol.models import ComputeOffer, DeploymentConfig, PrimeProviderOptions
from .config import SETTINGS_DIR
from .diagnostics import log_exception
from .prime_backend import (
    PrimeApiError,
    PrimeDiskOffer,
    select_prime_offer,
)
from .provider_options import prime_provider_options

PRIME_CACHE_DISKS_PATH = SETTINGS_DIR / "prime" / "disks.json"
PRIME_CACHE_DISK_NAME = "llp-cache"
PRIME_CACHE_DISK_SIZE_GB = 100
_READY_DISK_STATES = {"ACTIVE", "READY", "UNATTACHED"}
_FAILED_DISK_STATES = {"ERROR", "FAILED", "TERMINATED"}


@dataclass(frozen=True)
class StoredPrimeDisk:
    """One Launchpad-managed Prime disk remembered for later attaches."""

    id: str
    name: str = PRIME_CACHE_DISK_NAME
    cloud_id: str = ""
    provider_name: str = ""
    data_center: str = ""
    country: str = ""
    region: str = ""
    size_gb: int = PRIME_CACHE_DISK_SIZE_GB

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "cloud_id": self.cloud_id,
            "provider_name": self.provider_name,
            "data_center": self.data_center,
            "country": self.country,
            "region": self.region,
            "size_gb": self.size_gb,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StoredPrimeDisk | None:
        disk_id = str(payload.get("id") or "").strip()
        if not disk_id:
            return None
        try:
            size_gb = int(payload.get("size_gb") or PRIME_CACHE_DISK_SIZE_GB)
        except (TypeError, ValueError):
            size_gb = PRIME_CACHE_DISK_SIZE_GB
        return cls(
            id=disk_id,
            name=str(payload.get("name") or PRIME_CACHE_DISK_NAME),
            cloud_id=str(payload.get("cloud_id") or ""),
            provider_name=str(payload.get("provider_name") or ""),
            data_center=str(payload.get("data_center") or ""),
            country=str(payload.get("country") or ""),
            region=str(payload.get("region") or ""),
            size_gb=size_gb,
        )


def load_stored_prime_disks(path: Path | None = None) -> list[StoredPrimeDisk]:
    """Load remembered cache disks, ignoring a missing or corrupt store."""

    target = path or PRIME_CACHE_DISKS_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("disks") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    disks: list[StoredPrimeDisk] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        disk = StoredPrimeDisk.from_dict(item)
        if disk is None or disk.id in seen:
            continue
        seen.add(disk.id)
        disks.append(disk)
    return disks


def save_stored_prime_disks(
    disks: list[StoredPrimeDisk],
    path: Path | None = None,
) -> None:
    """Persist the managed cache-disk list."""

    target = path or PRIME_CACHE_DISKS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"disks": [disk.to_dict() for disk in disks]}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(target, 0o600)
        os.chmod(target.parent, 0o700)
    except OSError:
        pass


def remember_prime_disk(disk: StoredPrimeDisk, path: Path | None = None) -> None:
    """Insert or update one managed disk, keeping the newest last."""

    disks = [row for row in load_stored_prime_disks(path) if row.id != disk.id]
    disks.append(disk)
    save_stored_prime_disks(disks, path)


def forget_prime_disk(disk_id: str, path: Path | None = None) -> None:
    """Drop a managed disk that no longer exists remotely."""

    wanted = disk_id.strip()
    if not wanted:
        return
    remaining = [row for row in load_stored_prime_disks(path) if row.id != wanted]
    save_stored_prime_disks(remaining, path)


def disk_matches_gpu_offer(disk: StoredPrimeDisk | PrimeDiskOffer, offer: ComputeOffer) -> bool:
    """Return whether a disk and GPU offer share a provider location."""

    disk_provider = str(getattr(disk, "provider_name", "") or "").casefold()
    disk_center = str(getattr(disk, "data_center", "") or "").casefold()
    disk_cloud = str(getattr(disk, "cloud_id", "") or "").casefold()
    offer_provider = (offer.provider_name or "").casefold()
    offer_center = (offer.data_center or "").casefold()
    offer_cloud = (offer.cloud_id or "").casefold()
    if disk_provider and offer_provider and disk_provider != offer_provider:
        return False
    if disk_center and offer_center:
        return disk_center == offer_center
    if disk_cloud and offer_cloud:
        return disk_cloud == offer_cloud
    return bool(disk_provider and offer_provider)


def _disk_stock_unavailable(stock_status: str | None) -> bool:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        (stock_status or "available").casefold(),
    ).strip("_")
    return normalized in {
        "unavailable",
        "out_of_stock",
        "outofstock",
        "sold_out",
        "soldout",
        "not_available",
        "notavailable",
    }


def matching_disk_offer(
    disk_offers: list[PrimeDiskOffer],
    gpu_offer: ComputeOffer,
) -> PrimeDiskOffer | None:
    """Return the first available persistent-disk offer for a GPU location."""

    for disk in disk_offers:
        if _disk_stock_unavailable(disk.stock_status):
            continue
        if disk_matches_gpu_offer(disk, gpu_offer):
            return disk
    return None


def cache_disk_size_gb(offer: PrimeDiskOffer) -> int:
    """Clamp the default cache size to an availability row's bounds."""

    size = PRIME_CACHE_DISK_SIZE_GB
    if offer.minimum_size_gb is not None:
        size = max(size, offer.minimum_size_gb)
    if offer.maximum_size_gb is not None:
        size = min(size, offer.maximum_size_gb)
    return size


def wait_for_prime_disk_ready(
    backend: Any,
    disk_id: str,
    *,
    timeout_seconds: float = 300,
    poll_interval_seconds: float = 3,
) -> dict[str, Any]:
    """Wait until a newly created Prime disk can be attached to a pod."""

    deadline = time.monotonic() + timeout_seconds
    last_status = "UNKNOWN"
    while True:
        disk = backend.get_disk(disk_id)
        last_status = str(disk.get("status") or "UNKNOWN").strip().upper()
        if last_status in _READY_DISK_STATES:
            return disk
        if last_status in _FAILED_DISK_STATES:
            raise PrimeApiError(f"Prime cache disk {disk_id} entered {last_status}.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PrimeApiError(
                f"Timed out waiting for Prime cache disk {disk_id} "
                f"to become ready (last status: {last_status})."
            )
        time.sleep(min(poll_interval_seconds, remaining))


def _select_kwargs(
    config: DeploymentConfig,
    options: PrimeProviderOptions,
    required_image: str,
) -> dict[str, Any]:
    return {
        "offer_id": options.offer_id,
        "gpu_type": config.gpu_type,
        "gpu_count": config.gpu_count,
        "region": options.region,
        "required_vram_gb": config.required_vram_gb,
        "required_image": required_image,
    }


def _try_offer_for_disk(
    backend: Any,
    stored: StoredPrimeDisk,
    config: DeploymentConfig,
    options: PrimeProviderOptions,
    required_image: str,
    path: Path,
) -> ComputeOffer | None:
    try:
        backend.get_disk(stored.id)
    except PrimeApiError as exc:
        if exc.status_code == 404:
            forget_prime_disk(stored.id, path)
        return None
    except Exception:
        return None
    offers = backend.list_offers(
        gpu_type=config.gpu_type,
        gpu_count=config.gpu_count,
        region=options.region,
        disk_id=stored.id,
    )
    try:
        return select_prime_offer(offers, **_select_kwargs(config, options, required_image))
    except ValueError:
        return None


def _create_cache_disk(backend: Any, gpu_offer: ComputeOffer) -> StoredPrimeDisk | None:
    try:
        disk_offers = backend.list_disk_offers()
    except Exception:
        return None
    disk_offer = matching_disk_offer(list(disk_offers or []), gpu_offer)
    if disk_offer is None:
        return None
    size_gb = cache_disk_size_gb(disk_offer)
    try:
        created = backend.create_disk(
            disk_offer,
            size_gb=size_gb,
            name=PRIME_CACHE_DISK_NAME,
        )
    except Exception:
        return None
    disk_id = str((created or {}).get("id") or "").strip()
    if not disk_id:
        return None
    try:
        wait_for_prime_disk_ready(backend, disk_id)
    except Exception:
        try:
            backend.delete_disk(disk_id)
        except Exception:
            log_exception(f"Could not release Prime disk {disk_id} after a failed attach")
        return None
    return StoredPrimeDisk(
        id=disk_id,
        name=PRIME_CACHE_DISK_NAME,
        cloud_id=disk_offer.cloud_id,
        provider_name=disk_offer.provider_name,
        data_center=disk_offer.data_center,
        country=disk_offer.country,
        region=disk_offer.region,
        size_gb=size_gb,
    )


def resolve_prime_offer_and_disk(
    backend: Any,
    config: DeploymentConfig,
    *,
    required_image: str,
    path: Path | None = None,
) -> tuple[ComputeOffer, str | None, list[str]]:
    """Select a GPU offer and the persistent cache disk to attach, if any."""

    options = prime_provider_options(config)
    store_path = path or PRIME_CACHE_DISKS_PATH
    messages: list[str] = []
    explicit_disk = (options.disk_id or "").strip() or None
    select_kw = _select_kwargs(config, options, required_image)

    if explicit_disk:
        offers = backend.list_offers(
            gpu_type=config.gpu_type,
            gpu_count=config.gpu_count,
            region=options.region,
            disk_id=explicit_disk,
        )
        offer = select_prime_offer(offers, **select_kw)
        messages.append(f"Attaching Prime disk {explicit_disk}")
        return offer, explicit_disk, messages

    if options.auto_disk:
        stored_disks = list(reversed(load_stored_prime_disks(store_path)))
        for stored in stored_disks:
            offer = _try_offer_for_disk(
                backend,
                stored,
                config,
                options,
                required_image,
                store_path,
            )
            if offer is None:
                continue
            remember_prime_disk(stored, store_path)
            location = stored.data_center or stored.region or stored.provider_name or "-"
            messages.append(
                f"Reusing Prime cache disk {stored.id} ({location})"
            )
            return offer, stored.id, messages

    offers = backend.list_offers(
        gpu_type=config.gpu_type,
        gpu_count=config.gpu_count,
        region=options.region,
    )
    offer = select_prime_offer(offers, **select_kw)
    if not options.auto_disk:
        return offer, None, messages

    created = _create_cache_disk(backend, offer)
    if created is None:
        messages.append(
            "Prime cache disk unavailable; model weights will not persist across deploys"
        )
        return offer, None, messages
    try:
        disk_offers = backend.list_offers(
            gpu_type=config.gpu_type,
            gpu_count=config.gpu_count,
            region=options.region,
            disk_id=created.id,
        )
        offer = select_prime_offer(disk_offers, **select_kw)
    except Exception:
        try:
            backend.delete_disk(created.id)
        except Exception:
            log_exception(f"Could not release Prime disk {created.id} after a failed attach")
        messages.append(
            "Prime cache disk could not be paired with an available GPU; "
            "model weights will not persist across this deploy"
        )
        return offer, None, messages
    remember_prime_disk(created, store_path)
    location = created.data_center or created.region or created.provider_name or "-"
    messages.append(
        f"Created Prime cache disk {created.id} ({created.size_gb} GB, {location})"
    )
    return offer, created.id, messages


def bind_prime_disk(config: DeploymentConfig, disk_id: str | None) -> DeploymentConfig:
    """Return config with ``provider_options.disk_id`` set for pod create."""

    if not disk_id:
        return config
    options = prime_provider_options(config)
    if options.disk_id == disk_id:
        return config
    config.provider_options = replace(options, disk_id=disk_id)
    return config

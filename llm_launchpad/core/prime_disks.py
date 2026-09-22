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
# Disk headroom over the weights actually cached: model bytes plus room for a
# second quant during rotation plus HF bookkeeping. A fixed 100 GB disk fits
# small models but a download larger than that never fits on it, so the disk
# would be created, fail to help, and be recreated every deploy.
PRIME_CACHE_DISK_HEADROOM_FACTOR = 1.25
PRIME_CACHE_DISK_MIN_GB = 100
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


def _normalize_location(value: str | None) -> str:
    """Normalize provider location text across Prime's spelling variants."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def disk_matches_gpu_offer(disk: StoredPrimeDisk | PrimeDiskOffer, offer: ComputeOffer) -> bool:
    """Return whether a disk and GPU offer share a provider location."""

    disk_provider = _normalize_location(getattr(disk, "provider_name", ""))
    disk_center = _normalize_location(getattr(disk, "data_center", ""))
    disk_cloud = _normalize_location(getattr(disk, "cloud_id", ""))
    offer_provider = _normalize_location(offer.provider_name)
    offer_center = _normalize_location(offer.data_center)
    offer_cloud = _normalize_location(offer.cloud_id)
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


def _model_weights_gb(config: DeploymentConfig) -> float:
    """Return the planner's weight size for this deploy, or 0 when unknown."""
    assessment = config.placement_assessment
    memory = getattr(assessment, "memory", None)
    weights_gb = getattr(memory, "weights_gb", None)
    try:
        value = float(weights_gb) if weights_gb is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0 else 0.0


def _required_cache_disk_gb(weights_gb: float) -> int:
    """Disk size one model's weights need, headroom included.

    The single rule both sides of the decision use: what a new disk is created
    at, and what a remembered one has to clear to be reused.
    """

    return max(
        PRIME_CACHE_DISK_MIN_GB,
        int(weights_gb * PRIME_CACHE_DISK_HEADROOM_FACTOR + 0.999),
    )


def cache_disk_size_gb(
    offer: PrimeDiskOffer,
    config: DeploymentConfig | None = None,
) -> int:
    """Size the cache disk for the model being deployed.

    The old fixed 100 GB default is the floor, not the size: it is kept for
    models with no planner estimate, but a model whose weights alone exceed
    it gets headroom over its own weights instead of a disk its download can
    never fit on. Bounds from the availability row still win.
    """

    size = PRIME_CACHE_DISK_SIZE_GB
    if config is not None:
        weights_gb = _model_weights_gb(config)
        if weights_gb > 0:
            size = _required_cache_disk_gb(weights_gb)
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


def _disk_fits_model(stored: StoredPrimeDisk, config: DeploymentConfig) -> bool:
    """Return whether a remembered disk can hold this deploy's weights.

    Held to the same bar a freshly created disk is sized to. Comparing against
    the bare weight size accepted a disk with no room for the HF bookkeeping
    and quant rotation the headroom factor exists to cover, so a 100 GB disk
    was reused for 99 GB of weights that a new disk would have been given
    124 GB for -- and the download then filled it.
    """
    weights_gb = _model_weights_gb(config)
    if weights_gb <= 0:
        return True
    return stored.size_gb >= _required_cache_disk_gb(weights_gb)


def _try_offer_for_disk(
    backend: Any,
    stored: StoredPrimeDisk,
    config: DeploymentConfig,
    options: PrimeProviderOptions,
    required_image: str,
    path: Path,
) -> ComputeOffer | None:
    # A disk smaller than this deploy's weights can attach but never help:
    # skip it before spending API calls proving it pairs.
    if not _disk_fits_model(stored, config):
        return None
    try:
        disk = backend.get_disk(stored.id)
    except PrimeApiError as exc:
        if exc.status_code == 404:
            forget_prime_disk(stored.id, path)
        return None
    except Exception:
        return None
    # A disk that exists but is stuck provisioning/failed can never attach;
    # leaving it remembered means every future deploy tries it first.
    try:
        status = str((disk or {}).get("status") or "").strip().upper()
    except AttributeError:
        status = ""
    if status and status in _FAILED_DISK_STATES:
        forget_prime_disk(stored.id, path)
        return None
    try:
        offers = backend.list_offers(
            gpu_type=config.gpu_type,
            gpu_count=config.gpu_count,
            region=options.region,
            disk_id=stored.id,
        )
    except Exception:
        # A remembered disk is optional; continue with other disks or a fresh
        # GPU offer when the provider can no longer pair this disk.
        return None
    try:
        return select_prime_offer(offers, **_select_kwargs(config, options, required_image))
    except ValueError:
        return None


def _create_cache_disk(
    backend: Any,
    gpu_offer: ComputeOffer,
    config: DeploymentConfig | None = None,
    reasons: list[str] | None = None,
) -> StoredPrimeDisk | None:
    """Create a cache disk beside ``gpu_offer``, recording why if it cannot.

    Every failure here used to be swallowed into a bare ``return None``, so a
    revoked key, an exhausted quota and a region with no disk product all
    reached the user as the same sentence, with nothing in the logs either.
    The disk is what stops the next deploy re-downloading the weights, so
    "unavailable" without a reason is an unactionable bill.
    """

    def _record(reason: str) -> None:
        if reasons is not None:
            reasons.append(reason)

    try:
        disk_offers = backend.list_disk_offers()
    except Exception as exc:
        log_exception("Could not list Prime disk offers")
        _record(f"disk offers could not be listed ({exc})")
        return None
    disk_offer = matching_disk_offer(list(disk_offers or []), gpu_offer)
    if disk_offer is None:
        _record(f"no disk product in {gpu_offer.data_center or gpu_offer.region or 'this location'}")
        return None
    size_gb = cache_disk_size_gb(disk_offer, config)
    try:
        created = backend.create_disk(
            disk_offer,
            size_gb=size_gb,
            name=PRIME_CACHE_DISK_NAME,
        )
    except Exception as exc:
        log_exception(f"Could not create a {size_gb} GB Prime cache disk")
        _record(f"creating a {size_gb} GB disk failed ({exc})")
        return None
    disk_id = str((created or {}).get("id") or "").strip()
    if not disk_id:
        _record("Prime returned no disk id")
        return None
    try:
        wait_for_prime_disk_ready(backend, disk_id)
    except Exception as exc:
        log_exception(f"Prime disk {disk_id} never became ready")
        _record(f"disk {disk_id} never became ready ({exc})")
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

    disk_reasons: list[str] = []
    created = _create_cache_disk(backend, offer, config, disk_reasons)
    if created is None:
        reason = f": {disk_reasons[0]}" if disk_reasons else ""
        messages.append(
            "Prime cache disk unavailable; model weights will not persist "
            f"across deploys{reason}"
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


@dataclass(frozen=True)
class RetainedPrimeDisk:
    """One persistent disk that is still billing, as the account reports it."""

    id: str
    name: str = ""
    size_gb: int = 0
    location: str = ""
    status: str = ""
    managed: bool = False
    price_per_hour_usd: float = 0.0

    def describe(self) -> str:
        parts = [self.id]
        if self.name:
            parts.append(self.name)
        if self.size_gb:
            parts.append(f"{self.size_gb} GB")
        if self.location:
            parts.append(self.location)
        if self.status:
            parts.append(self.status)
        # The rate is the whole reason to look at this list, so it is not
        # optional formatting: a disk with no price shown reads as free.
        if self.price_per_hour_usd:
            parts.append(f"${self.price_per_hour_usd:.4f}/hr")
        parts.append("launchpad" if self.managed else "other")
        return " · ".join(parts)


def _disk_field(row: dict[str, Any], *names: str) -> Any:
    """Read a disk attribute from the row or its nested ``info`` block.

    Prime reports a disk's placement inside ``info`` (``dataCenterId``,
    ``country``) while its size and status sit at the top level. Reading only
    the top level left the location blank on every row, which a live disk in
    AP-JP-1 showed immediately.
    """
    nested = row.get("info")
    blocks = [row, nested if isinstance(nested, dict) else {}]
    for name in names:
        for block in blocks:
            if block.get(name) not in (None, ""):
                return block[name]
    return None


def _disk_price(row: dict[str, Any]) -> float:
    try:
        return float(_disk_field(row, "priceHr", "price_hr", "pricePerHour") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def list_retained_prime_disks(
    backend: Any, path: Path | None = None
) -> list[RetainedPrimeDisk]:
    """Every disk on the account, flagged by whether Launchpad created it.

    Disks outlive the pods they were attached to by design, so this is the
    only view that answers "what am I still paying for". Disks Launchpad does
    not manage are listed too rather than hidden: the point of the command is
    the bill, and a disk it has forgotten costs exactly as much as one it
    remembers.
    """
    managed = {disk.id for disk in load_stored_prime_disks(path)}
    retained: list[RetainedPrimeDisk] = []
    for row in backend.list_disks():
        disk_id = str(_disk_field(row, "id", "diskId") or "").strip()
        if not disk_id:
            continue
        try:
            size_gb = int(_disk_field(row, "size", "sizeGb", "size_gb") or 0)
        except (TypeError, ValueError):
            size_gb = 0
        retained.append(
            RetainedPrimeDisk(
                id=disk_id,
                name=str(_disk_field(row, "name") or ""),
                size_gb=size_gb,
                location=str(
                    _disk_field(
                        row, "dataCenterId", "dataCenter", "data_center",
                        "region", "country",
                    )
                    or ""
                ),
                status=str(_disk_field(row, "status") or "").upper(),
                managed=disk_id in managed,
                price_per_hour_usd=_disk_price(row),
            )
        )
    # Remembered disks the account no longer reports are stale local state,
    # not spend. Drop them so the list only ever shows what is billing.
    for disk_id in managed - {row.id for row in retained}:
        forget_prime_disk(disk_id, path)
    return sorted(retained, key=lambda disk: (not disk.managed, disk.id))


def delete_retained_prime_disk(
    backend: Any, disk_id: str, path: Path | None = None
) -> str:
    """Terminate one disk and forget it locally. Returns what happened."""
    disk_id = (disk_id or "").strip()
    if not disk_id:
        raise ValueError("A Prime disk id is required.")
    try:
        backend.delete_disk(disk_id)
    except Exception as exc:
        # Prime refuses while anything still holds the disk, including a pod
        # whose termination has not finished propagating. That is a wait, not
        # a dead end, and saying so is the difference between retrying and
        # assuming the disk cannot be removed.
        if "attached" in str(exc).casefold():
            raise RuntimeError(
                f"Prime disk {disk_id} is still attached to a pod. Stop the deployment "
                "using it, give the termination a moment to finish, then retry."
            ) from None
        raise
    forget_prime_disk(disk_id, path)
    return f"Terminated Prime disk {disk_id}."

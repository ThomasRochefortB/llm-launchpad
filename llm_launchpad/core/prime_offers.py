"""Prime Intellect runtime specs, offer parsing, compatibility filtering, and selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
import os
import re
from typing import Any
from collections.abc import Iterable

from ..protocol.enums import BackendType, QuoteAvailability
from ..protocol.models import ComputeOffer, DeploymentConfig
from .coerce import optional_float
from .deployment_preflight import price_cap_finding
from .llamacpp_planner import device_capacity_bytes, gib_to_bytes
from .modal_gpu import modal_gpu_capacity_gib
from .runtime_support import load_llamacpp_support_manifest, llamacpp_build_recipe

PRIME_DEFAULT_BOOTSTRAP_IMAGE = "ubuntu_22_cuda_12"
PRIME_VRAM_HEADROOM_FACTOR = 1.05


def _runtime_catalog() -> dict[str, Any]:
    try:
        payload = json.loads(
            files("llm_launchpad.data")
            .joinpath("prime_runtime.json")
            .read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


_PRIME_RUNTIME = _runtime_catalog()


@dataclass(frozen=True)
class PrimeLaunchSpec:
    """Resolved portable runtime and the Prime image an offer must support."""

    offer_image: str
    container_image: str = ""
    build_recipe: str | None = None


@dataclass(frozen=True)
class PrimeDiskOffer:
    """Normalized persistent-disk availability used by live validation."""

    cloud_id: str
    provider_name: str
    data_center: str
    country: str
    region: str
    stock_status: str
    price_per_gb_hour: float | None
    minimum_size_gb: int | None
    maximum_size_gb: int | None
    raw: dict[str, Any]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def prime_runtime(backend: BackendType) -> dict[str, Any]:
    """Return runtime metadata for one Prime serving backend."""

    runtimes = _PRIME_RUNTIME.get("runtimes")
    if isinstance(runtimes, dict):
        runtime = runtimes.get(backend.value)
        return runtime if isinstance(runtime, dict) else {}
    if backend == BackendType.VLLM:
        # Backward compatibility with the original single-runtime catalog.
        return _PRIME_RUNTIME
    return {}


def prime_networking_runtime() -> dict[str, Any]:
    """Return pinned metadata for Prime's reverse-tunnel client."""

    networking = _PRIME_RUNTIME.get("networking")
    return networking if isinstance(networking, dict) else {}


def default_prime_container_image(backend: BackendType) -> str:
    """Return the configured upstream runtime used by portable Prime launches."""

    backend_env = f"LLM_LAUNCHPAD_PRIME_{backend.value.upper()}_CONTAINER_IMAGE"
    return (
        os.getenv(backend_env, "").strip()
        or str(prime_runtime(backend).get("bootstrap_container_image") or "").strip()
    )


def preferred_prime_offer_image(backend: BackendType) -> str:
    """Return the Prime base image required by the portable runtime."""

    backend_env = f"LLM_LAUNCHPAD_PRIME_{backend.value.upper()}_OFFER_IMAGE"
    return (
        os.getenv(backend_env, "").strip()
        or os.getenv("LLM_LAUNCHPAD_PRIME_OFFER_IMAGE", "").strip()
        or str(prime_runtime(backend).get("bootstrap_base_image") or "").strip()
        or PRIME_DEFAULT_BOOTSTRAP_IMAGE
    )


def resolve_prime_launch_spec(config: DeploymentConfig) -> PrimeLaunchSpec:
    """Resolve Prime's single supported Ubuntu bootstrap runtime."""

    offer_image = preferred_prime_offer_image(config.backend)
    container_image = default_prime_container_image(config.backend)
    build_recipe = None
    if config.backend == BackendType.LLAMACPP and not os.getenv(
        "LLM_LAUNCHPAD_PRIME_LLAMACPP_CONTAINER_IMAGE", ""
    ).strip():
        runtime = load_llamacpp_support_manifest(config.gguf_architecture)
        if runtime.build_recipe:
            container_image = runtime.image_ref
            build_recipe = llamacpp_build_recipe(config.gguf_architecture)
    if not container_image:
        raise ValueError(
            f"Prime {config.backend.value} portable runtime image is not configured."
        )
    return PrimeLaunchSpec(
        offer_image=offer_image,
        container_image=container_image,
        build_recipe=build_recipe,
    )


def prime_offer_id(payload: dict[str, Any]) -> str:
    """Return the same six-character offer ID displayed by the Prime CLI."""

    memory = _dict(payload.get("memory"))
    vcpu = _dict(payload.get("vcpu"))
    location = f"{payload.get('country') or 'N/A'} - {payload.get('dataCenter') or 'N/A'}"
    value = (
        f"{payload.get('cloudId', '')}-{payload.get('gpuType', '')}-"
        f"{payload.get('socket') or 'N/A'}-{location}-{payload.get('provider') or 'N/A'}-"
        f"{memory.get('defaultCount')}-{vcpu.get('defaultCount')}-{payload.get('gpuCount', '')}"
    )
    return hashlib.md5(value.encode()).hexdigest()[:6]


def parse_prime_offer(payload: dict[str, Any]) -> ComputeOffer:
    """Normalize one Prime availability item."""

    prices = _dict(payload.get("prices"))
    price = prices.get("communityPrice")
    if price is None:
        price = prices.get("onDemand")
    disk = _dict(payload.get("disk"))
    vcpu = _dict(payload.get("vcpu"))
    memory = _dict(payload.get("memory"))
    images = payload.get("images")
    return ComputeOffer(
        id=prime_offer_id(payload),
        cloud_id=str(payload.get("cloudId") or ""),
        provider_name=str(payload.get("provider") or ""),
        gpu_type=str(payload.get("gpuType") or ""),
        gpu_count=int(payload.get("gpuCount") or 0),
        gpu_memory_gb=optional_float(payload.get("gpuMemory")),
        region=str(payload.get("region")) if payload.get("region") else None,
        data_center=str(payload.get("dataCenter")) if payload.get("dataCenter") else None,
        country=str(payload.get("country")) if payload.get("country") else None,
        socket=str(payload.get("socket")) if payload.get("socket") else None,
        security=str(payload.get("security")) if payload.get("security") else None,
        price_per_hour=optional_float(price),
        is_spot=bool(payload.get("isSpot", False)),
        is_variable_price=bool(prices.get("isVariable", False)),
        stock_status=str(payload.get("stockStatus")) if payload.get("stockStatus") else None,
        disk_default_gb=_optional_int(disk.get("defaultCount")),
        vcpu_default=_optional_int(vcpu.get("defaultCount")),
        memory_default_gb=_optional_int(memory.get("defaultCount")),
        images=tuple(str(item) for item in images) if isinstance(images, list) else (),
        raw=dict(payload),
    )


def _gpu_type_memory_gb(gpu_type: str) -> float | None:
    """Parse a per-GPU memory size encoded in a Prime GPU type name."""

    normalized = re.sub(r"[^A-Z0-9]+", "_", gpu_type.upper()).strip("_")
    if normalized == "CPU_NODE" or normalized.startswith("CPU_"):
        return None
    match = re.search(r"(?:^|_)(\d{1,3})GB(?:_|$)", normalized)
    return float(match.group(1)) if match is not None else None


def prime_offer_gpu_memory_gb(offer: ComputeOffer) -> float | None:
    """Return per-GPU memory from Prime metadata or its canonical GPU name.

    Prime sometimes reports aggregate node memory (for example 640 GB on an
    8x A100_80GB offer). Identity and VRAM math always use the per-GPU size.
    """

    named = _gpu_type_memory_gb(offer.gpu_type)
    reported = (
        float(offer.gpu_memory_gb)
        if offer.gpu_memory_gb is not None and offer.gpu_memory_gb > 0
        else None
    )
    count = max(1, int(offer.gpu_count or 1))
    if named is not None:
        if reported is None or (count > 1 and abs(reported - named * count) < 1.0):
            return named
        return min(named, reported)
    return reported


def prime_gpu_capacity_gib(gpu_type: str, reported_gib: float) -> float:
    """Bound nominal marketplace figures by known framebuffer capacities."""
    return min(reported_gib, modal_gpu_capacity_gib(gpu_type) or reported_gib)


def is_prime_gpu_offer(offer: ComputeOffer) -> bool:
    """Return whether an availability row describes usable GPU compute."""

    gpu_type = offer.gpu_type.strip().upper()
    return (
        offer.gpu_count > 0
        and gpu_type != "CPU_NODE"
        and not gpu_type.startswith("CPU_")
        and prime_offer_gpu_memory_gb(offer) is not None
    )


def prime_pod_hourly_rate(pod: dict[str, Any]) -> float | None:
    """Best-effort hourly price from a Prime pod payload (never a guess)."""
    candidates: list[Any] = []
    for key in (
        "pricePerHour",
        "price_per_hour",
        "hourlyPrice",
        "hourly_price",
        "costPerHour",
        "cost_per_hour",
        "price",
        "cost",
    ):
        candidates.append(pod.get(key))
        nested = pod.get("billing")
        if isinstance(nested, dict):
            candidates.append(nested.get(key))
        nested = pod.get("pricing")
        if isinstance(nested, dict):
            candidates.append(nested.get(key))
    for candidate in candidates:
        try:
            rate = optional_float(candidate)
        except Exception:
            continue
        if rate is not None and rate >= 0:
            return rate
    return None


def prime_pod_started_at(pod: dict[str, Any]) -> float | None:
    """Best-effort run start from a Prime pod payload, as wall-clock epoch."""
    for key in ("createdAt", "created_at", "created", "startTime", "startedAt", "started_at"):
        raw = pod.get(key)
        if isinstance(raw, (int, float)) and raw > 0:
            value = float(raw)
            if value > 1e12:
                value /= 1000.0
            return value
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            try:
                normalized = text.replace("Z", "+00:00")
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    return None


def cached_hourly_rate(cached: dict[str, Any]) -> float | None:
    try:
        rate = optional_float(cached.get("price_per_hour_usd"))
    except Exception:
        return None
    if rate is None or rate < 0:
        return None
    return rate


_UNAVAILABLE_STOCK = frozenset({
    "unavailable",
    "out_of_stock",
    "outofstock",
    "sold_out",
    "soldout",
    "not_available",
    "notavailable",
})
_AVAILABLE_STOCK = frozenset({"available", "in_stock", "instock"})


def prime_stock_availability(stock_status: str | None) -> QuoteAvailability:
    """Map a Prime stock status string onto a normalized availability state."""

    normalized = re.sub(r"[^a-z0-9]+", "_", (stock_status or "").casefold()).strip("_")
    if normalized in _UNAVAILABLE_STOCK:
        return QuoteAvailability.UNAVAILABLE
    if normalized in _AVAILABLE_STOCK:
        return QuoteAvailability.AVAILABLE
    return QuoteAvailability.UNKNOWN


def is_prime_offer_unavailable(stock_status: str | None) -> bool:
    """Return whether Prime marks an offer as unavailable under a known spelling."""

    return prime_stock_availability(stock_status) is QuoteAvailability.UNAVAILABLE


def supports_prime_image(offer: ComputeOffer, image: str) -> bool:
    """Return whether an availability row advertises one Prime image type."""

    wanted = image.strip().casefold()
    return bool(wanted) and any(
        candidate.strip().casefold() == wanted for candidate in offer.images
    )


def prime_offer_matches_location(offer: ComputeOffer, location: str | None) -> bool:
    """Match user-facing region, country, or data-center text locally."""

    wanted = str(location or "").strip().casefold()
    if not wanted:
        return True
    return wanted in {
        str(offer.region or "").strip().casefold(),
        str(offer.country or "").strip().casefold(),
        str(offer.data_center or "").strip().casefold(),
    }


def prime_offer_satisfies_vram(
    offer: ComputeOffer,
    required_vram_gb: float | None,
    *,
    per_device_required_gib: float | None = None,
) -> bool:
    """Check an assessed per-device budget, or the legacy aggregate estimate."""

    per_gpu_memory = prime_offer_gpu_memory_gb(offer)
    if not is_prime_gpu_offer(offer) or per_gpu_memory is None:
        return False
    if per_device_required_gib is not None:
        capacity = prime_gpu_capacity_gib(offer.gpu_type, per_gpu_memory)
        return gib_to_bytes(per_device_required_gib) <= device_capacity_bytes(capacity)
    if required_vram_gb is None or required_vram_gb <= 0:
        return True
    available_vram = per_gpu_memory * offer.gpu_count
    return available_vram >= required_vram_gb * PRIME_VRAM_HEADROOM_FACTOR


def is_compatible_prime_offer(
    offer: ComputeOffer,
    required_vram_gb: float | None = None,
    *,
    required_image: str = PRIME_DEFAULT_BOOTSTRAP_IMAGE,
) -> bool:
    """Return whether a Prime row is a deployable secure inference option."""

    return (
        prime_offer_satisfies_vram(offer, required_vram_gb)
        and supports_prime_image(offer, required_image)
        and not offer.is_spot
        and not offer.is_variable_price
        and (offer.security or "").casefold() == "secure_cloud"
        and not is_prime_offer_unavailable(offer.stock_status)
    )


def select_prime_offer(
    offers: Iterable[ComputeOffer],
    *,
    offer_id: str | None = None,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    region: str | None = None,
    required_vram_gb: float | None = None,
    per_device_required_gib: float | None = None,
    max_hourly_cost_usd: float | None = None,
    required_image: str = PRIME_DEFAULT_BOOTSTRAP_IMAGE,
) -> ComputeOffer:
    """Select an exact offer or the cheapest secure on-demand matching offer."""

    rows = list(offers)
    if offer_id:
        match = next(
            (row for row in rows if row.id.casefold() == offer_id.strip().casefold()),
            None,
        )
        if match is None:
            raise ValueError(f"Prime offer not found: {offer_id}")
        if gpu_type and match.gpu_type.casefold() != gpu_type.casefold():
            raise ValueError(f"Prime offer {offer_id} does not match the requested GPU type.")
        if gpu_count is not None and match.gpu_count != gpu_count:
            raise ValueError(f"Prime offer {offer_id} does not match the requested GPU count.")
        if region and not prime_offer_matches_location(match, region):
            raise ValueError(f"Prime offer {offer_id} does not match the requested region.")
        if not is_prime_gpu_offer(match):
            raise ValueError(f"Prime offer {offer_id} is not a GPU instance.")
        if not prime_offer_satisfies_vram(match, required_vram_gb, per_device_required_gib=per_device_required_gib):
            if per_device_required_gib is not None:
                raise ValueError(
                    f"Prime offer {offer_id} cannot provide {per_device_required_gib:.1f} GiB per GPU."
                )
            required = float(required_vram_gb or 0) * PRIME_VRAM_HEADROOM_FACTOR
            raise ValueError(
                f"Prime offer {offer_id} does not provide the required "
                f"~{required:.1f} GB of GPU memory."
            )
        if not supports_prime_image(match, required_image):
            raise ValueError(
                f"Prime offer {offer_id} does not support runtime image {required_image}."
            )
        if match.is_spot:
            raise ValueError(f"Prime offer {offer_id} is a spot instance.")
        if match.is_variable_price:
            raise ValueError(f"Prime offer {offer_id} has variable pricing.")
        if (match.security or "").casefold() != "secure_cloud":
            raise ValueError(f"Prime offer {offer_id} is not secure_cloud.")
        if is_prime_offer_unavailable(match.stock_status):
            raise ValueError(f"Prime offer {offer_id} is unavailable.")
        if finding := price_cap_finding(max_hourly_cost_usd, match.price_per_hour):
            raise ValueError(finding.message)
        return match

    matches = [
        row
        for row in rows
        if is_prime_gpu_offer(row)
        and supports_prime_image(row, required_image)
        and price_cap_finding(max_hourly_cost_usd, row.price_per_hour) is None
    ]
    matches = [row for row in matches if not row.is_spot and not row.is_variable_price]
    matches = [row for row in matches if (row.security or "").casefold() == "secure_cloud"]
    if gpu_type:
        matches = [row for row in matches if row.gpu_type.casefold() == gpu_type.casefold()]
    if gpu_count is not None:
        matches = [row for row in matches if row.gpu_count == gpu_count]
    if region:
        matches = [row for row in matches if prime_offer_matches_location(row, region)]
    matches = [
        row for row in matches if prime_offer_satisfies_vram(
            row, required_vram_gb, per_device_required_gib=per_device_required_gib,
        )
    ]
    matches = [row for row in matches if not is_prime_offer_unavailable(row.stock_status)]
    if not matches:
        raise ValueError(
            f"No secure on-demand Prime GPU offer supporting {required_image} "
            "matches the requested filters."
        )
    return min(
        matches,
        key=lambda row: (
            row.price_per_hour is None,
            row.price_per_hour if row.price_per_hour is not None else float("inf"),
            row.id,
        ),
    )

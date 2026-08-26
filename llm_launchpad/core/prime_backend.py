"""Prime Intellect REST integration for GPU offer and pod lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Iterable

import requests

from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import ComputeOffer, DeploymentConfig, EndpointInfo
from .config import SETTINGS_DIR
from .naming import infer_instance_from_app_name
from .prime_auth import PrimeConfig, load_prime_config
from .provider_options import prime_provider_options


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
PRIME_DEFAULT_BOOTSTRAP_IMAGE = "ubuntu_22_cuda_12"


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

    return (
        str(prime_runtime(backend).get("bootstrap_base_image") or "").strip()
        or PRIME_DEFAULT_BOOTSTRAP_IMAGE
    )


def resolve_prime_launch_spec(config: DeploymentConfig) -> PrimeLaunchSpec:
    """Resolve Prime's single supported Ubuntu bootstrap runtime."""

    offer_image = preferred_prime_offer_image(config.backend)
    container_image = default_prime_container_image(config.backend)
    if not container_image:
        raise ValueError(
            f"Prime {config.backend.value} portable runtime image is not configured."
        )
    return PrimeLaunchSpec(
        offer_image=offer_image,
        container_image=container_image,
    )


PRIME_POD_PREFIX = "llp-prime-"
PRIME_VRAM_HEADROOM_FACTOR = 1.05
PRIME_RUNTIME_CONTAINER_NAME = "llm-launchpad-runtime"
PRIME_RUNTIME_ROOT = "/opt/llm-launchpad"
PRIME_TUNNEL_LABEL = "llm-launchpad"
PRIME_TUNNEL_PID_PATH = f"{PRIME_RUNTIME_ROOT}/tunnel.pid"
PRIME_TUNNEL_LOG_PATH = f"{PRIME_RUNTIME_ROOT}/tunnel.log"
PRIME_BOOTSTRAP_SSH_KEY_NAME = "llm-launchpad-bootstrap"
PRIME_BOOTSTRAP_SSH_KEY_PATH = SETTINGS_DIR / "prime" / "bootstrap_ed25519"
PRIME_KNOWN_HOSTS_DIR = SETTINGS_DIR / "prime" / "known_hosts"


@dataclass(frozen=True)
class PrimeLaunchSpec:
    """Resolved portable runtime and the Prime image an offer must support."""

    offer_image: str
    container_image: str = ""


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


@dataclass(frozen=True)
class PrimeTunnel:
    """Prime Tunnel registration and its ephemeral client credentials."""

    tunnel_id: str
    hostname: str
    url: str
    name: str = ""
    frp_token: str = ""
    binding_secret: str = ""
    server_host: str = ""
    server_port: int = 7000
    expires_at: str = ""
    status: str = ""
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrimeApiError(RuntimeError):
    """Structured failure returned by the Prime API."""

    message: str
    status_code: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        return self.message


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _format_prime_error_value(value: Any, *, field: str = "") -> str:
    """Format Prime/FastAPI error payloads without echoing submitted values."""

    if isinstance(value, str):
        value = value.strip()
        return f"{field}: {value}" if field and value else value
    if isinstance(value, list):
        parts = [_format_prime_error_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    if not isinstance(value, dict):
        return ""

    location = value.get("loc")
    message = value.get("msg") or value.get("message")
    if message:
        location_text = ""
        if isinstance(location, (list, tuple)):
            location_text = ".".join(str(item) for item in location if str(item))
        elif location:
            location_text = str(location)
        return _format_prime_error_value(message, field=location_text or field)

    for key in ("detail", "error", "errors", "reason"):
        if key not in value:
            continue
        detail = _format_prime_error_value(value[key], field=field)
        if detail:
            return detail

    parts: list[str] = []
    for key, item in value.items():
        # FastAPI includes the rejected input and validation context. Neither is
        # needed to explain the failure, and an env-var input may be a secret.
        if key in {"input", "ctx", "url", "type", "loc"}:
            continue
        detail = _format_prime_error_value(item, field=str(key))
        if detail:
            parts.append(detail)
    return "; ".join(parts)


def _prime_error_detail(payload: Any, fallback_text: str = "") -> str:
    """Extract a concise validation message from any documented error shape."""

    detail = _format_prime_error_value(payload)
    if not detail:
        detail = fallback_text.strip()
    # Keep a malformed upstream response from flooding the TUI log view.
    return detail[:2000]


def _pod_tunnel_label(pod_id: str) -> str:
    """Return a stable label that associates a Prime Tunnel with one pod."""

    clean = re.sub(r"[^A-Za-z0-9_.:-]+", "-", pod_id.strip())
    if not clean:
        raise ValueError("Prime Tunnel requires a pod ID.")
    return f"pod:{clean}"


def _parse_prime_tunnel(payload: Any, *, credentials_required: bool) -> PrimeTunnel:
    """Validate and normalize one Prime Tunnel response."""

    if not isinstance(payload, dict):
        raise PrimeApiError("Prime Tunnel response was not an object.")
    tunnel_id = str(payload.get("tunnel_id") or "").strip()
    hostname = str(payload.get("hostname") or "").strip()
    url = str(payload.get("url") or "").strip().rstrip("/")
    frp_token = str(payload.get("frp_token") or "").strip()
    binding_secret = str(payload.get("binding_secret") or "").strip()
    server_host = str(payload.get("server_host") or "").strip()
    try:
        server_port = int(payload.get("server_port") or 7000)
    except (TypeError, ValueError) as exc:
        raise PrimeApiError("Prime Tunnel returned an invalid server port.") from exc
    required = {
        "tunnel ID": tunnel_id,
        "hostname": hostname,
        "HTTPS URL": url,
    }
    if credentials_required:
        required.update(
            {
                "frpc token": frp_token,
                "binding secret": binding_secret,
                "server host": server_host,
            }
        )
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise PrimeApiError(
            "Prime Tunnel response is missing " + ", ".join(missing) + "."
        )
    if not url.startswith("https://"):
        raise PrimeApiError("Prime Tunnel did not return an HTTPS URL.")
    raw_labels = payload.get("labels")
    labels = (
        tuple(str(label) for label in raw_labels if str(label).strip())
        if isinstance(raw_labels, list)
        else ()
    )
    return PrimeTunnel(
        tunnel_id=tunnel_id,
        hostname=hostname,
        url=url,
        name=str(payload.get("name") or "").strip(),
        frp_token=frp_token,
        binding_secret=binding_secret,
        server_host=server_host,
        server_port=server_port,
        expires_at=str(payload.get("expires_at") or "").strip(),
        status=str(payload.get("status") or "").strip(),
        labels=labels,
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
        gpu_memory_gb=_optional_float(payload.get("gpuMemory")),
        region=str(payload.get("region")) if payload.get("region") else None,
        data_center=str(payload.get("dataCenter")) if payload.get("dataCenter") else None,
        country=str(payload.get("country")) if payload.get("country") else None,
        socket=str(payload.get("socket")) if payload.get("socket") else None,
        security=str(payload.get("security")) if payload.get("security") else None,
        price_per_hour=_optional_float(price),
        is_spot=bool(payload.get("isSpot", False)),
        is_variable_price=bool(prices.get("isVariable", False)),
        stock_status=str(payload.get("stockStatus")) if payload.get("stockStatus") else None,
        disk_default_gb=_optional_int(disk.get("defaultCount")),
        vcpu_default=_optional_int(vcpu.get("defaultCount")),
        memory_default_gb=_optional_int(memory.get("defaultCount")),
        images=tuple(str(item) for item in images) if isinstance(images, list) else (),
        raw=dict(payload),
    )


def prime_offer_gpu_memory_gb(offer: ComputeOffer) -> float | None:
    """Return per-GPU memory from Prime metadata or its canonical GPU name."""

    if offer.gpu_memory_gb is not None and offer.gpu_memory_gb > 0:
        return offer.gpu_memory_gb
    normalized = re.sub(r"[^A-Z0-9]+", "_", offer.gpu_type.upper()).strip("_")
    if normalized == "CPU_NODE" or normalized.startswith("CPU_"):
        return None
    match = re.search(r"(?:^|_)(\d{1,3})GB(?:_|$)", normalized)
    if match is None:
        return None
    return float(match.group(1))


def is_prime_gpu_offer(offer: ComputeOffer) -> bool:
    """Return whether an availability row describes usable GPU compute."""

    gpu_type = offer.gpu_type.strip().upper()
    return (
        offer.gpu_count > 0
        and gpu_type != "CPU_NODE"
        and not gpu_type.startswith("CPU_")
        and prime_offer_gpu_memory_gb(offer) is not None
    )


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
) -> bool:
    """Check aggregate GPU memory with the catalog's safety headroom."""

    per_gpu_memory = prime_offer_gpu_memory_gb(offer)
    if not is_prime_gpu_offer(offer) or per_gpu_memory is None:
        return False
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
        and (offer.stock_status or "available").casefold() != "unavailable"
    )


def select_prime_offer(
    offers: Iterable[ComputeOffer],
    *,
    offer_id: str | None = None,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    region: str | None = None,
    required_vram_gb: float | None = None,
    required_image: str = PRIME_DEFAULT_BOOTSTRAP_IMAGE,
) -> ComputeOffer:
    """Select an exact offer or the cheapest secure on-demand matching offer."""

    rows = list(offers)
    if offer_id:
        match = next((row for row in rows if row.id.casefold() == offer_id.strip().casefold()), None)
        if match is None:
            raise ValueError(f"Prime offer not found: {offer_id}")
        if not is_prime_gpu_offer(match):
            raise ValueError(f"Prime offer {offer_id} is not a GPU instance.")
        if not prime_offer_satisfies_vram(match, required_vram_gb):
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
        if (match.stock_status or "available").casefold() == "unavailable":
            raise ValueError(f"Prime offer {offer_id} is unavailable.")
        return match

    matches = [
        row
        for row in rows
        if is_prime_gpu_offer(row) and supports_prime_image(row, required_image)
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
        row for row in matches if prime_offer_satisfies_vram(row, required_vram_gb)
    ]
    matches = [row for row in matches if (row.stock_status or "available").casefold() != "unavailable"]
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


class PrimeBackend:
    """Small requests-based client compatible with Prime CLI authentication state."""

    def __init__(
        self,
        config: PrimeConfig | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        ssh_private_key_path: Path | None = None,
    ) -> None:
        self.config = config or load_prime_config()
        self.session = session or requests.Session()
        self.timeout = timeout
        self.ssh_private_key_path = ssh_private_key_path or PRIME_BOOTSTRAP_SSH_KEY_PATH

    def preflight(self) -> tuple[bool, str]:
        if not self.config.api_key:
            return False, "Prime authentication missing. Run: prime login"
        try:
            self._request("GET", "/pods", params={"offset": 0, "limit": 1})
        except PrimeApiError as exc:
            return False, f"Prime authentication failed: {exc}"
        return True, ""

    @staticmethod
    def _public_key_identity(value: str) -> str:
        """Normalize an OpenSSH public key without its mutable comment."""

        parts = value.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else value.strip()

    def _ensure_bootstrap_key_files(self) -> str:
        """Create Launchpad's dedicated Prime bootstrap key when absent."""

        private_path = self.ssh_private_key_path
        public_path = Path(f"{private_path}.pub")
        private_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(private_path.parent, 0o700)
        except OSError:
            pass

        if not private_path.exists():
            try:
                result = subprocess.run(
                    [
                        "ssh-keygen",
                        "-q",
                        "-t",
                        "ed25519",
                        "-N",
                        "",
                        "-C",
                        PRIME_BOOTSTRAP_SSH_KEY_NAME,
                        "-f",
                        str(private_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("ssh-keygen is required for Prime portable launches.") from exc
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(f"Could not create the Prime bootstrap SSH key: {detail}")

        if not public_path.exists():
            try:
                result = subprocess.run(
                    ["ssh-keygen", "-y", "-f", str(private_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("ssh-keygen is required for Prime portable launches.") from exc
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError("Could not derive the Prime bootstrap public SSH key.")
            public_path.write_text(
                f"{result.stdout.strip()} {PRIME_BOOTSTRAP_SSH_KEY_NAME}\n",
                encoding="utf-8",
            )

        try:
            os.chmod(private_path, 0o600)
            os.chmod(public_path, 0o644)
        except OSError:
            pass
        return public_path.read_text(encoding="utf-8").strip()

    def ensure_bootstrap_ssh_key(self) -> str:
        """Return the Prime key ID for Launchpad's dedicated bootstrap key."""

        public_key = self._ensure_bootstrap_key_files()
        wanted = self._public_key_identity(public_key)
        payload = self._request("GET", "/ssh_keys/", params={"offset": 0, "limit": 100})
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if self._public_key_identity(str(row.get("publicKey") or "")) == wanted:
                key_id = str(row.get("id") or "").strip()
                if key_id:
                    return key_id
        created = self._request(
            "POST",
            "/ssh_keys/",
            json_payload={"name": PRIME_BOOTSTRAP_SSH_KEY_NAME, "publicKey": public_key},
        )
        key_id = str(created.get("id") or "").strip() if isinstance(created, dict) else ""
        if not key_id:
            raise PrimeApiError("Prime did not return an ID for the bootstrap SSH key.")
        return key_id

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        if not self.config.api_key:
            raise PrimeApiError("Prime API key is not configured.", status_code=401)
        url = f"{self.config.base_url}/api/v1/{path.lstrip('/')}"
        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=timeout or self.timeout,
            )
        except requests.RequestException as exc:
            raise PrimeApiError(f"Prime API request failed: {exc}") from exc
        if 200 <= response.status_code < 300:
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise PrimeApiError("Prime API returned malformed JSON.") from exc
        payload: Any = None
        try:
            payload = response.json()
        except ValueError:
            pass
        detail = _prime_error_detail(payload, response.text or "")
        labels = {401: "authentication failed", 403: "permission denied", 402: "insufficient balance"}
        label = labels.get(response.status_code, f"HTTP {response.status_code}")
        message = f"Prime API {label}"
        if detail:
            message = f"{message}: {detail}"
        raise PrimeApiError(message, status_code=response.status_code, detail=detail)

    def list_offers(
        self,
        *,
        gpu_type: str | None = None,
        gpu_count: int | None = None,
        region: str | None = None,
        disk_id: str | None = None,
    ) -> list[ComputeOffer]:
        params: dict[str, Any] = {}
        if gpu_type:
            params["gpu_type"] = gpu_type
        if gpu_count is not None:
            params["gpu_count"] = gpu_count
        if disk_id:
            params["disks"] = [disk_id]
        rows: list[ComputeOffer] = []
        page = 1
        while True:
            payload = self._request(
                "GET", "/availability/gpus", params={**params, "page": page, "page_size": 100}
            )
            page_rows = payload.get("items", []) if isinstance(payload, dict) else []
            rows.extend(parse_prime_offer(item) for item in page_rows if isinstance(item, dict))
            if len(page_rows) < 100:
                break
            if isinstance(payload, dict) and payload.get("totalCount") is not None:
                try:
                    total = int(payload.get("totalCount", 0))
                except (TypeError, ValueError):
                    total = 0
                if total and page * 100 >= total:
                    break
            page += 1
        # Prime's GPU endpoint also enumerates CPU_NODE. Launchpad only serves
        # GPU inference, so keep that provider quirk behind this adapter.
        rows = [row for row in rows if is_prime_gpu_offer(row)]
        # Prime's API only accepts its canonical region enum in `regions`.
        # Launchpad also accepts country and data-center labels, so location
        # matching belongs in the adapter instead of forwarding display text.
        return [row for row in rows if prime_offer_matches_location(row, region)]

    def list_disk_offers(self) -> list[PrimeDiskOffer]:
        """Return live persistent-disk availability without leaking provider shapes."""

        rows: list[PrimeDiskOffer] = []
        page = 1
        while True:
            payload = self._request(
                "GET",
                "/availability/disks",
                params={"page": page, "page_size": 100},
            )
            page_rows = payload.get("items", []) if isinstance(payload, dict) else []
            for item in page_rows:
                if not isinstance(item, dict):
                    continue
                spec = _dict(item.get("spec"))
                rows.append(
                    PrimeDiskOffer(
                        cloud_id=str(item.get("cloudId") or ""),
                        provider_name=str(item.get("provider") or ""),
                        data_center=str(item.get("dataCenter") or ""),
                        country=str(item.get("country") or ""),
                        region=str(item.get("region") or ""),
                        stock_status=str(item.get("stockStatus") or ""),
                        price_per_gb_hour=_optional_float(spec.get("pricePerUnit")),
                        minimum_size_gb=_optional_int(spec.get("minCount")),
                        maximum_size_gb=_optional_int(spec.get("maxCount")),
                        raw=item,
                    )
                )
            if len(page_rows) < 100:
                break
            if isinstance(payload, dict) and payload.get("totalCount") is not None:
                try:
                    total = int(payload.get("totalCount", 0))
                except (TypeError, ValueError):
                    total = 0
                if total and page * 100 >= total:
                    break
            page += 1
        return rows

    def create_disk(
        self,
        offer: PrimeDiskOffer,
        *,
        size_gb: int,
        name: str,
    ) -> dict[str, Any]:
        """Create one persistent disk from an exact availability row."""

        if size_gb <= 0:
            raise ValueError("Prime disk size must be greater than zero.")
        if offer.minimum_size_gb is not None and size_gb < offer.minimum_size_gb:
            raise ValueError(
                f"Prime disk size must be at least {offer.minimum_size_gb} GB."
            )
        if offer.maximum_size_gb is not None and size_gb > offer.maximum_size_gb:
            raise ValueError(
                f"Prime disk size must be at most {offer.maximum_size_gb} GB."
            )
        payload: dict[str, Any] = {
            "disk": {
                "size": size_gb,
                "name": name,
                "country": offer.country or None,
                "cloudId": offer.cloud_id or None,
                "dataCenterId": offer.data_center or None,
            },
            "provider": {"type": offer.provider_name},
        }
        payload["disk"] = {
            key: value for key, value in payload["disk"].items() if value is not None
        }
        if self.config.team_id:
            payload["team"] = {"teamId": self.config.team_id}
        result = self._request("POST", "/disks", json_payload=payload, timeout=60)
        if not isinstance(result, dict):
            raise PrimeApiError("Prime create-disk response was not an object.")
        return result

    def get_disk(self, disk_id: str) -> dict[str, Any]:
        """Return one persistent disk."""

        payload = self._request("GET", f"/disks/{disk_id}")
        if not isinstance(payload, dict):
            raise PrimeApiError("Prime disk response was not an object.")
        return payload

    def delete_disk(self, disk_id: str) -> None:
        """Permanently terminate one persistent disk."""

        self._request("DELETE", f"/disks/{disk_id}")

    def create_tunnel(
        self,
        pod_id: str,
        *,
        name: str,
        local_port: int = 8000,
    ) -> PrimeTunnel:
        """Register an authenticated Prime reverse tunnel for one pod."""

        if not 1 <= local_port <= 65535:
            raise ValueError("Prime Tunnel local port must be between 1 and 65535.")
        payload: dict[str, Any] = {
            "name": (name.strip() or f"llm-launchpad-{pod_id}")[:64],
            "local_port": local_port,
            "labels": [PRIME_TUNNEL_LABEL, _pod_tunnel_label(pod_id)],
        }
        if self.config.team_id:
            payload["teamId"] = self.config.team_id
        result = self._request("POST", "/tunnel", json_payload=payload, timeout=60)
        return _parse_prime_tunnel(result, credentials_required=True)

    def get_tunnel(self, tunnel_id: str) -> PrimeTunnel | None:
        """Return an active Prime Tunnel, or ``None`` after it is removed."""

        try:
            result = self._request("GET", f"/tunnel/{tunnel_id}")
        except PrimeApiError as exc:
            if exc.status_code == 404:
                return None
            raise
        return _parse_prime_tunnel(result, credentials_required=False)

    def list_tunnels(self) -> list[PrimeTunnel]:
        """List active Prime Tunnels for the configured user or team."""

        rows: list[PrimeTunnel] = []
        page = 1
        per_page = 100
        while True:
            params: dict[str, Any] = {"page": page, "perPage": per_page}
            if self.config.team_id:
                params["teamId"] = self.config.team_id
            result = self._request("GET", "/tunnel", params=params)
            payload_rows = result.get("tunnels", []) if isinstance(result, dict) else []
            rows.extend(
                _parse_prime_tunnel(item, credentials_required=False)
                for item in payload_rows
                if isinstance(item, dict)
            )
            has_next = bool(result.get("has_next")) if isinstance(result, dict) else False
            total = int(result.get("total", len(rows))) if isinstance(result, dict) else 0
            if not has_next and len(rows) >= total:
                break
            if not payload_rows:
                break
            page += 1
        return rows

    def delete_tunnel(self, tunnel_id: str) -> None:
        """Delete one Prime Tunnel, tolerating an already-removed registration."""

        try:
            self._request("DELETE", f"/tunnel/{tunnel_id}")
        except PrimeApiError as exc:
            if exc.status_code != 404:
                raise

    def delete_tunnels_for_pod(self, pod_id: str) -> list[str]:
        """Delete every Launchpad tunnel associated with a pod."""

        pod_label = _pod_tunnel_label(pod_id)
        tunnel_ids = [
            tunnel.tunnel_id
            for tunnel in self.list_tunnels()
            if PRIME_TUNNEL_LABEL in tunnel.labels and pod_label in tunnel.labels
        ]
        for tunnel_id in tunnel_ids:
            self.delete_tunnel(tunnel_id)
        return tunnel_ids

    def create_pod(self, config: DeploymentConfig, offer: ComputeOffer) -> dict[str, Any]:
        options = prime_provider_options(config)
        launch = resolve_prime_launch_spec(config)
        if not supports_prime_image(offer, launch.offer_image):
            raise ValueError(
                f"Prime offer {offer.id} does not support runtime image {launch.offer_image}."
            )
        raw = offer.raw
        pod: dict[str, Any] = {
            "name": config.app_name,
            "cloudId": offer.cloud_id,
            "gpuType": offer.gpu_type,
            "socket": offer.socket or raw.get("socket") or "PCIe",
            "gpuCount": offer.gpu_count,
            "image": launch.offer_image,
            "dataCenterId": offer.data_center,
            "country": offer.country,
            "security": offer.security,
            "diskSize": offer.disk_default_gb,
            "vcpus": offer.vcpu_default,
            "memory": offer.memory_default_gb,
            "autoRestart": False,
        }
        pod["sshKeyId"] = self.ensure_bootstrap_ssh_key()
        pod = {key: value for key, value in pod.items() if value is not None}
        payload: dict[str, Any] = {
            "pod": pod,
            "provider": {"type": offer.provider_name},
        }
        if options.disk_id:
            payload["disks"] = [options.disk_id]
        if self.config.team_id:
            payload["team"] = {"teamId": self.config.team_id}
        result = self._request("POST", "/pods", json_payload=payload, timeout=60)
        if not isinstance(result, dict):
            raise PrimeApiError("Prime create-pod response was not an object.")
        return result

    @staticmethod
    def runtime_env(config: DeploymentConfig) -> dict[str, str]:
        options = prime_provider_options(config)
        if config.backend == BackendType.VLLM:
            env: dict[str, str] = {
                "MODEL_NAME": (config.model_name or "").strip(),
                "SERVED_MODEL_NAME": (config.served_model_name or "").strip(),
                "N_GPU": str(config.n_gpu or config.gpu_count or 1),
                "TRUST_REMOTE_CODE": "true" if config.trust_remote_code else "false",
                "FAST_BOOT": "true" if config.fast_boot else "false",
                "VLLM_API_KEY": (config.endpoint_api_key or "").strip(),
            }
            optional = {
                "MODEL_REVISION": config.model_revision,
                "REASONING_PARSER": config.reasoning_parser,
                "TOOL_CALL_PARSER": config.tool_call_parser,
                "DEFAULT_CHAT_TEMPLATE_KWARGS": config.default_chat_template_kwargs,
            }
            env.update(
                {
                    key: value.strip()
                    for key, value in optional.items()
                    if value and value.strip()
                }
            )
            if options.disk_id:
                env["HF_HOME"] = "/data/huggingface"
                env["VLLM_CACHE_ROOT"] = "/data/vllm"
        else:
            extra_args = shlex.split(config.server_args or "")
            env = {
                "REPO_ID": (config.repo_id or "").strip(),
                "QUANT": (config.quant or "").strip(),
                "SERVED_MODEL_NAME": (config.served_model_name or "").strip(),
                "N_GPU_LAYERS": (
                    str(config.n_gpu_layers)
                    if config.n_gpu_layers is not None
                    else "all"
                ),
                "LLAMACPP_API_KEY": (config.endpoint_api_key or "").strip(),
                "LLAMACPP_SERVER_ARGS": "\n".join(extra_args),
            }
            if options.disk_id:
                env["LLAMA_CACHE"] = "/data/llama.cpp"
        try:
            from huggingface_hub import get_token

            hf_token = (get_token() or "").strip()
        except Exception:
            hf_token = ""
        if hf_token:
            env["HF_TOKEN"] = hf_token
        return {key: value for key, value in env.items() if value}

    @staticmethod
    def _ssh_connection(pod: dict[str, Any]) -> tuple[str, str | None]:
        value = pod.get("sshConnection")
        if isinstance(value, list):
            value = next((item for item in value if item), "")
        tokens = shlex.split(str(value or ""))
        target = next((token for token in tokens if "@" in token and not token.startswith("-")), "")
        if not re.fullmatch(r"[A-Za-z0-9._-]+@[A-Za-z0-9.:\[\]-]+", target):
            raise ValueError("Prime pod is active but has no valid SSH connection.")
        port: str | None = None
        for index, token in enumerate(tokens[:-1]):
            if token == "-p" and tokens[index + 1].isdigit():
                port = tokens[index + 1]
                break
        return target, port

    def _ssh_args(self, pod: dict[str, Any]) -> list[str]:
        if not self.ssh_private_key_path.exists():
            raise ValueError(
                "Launchpad's Prime bootstrap SSH key is missing; the pod cannot be managed."
            )
        target, port = self._ssh_connection(pod)
        pod_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(pod.get("id") or "pod"))
        PRIME_KNOWN_HOSTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(PRIME_KNOWN_HOSTS_DIR, 0o700)
        except OSError:
            pass
        args = [
            "ssh",
            "-i",
            str(self.ssh_private_key_path),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            f"UserKnownHostsFile={PRIME_KNOWN_HOSTS_DIR / pod_id}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=20",
        ]
        if port:
            args.extend(["-p", port])
        args.append(target)
        return args

    def _run_ssh(
        self,
        pod: dict[str, Any],
        command: str,
        *,
        input_text: str | None = None,
        timeout: float = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [*self._ssh_args(pod), command],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("OpenSSH is required for Prime portable launches.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out connecting to the Prime pod over SSH.") from exc

    @staticmethod
    def _privileged_shell_command(command: str) -> str:
        """Run as root directly, or use passwordless sudo for non-root images."""

        quoted = shlex.quote(command)
        return (
            'if [ "$(id -u)" -eq 0 ]; then '
            f"sh -c {quoted}; "
            "elif command -v sudo >/dev/null 2>&1; then "
            f"sudo -n sh -c {quoted}; "
            "else echo 'Prime image requires root or passwordless sudo' >&2; exit 127; fi"
        )

    def _run_privileged_ssh(
        self,
        pod: dict[str, Any],
        command: str,
        *,
        input_text: str | None = None,
        timeout: float = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run a privileged shell command across Prime image user variants."""

        return self._run_ssh(
            pod,
            self._privileged_shell_command(command),
            input_text=input_text,
            timeout=timeout,
        )

    def _write_remote_runtime_file(
        self,
        pod: dict[str, Any],
        filename: str,
        content: str,
        *,
        mode: str,
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", filename):
            raise ValueError(f"Invalid Prime runtime filename: {filename}")
        directory = self._run_privileged_ssh(
            pod,
            f"install -d -m 700 {PRIME_RUNTIME_ROOT}",
        )
        if directory.returncode != 0:
            raise RuntimeError(
                "Could not prepare the Prime runtime directory: "
                f"{(directory.stderr or directory.stdout).strip()}"
            )
        path = f"{PRIME_RUNTIME_ROOT}/{filename}"
        written = self._run_privileged_ssh(
            pod,
            f"umask 077; cat > {path}",
            input_text=content,
        )
        if written.returncode != 0:
            raise RuntimeError(
                f"Could not upload {filename} to the Prime pod: "
                f"{(written.stderr or written.stdout).strip()}"
            )
        changed = self._run_privileged_ssh(pod, f"chmod {mode} {path}")
        if changed.returncode != 0:
            raise RuntimeError(f"Could not secure {filename} on the Prime pod.")

    @staticmethod
    def _docker_env_file(config: DeploymentConfig) -> str:
        runtime_env = PrimeBackend.runtime_env(config)
        values: dict[str, str] = {}
        endpoint_api_key = str(config.endpoint_api_key or "").strip()
        if not endpoint_api_key:
            raise ValueError("Prime portable runtimes require an endpoint API key.")
        if runtime_env.get("HF_TOKEN"):
            values["HF_TOKEN"] = runtime_env["HF_TOKEN"]
        if config.backend == BackendType.LLAMACPP:
            values["LLAMA_ARG_API_KEY"] = endpoint_api_key
            if prime_provider_options(config).disk_id:
                values["LLAMA_CACHE"] = "/data/llama.cpp"
        else:
            values["VLLM_API_KEY"] = endpoint_api_key
            if prime_provider_options(config).disk_id:
                values["HF_HOME"] = "/data/huggingface"
                values["VLLM_CACHE_ROOT"] = "/data/vllm"
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"Prime runtime value {key} may not contain newlines.")
        return "".join(f"{key}={value}\n" for key, value in values.items() if value)

    @staticmethod
    def _probe_config(config: DeploymentConfig) -> str:
        api_key = str(config.endpoint_api_key or "")
        if any(character in api_key for character in {'"', "\\", "\n", "\r"}):
            raise ValueError(
                "Endpoint API keys may not contain quotes, backslashes, or newlines on Prime."
            )
        return (
            "silent\nshow-error\nfail\nmax-time = 5\n"
            f'header = "Authorization: Bearer {api_key}"\n'
        )

    @staticmethod
    def _bootstrap_docker_command(
        config: DeploymentConfig,
        launch: PrimeLaunchSpec,
    ) -> list[str]:
        options = prime_provider_options(config)
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            PRIME_RUNTIME_CONTAINER_NAME,
            "--restart",
            "unless-stopped",
            "--gpus",
            "all",
            "--ipc=host",
            "-p",
            "8000:8000" if options.allow_insecure_http else "127.0.0.1:8000:8000",
            "--env-file",
            f"{PRIME_RUNTIME_ROOT}/runtime.env",
            "-v",
            "llm-launchpad-hf-cache:/root/.cache/huggingface",
        ]
        if options.disk_id:
            command.extend(["-v", "/data:/data"])

        if config.backend == BackendType.LLAMACPP:
            repo = str(config.repo_id or "").strip()
            quant = str(config.quant or "").strip()
            hf_repo = f"{repo}:{quant}" if quant else repo
            llama_args = [
                "/app/llama-server",
                "--hf-repo",
                hf_repo,
                "--n-gpu-layers",
                str(config.n_gpu_layers) if config.n_gpu_layers is not None else "all",
                *shlex.split(config.server_args or ""),
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--alias",
                str(config.served_model_name or repo.rsplit("/", 1)[-1]),
            ]
            inner = f'exec {shlex.join(llama_args)} --api-key "$LLAMA_ARG_API_KEY"'
            if not options.disk_id:
                command.extend(
                    ["-v", "llm-launchpad-llama-cache:/root/.cache/llama.cpp"]
                )
            command.extend(
                ["--entrypoint", "/bin/sh", launch.container_image, "-lc", inner]
            )
            return command

        model_name = str(config.model_name or "").strip()
        served_name = str(config.served_model_name or model_name.rsplit("/", 1)[-1])
        vllm_args = [
            "vllm",
            "serve",
            model_name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--uvicorn-log-level",
            "info",
            "--served-model-name",
            served_name,
            "--tensor-parallel-size",
            str(config.n_gpu or config.gpu_count or 1),
        ]
        if config.model_revision:
            vllm_args.extend(["--revision", config.model_revision])
        if config.trust_remote_code:
            vllm_args.append("--trust-remote-code")
        if config.fast_boot:
            vllm_args.append("--enforce-eager")
        if config.reasoning_parser:
            vllm_args.extend(["--reasoning-parser", config.reasoning_parser])
        if config.tool_call_parser:
            vllm_args.extend(
                ["--enable-auto-tool-choice", "--tool-call-parser", config.tool_call_parser]
            )
        if config.default_chat_template_kwargs:
            vllm_args.extend(
                ["--default-chat-template-kwargs", config.default_chat_template_kwargs]
            )
        inner = f'exec {shlex.join(vllm_args)} --api-key "$VLLM_API_KEY"'
        command.extend(
            [
                "--entrypoint",
                "/bin/bash",
                launch.container_image,
                "-lc",
                inner,
            ]
        )
        return command

    @staticmethod
    def _bootstrap_script(config: DeploymentConfig, launch: PrimeLaunchSpec) -> str:
        docker_command = PrimeBackend._bootstrap_docker_command(config, launch)
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"rm -f {PRIME_RUNTIME_ROOT}/bootstrap.exit",
                (
                    "trap 'status=$?; printf \"%s\\n\" \"$status\" > "
                    f"{PRIME_RUNTIME_ROOT}/bootstrap.exit' EXIT"
                ),
                (
                    "for attempt in $(seq 1 36); do "
                    "if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; "
                    "then break; fi; "
                    'echo "Waiting for the Prime image to finish installing Docker '
                    '($attempt/36)"; sleep 5; done'
                ),
                (
                    "command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || "
                    "{ echo 'Prime image did not provide a ready Docker runtime after 180 seconds' "
                    ">&2; exit 127; }"
                ),
                "nvidia-smi >/dev/null",
                f"docker pull {shlex.quote(launch.container_image)}",
                f"docker rm -f {PRIME_RUNTIME_CONTAINER_NAME} >/dev/null 2>&1 || true",
                shlex.join(docker_command),
                "",
            ]
        )

    def start_bootstrap_runtime(
        self,
        config: DeploymentConfig,
        pod: dict[str, Any],
    ) -> None:
        """Upload and asynchronously start a portable inference runtime."""

        launch = resolve_prime_launch_spec(config)
        self._write_remote_runtime_file(
            pod,
            "runtime.env",
            self._docker_env_file(config),
            mode="600",
        )
        self._write_remote_runtime_file(
            pod,
            "probe.curl",
            self._probe_config(config),
            mode="600",
        )
        self._write_remote_runtime_file(
            pod,
            "bootstrap.sh",
            self._bootstrap_script(config, launch),
            mode="700",
        )
        started = self._run_privileged_ssh(
            pod,
            (
                "nohup "
                f"{PRIME_RUNTIME_ROOT}/bootstrap.sh > {PRIME_RUNTIME_ROOT}/bootstrap.log "
                "2>&1 </dev/null &"
            ),
        )
        if started.returncode != 0:
            raise RuntimeError(
                "Could not start the Prime runtime bootstrap: "
                f"{(started.stderr or started.stdout).strip()}"
            )

    @staticmethod
    def _tunnel_config(tunnel: PrimeTunnel, *, local_port: int = 8000) -> str:
        """Build the minimal frpc configuration documented by Prime Tunnel."""

        if not 1 <= local_port <= 65535:
            raise ValueError("Prime Tunnel local port must be between 1 and 65535.")
        if not 1 <= tunnel.server_port <= 65535:
            raise ValueError("Prime Tunnel returned an invalid server port.")

        def toml_string(value: str) -> str:
            return json.dumps(value, ensure_ascii=True)

        return "\n".join(
            [
                f"serverAddr = {toml_string(tunnel.server_host)}",
                f"serverPort = {tunnel.server_port}",
                f"user = {toml_string(tunnel.tunnel_id)}",
                'auth.method = "token"',
                f"auth.token = {toml_string(tunnel.frp_token)}",
                f"metadatas.binding_secret = {toml_string(tunnel.binding_secret)}",
                "transport.tcpMux = true",
                "transport.tcpMuxKeepaliveInterval = 30",
                "transport.poolCount = 10",
                "transport.dialServerKeepalive = 60",
                'log.to = "console"',
                'log.level = "info"',
                "[[proxies]]",
                f"name = {toml_string(tunnel.tunnel_id)}",
                'type = "http"',
                'localIP = "127.0.0.1"',
                f"localPort = {local_port}",
                f"subdomain = {toml_string(tunnel.tunnel_id)}",
                "",
            ]
        )

    @staticmethod
    def _tunnel_bootstrap_script() -> str:
        """Download Prime's pinned frpc build, verify it, and start the tunnel."""

        networking = prime_networking_runtime()
        version = str(networking.get("frpc_version") or "").strip()
        amd64_checksum = str(
            networking.get("frpc_linux_amd64_sha256") or ""
        ).strip()
        arm64_checksum = str(
            networking.get("frpc_linux_arm64_sha256") or ""
        ).strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError("Prime Tunnel frpc version is not configured correctly.")
        for checksum in (amd64_checksum, arm64_checksum):
            if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError("Prime Tunnel frpc checksum is not configured correctly.")
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'frpc_version="{version}"',
                'case "$(uname -m)" in',
                f'  x86_64|amd64) frpc_arch="amd64"; frpc_sha256="{amd64_checksum}" ;;',
                f'  aarch64|arm64) frpc_arch="arm64"; frpc_sha256="{arm64_checksum}" ;;',
                '  *) echo "Unsupported Prime Tunnel architecture: $(uname -m)" >&2; exit 1 ;;',
                "esac",
                'work_dir="$(mktemp -d /tmp/llm-launchpad-frpc.XXXXXX)"',
                'trap \'rm -rf "$work_dir"\' EXIT',
                'archive="$work_dir/frp.tar.gz"',
                (
                    'curl --fail --location --retry 3 --proto "=https" '
                    '--proto-redir "=https" --output "$archive" '
                    '"https://github.com/fatedier/frp/releases/download/v${frpc_version}/'
                    'frp_${frpc_version}_linux_${frpc_arch}.tar.gz"'
                ),
                'printf "%s  %s\\n" "$frpc_sha256" "$archive" | sha256sum --check --status',
                (
                    'tar -xzf "$archive" -C "$work_dir" --strip-components=1 '
                    '"frp_${frpc_version}_linux_${frpc_arch}/frpc"'
                ),
                f'install -m 755 "$work_dir/frpc" {PRIME_RUNTIME_ROOT}/frpc',
                f'if [ -s {PRIME_TUNNEL_PID_PATH} ]; then',
                f'  old_pid="$(cat {PRIME_TUNNEL_PID_PATH})"',
                '  if kill -0 "$old_pid" 2>/dev/null; then kill "$old_pid"; fi',
                "fi",
                (
                    f'nohup {PRIME_RUNTIME_ROOT}/frpc -c {PRIME_RUNTIME_ROOT}/tunnel.toml '
                    f'> {PRIME_TUNNEL_LOG_PATH} 2>&1 </dev/null &'
                ),
                f'printf "%s\\n" "$!" > {PRIME_TUNNEL_PID_PATH}',
                "",
            ]
        )

    def start_tunnel(
        self,
        pod: dict[str, Any],
        tunnel: PrimeTunnel,
        *,
        local_port: int = 8000,
    ) -> None:
        """Install and asynchronously start Prime's reverse-tunnel client."""

        self._write_remote_runtime_file(
            pod,
            "tunnel.toml",
            self._tunnel_config(tunnel, local_port=local_port),
            mode="600",
        )
        self._write_remote_runtime_file(
            pod,
            "tunnel-bootstrap.sh",
            self._tunnel_bootstrap_script(),
            mode="700",
        )
        started = self._run_privileged_ssh(
            pod,
            f"{PRIME_RUNTIME_ROOT}/tunnel-bootstrap.sh",
            timeout=180,
        )
        if started.returncode != 0:
            detail = (started.stderr or started.stdout).strip()
            raise RuntimeError(
                "Could not start Prime Tunnel: " + self._redact_runtime_log(detail)
            )

    def tunnel_runtime_logs(self, pod: dict[str, Any], tail: int = 50) -> list[str]:
        """Return sanitized Prime Tunnel client logs from the pod."""

        count = min(500, max(1, int(tail)))
        result = self._run_privileged_ssh(
            pod,
            f"tail -n {count} {PRIME_TUNNEL_LOG_PATH}",
            timeout=20,
        )
        lines = (result.stdout or result.stderr).splitlines()
        return [self._redact_runtime_log(line) for line in lines]

    def tunnel_runtime_status(
        self,
        pod: dict[str, Any],
        tunnel_id: str,
    ) -> tuple[bool, bool, str]:
        """Return ``(ready, failed, detail)`` for the pod-side tunnel client."""

        process = self._run_privileged_ssh(
            pod,
            (
                f"test -s {PRIME_TUNNEL_PID_PATH} && "
                f"kill -0 $(cat {PRIME_TUNNEL_PID_PATH})"
            ),
            timeout=20,
        )
        logs = self.tunnel_runtime_logs(pod, tail=30)
        lowered_logs = "\n".join(logs).casefold()
        if process.returncode != 0:
            detail = logs[-1] if logs else "Prime Tunnel client exited"
            return False, True, detail
        tunnel = self.get_tunnel(tunnel_id)
        if tunnel is None:
            return False, True, "Prime Tunnel registration no longer exists"
        status = tunnel.status.casefold()
        if status in {"failed", "error", "expired", "terminated", "deleted"}:
            return False, True, f"Prime Tunnel entered {status} state"
        if status in {"connected", "active", "running"} or "start proxy success" in lowered_logs:
            return True, False, "secure HTTPS tunnel is connected"
        return False, False, f"waiting for Prime Tunnel ({status or 'pending'})"

    @staticmethod
    def _redact_runtime_log(value: str) -> str:
        value = re.sub(r"(--api-key(?:=|\s+))\S+", r"\1[redacted]", value)
        value = re.sub(
            (
                r"((?:HF_TOKEN|VLLM_API_KEY|LLAMA_ARG_API_KEY|"
                r"frp_token|binding_secret)=)\S+"
            ),
            r"\1[redacted]",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            (
                r"((?:['\"]?(?:api_key|token|frp_token|binding_secret)['\"]?)"
                r"\s*[=:]\s*)"
                r"(\[[^\]]*\]|'[^']*'|\"[^\"]*\"|\S+)"
            ),
            r"\1[redacted]",
            value,
            flags=re.IGNORECASE,
        )
        return value

    def bootstrap_runtime_status(self, pod: dict[str, Any]) -> tuple[bool, bool, str]:
        """Return ``(ready, failed, detail)`` for a portable runtime."""

        probe = self._run_privileged_ssh(
            pod,
            (
                f"curl --config {PRIME_RUNTIME_ROOT}/probe.curl "
                "http://127.0.0.1:8000/v1/models"
            ),
            timeout=20,
        )
        if probe.returncode == 0:
            return True, False, "OpenAI-compatible endpoint is ready"

        state = self._run_privileged_ssh(
            pod,
            (
                "docker inspect "
                "--format='{{.State.Status}}|{{.State.ExitCode}}|{{.RestartCount}}' "
                f"{PRIME_RUNTIME_CONTAINER_NAME}"
            ),
            timeout=20,
        )
        state_fields = state.stdout.strip().split("|") if state.returncode == 0 else []
        container_state = state_fields[0].casefold() if state_fields else ""
        try:
            exit_code = int(state_fields[1]) if len(state_fields) > 1 else 0
        except ValueError:
            exit_code = 0
        try:
            restart_count = int(state_fields[2]) if len(state_fields) > 2 else 0
        except ValueError:
            restart_count = 0
        if container_state == "restarting" or restart_count > 0:
            logs = self.bootstrap_runtime_logs(pod, tail=30)
            detail = logs[-1] if logs else f"last exit code {exit_code}"
            return (
                False,
                True,
                f"runtime container restart loop (count {restart_count}): {detail}",
            )
        if container_state in {"exited", "dead", "removing"}:
            logs = self.bootstrap_runtime_logs(pod, tail=30)
            detail = logs[-1] if logs else (
                f"runtime container is {container_state} (exit code {exit_code})"
            )
            return False, True, detail
        if container_state == "running":
            network = self._run_privileged_ssh(
                pod,
                (
                    "docker stats --no-stream --format='{{.NetIO}}' "
                    f"{PRIME_RUNTIME_CONTAINER_NAME}"
                ),
                timeout=20,
            )
            detail = "runtime container is loading the model"
            if network.returncode == 0 and network.stdout.strip():
                detail += f" (network {network.stdout.strip()})"
            return False, False, detail

        bootstrap_exit = self._run_privileged_ssh(
            pod,
            f"cat {PRIME_RUNTIME_ROOT}/bootstrap.exit",
            timeout=20,
        )
        if bootstrap_exit.returncode == 0 and bootstrap_exit.stdout.strip() not in {"", "0"}:
            logs = self.bootstrap_runtime_logs(pod, tail=30)
            detail = logs[-1] if logs else "runtime bootstrap failed"
            return False, True, detail
        return False, False, "pulling the runtime image"

    def bootstrap_runtime_logs(self, pod: dict[str, Any], tail: int = 200) -> list[str]:
        """Read portable runtime logs over SSH without exposing env files."""

        count = min(5000, max(1, int(tail)))
        result = self._run_privileged_ssh(
            pod,
            (
                f"docker logs --tail {count} {PRIME_RUNTIME_CONTAINER_NAME} "
                f"2>&1 || tail -n {count} {PRIME_RUNTIME_ROOT}/bootstrap.log"
            ),
            timeout=30,
        )
        lines = (result.stdout or result.stderr).splitlines()
        return [self._redact_runtime_log(line) for line in lines]

    def list_pods(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        limit = 100
        while True:
            payload = self._request("GET", "/pods", params={"offset": offset, "limit": limit})
            page_rows = payload.get("data", []) if isinstance(payload, dict) else []
            rows.extend(row for row in page_rows if isinstance(row, dict))
            total = int(
                payload.get("total_count", payload.get("totalCount", len(page_rows)))
            ) if isinstance(payload, dict) else 0
            offset += len(page_rows)
            if not page_rows or offset >= total:
                break
        return rows

    def list_deployments(self) -> list[EndpointInfo]:
        from .connection_store import load_connection_entries

        cached_entries = load_connection_entries()
        results: list[EndpointInfo] = []
        for row in self.list_pods():
            name = str(row.get("name") or "")
            cached = cached_entries.get(name, {})
            cached_provider = str(cached.get("provider") or "")
            if (
                not name.startswith(PRIME_POD_PREFIX)
                and cached_provider != ComputeProvider.PRIME.value
            ):
                continue
            instance_name = str(cached.get("instance_name") or "").strip() or None
            backend_value = str(cached.get("backend") or "").strip()
            try:
                backend = BackendType(backend_value)
            except ValueError:
                backend = (
                    BackendType.LLAMACPP
                    if name.startswith("llp-prime-llamacpp-")
                    else BackendType.VLLM
                )
            results.append(
                EndpointInfo(
                    name=name,
                    app_id=str(row.get("id") or ""),
                    state=str(row.get("status") or "unknown").lower(),
                    backend=backend,
                    instance_name=instance_name
                    or infer_instance_from_app_name(name, backend),
                    provider=ComputeProvider.PRIME,
                )
            )
        return results

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/pods/{pod_id}")
        if not isinstance(payload, dict):
            raise PrimeApiError("Prime pod response was not an object.")
        return payload

    def get_pod_logs(self, pod_id: str, tail: int = 200) -> list[str]:
        pod = self.get_pod(pod_id)
        return self.bootstrap_runtime_logs(pod, tail=tail)

    def delete_pod(self, pod_id: str) -> None:
        tunnel_error: Exception | None = None
        try:
            self.delete_tunnels_for_pod(pod_id)
        except Exception as exc:
            tunnel_error = exc
        self._request("DELETE", f"/pods/{pod_id}")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", pod_id or "pod")
        try:
            (PRIME_KNOWN_HOSTS_DIR / safe_id).unlink(missing_ok=True)
        except OSError:
            pass
        if tunnel_error is not None:
            raise PrimeApiError(
                f"Prime pod was terminated, but its tunnel cleanup failed: {tunnel_error}"
            ) from tunnel_error

    @staticmethod
    def endpoint_url(
        pod: dict[str, Any],
        *,
        allow_insecure_http: bool = False,
        allow_direct_ip: bool = False,
    ) -> str:
        mappings = pod.get("primePortMapping")
        rows = mappings if isinstance(mappings, list) else []
        mapping = next(
            (
                row
                for row in rows
                if isinstance(row, dict) and str(row.get("internal") or "") == "8000"
            ),
            None,
        )
        if mapping is None:
            mapping = next(
                (
                    row
                    for row in rows
                    if isinstance(row, dict) and str(row.get("internal") or "") == "*"
                ),
                None,
            )
        if not isinstance(mapping, dict):
            if not allow_direct_ip:
                raise ValueError("Prime pod does not expose the inference port (internal 8000).")
            ip = pod.get("ip")
            if isinstance(ip, list):
                ip = next((item for item in ip if item), "")
            host = str(ip or "").strip()
            if not host:
                raise ValueError("Prime pod is missing a public IP address.")
            if not allow_insecure_http:
                raise ValueError(
                    "Prime's default image exposes this endpoint over HTTP; enable "
                    "the explicit HTTP opt-in to use it."
                )
            return f"http://{host}:8000"
        external = str(mapping.get("external") or "").strip()
        if external == "*":
            external = "8000"
        if external.startswith("https://"):
            return external.rstrip("/")
        if external.startswith("http://"):
            if not allow_insecure_http:
                raise ValueError("Prime endpoint is HTTP-only; pass --allow-insecure-http to use it.")
            return external.rstrip("/")
        ip = pod.get("ip")
        if isinstance(ip, list):
            ip = next((item for item in ip if item), "")
        host = str(ip or "").strip()
        if not host or not external:
            raise ValueError("Prime pod is missing a public IP or external port mapping.")
        protocol = str(mapping.get("protocol") or "").casefold()
        scheme = "https" if protocol == "https" else "http"
        if scheme == "http" and not allow_insecure_http:
            raise ValueError("Prime endpoint is HTTP-only; pass --allow-insecure-http to use it.")
        return f"{scheme}://{host}:{external}"

    def public_endpoint_ready(self, endpoint: str, api_key: str) -> tuple[bool, str]:
        """Probe the public runtime endpoint without forwarding Prime credentials."""

        try:
            response = requests.get(
                f"{endpoint.rstrip('/')}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        except requests.RequestException as exc:
            return False, str(exc)
        if 200 <= response.status_code < 300:
            return True, ""
        return False, f"HTTP {response.status_code}"

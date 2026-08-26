"""Safety and reporting primitives for opt-in Prime live validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
import time
from typing import Any, Callable


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s'\"]+"),
    re.compile(
        r"(?i)((?:['\"]?(?:api[_-]?key|token|frp[_-]?token|binding[_-]?secret)"
        r"['\"]?)\s*[=:]\s*)"
        r"(\[[^\]]*\]|'[^']*'|\"[^\"]*\"|\S+)"
    ),
    re.compile(
        r"(?i)((?:VLLM_API_KEY|LLAMACPP_API_KEY|LLAMA_ARG_API_KEY|HF_TOKEN|"
        r"FRP_TOKEN|BINDING_SECRET)=)\S+"
    ),
)


def utc_now_iso() -> str:
    """Return a compact UTC timestamp suitable for live-test reports."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_live_value(value: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove known credentials and common bearer-token forms from report text."""

    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[redacted]", redacted)
    return redacted


@dataclass
class BillableResource:
    """One live Prime resource whose cost accrues over time."""

    kind: str
    resource_id: str
    hourly_rate_usd: float
    created_monotonic: float
    closed_monotonic: float | None = None

    def estimated_cost(self, now: float) -> float:
        end = self.closed_monotonic if self.closed_monotonic is not None else now
        seconds = max(0.0, end - self.created_monotonic)
        return self.hourly_rate_usd * seconds / 3600.0


class BudgetExceeded(RuntimeError):
    """Raised before a live action would cross the configured safety budget."""


@dataclass
class PrimeBudgetGuard:
    """Track accrued provider cost and reserve a cleanup safety margin."""

    cap_usd: float = 3.0
    cleanup_reserve_usd: float = 0.30
    clock: Callable[[], float] = time.monotonic
    resources: dict[str, BillableResource] = field(default_factory=dict)

    @property
    def operational_cap_usd(self) -> float:
        return max(0.0, self.cap_usd - self.cleanup_reserve_usd)

    def register(self, kind: str, resource_id: str, hourly_rate_usd: float) -> None:
        if resource_id in self.resources:
            return
        self.resources[resource_id] = BillableResource(
            kind=kind,
            resource_id=resource_id,
            hourly_rate_usd=max(0.0, hourly_rate_usd),
            created_monotonic=self.clock(),
        )

    def close(self, resource_id: str) -> None:
        resource = self.resources.get(resource_id)
        if resource is not None and resource.closed_monotonic is None:
            resource.closed_monotonic = self.clock()

    def estimated_cost_usd(self) -> float:
        now = self.clock()
        return sum(resource.estimated_cost(now) for resource in self.resources.values())

    def require_capacity(
        self,
        *,
        hourly_rate_usd: float,
        maximum_runtime_seconds: float,
        description: str,
    ) -> None:
        projected = self.estimated_cost_usd() + (
            max(0.0, hourly_rate_usd) * max(0.0, maximum_runtime_seconds) / 3600.0
        )
        if projected > self.operational_cap_usd:
            raise BudgetExceeded(
                f"{description} would project ${projected:.2f}, above the "
                f"${self.operational_cap_usd:.2f} operational cutoff."
            )

    def assert_below_cap(self) -> None:
        estimated = self.estimated_cost_usd()
        if estimated > self.cap_usd:
            raise BudgetExceeded(
                f"Estimated Prime spend ${estimated:.2f} exceeded the ${self.cap_usd:.2f} cap."
            )


@dataclass
class PrimeResourceLedger:
    """IDs that must be removed even when a live stage fails or is interrupted."""

    pod_ids: set[str] = field(default_factory=set)
    disk_ids: set[str] = field(default_factory=set)
    cleanup_errors: list[str] = field(default_factory=list)

    def add_pod(self, pod_id: str) -> None:
        if pod_id:
            self.pod_ids.add(pod_id)

    def add_disk(self, disk_id: str) -> None:
        if disk_id:
            self.disk_ids.add(disk_id)

    def close_pod(self, pod_id: str) -> None:
        self.pod_ids.discard(pod_id)

    def close_disk(self, disk_id: str) -> None:
        self.disk_ids.discard(disk_id)


@dataclass
class PrimeLiveStage:
    """Serializable evidence for one validation stage."""

    name: str
    success: bool = False
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str = ""
    duration_seconds: float = 0.0
    offer_id: str = ""
    gpu_type: str = ""
    gpu_count: int = 0
    hourly_rate_usd: float | None = None
    pod_ids: list[str] = field(default_factory=list)
    disk_id: str = ""
    endpoint_scheme: str = ""
    auth_statuses: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class PrimeLiveReport:
    """Top-level, redacted report produced by a live validation run."""

    run_id: str
    commit: str
    budget_cap_usd: float
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str = ""
    estimated_spend_usd: float = 0.0
    stages: list[PrimeLiveStage] = field(default_factory=list)
    cleanup: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, secrets: tuple[str, ...] = ()) -> dict[str, Any]:
        """Serialize and recursively redact every string value."""

        def clean(value: Any) -> Any:
            if isinstance(value, str):
                return redact_live_value(value, secrets)
            if isinstance(value, dict):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return value

        return clean(asdict(self))

"""Offline demo TUI with fake deployments for local UI iteration.

Launches the real Textual screens (Home, Manage, Endpoint Actions, Connection
Info, Status, Logs, Stop, Jobs, Storage) backed entirely by in-memory fake
deployments. No Modal, Prime Intellect, Vast.ai, Hugging Face, or Artificial
Analysis network/CLI calls are made, and nothing is written to
``~/.llm_launchpad/`` or OpenCode config.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import threading
import time
from typing import Any

from ..core.job_store import JobRecord, JobStatus, StoredEvent
from ..core.provider_billing import (
    BalanceKind,
    BillingCharge,
    BillingStatus,
    ProviderBilling,
    StorageBilling,
)
from ..core.provider_readiness import ProviderReadiness, ProviderReadinessStage
from ..protocol.enums import (
    BackendType,
    ComputeProvider,
    DeploymentState,
    OperationType,
    VisionVerification,
)
from ..protocol.events import (
    BaseEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import (
    DeploymentConfig,
    EndpointInfo,
    FleetDiscovery,
    ProviderListing,
    ServingSnapshot,
    ServingStats,
    StorageSnapshot,
    StoredModelInfo,
    VisionCapabilities,
)
from .app import ProviderReadinessUpdated, TuiApp
from .screens.manage import ManageScreen
from .screens.monitor import MonitorScreen
from .workers import StorageLoaded, _dispatch_event


def _demo_key(row: EndpointInfo) -> tuple[str, str, str]:
    return (row.provider.value, row.app_id or "", row.name or "")


def build_demo_endpoints(*, now: float | None = None) -> list[EndpointInfo]:
    """Build a representative set of fake deployments across all providers."""
    moment = time.time() if now is None else now
    prime_started = moment - 5610.0  # 1h 33m
    vast_started = moment - 485.0  # 8m 05s
    return [
        EndpointInfo(
            name="prime-vllm-qwen3-32b",
            app_id="pod-demo-prime-1",
            state="active",
            backend=BackendType.VLLM,
            instance_name="qwen3-32b",
            web_url="https://demo-prime-qwen3-32b.example.invalid/v1",
            served_model_name="qwen3-32b",
            display_name="Qwen3 32B",
            model_name="Qwen/Qwen3-32B",
            runtime_status="healthy",
            runtime_status_detail="HTTP 200",
            provider=ComputeProvider.PRIME,
            endpoint_api_key="lp-demo-prime-key-7f9a2c",
            max_context_tokens=32768,
            max_output_tokens=8192,
            vision=VisionCapabilities(
                supported=True,
                enabled=True,
                verification=VisionVerification.PASSED,
                message="Verified with a demo image request.",
            ),
            serving=ServingSnapshot(
                stats=ServingStats(
                    captured_at=time.monotonic(),
                    prompt_tokens=182_400.0,
                    generation_tokens=61_920.0,
                    requests_running=2.0,
                    requests_waiting=0.0,
                    requests_finished=148.0,
                    kv_cache_usage=0.34,
                    avg_ttft_seconds=0.18,
                    runtime_started_at=prime_started,
                ),
                total_prompt_tokens=182_400.0,
                total_generation_tokens=61_920.0,
                tokens_per_second=84.6,
                observed_at=moment - 4.0,
            ),
            run_started_at_epoch=prime_started,
            hourly_cost_usd=2.45,
            cumulative_cost_usd=3.82,
            cost_estimated=False,
            cost_tracked_since_epoch=prime_started,
        ),
        EndpointInfo(
            name="llamacpp-glm-4-7-flash",
            app_id="ap-demo-modal-1",
            state="deployed",
            backend=BackendType.LLAMACPP,
            instance_name="glm-4-7-flash",
            web_url="https://demo--llamacpp-glm-4-7-flash.modal.run",
            served_model_name="glm-4.7-flash",
            display_name="GLM-4.7 Flash (Q4_K_M)",
            repo_id="unsloth/GLM-4.7-Flash-GGUF",
            quant="Q4_K_M",
            runtime_status="unchecked",
            runtime_status_detail="health not checked",
            provider=ComputeProvider.MODAL,
            endpoint_api_key="lp-demo-modal-key-3b8d1e",
            max_context_tokens=65536,
            vision=VisionCapabilities(
                supported=False,
                enabled=False,
                message="Text-only GGUF model.",
            ),
            serving=ServingSnapshot(
                total_prompt_tokens=48_200.0,
                total_generation_tokens=14_800.0,
                observed_at=moment - 1920.0,
            ),
            hourly_cost_usd=1.10,
            cumulative_cost_usd=0.64,
            cost_estimated=True,
            cost_tracked_since_epoch=moment - 14_400.0,
        ),
        EndpointInfo(
            name="vast-llamacpp-deepseek-r1",
            app_id="vast-9182734",
            state="starting",
            backend=BackendType.LLAMACPP,
            instance_name="deepseek-r1-32b",
            served_model_name="deepseek-r1-32b",
            display_name="DeepSeek-R1 Distill 32B (Q4_K_M)",
            repo_id="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
            quant="Q4_K_M",
            runtime_status="in_progress",
            runtime_status_detail="Loading GGUF shards into GPU memory",
            provider=ComputeProvider.VAST,
            endpoint_api_key="lp-demo-vast-key-9c4e0a",
            max_context_tokens=32768,
            run_started_at_epoch=vast_started,
            hourly_cost_usd=0.89,
            cumulative_cost_usd=0.12,
            cost_estimated=False,
            cost_tracked_since_epoch=vast_started,
        ),
        EndpointInfo(
            name="vllm-llama-3-3-70b",
            app_id="ap-demo-modal-2",
            state="stopped",
            backend=BackendType.VLLM,
            instance_name="llama-3-3-70b",
            served_model_name="llama-3.3-70b",
            display_name="Llama 3.3 70B Instruct",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            runtime_status="in_progress",
            provider=ComputeProvider.MODAL,
            serving=ServingSnapshot(
                total_prompt_tokens=310_000.0,
                total_generation_tokens=95_400.0,
                observed_at=moment - 21_600.0,
            ),
            hourly_cost_usd=3.95,
            cumulative_cost_usd=5.18,
            cost_estimated=True,
            cost_tracked_since_epoch=moment - 86_400.0,
        ),
    ]


def build_demo_storage_snapshot() -> StorageSnapshot:
    """Fake cached-model inventory so Storage and Home billing stay offline."""
    return StorageSnapshot(
        llamacpp_models=[
            StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id="unsloth/GLM-4.7-Flash-GGUF",
                revision="a8f31c9e442b81f090e7a2b5d1c3e8f60a1b2c3d",
                quant="Q4_K_M",
                size_bytes=18_420_000_000,
                file_count=1,
                source_volume="llamacpp-models",
            ),
            StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id="unsloth/GLM-4.7-Flash-GGUF",
                revision="a8f31c9e442b81f090e7a2b5d1c3e8f60a1b2c3d",
                quant="Q8_0",
                size_bytes=32_150_000_000,
                file_count=1,
                source_volume="llamacpp-models",
            ),
            StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
                revision="main",
                quant="Q4_K_M",
                size_bytes=19_850_000_000,
                file_count=1,
                source_volume="llamacpp-models",
                incomplete=True,
            ),
        ],
        vllm_models=[
            StoredModelInfo(
                backend=BackendType.VLLM,
                model_id="Qwen/Qwen3-32B",
                revision="main",
                size_bytes=64_900_000_000,
                file_count=18,
                source_volume="vllm-models",
            ),
        ],
    )


def build_demo_billing() -> tuple[ProviderBilling, ...]:
    """Fake billing snapshots for Home's Usage & Balance panel."""
    return (
        ProviderBilling(
            provider=ComputeProvider.MODAL,
            status=BillingStatus.READY,
            kind=BalanceKind.SPEND_MTD,
            amount_usd=14.20,
            charges=(
                BillingCharge("gpu", 12.85),
                BillingCharge("storage", 1.35),
            ),
            storage=StorageBilling(metered_usd=1.35, after_allowance_usd=0.0),
        ),
        ProviderBilling(
            provider=ComputeProvider.PRIME,
            status=BillingStatus.READY,
            kind=BalanceKind.BALANCE,
            amount_usd=42.50,
        ),
        ProviderBilling(
            provider=ComputeProvider.VAST,
            status=BillingStatus.READY,
            kind=BalanceKind.CREDIT,
            amount_usd=18.75,
        ),
    )


def _stored_events(job_id: str, events: list[BaseEvent], start_epoch: float) -> list[StoredEvent]:
    _ = job_id
    return [
        StoredEvent(
            seq=index + 1,
            timestamp=start_epoch + index * 2.0,
            type=type(event).__name__,
            event=event,
        )
        for index, event in enumerate(events)
    ]


class _DemoJobStore:
    """In-memory JobStore substitute for the Manage -> Jobs tab."""

    def __init__(self, *, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        base_dt = datetime.fromtimestamp(moment)
        prime_config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="prime-vllm-qwen3-32b",
            instance_name="qwen3-32b",
            model_name="Qwen/Qwen3-32B",
            served_model_name="qwen3-32b",
            gpu_type="H100_80GB",
            gpu_count=1,
            max_context_tokens=32768,
            endpoint_api_key="lp-demo-prime-key-7f9a2c",
        )
        modal_config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            app_name="llamacpp-glm-4-7-flash",
            instance_name="glm-4-7-flash",
            repo_id="unsloth/GLM-4.7-Flash-GGUF",
            quant="Q4_K_M",
            served_model_name="glm-4.7-flash",
            gpu_type="L40S",
            gpu_count=1,
            max_context_tokens=65536,
            endpoint_api_key="lp-demo-modal-key-3b8d1e",
        )
        vast_failed_config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.VAST,
            app_name="vast-llamacpp-qwen3-72b",
            instance_name="qwen3-72b",
            repo_id="unsloth/Qwen3-72B-GGUF",
            quant="Q4_K_M",
            served_model_name="qwen3-72b",
            gpu_type="RTX 4090",
            gpu_count=2,
            max_context_tokens=32768,
        )
        self._jobs: dict[str, JobRecord] = {
            "job-demo-prime-1": JobRecord(
                id="job-demo-prime-1",
                status=JobStatus.SUCCEEDED,
                config=prime_config,
                outcome="Ready (demo)",
                app_name=prime_config.app_name or "",
                provider=ComputeProvider.PRIME.value,
                backend=BackendType.VLLM.value,
                resource_app_id="pod-demo-prime-1",
                event_count=5,
                updated_at=moment - 5600.0,
            ),
            "job-demo-modal-1": JobRecord(
                id="job-demo-modal-1",
                status=JobStatus.SUCCEEDED,
                config=modal_config,
                outcome="Ready (demo)",
                app_name=modal_config.app_name or "",
                provider=ComputeProvider.MODAL.value,
                backend=BackendType.LLAMACPP.value,
                resource_app_id="ap-demo-modal-1",
                event_count=4,
                updated_at=moment - 14_350.0,
            ),
            "job-demo-vast-1": JobRecord(
                id="job-demo-vast-1",
                status=JobStatus.FAILED,
                config=vast_failed_config,
                outcome="Failed — resource stopped. Demo host timed out during image pull.",
                app_name=vast_failed_config.app_name or "",
                provider=ComputeProvider.VAST.value,
                backend=BackendType.LLAMACPP.value,
                event_count=4,
                updated_at=moment - 22_000.0,
            ),
        }
        self._events: dict[str, list[StoredEvent]] = {
            "job-demo-prime-1": _stored_events(
                "job-demo-prime-1",
                [
                    StateChangeEvent(
                        current=DeploymentState.DEPLOYING,
                        operation=OperationType.DEPLOY,
                        detail="Provisioning Prime Intellect pod (1x H100 80GB) [demo]",
                        timestamp=base_dt - timedelta(seconds=5660),
                    ),
                    LogEvent(
                        line="[demo] Allocated pod pod-demo-prime-1 in us-east-1.",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                        timestamp=base_dt - timedelta(seconds=5645),
                    ),
                    StateChangeEvent(
                        current=DeploymentState.WARMING_UP,
                        operation=OperationType.WARMUP,
                        detail="Waiting for vLLM OpenAI endpoint to pass /health",
                        timestamp=base_dt - timedelta(seconds=5625),
                    ),
                    LogEvent(
                        line="[demo] Warmup certified: qwen3-32b answered /v1/chat/completions (TTFT 0.18s).",
                        operation=OperationType.WARMUP,
                        is_milestone=True,
                        timestamp=base_dt - timedelta(seconds=5605),
                    ),
                    OperationCompleteEvent(
                        operation=OperationType.DEPLOY,
                        success=True,
                        detail="Ready (demo)",
                        timestamp=base_dt - timedelta(seconds=5600),
                    ),
                ],
                moment - 5660.0,
            ),
            "job-demo-modal-1": _stored_events(
                "job-demo-modal-1",
                [
                    StateChangeEvent(
                        current=DeploymentState.DEPLOYING,
                        operation=OperationType.DEPLOY,
                        detail="Deploying Modal llama.cpp app (1x L40S) [demo]",
                        timestamp=base_dt - timedelta(seconds=14_400),
                    ),
                    LogEvent(
                        line="[demo] Reused cached GGUF unsloth/GLM-4.7-Flash-GGUF (Q4_K_M).",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                        timestamp=base_dt - timedelta(seconds=14_380),
                    ),
                    LogEvent(
                        line="[demo] Endpoint live at https://demo--llamacpp-glm-4-7-flash.modal.run",
                        operation=OperationType.WARMUP,
                        is_milestone=True,
                        timestamp=base_dt - timedelta(seconds=14_355),
                    ),
                    OperationCompleteEvent(
                        operation=OperationType.DEPLOY,
                        success=True,
                        detail="Ready (demo)",
                        timestamp=base_dt - timedelta(seconds=14_350),
                    ),
                ],
                moment - 14_400.0,
            ),
            "job-demo-vast-1": _stored_events(
                "job-demo-vast-1",
                [
                    StateChangeEvent(
                        current=DeploymentState.DEPLOYING,
                        operation=OperationType.DEPLOY,
                        detail="Renting Vast.ai 2x RTX 4090 [demo]",
                        timestamp=base_dt - timedelta(seconds=22_060),
                    ),
                    LogEvent(
                        line="[demo] Pulling llama.cpp server image on host #88412...",
                        operation=OperationType.DEPLOY,
                        timestamp=base_dt - timedelta(seconds=22_040),
                    ),
                    LogEvent(
                        line="[demo] Host timed out during container startup; destroyed rental.",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                        timestamp=base_dt - timedelta(seconds=22_005),
                    ),
                    OperationCompleteEvent(
                        operation=OperationType.DEPLOY,
                        success=False,
                        exit_code=1,
                        detail="Demo host timed out during image pull.",
                        timestamp=base_dt - timedelta(seconds=22_000),
                    ),
                ],
                moment - 22_060.0,
            ),
        }
        self._results: dict[str, dict[str, str]] = {
            "job-demo-prime-1": {"url": "https://demo-prime-qwen3-32b.example.invalid/v1"},
            "job-demo-modal-1": {"url": "https://demo--llamacpp-glm-4-7-flash.modal.run"},
        }

    def list_jobs(self, *, include_terminal: bool = True, limit: int = 200) -> list[JobRecord]:
        records = list(self._jobs.values())
        if not include_terminal:
            records = [record for record in records if not record.terminal]
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return records[:limit]

    def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def get_events(self, job_id: str, *, after_seq: int = 0, limit: int = 5000) -> list[StoredEvent]:
        events = [event for event in self._events.get(job_id, []) if event.seq > after_seq]
        return events[:limit]

    def get_result(self, job_id: str) -> dict[str, str] | None:
        result = self._results.get(job_id)
        return dict(result) if result is not None else None

    def request_cancel(self, job_id: str) -> bool:
        _ = job_id
        return False

    def reconcile_workers(self) -> None:
        return None

    def import_journal_entries(self) -> None:
        return None

    def pid_alive(self, pid: int | None) -> bool:
        _ = pid
        return False


class DemoTuiApp(TuiApp):
    """TUI configured with fake in-memory deployments for offline UI iteration."""

    TITLE = "llm-launchpad [DEMO]"
    SUB_TITLE = "Fake deployments · no cloud resources"
    is_demo_mode = True

    def __init__(
        self,
        *,
        mouse_enabled: bool | None = None,
        open_manage: bool = False,
        **kwargs: Any,
    ) -> None:
        self._open_manage = open_manage
        self._demo_lock = threading.RLock()
        self._demo_endpoints: list[EndpointInfo] = build_demo_endpoints()
        self._demo_storage = build_demo_storage_snapshot()
        self._demo_billing = build_demo_billing()
        self._demo_job_store = _DemoJobStore()
        self._demo_refresh_ticks = 0
        super().__init__(mouse_enabled=mouse_enabled, **kwargs)
        self._username = "demo"
        self._provider_readiness = {
            provider: ProviderReadiness(
                provider=provider,
                stage=ProviderReadinessStage.READY,
                detail="demo mode (offline)",
                profile="demo" if provider == ComputeProvider.MODAL else None,
            )
            for provider in ComputeProvider
        }
        self._storage_snapshot_cache = self._demo_storage
        self._storage_snapshot_cached_at_epoch = time.time()

    # -- Disable real disk caches, journal, and OpenCode writes -----------

    def _load_persisted_storage_cache(self) -> None:
        self._storage_snapshot_cache = self._demo_storage
        self._storage_snapshot_cached_at_epoch = time.time()

    def _persist_storage_snapshot(self, snapshot: StorageSnapshot) -> None:
        _ = snapshot

    def _load_persisted_deploy_connection_cache(self) -> None:
        self._deploy_connection_cache = {}

    def _persist_deploy_connection_cache(self) -> None:
        return None

    def _merge_deploy_connection_cache(self, rows: list[EndpointInfo]) -> None:
        _ = rows

    def _warn_about_abandoned_deployments(self) -> None:
        return None

    def _reconcile_durable_jobs(self) -> None:
        return None

    def _get_job_store(self) -> Any:
        return self._demo_job_store

    def _sync_opencode(self, **kwargs: Any) -> None:
        monitor = kwargs.get("monitor")
        if monitor is not None:
            _dispatch_event(monitor, LogEvent(line="[demo] Skipped OpenCode config sync."))

    # -- Startup & provider readiness -------------------------------------

    def _enter_main_menu(self) -> None:
        super()._enter_main_menu()
        if self._open_manage:
            self.push_screen(ManageScreen())

    def refresh_provider_readiness(self) -> None:
        snapshot = self.provider_readiness_snapshot()
        self.post_message(ProviderReadinessUpdated(snapshot))

    # -- Fake fleet discovery & live metrics ------------------------------

    def discover_fleet(self) -> FleetDiscovery:
        now = time.time()
        with self._demo_lock:
            self._demo_refresh_ticks += 1
            self._tick_live_demo_metrics(now)
            rows_by_provider: dict[ComputeProvider, list[EndpointInfo]] = {
                provider: [] for provider in ComputeProvider
            }
            for row in self._demo_endpoints:
                rows_by_provider[row.provider].append(self._clone_endpoint(row))
            listings = tuple(
                ProviderListing(
                    provider=provider,
                    rows=tuple(rows_by_provider[provider]),
                    retrieved_at_epoch=now,
                )
                for provider in ComputeProvider
            )
            discovery = FleetDiscovery(listings=listings)
        with self._provider_listing_lock:
            self._provider_listings = {listing.provider: listing for listing in listings}
            self._last_fleet_discovery = discovery
        return discovery

    def _tick_live_demo_metrics(self, now: float) -> None:
        """Advance live Prime/Vast counters gently so refreshes look alive."""
        for row in self._demo_endpoints:
            if row.provider == ComputeProvider.PRIME and row.state in {"active", "running"} and row.serving is not None:
                added_prompt = 640.0
                added_gen = 290.0
                rate = 78.0 + float((self._demo_refresh_ticks * 7) % 19)
                stats = replace(
                    row.serving.stats,
                    captured_at=time.monotonic(),
                    prompt_tokens=(row.serving.stats.prompt_tokens or 0.0) + added_prompt,
                    generation_tokens=(row.serving.stats.generation_tokens or 0.0) + added_gen,
                    requests_running=float(1 + (self._demo_refresh_ticks % 3)),
                    kv_cache_usage=min(0.85, 0.28 + 0.04 * (self._demo_refresh_ticks % 5)),
                )
                row.serving = ServingSnapshot(
                    stats=stats,
                    total_prompt_tokens=row.serving.total_prompt_tokens + added_prompt,
                    total_generation_tokens=row.serving.total_generation_tokens + added_gen,
                    tokens_per_second=rate,
                    observed_at=now,
                )
                if row.hourly_cost_usd and row.run_started_at_epoch:
                    row.cumulative_cost_usd = round(
                        row.hourly_cost_usd * max(0.0, now - row.run_started_at_epoch) / 3600.0,
                        4,
                    )

    @staticmethod
    def _clone_endpoint(row: EndpointInfo) -> EndpointInfo:
        cloned = replace(row)
        if row.serving is not None:
            cloned.serving = replace(row.serving, stats=replace(row.serving.stats))
        if row.vision is not None:
            cloned.vision = replace(row.vision)
        return cloned

    def _find_demo_row(self, row: EndpointInfo) -> EndpointInfo | None:
        target_key = _demo_key(row)
        for candidate in self._demo_endpoints:
            if _demo_key(candidate) == target_key:
                return candidate
        for candidate in self._demo_endpoints:
            if (
                candidate.provider == row.provider
                and candidate.name
                and candidate.name == row.name
            ):
                return candidate
        return None

    def annotate_serving_stats(self, rows: list[EndpointInfo]) -> None:
        """Sync Manage rows from the in-memory fake fleet without network/disk IO."""
        now = time.time()
        with self._demo_lock:
            self._demo_refresh_ticks += 1
            self._tick_live_demo_metrics(now)
            for row in rows:
                source = self._find_demo_row(row)
                if source is None:
                    continue
                cloned = self._clone_endpoint(source)
                row.serving = cloned.serving
                row.live_metrics_checked_at = cloned.live_metrics_checked_at
                row.live_metrics_error = cloned.live_metrics_error
                row.run_started_at_epoch = cloned.run_started_at_epoch
                row.hourly_cost_usd = cloned.hourly_cost_usd
                row.cumulative_cost_usd = cloned.cumulative_cost_usd
                row.cost_estimated = cloned.cost_estimated
                row.cost_tracked_since_epoch = cloned.cost_tracked_since_epoch
                row.runtime_status = cloned.runtime_status
                row.runtime_status_detail = cloned.runtime_status_detail
                row.runtime_checked_at = cloned.runtime_checked_at

    def fetch_live_metrics(self, row: EndpointInfo) -> EndpointInfo:
        """Simulate an explicit single-endpoint metrics fetch (including Modal wake)."""
        now = time.time()
        with self._demo_lock:
            target = self._find_demo_row(row)
            if target is None:
                row.live_metrics_checked_at = now
                row.live_metrics_error = "demo endpoint not found"
                return row
            prior_prompt = target.serving.total_prompt_tokens if target.serving else 0.0
            prior_gen = target.serving.total_generation_tokens if target.serving else 0.0
            added_prompt = 1_120.0
            added_gen = 480.0
            started = target.run_started_at_epoch or (now - 95.0)
            target.run_started_at_epoch = started
            target.runtime_status = "healthy"
            target.runtime_status_detail = "HTTP 200 (demo live fetch)"
            target.runtime_checked_at = now
            target.live_metrics_checked_at = now
            target.live_metrics_error = None
            target.serving = ServingSnapshot(
                stats=ServingStats(
                    captured_at=time.monotonic(),
                    prompt_tokens=prior_prompt + added_prompt,
                    generation_tokens=prior_gen + added_gen,
                    requests_running=1.0,
                    requests_waiting=0.0,
                    kv_cache_usage=0.22 if target.backend == BackendType.VLLM else None,
                    avg_ttft_seconds=0.21 if target.backend == BackendType.VLLM else None,
                    reported_tokens_per_second=(
                        62.4 if target.backend == BackendType.LLAMACPP else None
                    ),
                    runtime_started_at=started,
                ),
                total_prompt_tokens=prior_prompt + added_prompt,
                total_generation_tokens=prior_gen + added_gen,
                tokens_per_second=62.4 if target.backend == BackendType.LLAMACPP else 89.2,
                observed_at=now,
            )
            if target.cumulative_cost_usd is not None and target.hourly_cost_usd:
                target.cumulative_cost_usd = round(
                    target.cumulative_cost_usd + target.hourly_cost_usd * (30.0 / 3600.0),
                    4,
                )
            return self._clone_endpoint(target)

    def annotate_home_deployments(self, rows: list[EndpointInfo]) -> None:
        """Keep Home screen fleet statuses aligned with in-memory fake state."""
        self.annotate_serving_stats(rows)

    def load_demo_billing(self) -> tuple[ProviderBilling, ...]:
        """Return offline demo billing snapshots for Home's Usage & Balance panel."""
        return self._demo_billing

    # -- Simulated Manage operations --------------------------------------

    def _run_status(
        self,
        endpoint: EndpointInfo,
        url: str,
        timeout: int,
        app_name: str,
        monitor: MonitorScreen,
    ) -> None:
        _ = timeout
        now = time.time()
        _dispatch_event(monitor, LogEvent(line=f"[demo] Target app: {app_name}", operation=OperationType.STATUS))
        _dispatch_event(
            monitor,
            StateChangeEvent(
                current=DeploymentState.WARMING_UP,
                operation=OperationType.STATUS,
                detail=f"Probing {url}",
            ),
        )
        _dispatch_event(
            monitor,
            LogEvent(
                line=f"[demo] GET {url.rstrip('/')}/health -> 200 OK (42 ms)",
                operation=OperationType.STATUS,
                is_milestone=True,
            ),
        )
        model_id = endpoint.served_model_name or endpoint.model_name or "demo-model"
        _dispatch_event(
            monitor,
            LogEvent(
                line=f"[demo] GET {url.rstrip('/')}/models -> serving {model_id}",
                operation=OperationType.STATUS,
                is_milestone=True,
            ),
        )
        with self._demo_lock:
            stored = self._find_demo_row(endpoint)
            if stored is not None:
                stored.runtime_status = "healthy"
                stored.runtime_status_detail = "HTTP 200 (demo)"
                stored.runtime_checked_at = now
        self._invalidate_endpoint_snapshot()
        _dispatch_event(
            monitor,
            OperationCompleteEvent(
                operation=OperationType.STATUS,
                success=True,
                detail=f"Endpoint {app_name} is healthy (demo).",
            ),
        )

    def _run_logs(
        self,
        backend: BackendType,
        follow: bool,
        app_ref: str,
        app_name: str,
        monitor: MonitorScreen,
        provider: ComputeProvider = ComputeProvider.MODAL,
    ) -> None:
        _ = follow
        target_label = app_name if app_ref == app_name else f"{app_name} ({app_ref})"
        _dispatch_event(monitor, LogEvent(line=f"[demo] Target app: {target_label}", operation=OperationType.LOGS))
        _dispatch_event(
            monitor,
            StateChangeEvent(
                current=DeploymentState.RUNNING,
                operation=OperationType.LOGS,
                detail=f"Streaming demo logs ({provider.display_name}/{backend.value})",
            ),
        )
        if backend == BackendType.VLLM:
            sample_lines = (
                "INFO 11-25 13:58:02 [api_server.py:912] vLLM API server version 0.8.5 (demo)",
                "INFO 11-25 13:58:14 [engine.py:291] KV cache blocks: 14,820 GPU · max context: 32,768",
                "INFO 11-25 13:59:41 [metrics.py:118] Avg prompt throughput: 412.4 tok/s, Avg generation throughput: 84.6 tok/s, Running: 2 reqs, GPU KV cache usage: 34.0%",
                'INFO 11-25 14:00:08 [uvicorn.access] 127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
            )
        else:
            sample_lines = (
                "main: server is listening on http://0.0.0.0:8000 (demo llama.cpp)",
                "srv  load_model: offloaded 48/48 layers to GPU · flash_attn=1 · kv_cache=f16",
                "slot update_slots: id 0 | task 42 | prompt eval time = 184.2 ms / 512 tokens (2779.6 tok/s)",
                "slot release: id 0 | task 42 | eval time = 4120.5 ms / 256 tokens (62.1 tok/s)",
            )
        for line in sample_lines:
            _dispatch_event(monitor, LogEvent(line=line, operation=OperationType.LOGS))
        _dispatch_event(
            monitor,
            OperationCompleteEvent(
                operation=OperationType.LOGS,
                success=True,
                detail="End of demo log stream.",
            ),
        )

    def _run_stop(
        self,
        backend: BackendType,
        app_ref: str,
        app_name: str,
        monitor: MonitorScreen,
        provider: ComputeProvider = ComputeProvider.MODAL,
    ) -> None:
        _ = backend
        target_label = app_name if app_ref == app_name else f"{app_name} ({app_ref})"
        _dispatch_event(monitor, LogEvent(line=f"[demo] Target app: {target_label}", operation=OperationType.STOP))
        _dispatch_event(
            monitor,
            StateChangeEvent(
                current=DeploymentState.STOPPED,
                operation=OperationType.STOP,
                detail=f"Stopping demo deployment on {provider.display_name}",
            ),
        )
        with self._demo_lock:
            for candidate in self._demo_endpoints:
                if candidate.provider == provider and (
                    candidate.name == app_name or (candidate.app_id and candidate.app_id == app_ref)
                ):
                    candidate.state = "stopped"
                    candidate.runtime_status = "in_progress"
                    candidate.run_started_at_epoch = None
                    if candidate.serving is not None:
                        candidate.serving = ServingSnapshot(
                            total_prompt_tokens=candidate.serving.total_prompt_tokens,
                            total_generation_tokens=candidate.serving.total_generation_tokens,
                            observed_at=candidate.serving.observed_at,
                        )
                    break
        self._invalidate_endpoint_snapshot()
        live_rows, _prune_providers = self._visible_rows_and_prune_scope()
        self._cache_endpoint_snapshot(live_rows)
        _dispatch_event(
            monitor,
            LogEvent(
                line=f"[demo] Marked {app_name} stopped in-memory (no cloud API called).",
                operation=OperationType.STOP,
                is_milestone=True,
            ),
        )
        _dispatch_event(
            monitor,
            OperationCompleteEvent(
                operation=OperationType.STOP,
                success=True,
                detail=f"Stopped {app_name} (demo).",
            ),
        )

    # -- Offline storage & guarded mutations ------------------------------

    def begin_storage_refresh(self, receiver: Any, force: bool = False) -> None:
        _ = force
        poster = getattr(receiver, "post_message", None)
        if callable(poster):
            poster(StorageLoaded(snapshot=self._demo_storage))

    def begin_prime_disks_refresh(self, receiver: Any) -> None:
        from ..core.prime_disks import RetainedPrimeDisk
        from .workers import PrimeDisksLoaded

        poster = getattr(receiver, "post_message", None)
        if callable(poster):
            poster(
                PrimeDisksLoaded(
                    disks=[
                        RetainedPrimeDisk(
                            id="disk-demo-us-east-1",
                            name="llp-cache-qwen3",
                            size_gb=250,
                            status="ACTIVE",
                            location="us-east-1",
                            price_per_hour_usd=0.0342,
                            managed=True,
                        ),
                        RetainedPrimeDisk(
                            id="disk-demo-eu-west-2",
                            name="llp-cache-deepseek",
                            size_gb=150,
                            status="UNATTACHED",
                            location="eu-west-2",
                            price_per_hour_usd=0.0205,
                            managed=True,
                        ),
                    ]
                )
            )

    def begin_deploy(self, config: DeploymentConfig) -> None:
        _ = config
        self.notify(
            "Demo mode does not start real cloud deployments.",
            severity="warning",
            timeout=5,
        )

    def begin_storage_predownload(
        self,
        backend: BackendType,
        model_id: str,
        quant: str | None = None,
        revision: str | None = None,
    ) -> None:
        _ = (backend, model_id, quant, revision)
        self.notify("Demo mode does not download to cloud storage.", severity="warning", timeout=4)

    def begin_storage_delete(self, model: StoredModelInfo) -> None:
        _ = model
        self.notify("Demo mode does not delete cloud storage.", severity="warning", timeout=4)

    def begin_prime_disk_delete(self, disk_id: str, receiver: Any) -> None:
        _ = (disk_id, receiver)
        self.notify("Demo mode does not delete Prime disks.", severity="warning", timeout=4)

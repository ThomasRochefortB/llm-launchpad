#!/usr/bin/env python3
"""Run the opt-in, TUI-driven Prime Intellect live validation suite."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable
from urllib.parse import urlsplit

import requests
from textual.widgets import Button, Input, OptionList, Select, Switch

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.naming import build_deployment_name
from llm_launchpad.core.opencode import provider_id_for_app
from llm_launchpad.core.prime_backend import (
    PRIME_DEFAULT_BOOTSTRAP_IMAGE,
    PrimeBackend,
    PrimeDiskOffer,
    is_compatible_prime_offer,
    preferred_prime_offer_image,
    select_prime_offer,
)
from llm_launchpad.core.prime_live import (
    PrimeBudgetGuard,
    PrimeLiveReport,
    PrimeLiveStage,
    PrimeResourceLedger,
    redact_live_value,
    utc_now_iso,
)
from llm_launchpad.core.provider_options import prime_provider_options
from llm_launchpad.core.quick_deploy import (
    get_quick_deploy_profile,
    list_quick_deploy_models,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    ComputeAvailabilitySnapshot,
    ComputeOffer,
    DeploymentConfig,
    InferencePlan,
    StorageSnapshot,
)
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.deploy import (
    LlamaCppDeployScreen,
    PrimeOffersLoaded,
    VllmDeployScreen,
)
from llm_launchpad.tui.screens.fast_deploy import (
    FastDeployAvailabilityLoaded,
    FastDeployScreen,
)
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen
from llm_launchpad.tui.workers import OperationDone, StorageLoaded


VLLM_MODEL = "Qwen/Qwen3-0.6B"
LLAMA_MODEL = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
LLAMA_QUANT = "Q4_K_M"
LIVE_STAGES = (
    "portable_vllm_and_auth",
    "portable_llamacpp_and_auth",
    "persistent_disk_cache_reuse",
    "multi_gpu_vllm",
    "management_restart_and_tunnel_recovery",
    "failed_deployment_cleanup",
    "qwen38_27b_quick_deploy",
)
_ACTIVE_LIVE_RUN: tuple[
    PrimeBackend,
    PrimeResourceLedger,
    PrimeBudgetGuard,
    str,
] | None = None


class CaptureMonitorScreen(MonitorScreen):
    """Monitor that preserves its final event for the live harness."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.last_done: OperationDone | None = None

    def on_operation_done(self, message: OperationDone) -> None:
        self.last_done = message
        super().on_operation_done(message)


class LiveFastDeployScreen(FastDeployScreen):
    """Fast Deploy screen backed by one already-fetched live snapshot."""

    def __init__(self, snapshot: ComputeAvailabilitySnapshot) -> None:
        super().__init__()
        self.live_snapshot = snapshot

    def _run_load_availability(self, request_id: int, *, purpose: str = "infra") -> None:
        self.post_message(
            FastDeployAvailabilityLoaded(
                self.live_snapshot,
                request_id=request_id,
                purpose=purpose,
            )
        )


class LiveTuiApp(TuiApp):
    """Real TUI deployment flow with deterministic test-only routing."""

    CSS_PATH = Path(__file__).resolve().parents[1] / "llm_launchpad" / "tui" / "theme.tcss"

    def __init__(
        self,
        backend: BackendType,
        *,
        quick_plan: InferencePlan | None = None,
        fast_snapshot: ComputeAvailabilitySnapshot | None = None,
        fast_provider: ComputeProvider | None = None,
        fast_profile_id: str | None = None,
    ) -> None:
        super().__init__(mouse_enabled=False)
        self.live_backend = backend
        self.quick_plan = quick_plan
        self.fast_snapshot = fast_snapshot
        self.fast_provider = fast_provider
        self.fast_profile_id = fast_profile_id
        self.deployed_config: DeploymentConfig | None = None
        self.capture_monitor: CaptureMonitorScreen | None = None
        self.live_notifications: list[tuple[str, str]] = []

    def on_mount(self) -> None:
        if self.fast_snapshot is not None:
            self.push_screen(LiveFastDeployScreen(self.fast_snapshot))
            return
        if self.quick_plan is not None:
            self.push_screen(QuickDeployScreen(self.quick_plan))
            return
        screen = (
            VllmDeployScreen()
            if self.live_backend == BackendType.VLLM
            else LlamaCppDeployScreen()
        )
        self.push_screen(screen)

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        _ = force
        poster = getattr(receiver, "post_message", None)
        if callable(poster):
            poster(StorageLoaded(StorageSnapshot(llamacpp_models=[], vllm_models=[])))

    def begin_deploy(self, config: DeploymentConfig) -> None:
        if not config.app_name:
            config.app_name = build_deployment_name(
                config.provider, config.backend, config.instance_name
            )
        self.deployed_config = config
        monitor = CaptureMonitorScreen(
            title="Deploy",
            deploy_backend=config.backend,
            summarize_backend_logs=True,
            show_debug_logs=config.show_debug_logs,
        )
        self.capture_monitor = monitor
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_deploy(config, monitor),
            name="deploy-worker",
            thread=True,
        )

    def begin_capture_stop(self, config: DeploymentConfig, pod_id: str) -> None:
        monitor = CaptureMonitorScreen(title="Stop")
        self.capture_monitor = monitor
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_stop(
                config.backend,
                pod_id,
                config.app_name or "",
                monitor,
                provider=ComputeProvider.PRIME,
            ),
            name="stop-worker",
            thread=True,
        )

    def _sync_opencode(self, **_: object) -> None:
        """Keep live validation from changing an unrelated OpenCode config."""

    def notify(
        self,
        message: object,
        *,
        severity: str = "information",
        **kwargs: object,
    ) -> None:
        self.live_notifications.append((str(message), severity))
        super().notify(str(message), severity=severity, **kwargs)


def _commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _report_path(value: str | None, run_id: str) -> Path:
    return Path(value) if value else Path(f"/tmp/llm-launchpad-prime-{run_id}.json")


def _write_report(path: Path, report: PrimeLiveReport, secrets: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(secrets), indent=2, sort_keys=True) + "\n")


def _root_endpoint(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _probe_auth(endpoint: str, api_key: str, model: str) -> tuple[dict[str, int], str]:
    base = f"{_root_endpoint(endpoint)}/v1"
    statuses: dict[str, int] = {}
    statuses["models_without_token"] = requests.get(f"{base}/models", timeout=20).status_code
    statuses["models_wrong_token"] = requests.get(
        f"{base}/models",
        headers={"Authorization": "Bearer deliberately-wrong"},
        timeout=20,
    ).status_code
    correct_models = requests.get(
        f"{base}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    statuses["models_correct_token"] = correct_models.status_code
    sentinel = f"PRIME-{int(time.time())}"
    completion = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply with exactly this identifier and nothing else: {sentinel}",
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=180,
    )
    statuses["chat_correct_token"] = completion.status_code
    if statuses["models_without_token"] not in {401, 403}:
        raise AssertionError("OpenAI models endpoint accepted a missing bearer token.")
    if statuses["models_wrong_token"] not in {401, 403}:
        raise AssertionError("OpenAI models endpoint accepted an incorrect bearer token.")
    if correct_models.status_code != 200:
        raise AssertionError(f"Correct bearer token returned HTTP {correct_models.status_code}.")
    if completion.status_code != 200:
        raise AssertionError(f"Authenticated chat returned HTTP {completion.status_code}.")
    if sentinel not in completion.text:
        raise AssertionError("Authenticated chat did not return the requested sentinel.")
    return statuses, sentinel


def _choose_offer(
    prime: PrimeBackend,
    backend: BackendType,
    *,
    gpu_count: int,
    disk_id: str | None = None,
    required_vram_gb: float,
) -> ComputeOffer:
    offers = prime.list_offers(gpu_count=gpu_count, disk_id=disk_id)
    return select_prime_offer(
        offers,
        gpu_count=gpu_count,
        required_vram_gb=required_vram_gb,
        required_image=preferred_prime_offer_image(backend),
    )


def _choose_disk_pair(prime: PrimeBackend) -> tuple[PrimeDiskOffer, ComputeOffer]:
    gpu_offers = prime.list_offers(gpu_count=1)
    candidates: list[tuple[float, PrimeDiskOffer, ComputeOffer]] = []
    for disk in prime.list_disk_offers():
        if (disk.stock_status or "available").casefold() == "unavailable":
            continue
        for gpu in gpu_offers:
            if gpu.provider_name.casefold() != disk.provider_name.casefold():
                continue
            if gpu.data_center.casefold() != disk.data_center.casefold():
                continue
            if not is_compatible_prime_offer(
                gpu,
                0.75,
                required_image=PRIME_DEFAULT_BOOTSTRAP_IMAGE,
            ):
                continue
            candidates.append((gpu.price_per_hour or float("inf"), disk, gpu))
    if not candidates:
        raise RuntimeError("No live Prime GPU and persistent-disk availability intersect.")
    _, disk, gpu = min(candidates, key=lambda item: (item[0], item[1].data_center))
    return disk, gpu


def _matching_pod(prime: PrimeBackend, app_name: str) -> dict[str, Any] | None:
    return next((pod for pod in prime.list_pods() if pod.get("name") == app_name), None)


async def _wait_for_deploy(
    app: LiveTuiApp,
    prime: PrimeBackend,
    offer: ComputeOffer,
    ledger: PrimeResourceLedger,
    budget: PrimeBudgetGuard,
    stage: PrimeLiveStage,
    timeout_seconds: int,
) -> tuple[DeploymentConfig, dict[str, Any], list[str]]:
    started = time.monotonic()
    pod: dict[str, Any] | None = None
    while time.monotonic() - started < timeout_seconds:
        monitor = app.capture_monitor
        config = app.deployed_config
        if config is not None and config.app_name and pod is None:
            try:
                pod = await asyncio.to_thread(_matching_pod, prime, config.app_name)
            except Exception:
                pod = None
            if pod:
                pod_id = str(pod.get("id") or "")
                ledger.add_pod(pod_id)
                budget.register("pod", pod_id, offer.price_per_hour or 0.0)
                if pod_id and pod_id not in stage.pod_ids:
                    stage.pod_ids.append(pod_id)
        budget.assert_below_cap()
        if monitor is not None and monitor.last_done is not None:
            lines = list(monitor.log_viewer.log_widget.lines)
            if not monitor.last_done.success:
                raise RuntimeError(monitor.last_done.detail or "TUI deployment failed.")
            if config is None:
                raise RuntimeError("TUI completed without preserving its deployment config.")
            if pod is None:
                pod = await asyncio.to_thread(_matching_pod, prime, config.app_name or "")
            if pod is None:
                raise RuntimeError("TUI completed but the Prime pod could not be resolved.")
            pod_id = str(pod.get("id") or "")
            if pod_id:
                pod = await asyncio.to_thread(prime.get_pod, pod_id)
            return config, pod, lines
        await asyncio.sleep(2)
    raise TimeoutError(f"TUI deployment exceeded {timeout_seconds} seconds.")


async def _wait_for_stop(
    app: LiveTuiApp,
    prime: PrimeBackend,
    pod_id: str,
    ledger: PrimeResourceLedger,
    budget: PrimeBudgetGuard,
    timeout_seconds: int = 180,
) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        monitor = app.capture_monitor
        if monitor is not None and monitor.last_done is not None:
            if not monitor.last_done.success:
                raise RuntimeError(monitor.last_done.detail or "TUI stop failed.")
            break
        await asyncio.sleep(1)
    else:
        raise TimeoutError("TUI stop did not complete.")
    await _wait_for_resource_cleanup(prime, pod_id, ledger, budget)


async def _wait_for_resource_cleanup(
    prime: PrimeBackend,
    pod_id: str,
    ledger: PrimeResourceLedger,
    budget: PrimeBudgetGuard,
) -> None:
    """Wait until both a pod and its associated tunnel are gone."""

    for _ in range(30):
        rows = await asyncio.to_thread(prime.list_pods)
        if not any(str(row.get("id") or "") == pod_id for row in rows):
            tunnels = await asyncio.to_thread(prime.list_tunnels)
            if not any(f"pod:{pod_id}" in tunnel.labels for tunnel in tunnels):
                ledger.close_pod(pod_id)
                budget.close(pod_id)
                return
        await asyncio.sleep(2)
    raise TimeoutError(f"Prime pod or tunnel for {pod_id} remained live after TUI stop.")


async def _configure_and_start(
    pilot: Any,
    app: LiveTuiApp,
    offer: ComputeOffer,
    *,
    run_id: str,
    stage_slug: str,
    disk_id: str | None,
    vllm_model: str = VLLM_MODEL,
    quick_plan: InferencePlan | None = None,
    fast_provider: ComputeProvider | None = None,
    fast_profile_id: str | None = None,
) -> None:
    screen: object = app.screen
    deploy_screens = (
        VllmDeployScreen,
        LlamaCppDeployScreen,
        QuickDeployScreen,
        FastDeployScreen,
    )
    if not isinstance(screen, deploy_screens):
        app.push_screen(
            LiveFastDeployScreen(app.fast_snapshot)
            if app.fast_snapshot is not None
            else QuickDeployScreen(quick_plan)
            if quick_plan is not None
            else (
                VllmDeployScreen()
                if app.live_backend == BackendType.VLLM
                else LlamaCppDeployScreen()
            )
        )
    for _ in range(50):
        if isinstance(screen, deploy_screens):
            break
        await pilot.pause()
        await asyncio.sleep(0.1)
        screen = app.screen
    else:
        raise RuntimeError(f"TUI deploy form did not mount; current screen is {type(screen).__name__}.")

    if isinstance(screen, FastDeployScreen):
        if fast_provider is None or not fast_profile_id:
            raise RuntimeError("Fast Deploy validation requires a provider and profile ID.")
        models = list_quick_deploy_models()
        model = next(
            (
                candidate
                for candidate in models
                if any(profile.id == fast_profile_id for profile in candidate.profiles)
            ),
            None,
        )
        if model is None:
            raise RuntimeError(f"Fast Deploy model for {fast_profile_id!r} was not found.")
        option_list = screen.query_one("#fast-deploy-list", OptionList)
        model_index = next(
            (
                index
                for index in range(option_list.option_count)
                if option_list.get_option_at_index(index).id == model.id
            ),
            None,
        )
        if model_index is None:
            raise RuntimeError("Qwen3.8 did not appear in the Fast Deploy model picker.")
        option_list.highlighted = model_index
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            if screen._phase == "infra" and screen._infra_rows:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Fast Deploy infrastructure choices did not load.")
        selected = next(
            (
                row
                for row in screen._infra_rows.values()
                if row.plan.quote.provider == fast_provider
                and row.profile.id == fast_profile_id
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(
                f"Fast Deploy did not show {fast_provider.display_name} for {fast_profile_id}."
            )
        option_list = screen.query_one("#fast-deploy-list", OptionList)
        infra_index = next(
            index
            for index in range(option_list.option_count)
            if option_list.get_option_at_index(index).id == selected.plan.quote.id
        )
        option_list.highlighted = infra_index
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            if isinstance(app.screen, QuickDeployScreen):
                screen = app.screen
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Fast Deploy did not open its confirmation screen.")

    instance_name = f"{run_id}-{stage_slug}"
    if isinstance(screen, QuickDeployScreen):
        screen.query_one("#toggle-advanced-quick", Button).press()
        await pilot.pause()
        screen.query_one("#quick-instance-name", Input).value = instance_name
        screen.query_one("#quick-warmup", Switch).value = True
        screen.query_one("#quick-prime-insecure-http", Switch).value = False
        screen.query_one("#quick-prime-keep-failed", Switch).value = False
        screen.query_one("#quick-prime-auto-disk", Switch).value = False
        screen.query_one("#quick-deploy-btn", Button).press()
    elif isinstance(screen, VllmDeployScreen):
        screen.query_one("#model-name", Input).value = vllm_model
        screen.on_prime_offers_loaded(PrimeOffersLoaded([offer]))
        screen.query_one("#provider-vllm", Select).value = ComputeProvider.PRIME.value
        await pilot.pause()
        screen.query_one("#toggle-advanced-vllm", Button).press()
        await pilot.pause()
        screen.query_one("#instance-name-vllm", Input).value = instance_name
        screen.query_one("#chat-template-kwargs", Input).value = '{"enable_thinking": false}'
        if disk_id:
            screen.query_one("#prime-disk-id", Input).value = disk_id
        screen.query_one("#prime-insecure-http", Switch).value = False
        screen.query_one("#warmup-vllm", Switch).value = True
        screen.on_prime_offers_loaded(PrimeOffersLoaded([offer]))
        screen.query_one("#prime-offer-vllm", Select).value = offer.id
        await pilot.pause()
        screen.query_one("#deploy-vllm-btn", Button).press()
    else:
        assert isinstance(screen, LlamaCppDeployScreen)
        screen.query_one("#repo-id", Input).value = LLAMA_MODEL
        screen.query_one("#quant", Input).value = LLAMA_QUANT
        screen.on_prime_offers_loaded(PrimeOffersLoaded([offer]))
        screen.query_one("#provider-llama", Select).value = ComputeProvider.PRIME.value
        await pilot.pause()
        screen.query_one("#toggle-advanced-llama", Button).press()
        await pilot.pause()
        screen.query_one("#instance-name-llama", Input).value = instance_name
        screen.query_one("#server-args", Input).value = "--ctx-size 4096"
        if disk_id:
            screen.query_one("#prime-disk-id-llama", Input).value = disk_id
        screen.query_one("#prime-insecure-http-llama", Switch).value = False
        screen.query_one("#warmup", Switch).value = True
        screen.on_prime_offers_loaded(PrimeOffersLoaded([offer]))
        screen.query_one("#prime-offer-llama", Select).value = offer.id
        await pilot.pause()
        screen.query_one("#deploy-btn", Button).press()

    for _ in range(30):
        await pilot.pause()
        if app.deployed_config is not None:
            return
        await asyncio.sleep(0.1)
    detail = (
        f" Last TUI notification: {app.live_notifications[-1][0]}"
        if app.live_notifications
        else ""
    )
    raise RuntimeError(f"TUI deploy form did not submit.{detail}")


async def _run_tui_stage(
    prime: PrimeBackend,
    budget: PrimeBudgetGuard,
    ledger: PrimeResourceLedger,
    stage: PrimeLiveStage,
    *,
    backend: BackendType,
    offer: ComputeOffer,
    run_id: str,
    stage_slug: str,
    disk_id: str | None = None,
    timeout_seconds: int = 1200,
    inspect: Callable[[DeploymentConfig, dict[str, Any]], dict[str, Any]] | None = None,
    inspect_stops_pod: bool = False,
    secrets: list[str] | None = None,
    quick_plan: InferencePlan | None = None,
    fast_snapshot: ComputeAvailabilitySnapshot | None = None,
    fast_provider: ComputeProvider | None = None,
    fast_profile_id: str | None = None,
) -> tuple[dict[str, Any], DeploymentConfig]:
    budget.require_capacity(
        hourly_rate_usd=offer.price_per_hour or 0.0,
        maximum_runtime_seconds=timeout_seconds,
        description=stage.name,
    )
    started = time.monotonic()
    app = LiveTuiApp(
        backend,
        quick_plan=quick_plan,
        fast_snapshot=fast_snapshot,
        fast_provider=fast_provider,
        fast_profile_id=fast_profile_id,
    )
    pod: dict[str, Any] = {}
    config: DeploymentConfig | None = None
    try:
        async with app.run_test(size=(150, 48)) as pilot:
            await _configure_and_start(
                pilot,
                app,
                offer,
                run_id=run_id,
                stage_slug=stage_slug,
                disk_id=disk_id,
                quick_plan=quick_plan,
                fast_provider=fast_provider,
                fast_profile_id=fast_profile_id,
            )
            config, pod, log_lines = await _wait_for_deploy(
                app,
                prime,
                offer,
                ledger,
                budget,
                stage,
                timeout_seconds,
            )
            pod_id = str(pod.get("id") or "")
            entry = app._deploy_connection_cache.get(config.app_name or "", {})
            endpoint = str(entry.get("base_url") or "")
            api_key = str(config.endpoint_api_key or "")
            if not endpoint or not api_key:
                raise RuntimeError("TUI did not preserve the endpoint URL and generated key.")
            if secrets is not None and api_key not in secrets:
                secrets.append(api_key)
            statuses, sentinel = await asyncio.to_thread(
                _probe_auth,
                endpoint,
                api_key,
                config.served_model_name or config.repo_id or config.model_name or "",
            )
            stage.auth_statuses = statuses
            stage.endpoint_scheme = urlsplit(endpoint).scheme
            stage.evidence["chat_sentinel"] = sentinel
            stage.evidence["tui_milestones"] = [
                redact_live_value(line, (api_key,)) for line in log_lines[-80:]
            ]
            if stage.endpoint_scheme != "https":
                raise AssertionError("Prime deployment did not return an HTTPS endpoint.")
            tunnels = await asyncio.to_thread(prime.list_tunnels)
            matching = [
                tunnel for tunnel in tunnels if f"pod:{pod_id}" in tunnel.labels
            ]
            if len(matching) != 1:
                raise AssertionError(
                    f"Expected one Prime Tunnel for {pod_id}, found {len(matching)}."
                )
            stage.evidence["tunnel_id"] = matching[0].tunnel_id
            stage.evidence["tunnel_status"] = matching[0].status
            stage.evidence["tunnel_expires_at"] = matching[0].expires_at
            if inspect is not None:
                stage.evidence.update(await asyncio.to_thread(inspect, config, pod))
            if inspect_stops_pod:
                await _wait_for_resource_cleanup(prime, pod_id, ledger, budget)
            else:
                app.begin_capture_stop(config, pod_id)
                await _wait_for_stop(app, prime, pod_id, ledger, budget)
        stage.success = True
        return pod, config
    except Exception:
        if config is not None and pod:
            pod_id = str(pod.get("id") or "")
            if pod_id:
                try:
                    await asyncio.to_thread(prime.delete_pod, pod_id)
                    ledger.close_pod(pod_id)
                    budget.close(pod_id)
                except Exception as cleanup_exc:
                    ledger.cleanup_errors.append(str(cleanup_exc))
        raise
    finally:
        stage.duration_seconds = round(time.monotonic() - started, 3)
        stage.finished_at = utc_now_iso()


async def _run_expected_failure_cleanup_stage(
    prime: PrimeBackend,
    budget: PrimeBudgetGuard,
    ledger: PrimeResourceLedger,
    stage: PrimeLiveStage,
    *,
    offer: ComputeOffer,
    run_id: str,
    timeout_seconds: int = 900,
) -> None:
    """Prove a real TUI runtime failure terminates its billable Prime resources."""

    budget.require_capacity(
        hourly_rate_usd=offer.price_per_hour or 0.0,
        maximum_runtime_seconds=timeout_seconds,
        description=stage.name,
    )
    started = time.monotonic()
    app = LiveTuiApp(BackendType.VLLM)
    try:
        async with app.run_test(size=(150, 48)) as pilot:
            await _configure_and_start(
                pilot,
                app,
                offer,
                run_id=run_id,
                stage_slug="expected-failure",
                disk_id=None,
                vllm_model="llm-launchpad/definitely-not-a-real-model-e2e",
            )
            try:
                await _wait_for_deploy(
                    app,
                    prime,
                    offer,
                    ledger,
                    budget,
                    stage,
                    timeout_seconds,
                )
            except RuntimeError as exc:
                failure_detail = str(exc)
            else:
                raise AssertionError("The deliberately invalid model unexpectedly deployed.")

            monitor = app.capture_monitor
            log_lines = list(monitor.log_viewer.log_widget.lines) if monitor else []
            if not stage.pod_ids:
                for line in log_lines:
                    match = re.search(r"Prime pod created:\s*([A-Za-z0-9-]+)", line)
                    if match:
                        stage.pod_ids.append(match.group(1))
                        ledger.add_pod(match.group(1))
                        budget.register(
                            "pod", match.group(1), offer.price_per_hour or 0.0
                        )
                        break
            if not stage.pod_ids:
                raise AssertionError("Failure test did not create a Prime pod.")
            for pod_id in stage.pod_ids:
                await _wait_for_resource_cleanup(prime, pod_id, ledger, budget)
            stage.evidence["expected_failure_observed"] = True
            stage.evidence["failure_detail"] = redact_live_value(failure_detail)
            stage.evidence["tui_milestones"] = [
                redact_live_value(line) for line in log_lines[-80:]
            ]
            stage.evidence["automatic_cleanup_verified"] = True
        stage.success = True
    finally:
        stage.duration_seconds = round(time.monotonic() - started, 3)
        stage.finished_at = utc_now_iso()


def _cache_snapshot(prime: PrimeBackend, pod: dict[str, Any], run_id: str, create: bool) -> str:
    marker = f"/data/llm-launchpad-e2e-{run_id}"
    prefix = f"printf '%s\\n' {run_id} | tee {marker} >/dev/null; " if create else ""
    command = (
        prefix
        + f"test -f {marker}; "
        + "find /data/llama.cpp -type f -name '*.gguf' "
        + "-exec stat -c '%n %s %Y' {} \\; -exec sha256sum {} \\; | sort"
    )
    result = prime._run_privileged_ssh(pod, command, timeout=300)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError((result.stderr or result.stdout or "cache snapshot failed").strip())
    return result.stdout.strip()


def _multi_gpu_evidence(prime: PrimeBackend, pod: dict[str, Any]) -> dict[str, Any]:
    result = prime._run_ssh(
        pod,
        "nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    gpu_rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(gpu_rows) != 2:
        raise AssertionError(f"Expected two visible GPUs, found {len(gpu_rows)}.")
    for row in gpu_rows:
        try:
            used_mib = int(row.rsplit(",", 1)[1].strip())
        except (IndexError, ValueError) as exc:
            raise AssertionError(f"Could not parse GPU memory evidence: {row}") from exc
        if used_mib <= 0:
            raise AssertionError(f"GPU has no model memory allocated: {row}")
    logs = prime.bootstrap_runtime_logs(pod, tail=500)
    joined = "\n".join(logs)
    if not re.search(r"(?i)(world[_ ]size\s*[=:]\s*2|tensor.parallel[^\n]*2)", joined):
        raise AssertionError("vLLM logs did not report tensor-parallel world size 2.")
    return {"nvidia_smi": gpu_rows, "tensor_parallel_log_verified": True}


def _run_isolated_cli(
    args: list[str],
    *,
    opencode_root: Path,
    secrets: tuple[str, ...],
    timeout: int = 180,
) -> dict[str, Any]:
    """Run one real CLI command while keeping OpenCode writes in a temporary root."""

    launcher = """
from pathlib import Path
import os
from llm_launchpad.core import opencode

root = Path(os.environ["LLM_LAUNCHPAD_LIVE_OPENCODE_ROOT"])
opencode.OPENCODE_CONFIG_PATH = root / "opencode.json"
opencode.OPENCODE_JSONC_CONFIG_PATH = root / "opencode.jsonc"
opencode.OPENCODE_REGISTRY_PATH = root / "opencode_registry.json"

from llm_launchpad.cli.main import main
main()
"""
    env = os.environ.copy()
    env["LLM_LAUNCHPAD_LIVE_OPENCODE_ROOT"] = str(opencode_root)
    env["OPENCODE_CONFIG"] = str(opencode_root / "opencode.json")
    result = subprocess.run(
        [sys.executable, "-c", launcher, *args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    redacted_lines = [
        redact_live_value(line, secrets)
        for line in combined.splitlines()
        if line.strip()
    ]
    evidence = {
        "command": "llm-launchpad " + " ".join(args),
        "exit_code": result.returncode,
        "output_tail": redacted_lines[-20:],
    }
    if result.returncode != 0:
        detail = redacted_lines[-1] if redacted_lines else "no command output"
        raise RuntimeError(f"{evidence['command']} failed: {detail}")
    return evidence


def _run_isolated_opencode(
    config: DeploymentConfig,
    *,
    opencode_root: Path,
    secrets: tuple[str, ...],
    timeout: int = 240,
) -> dict[str, Any]:
    """Send a real prompt through the isolated Launchpad OpenCode provider."""

    executable = shutil.which("opencode")
    if not executable:
        raise RuntimeError("OpenCode is not installed for live validation.")

    app_name = str(config.app_name or "").strip()
    model_id = str(config.served_model_name or config.model_name or "").strip()
    if not app_name or not model_id:
        raise RuntimeError("OpenCode validation requires an app name and served model.")
    provider_id = provider_id_for_app(app_name)
    model_reference = f"{provider_id}/{model_id}"

    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(opencode_root / "opencode.json")
    env["XDG_CONFIG_HOME"] = str(opencode_root / "xdg-config")
    env["XDG_DATA_HOME"] = str(opencode_root / "xdg-data")
    env["XDG_CACHE_HOME"] = str(opencode_root / "xdg-cache")
    env["XDG_STATE_HOME"] = str(opencode_root / "xdg-state")
    env["NO_COLOR"] = "1"

    models = subprocess.run(
        [executable, "models", provider_id],
        cwd=opencode_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    models_output = "\n".join(
        part for part in (models.stdout, models.stderr) if part
    ).strip()
    if models.returncode != 0:
        detail = redact_live_value(models_output or "no command output", secrets)
        raise RuntimeError(f"opencode models failed: {detail}")
    if model_reference not in models_output:
        raise AssertionError(
            f"OpenCode did not list the synced model {model_reference}."
        )

    sentinel = f"PRIME-OPENCODE-{int(time.time())}"
    prompt = f"Reply with exactly this identifier and nothing else: {sentinel}"
    run_result = subprocess.run(
        [
            executable,
            "run",
            "--pure",
            "--format",
            "json",
            "--model",
            model_reference,
            prompt,
        ],
        cwd=opencode_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    combined = "\n".join(
        part for part in (run_result.stdout, run_result.stderr) if part
    ).strip()
    redacted_lines = [
        redact_live_value(line, secrets)
        for line in combined.splitlines()
        if line.strip()
    ]
    if run_result.returncode != 0:
        detail = redacted_lines[-1] if redacted_lines else "no command output"
        raise RuntimeError(f"opencode run failed: {detail}")
    if sentinel not in combined:
        raise AssertionError("OpenCode did not return the requested sentinel.")
    return {
        "model_reference": model_reference,
        "models_exit_code": models.returncode,
        "run_exit_code": run_result.returncode,
        "chat_sentinel": sentinel,
        "output_tail": redacted_lines[-20:],
    }


def _llamacpp_management_evidence(
    config: DeploymentConfig,
    *,
    provider: ComputeProvider,
) -> dict[str, Any]:
    """Exercise fresh-process management commands and a real OpenCode request."""

    app_name = str(config.app_name or "").strip()
    api_key = str(config.endpoint_api_key or "").strip()
    if not app_name:
        raise RuntimeError("Management validation requires an app name.")
    secrets = (api_key,) if api_key else ()
    command_evidence: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix=f"llm-launchpad-{provider.value}-llamacpp-cli-"
    ) as temp_root:
        opencode_root = Path(temp_root)
        command_specs = {
            "list": ["list", "--provider", provider.value],
            "status": [
                "status",
                "--provider",
                provider.value,
                "--backend",
                BackendType.LLAMACPP.value,
                "--app-name",
                app_name,
                "--timeout",
                "180",
            ],
            "logs": [
                "logs",
                "--provider",
                provider.value,
                "--backend",
                BackendType.LLAMACPP.value,
                "--app-name",
                app_name,
                "--no-follow",
            ],
            "warmup": [
                "warmup",
                "--provider",
                provider.value,
                "--backend",
                BackendType.LLAMACPP.value,
                "--app-name",
                app_name,
                "--timeout",
                "180",
                "--no-tail-logs",
            ],
            "opencode_dry_run": [
                "opencode",
                "sync",
                "--provider",
                provider.value,
                "--app-name",
                app_name,
                "--dry-run",
            ],
            "opencode_sync": [
                "opencode",
                "sync",
                "--provider",
                provider.value,
                "--app-name",
                app_name,
            ],
        }
        for name, cli_args in command_specs.items():
            command_evidence[name] = _run_isolated_cli(
                cli_args,
                opencode_root=opencode_root,
                secrets=secrets,
                timeout=300,
            )
        command_evidence["opencode_run"] = _run_isolated_opencode(
            config,
            opencode_root=opencode_root,
            secrets=secrets,
            timeout=300,
        )
    return command_evidence


def _management_and_tunnel_recovery_evidence(
    prime: PrimeBackend,
    config: DeploymentConfig,
    pod: dict[str, Any],
) -> dict[str, Any]:
    """Exercise management and OpenCode, then recover a killed tunnel client."""

    app_name = str(config.app_name or "")
    pod_id = str(pod.get("id") or "")
    api_key = str(config.endpoint_api_key or "")
    if not app_name or not pod_id or not api_key:
        raise RuntimeError("Management validation requires app, pod, and endpoint credentials.")

    matching = [
        tunnel
        for tunnel in prime.list_tunnels()
        if f"pod:{pod_id}" in tunnel.labels
    ]
    if len(matching) != 1:
        raise AssertionError(f"Expected one tunnel before recovery, found {len(matching)}.")
    tunnel = matching[0]
    stopped = prime._run_privileged_ssh(
        pod,
        (
            "test -s /opt/llm-launchpad/tunnel.pid; "
            "kill $(cat /opt/llm-launchpad/tunnel.pid); "
            "for attempt in $(seq 1 20); do "
            "kill -0 $(cat /opt/llm-launchpad/tunnel.pid) 2>/dev/null || exit 0; "
            "sleep 1; done; exit 1"
        ),
        timeout=30,
    )
    if stopped.returncode != 0:
        raise RuntimeError("Could not stop the Prime Tunnel client for recovery testing.")
    ready, failed, disconnected_detail = prime.tunnel_runtime_status(pod, tunnel.tunnel_id)
    if ready or not failed:
        raise AssertionError("Killed Prime Tunnel client was not detected as disconnected.")

    restarted = prime._run_privileged_ssh(
        pod,
        "/opt/llm-launchpad/tunnel-bootstrap.sh",
        timeout=180,
    )
    if restarted.returncode != 0:
        detail = (restarted.stderr or restarted.stdout or "tunnel restart failed").strip()
        raise RuntimeError(PrimeBackend._redact_runtime_log(detail))
    deadline = time.monotonic() + 120
    recovered_detail = ""
    while time.monotonic() < deadline:
        ready, failed, recovered_detail = prime.tunnel_runtime_status(pod, tunnel.tunnel_id)
        if ready:
            break
        if failed:
            raise RuntimeError(f"Prime Tunnel recovery failed: {recovered_detail}")
        time.sleep(2)
    else:
        raise TimeoutError(f"Prime Tunnel did not recover: {recovered_detail}")

    endpoint = str(tunnel.url or "")
    auth_statuses, _ = _probe_auth(
        endpoint,
        api_key,
        config.served_model_name or config.model_name or "",
    )

    command_evidence: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="llm-launchpad-prime-cli-") as temp_root:
        opencode_root = Path(temp_root)
        command_specs = {
            "list": ["list", "--provider", "prime"],
            "status": [
                "status", "--provider", "prime", "--backend", "vllm",
                "--app-name", app_name, "--timeout", "60",
            ],
            "logs": [
                "logs", "--provider", "prime", "--backend", "vllm",
                "--app-name", app_name, "--no-follow",
            ],
            "warmup": [
                "warmup", "--provider", "prime", "--backend", "vllm",
                "--app-name", app_name, "--timeout", "60", "--no-tail-logs",
            ],
            "opencode_dry_run": [
                "opencode", "sync", "--provider", "prime",
                "--app-name", app_name, "--dry-run",
            ],
            "opencode_sync": [
                "opencode", "sync", "--provider", "prime",
                "--app-name", app_name,
            ],
        }
        for name, cli_args in command_specs.items():
            command_evidence[name] = _run_isolated_cli(
                cli_args,
                opencode_root=opencode_root,
                secrets=(api_key,),
            )
        command_evidence["opencode_run"] = _run_isolated_opencode(
            config,
            opencode_root=opencode_root,
            secrets=(api_key,),
        )
        command_evidence["stop"] = _run_isolated_cli(
            [
                "stop", "--provider", "prime", "--backend", "vllm",
                "--app-name", app_name, "--yes",
            ],
            opencode_root=opencode_root,
            secrets=(api_key,),
        )

    return {
        "fresh_process_commands": command_evidence,
        "tunnel_disconnect_detected": True,
        "tunnel_disconnect_detail": disconnected_detail,
        "tunnel_recovered": True,
        "tunnel_recovery_detail": recovered_detail,
        "post_recovery_auth_statuses": auth_statuses,
    }


def _qwen38_quick_deploy_evidence(
    prime: PrimeBackend,
    config: DeploymentConfig,
    pod: dict[str, Any],
    *,
    expected_offer_id: str,
) -> dict[str, Any]:
    """Verify that the deployed config came from the curated Qwen3.8 bundle."""

    profile = get_quick_deploy_profile("qwen3-8-27b-q2xl-cheap-l4")
    if config.backend != BackendType.LLAMACPP:
        raise AssertionError("Qwen3.8 quick deploy did not select llama.cpp.")
    if config.repo_id != profile.repo_id or config.quant != profile.quant:
        raise AssertionError("Qwen3.8 quick deploy changed its curated model or quantization.")
    if tuple(shlex.split(config.server_args or "")) != profile.server_args:
        raise AssertionError("Qwen3.8 quick deploy changed its curated server arguments.")
    if config.required_vram_gb != profile.required_vram_gb:
        raise AssertionError("Qwen3.8 quick deploy lost its model-specific VRAM requirement.")
    options = prime_provider_options(config)
    if options.offer_id != expected_offer_id:
        raise AssertionError("Qwen3.8 quick deploy did not preserve the selected Prime offer.")
    if options.auto_disk:
        raise AssertionError("The disposable live validation unexpectedly enabled a cache disk.")

    gpu = prime._run_ssh(
        pod,
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used "
        "--format=csv,noheader,nounits",
        timeout=60,
    )
    if gpu.returncode != 0:
        raise RuntimeError((gpu.stderr or gpu.stdout).strip())
    gpu_rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if len(gpu_rows) != config.gpu_count:
        raise AssertionError(
            f"Expected {config.gpu_count} visible GPU(s), found {len(gpu_rows)}."
        )
    logs = "\n".join(prime.bootstrap_runtime_logs(pod, tail=300))
    if "Qwen3.8-27B" not in logs:
        raise AssertionError("llama.cpp logs did not identify the Qwen3.8 27B model.")
    return {
        "quick_deploy_profile_id": profile.id,
        "repo_id": config.repo_id,
        "quant": config.quant,
        "server_args": list(profile.server_args),
        "required_vram_gb": config.required_vram_gb,
        "selected_offer_id": options.offer_id,
        "nvidia_smi": gpu_rows,
        "model_log_verified": True,
        "fresh_process_commands": _llamacpp_management_evidence(
            config,
            provider=ComputeProvider.PRIME,
        ),
    }


async def _wait_disk_ready(prime: PrimeBackend, disk_id: str, timeout_seconds: int = 300) -> dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        disk = await asyncio.to_thread(prime.get_disk, disk_id)
        status = str(disk.get("status") or "").upper()
        if status in {"ACTIVE", "READY", "UNATTACHED"}:
            return disk
        if status in {"ERROR", "FAILED", "TERMINATED"}:
            raise RuntimeError(f"Prime disk entered {status}.")
        await asyncio.sleep(3)
    raise TimeoutError("Prime disk did not become ready.")


async def _wait_disk_detached(prime: PrimeBackend, disk_id: str, timeout_seconds: int = 300) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        disk = await asyncio.to_thread(prime.get_disk, disk_id)
        pods = disk.get("pods") if isinstance(disk.get("pods"), list) else []
        status = str(disk.get("status") or "").upper()
        if not pods and status in {"ACTIVE", "READY", "UNATTACHED"}:
            return
        await asyncio.sleep(3)
    raise TimeoutError("Prime disk did not detach after pod termination.")


async def _cleanup(
    prime: PrimeBackend,
    ledger: PrimeResourceLedger,
    budget: PrimeBudgetGuard,
    run_id: str,
) -> None:
    pods_discovered = False
    try:
        pods = await asyncio.to_thread(prime.list_pods)
        pods_discovered = True
    except Exception as exc:
        ledger.cleanup_errors.append(f"pod discovery: {exc}")
        pods = []
    for pod in pods:
        pod_id = str(pod.get("id") or "")
        name = str(pod.get("name") or "")
        if pod_id not in ledger.pod_ids and run_id not in name:
            continue
        try:
            await asyncio.to_thread(prime.delete_pod, pod_id)
            ledger.close_pod(pod_id)
            budget.close(pod_id)
        except Exception as exc:
            ledger.cleanup_errors.append(f"pod {pod_id}: {exc}")
    if pods_discovered:
        active_pod_ids = {str(pod.get("id") or "") for pod in pods}
        for pod_id in list(ledger.pod_ids):
            if pod_id not in active_pod_ids:
                ledger.close_pod(pod_id)
                budget.close(pod_id)
    try:
        tunnels = await asyncio.to_thread(prime.list_tunnels)
    except Exception as exc:
        ledger.cleanup_errors.append(f"tunnel discovery: {exc}")
        tunnels = []
    for tunnel in tunnels:
        if run_id not in tunnel.name:
            continue
        try:
            await asyncio.to_thread(prime.delete_tunnel, tunnel.tunnel_id)
        except Exception as exc:
            ledger.cleanup_errors.append(f"tunnel {tunnel.tunnel_id}: {exc}")
    for disk_id in list(ledger.disk_ids):
        try:
            await _wait_disk_detached(prime, disk_id)
            await asyncio.to_thread(prime.delete_disk, disk_id)
            ledger.close_disk(disk_id)
            budget.close(disk_id)
        except Exception as exc:
            ledger.cleanup_errors.append(f"disk {disk_id}: {exc}")


def _emergency_cleanup() -> list[str]:
    """Best-effort synchronous cleanup if asyncio is interrupted repeatedly."""

    ModalBackend.terminate_all()
    if _ACTIVE_LIVE_RUN is None:
        return []
    prime, ledger, budget, run_id = _ACTIVE_LIVE_RUN
    errors: list[str] = []
    try:
        pods = prime.list_pods()
    except Exception as exc:
        errors.append(f"emergency pod discovery: {exc}")
        pods = []
    for pod in pods:
        pod_id = str(pod.get("id") or "")
        name = str(pod.get("name") or "")
        if pod_id not in ledger.pod_ids and run_id not in name:
            continue
        try:
            prime.delete_pod(pod_id)
            ledger.close_pod(pod_id)
            budget.close(pod_id)
        except Exception as exc:
            errors.append(f"emergency pod {pod_id}: {exc}")
    try:
        tunnels = prime.list_tunnels()
    except Exception as exc:
        errors.append(f"emergency tunnel discovery: {exc}")
        tunnels = []
    for tunnel in tunnels:
        if run_id not in tunnel.name:
            continue
        try:
            prime.delete_tunnel(tunnel.tunnel_id)
        except Exception as exc:
            errors.append(f"emergency tunnel {tunnel.tunnel_id}: {exc}")
    for disk_id in list(ledger.disk_ids):
        last_error = ""
        for _ in range(15):
            try:
                prime.delete_disk(disk_id)
                ledger.close_disk(disk_id)
                budget.close(disk_id)
                last_error = ""
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(2)
        if last_error:
            errors.append(f"emergency disk {disk_id}: {last_error}")
    ledger.cleanup_errors.extend(errors)
    return errors


async def run(args: argparse.Namespace) -> int:
    global _ACTIVE_LIVE_RUN

    prime = PrimeBackend()
    ok, detail = prime.preflight()
    if not ok:
        raise RuntimeError(detail)
    run_id = f"e2e-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    path = _report_path(args.report, run_id)
    report = PrimeLiveReport(
        run_id=run_id,
        commit=_commit_sha(),
        budget_cap_usd=args.budget_usd,
    )
    budget = PrimeBudgetGuard(cap_usd=args.budget_usd)
    ledger = PrimeResourceLedger()
    _ACTIVE_LIVE_RUN = (prime, ledger, budget, run_id)
    secrets: list[str] = []
    failed = False
    suite_error = ""
    selected_stages = set(args.stages or LIVE_STAGES)

    async def execute(stage: PrimeLiveStage, action: Callable[[], Any]) -> Any:
        nonlocal failed
        report.stages.append(stage)
        print(f"[{stage.name}] starting", flush=True)
        try:
            result = await action()
            print(f"[{stage.name}] passed", flush=True)
            return result
        except Exception as exc:
            stage.error = redact_live_value(
                f"{type(exc).__name__}: {exc}", tuple(secrets)
            )
            stage.success = False
            failed = True
            print(f"[{stage.name}] failed: {stage.error}", flush=True)
            traceback.print_exc()
            raise
        finally:
            report.estimated_spend_usd = round(budget.estimated_cost_usd(), 6)
            _write_report(path, report, tuple(secrets))

    try:
        existing = [pod for pod in prime.list_pods() if run_id in str(pod.get("name") or "")]
        if existing:
            raise RuntimeError(f"Existing resources conflict with run ID {run_id}.")

        if "portable_vllm_and_auth" in selected_stages:
            offer = _choose_offer(
                prime,
                BackendType.VLLM,
                gpu_count=1,
                required_vram_gb=1.5,
            )
            stage = PrimeLiveStage(
                name="portable_vllm_and_auth",
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                gpu_count=offer.gpu_count,
                hourly_rate_usd=offer.price_per_hour,
            )
            await execute(
                stage,
                lambda: _run_tui_stage(
                    prime,
                    budget,
                    ledger,
                    stage,
                    backend=BackendType.VLLM,
                    offer=offer,
                    run_id=run_id,
                    stage_slug="portable-vllm",
                    secrets=secrets,
                ),
            )

        if "portable_llamacpp_and_auth" in selected_stages:
            offer = _choose_offer(
                prime,
                BackendType.LLAMACPP,
                gpu_count=1,
                required_vram_gb=0.75,
            )
            stage = PrimeLiveStage(
                name="portable_llamacpp_and_auth",
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                gpu_count=offer.gpu_count,
                hourly_rate_usd=offer.price_per_hour,
            )
            await execute(
                stage,
                lambda: _run_tui_stage(
                    prime,
                    budget,
                    ledger,
                    stage,
                    backend=BackendType.LLAMACPP,
                    offer=offer,
                    run_id=run_id,
                    stage_slug="portable-llamacpp",
                    secrets=secrets,
                ),
            )

        if "persistent_disk_cache_reuse" in selected_stages:
            disk_offer, _ = _choose_disk_pair(prime)
            size_gb = max(30, disk_offer.minimum_size_gb or 0)
            if disk_offer.maximum_size_gb is not None and size_gb > disk_offer.maximum_size_gb:
                raise RuntimeError("Compatible Prime disk availability cannot provide 30 GB.")
            disk_stage = PrimeLiveStage(name="persistent_disk_cache_reuse")
            report.stages.append(disk_stage)
            print(f"[{disk_stage.name}] starting", flush=True)
            disk_started = time.monotonic()
            try:
                projected_rate = (disk_offer.price_per_gb_hour or 0.0) * size_gb
                budget.require_capacity(
                    hourly_rate_usd=projected_rate,
                    maximum_runtime_seconds=1800,
                    description="persistent disk",
                )
                disk = await asyncio.to_thread(
                    prime.create_disk,
                    disk_offer,
                    size_gb=size_gb,
                    name=f"llp-{run_id}",
                )
                disk_id = str(disk.get("id") or "")
                if not disk_id:
                    raise RuntimeError("Prime did not return a disk ID.")
                ledger.add_disk(disk_id)
                budget.register("disk", disk_id, float(disk.get("priceHr") or projected_rate))
                disk_stage.disk_id = disk_id
                await _wait_disk_ready(prime, disk_id)
                offer = _choose_offer(
                    prime,
                    BackendType.LLAMACPP,
                    gpu_count=1,
                    disk_id=disk_id,
                    required_vram_gb=0.75,
                )
                disk_stage.offer_id = offer.id
                disk_stage.gpu_type = offer.gpu_type
                disk_stage.gpu_count = offer.gpu_count
                disk_stage.hourly_rate_usd = offer.price_per_hour
                first_snapshot: dict[str, str] = {}

                def inspect_first(_: DeploymentConfig, pod: dict[str, Any]) -> dict[str, Any]:
                    snapshot = _cache_snapshot(prime, pod, run_id, True)
                    first_snapshot["value"] = snapshot
                    return {"first_cache_snapshot": snapshot}

                await _run_tui_stage(
                    prime,
                    budget,
                    ledger,
                    disk_stage,
                    backend=BackendType.LLAMACPP,
                    offer=offer,
                    run_id=run_id,
                    stage_slug="disk-first",
                    disk_id=disk_id,
                    timeout_seconds=900,
                    inspect=inspect_first,
                    secrets=secrets,
                )
                await _wait_disk_detached(prime, disk_id)

                def inspect_second(_: DeploymentConfig, pod: dict[str, Any]) -> dict[str, Any]:
                    snapshot = _cache_snapshot(prime, pod, run_id, False)
                    if snapshot != first_snapshot.get("value"):
                        raise AssertionError("Persistent llama.cpp cache changed across pods.")
                    return {"second_cache_snapshot": snapshot, "cache_reused": True}

                await _run_tui_stage(
                    prime,
                    budget,
                    ledger,
                    disk_stage,
                    backend=BackendType.LLAMACPP,
                    offer=offer,
                    run_id=run_id,
                    stage_slug="disk-second",
                    disk_id=disk_id,
                    timeout_seconds=900,
                    inspect=inspect_second,
                    secrets=secrets,
                )
                await _wait_disk_detached(prime, disk_id)
                await asyncio.to_thread(prime.delete_disk, disk_id)
                ledger.close_disk(disk_id)
                budget.close(disk_id)
                disk_stage.success = True
                print(f"[{disk_stage.name}] passed", flush=True)
            except Exception as exc:
                disk_stage.error = redact_live_value(
                    f"{type(exc).__name__}: {exc}", tuple(secrets)
                )
                disk_stage.success = False
                failed = True
                print(f"[{disk_stage.name}] failed: {disk_stage.error}", flush=True)
                traceback.print_exc()
                raise
            finally:
                disk_stage.duration_seconds = round(time.monotonic() - disk_started, 3)
                disk_stage.finished_at = utc_now_iso()
                report.estimated_spend_usd = round(budget.estimated_cost_usd(), 6)
                _write_report(path, report, tuple(secrets))

        if "multi_gpu_vllm" in selected_stages:
            offer = _choose_offer(
                prime,
                BackendType.VLLM,
                gpu_count=2,
                required_vram_gb=1.5,
            )
            stage = PrimeLiveStage(
                name="multi_gpu_vllm",
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                gpu_count=offer.gpu_count,
                hourly_rate_usd=offer.price_per_hour,
            )
            await execute(
                stage,
                lambda: _run_tui_stage(
                    prime,
                    budget,
                    ledger,
                    stage,
                    backend=BackendType.VLLM,
                    offer=offer,
                    run_id=run_id,
                    stage_slug="multi-gpu",
                    timeout_seconds=900,
                    inspect=lambda _config, pod: _multi_gpu_evidence(prime, pod),
                    secrets=secrets,
                ),
            )

        if "management_restart_and_tunnel_recovery" in selected_stages:
            offer = _choose_offer(
                prime,
                BackendType.VLLM,
                gpu_count=1,
                required_vram_gb=1.5,
            )
            stage = PrimeLiveStage(
                name="management_restart_and_tunnel_recovery",
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                gpu_count=offer.gpu_count,
                hourly_rate_usd=offer.price_per_hour,
            )
            await execute(
                stage,
                lambda: _run_tui_stage(
                    prime,
                    budget,
                    ledger,
                    stage,
                    backend=BackendType.VLLM,
                    offer=offer,
                    run_id=run_id,
                    stage_slug="management-recovery",
                    timeout_seconds=900,
                    inspect=lambda config, pod: _management_and_tunnel_recovery_evidence(
                        prime, config, pod
                    ),
                    inspect_stops_pod=True,
                    secrets=secrets,
                ),
            )

        if "failed_deployment_cleanup" in selected_stages:
            offer = _choose_offer(
                prime,
                BackendType.VLLM,
                gpu_count=1,
                required_vram_gb=1.5,
            )
            stage = PrimeLiveStage(
                name="failed_deployment_cleanup",
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                gpu_count=offer.gpu_count,
                hourly_rate_usd=offer.price_per_hour,
            )
            await execute(
                stage,
                lambda: _run_expected_failure_cleanup_stage(
                    prime,
                    budget,
                    ledger,
                    stage,
                    offer=offer,
                    run_id=run_id,
                ),
            )

        if "qwen38_27b_quick_deploy" in selected_stages:
            profile = get_quick_deploy_profile("qwen3-8-27b-q2xl-cheap-l4")
            if profile.required_vram_gb is None:
                raise RuntimeError("Qwen3.8 quick-deploy profile has no VRAM requirement.")
            offer = _choose_offer(
                prime,
                BackendType.LLAMACPP,
                gpu_count=profile.gpu_count,
                required_vram_gb=profile.required_vram_gb,
            )
            snapshot = replace(
                aggregate_compute_availability(prime_offers=(offer,)),
                providers=(ComputeProvider.PRIME,),
            )
            stage = PrimeLiveStage(
                name="qwen38_27b_quick_deploy",
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                gpu_count=offer.gpu_count,
                hourly_rate_usd=offer.price_per_hour,
            )
            await execute(
                stage,
                lambda: _run_tui_stage(
                    prime,
                    budget,
                    ledger,
                    stage,
                    backend=BackendType.LLAMACPP,
                    offer=offer,
                    run_id=run_id,
                    stage_slug="qwen38-27b",
                    timeout_seconds=1800,
                    inspect=lambda config, pod: _qwen38_quick_deploy_evidence(
                        prime,
                        config,
                        pod,
                        expected_offer_id=offer.id,
                    ),
                    secrets=secrets,
                    fast_snapshot=snapshot,
                    fast_provider=ComputeProvider.PRIME,
                    fast_profile_id=profile.id,
                ),
            )
    except Exception as exc:
        suite_error = redact_live_value(
            f"{type(exc).__name__}: {exc}", tuple(secrets)
        )
        print(f"[suite] failed: {suite_error}", flush=True)
        traceback.print_exc()
        failed = True
    finally:
        cleanup_cancelled = False
        cleanup_task = asyncio.create_task(_cleanup(prime, ledger, budget, run_id))
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            cleanup_cancelled = True
            ModalBackend.terminate_all()
            await cleanup_task
        report.finished_at = utc_now_iso()
        report.estimated_spend_usd = round(budget.estimated_cost_usd(), 6)
        report.cleanup = {
            "open_pod_ids": sorted(ledger.pod_ids),
            "open_disk_ids": sorted(ledger.disk_ids),
            "errors": [redact_live_value(error, tuple(secrets)) for error in ledger.cleanup_errors],
            "suite_error": suite_error,
        }
        if ledger.pod_ids or ledger.disk_ids or ledger.cleanup_errors:
            failed = True
        _write_report(path, report, tuple(secrets))
        print(f"Report: {path}", flush=True)
        print(f"Estimated Prime spend: ${report.estimated_spend_usd:.4f}", flush=True)
        _ACTIVE_LIVE_RUN = None
        if cleanup_cancelled:
            raise asyncio.CancelledError
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that the suite creates billable Prime resources.",
    )
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--report", help="JSON report path (defaults to /tmp).")
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        choices=LIVE_STAGES,
        help="Run one stage; repeat to run a subset. Defaults to all stages.",
    )
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("--confirm-live is required")
    if args.budget_usd <= 0:
        parser.error("--budget-usd must be greater than zero")
    return args


def main() -> int:
    try:
        return asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        errors = _emergency_cleanup()
        if errors:
            print(f"Interrupted; emergency cleanup errors: {errors}", file=sys.stderr)
        else:
            print("Interrupted; tracked Prime resources were cleaned up.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

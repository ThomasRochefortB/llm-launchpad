#!/usr/bin/env python3
"""Run an opt-in, TUI-driven Modal Fast Deploy live validation."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import traceback
from typing import Any
from urllib.parse import urlsplit

import requests
from textual.widgets import Button, Input, OptionList, Switch

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.modal_auth import get_modal_auth_status
from llm_launchpad.core.modal_gpu import fetch_modal_gpu_catalog
from llm_launchpad.core.naming import build_deployment_name
from llm_launchpad.core.quick_deploy import (
    get_quick_deploy_profile,
    list_quick_deploy_models,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    ComputeAvailabilitySnapshot,
    DeploymentConfig,
    StorageSnapshot,
)
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen
from llm_launchpad.tui.workers import OperationDone, StorageLoaded
from scripts.validate_prime_live import (
    LiveFastDeployScreen,
    _llamacpp_management_evidence,
)


PROFILE_ID = "qwen3-8-27b-q2xl-cheap-l4"
STOPPED_STATES = {"stopped", "stopping", "terminated", "archived"}


class CaptureMonitorScreen(MonitorScreen):
    """Monitor that preserves the final operation result."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.last_done: OperationDone | None = None

    def on_operation_done(self, message: OperationDone) -> None:
        self.last_done = message
        super().on_operation_done(message)


class ModalFastDeployApp(TuiApp):
    """Real Modal TUI workflow isolated from the user's OpenCode config."""

    CSS_PATH = Path(__file__).resolve().parents[1] / "llm_launchpad" / "tui" / "theme.tcss"

    def __init__(self, snapshot: ComputeAvailabilitySnapshot) -> None:
        super().__init__(mouse_enabled=False)
        self.snapshot = snapshot
        self.deployed_config: DeploymentConfig | None = None
        self.capture_monitor: CaptureMonitorScreen | None = None
        self.live_notifications: list[tuple[str, str]] = []

    def on_mount(self) -> None:
        self.push_screen(LiveFastDeployScreen(self.snapshot))

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        _ = force
        poster = getattr(receiver, "post_message", None)
        if callable(poster):
            poster(StorageLoaded(StorageSnapshot(llamacpp_models=[], vllm_models=[])))

    def begin_deploy(self, config: DeploymentConfig) -> None:
        if not config.app_name:
            config.app_name = build_deployment_name(
                config.provider,
                config.backend,
                config.instance_name,
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

    def begin_capture_stop(self, config: DeploymentConfig) -> None:
        monitor = CaptureMonitorScreen(title="Stop")
        self.capture_monitor = monitor
        self.push_screen(monitor)

        def start_stop() -> None:
            self.run_worker(
                lambda: self._run_stop(
                    config.backend,
                    config.app_name or "",
                    config.app_name or "",
                    monitor,
                    provider=ComputeProvider.MODAL,
                ),
                name="stop-worker",
                thread=True,
            )

        self.call_after_refresh(start_stop)

    def _sync_opencode(self, **_: object) -> None:
        """Keep the live TUI from changing the user's OpenCode config."""

    def notify(
        self,
        message: object,
        *,
        severity: str = "information",
        **kwargs: object,
    ) -> None:
        self.live_notifications.append((str(message), severity))
        super().notify(str(message), severity=severity, **kwargs)


async def _navigate_and_submit(
    pilot: Any,
    app: ModalFastDeployApp,
    *,
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(app.screen, FastDeployScreen):
        app.push_screen(LiveFastDeployScreen(app.snapshot))
    for _ in range(50):
        await pilot.pause()
        if isinstance(app.screen, FastDeployScreen):
            break
        await asyncio.sleep(0.1)
    else:
        stack = [type(candidate).__name__ for candidate in app.screen_stack]
        raise RuntimeError(
            f"Fast Deploy model picker did not mount; current={type(app.screen).__name__}, "
            f"stack={stack}, notifications={app.live_notifications[-3:]}"
        )

    screen = app.screen
    assert isinstance(screen, FastDeployScreen)
    model = next(
        (
            candidate
            for candidate in list_quick_deploy_models()
            if any(profile.id == PROFILE_ID for profile in candidate.profiles)
        ),
        None,
    )
    if model is None:
        raise RuntimeError("Qwen3.8 27B is missing from the Fast Deploy catalog.")
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
        raise RuntimeError("Qwen3.8 27B is missing from the rendered model picker.")
    option_list.highlighted = model_index
    await pilot.press("enter")

    for _ in range(50):
        await pilot.pause()
        if screen._phase == "infra" and screen._infra_rows:
            break
        await asyncio.sleep(0.1)
    else:
        raise RuntimeError("Modal infrastructure choices did not load.")
    selected = next(
        (
            row
            for row in screen._infra_rows.values()
            if row.plan.quote.provider == ComputeProvider.MODAL
            and row.profile.id == PROFILE_ID
        ),
        None,
    )
    if selected is None:
        raise RuntimeError("No live Modal placement fits the Qwen3.8 27B recipe.")
    option_list = screen.query_one("#fast-deploy-list", OptionList)
    option_list.highlighted = next(
        index
        for index in range(option_list.option_count)
        if option_list.get_option_at_index(index).id == selected.plan.quote.id
    )
    await pilot.press("enter")

    for _ in range(50):
        await pilot.pause()
        if isinstance(app.screen, QuickDeployScreen):
            break
        await asyncio.sleep(0.1)
    else:
        raise RuntimeError("Fast Deploy confirmation screen did not mount.")
    confirm = app.screen
    assert isinstance(confirm, QuickDeployScreen)
    confirm.query_one("#toggle-advanced-quick", Button).press()
    await pilot.pause()
    confirm.query_one("#quick-instance-name", Input).value = f"{run_id}-modal"
    confirm.query_one("#quick-warmup", Switch).value = True
    confirm.query_one("#quick-deploy-btn", Button).press()

    for _ in range(30):
        await pilot.pause()
        if app.deployed_config is not None:
            return {
                "model_id": model.id,
                "profile_id": selected.profile.id,
                "plan_id": selected.plan.quote.id,
                "gpu_type": selected.plan.quote.gpu_type,
                "gpu_count": selected.plan.quote.gpu_count,
                "price_per_hour_usd": selected.plan.quote.price_per_hour_usd,
            }
        await asyncio.sleep(0.1)
    detail = app.live_notifications[-1][0] if app.live_notifications else "no notification"
    raise RuntimeError(f"Fast Deploy confirmation did not submit: {detail}")


async def _wait_operation(
    app: ModalFastDeployApp,
    *,
    timeout_seconds: int,
    label: str,
) -> list[str]:
    started = time.monotonic()
    last_reported = 0
    while time.monotonic() - started < timeout_seconds:
        monitor = app.capture_monitor
        lines = (
            list(monitor.log_viewer.log_widget.lines)
            if monitor is not None and monitor.is_mounted
            else []
        )
        if len(lines) > last_reported:
            for line in lines[last_reported:]:
                if line.strip():
                    print(f"[{label}] {line}", flush=True)
            last_reported = len(lines)
        if monitor is not None and monitor.last_done is not None:
            if not monitor.last_done.success:
                raise RuntimeError(monitor.last_done.detail or f"Modal {label} failed.")
            return lines
        await asyncio.sleep(2)
    raise TimeoutError(f"Modal {label} exceeded {timeout_seconds} seconds.")


def _probe_endpoint(config: DeploymentConfig, endpoint: str) -> dict[str, Any]:
    if urlsplit(endpoint).scheme != "https":
        raise AssertionError("Modal Fast Deploy did not return an HTTPS endpoint.")
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    models = requests.get(f"{base}/v1/models", timeout=300)
    sentinel = f"MODAL-QWEN38-{int(time.time())}"
    completion = requests.post(
        f"{base}/v1/chat/completions",
        json={
            "model": config.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply with exactly this identifier: {sentinel}",
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=600,
    )
    if models.status_code != 200:
        raise AssertionError(f"Modal models endpoint returned HTTP {models.status_code}.")
    if completion.status_code != 200:
        raise AssertionError(
            f"Modal chat endpoint returned HTTP {completion.status_code}: "
            f"{completion.text[-500:]}"
        )
    if sentinel not in completion.text:
        raise AssertionError("Modal chat endpoint did not return the requested sentinel.")
    return {
        "endpoint_scheme": "https",
        "models_status": models.status_code,
        "chat_status": completion.status_code,
        "chat_sentinel": sentinel,
    }


def _matching_live_apps(app_name: str) -> list[dict[str, str]]:
    rows = ModalBackend.list_apps()
    if rows is None:
        raise RuntimeError("Could not list Modal apps during cleanup verification.")
    return [
        {"name": row.name, "state": row.state}
        for row in rows
        if (row.name or "") == app_name
        and (row.state or "").strip().casefold() not in STOPPED_STATES
    ]


async def _wait_no_live_apps(
    app_name: str,
    *,
    timeout_seconds: int = 180,
) -> list[dict[str, str]]:
    started = time.monotonic()
    remaining: list[dict[str, str]] = []
    while time.monotonic() - started < timeout_seconds:
        remaining = await asyncio.to_thread(_matching_live_apps, app_name)
        if not remaining:
            return []
        await asyncio.sleep(2)
    raise TimeoutError(f"Modal app remained live after stop: {remaining}")


async def run(report_path: Path) -> int:
    auth = get_modal_auth_status()
    if not auth.authenticated:
        raise RuntimeError("Modal authentication is required for live validation.")
    profile = get_quick_deploy_profile(PROFILE_ID)
    snapshot = replace(
        aggregate_compute_availability(
            modal_catalog=fetch_modal_gpu_catalog(force_refresh=True)
        ),
        providers=(ComputeProvider.MODAL,),
    )
    run_id = f"qwen38-{int(time.time())}"
    report: dict[str, Any] = {
        "provider": ComputeProvider.MODAL.value,
        "profile": auth.profile,
        "run_id": run_id,
        "started_at_epoch": time.time(),
        "success": False,
        "evidence": {},
        "cleanup": {},
    }
    app = ModalFastDeployApp(snapshot)
    config: DeploymentConfig | None = None
    stopped = False
    error: str | None = None
    try:
        async with app.run_test(size=(150, 48)) as pilot:
            report["evidence"]["fast_deploy_selection"] = await _navigate_and_submit(
                pilot,
                app,
                run_id=run_id,
            )
            config = app.deployed_config
            if config is None:
                raise RuntimeError("TUI did not preserve the Modal deployment config.")
            lines = await _wait_operation(
                app,
                timeout_seconds=1800,
                label="deploy",
            )
            entry = app._deploy_connection_cache.get(config.app_name or "", {})
            endpoint = str(entry.get("base_url") or "")
            if not endpoint:
                raise RuntimeError("TUI did not preserve the Modal endpoint URL.")
            if config.repo_id != profile.repo_id or config.quant != profile.quant:
                raise AssertionError("Fast Deploy changed the curated model or quantization.")
            if config.backend != BackendType.LLAMACPP:
                raise AssertionError("Fast Deploy did not select llama.cpp.")
            report["evidence"]["config"] = {
                key: value
                for key, value in asdict(config).items()
                if key not in {"endpoint_api_key", "provider_options"}
            }
            report["evidence"]["endpoint"] = await asyncio.to_thread(
                _probe_endpoint,
                config,
                endpoint,
            )
            report["evidence"]["management"] = await asyncio.to_thread(
                _llamacpp_management_evidence,
                config,
                provider=ComputeProvider.MODAL,
            )
            report["evidence"]["tui_milestones"] = lines[-100:]
            app.begin_capture_stop(config)
            await _wait_operation(app, timeout_seconds=300, label="stop")
            stopped = True
        remaining = await _wait_no_live_apps(config.app_name or "")
        report["cleanup"] = {"matching_live_apps": remaining}
        report["success"] = True
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["error"] = error
        traceback.print_exc()
        return 1
    finally:
        if config is not None and config.app_name and not stopped:
            try:
                events = app._orchestrator.stop_app(
                    config.backend,
                    app_name=config.app_name,
                    provider=ComputeProvider.MODAL,
                )
                list(events)
                stopped = True
            except Exception as cleanup_exc:
                report["cleanup"]["fallback_stop_error"] = str(cleanup_exc)
        if config is not None and config.app_name:
            try:
                report["cleanup"]["matching_live_apps"] = await _wait_no_live_apps(
                    config.app_name,
                )
            except Exception as cleanup_exc:
                report["cleanup"]["verification_error"] = str(cleanup_exc)
        report["finished_at_epoch"] = time.time()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Report: {report_path}", flush=True)
        if error:
            print(f"Failed: {error}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that this creates a billable Modal deployment.",
    )
    parser.add_argument(
        "--report",
        default=f"/tmp/llm-launchpad-modal-qwen38-{int(time.time())}.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_live:
        print("Refusing to create billable resources without --confirm-live.")
        return 2
    return asyncio.run(run(Path(args.report)))


if __name__ == "__main__":
    raise SystemExit(main())

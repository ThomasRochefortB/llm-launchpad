"""Main Textual App: screen routing, keybindings, worker orchestration.

TuiApp is the entry point for the interactive ``llm-launchpad`` TUI. It owns
the screen stack and bridges user actions to Core via threaded workers.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from typing import Optional

from textual.app import App
from textual.binding import Binding

from ..core.backend import ModalBackend
from ..core.benchmark import benchmark_config_from_endpoint, parse_concurrency_values
from ..core.config import SETTINGS_DIR
from ..core.hf_models import fetch_gguf_quant_metadata, list_llamacpp_candidates, list_vllm_candidates
from ..core.naming import (
    build_app_name,
    legacy_app_name,
)
from ..core.quick_deploy import QuickDeployProfile
from ..core.opencode import (
    build_openai_connection_payload,
    resolve_connection_for_app,
    resolve_connections_for_rows,
    sync_opencode_config,
    visible_launchpad_rows,
)
from ..core.orchestrator import Orchestrator
from ..protocol.enums import BackendType, OperationType
from ..protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from ..protocol.models import BenchmarkConfig
from ..protocol.models import EndpointInfo
from ..protocol.models import DeploymentConfig
from ..protocol.models import StoredModelInfo
from ..protocol.models import StorageSnapshot

from .screens.main_menu import MainMenuScreen
from .screens.deploy import BackendSelectScreen
from .screens.manage import ManageScreen
from .screens.monitor import MonitorScreen
from .screens.quick_deploy import QuickDeployScreen
from .screens.storage import StorageScreen
from .screens.settings import SettingsScreen
from .workers import (
    LlamaCppModelsFailed,
    LlamaCppModelsLoaded,
    LlamaCppQuantsFailed,
    LlamaCppQuantsLoaded,
    StorageFailed,
    StorageLoaded,
    VllmModelsFailed,
    VllmModelsLoaded,
    _dispatch_event,
)


def _deploy_connection_summary_lines(config: DeploymentConfig, server_url: str) -> list[str]:
    """Build a compact post-deploy connection summary for OpenAI-compatible clients."""
    payload = _deploy_connection_summary_payload(config, server_url)
    base_url = str(payload["base_url"])
    model_id = str(payload["model_id"])
    display_name = str(payload["display_name"])

    return [
        "=== OpenAI-compatible ===",
        f"Base URL: {base_url}",
        f"Model ID: {model_id}",
        f"Display name: {display_name}",
        "API key: (leave blank; no auth by default)",
        "=========================",
    ]


def _deploy_connection_summary_payload(
    config: DeploymentConfig,
    server_url: str,
) -> dict[str, str]:
    """Structured OpenAI-compatible connection summary for a completed deploy."""
    return build_openai_connection_payload(config, server_url)


def _osc_52_sequence(text: str) -> str:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"\x1b]52;c;{payload}\a"


def _tmux_passthrough_sequence(text: str) -> str:
    osc = _osc_52_sequence(text).replace("\x1b", "\x1b\x1b")
    return f"\x1bPtmux;{osc}\x1b\\"


def _screen_passthrough_sequence(text: str) -> str:
    return f"\x1bP{_osc_52_sequence(text)}\x1b\\"


class TuiApp(App):
    """llm-launchpad interactive terminal UI."""

    TITLE = "llm-launchpad"
    SUB_TITLE = "Modal LLM backends"

    CSS_PATH = "theme.tcss"

    BINDINGS = [
        Binding("ctrl+c", "request_quit", show=False, priority=True, system=True),
        Binding("ctrl+t", "toggle_mouse_mode", "Mouse", show=True),
    ]
    _CTRL_C_CONFIRM_WINDOW_SECONDS = 10.0
    _STORAGE_CACHE_TTL_SECONDS = 20.0

    def __init__(self, *, mouse_enabled: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.mouse_enabled = mouse_enabled
        self._ctrl_c_last_requested_at = 0.0
        self._orchestrator = Orchestrator()
        self._username: str = ""
        self._version: str = ""
        self._storage_snapshot_cache: StorageSnapshot | None = None
        self._storage_snapshot_cached_at_epoch: float = 0.0
        self._storage_refresh_inflight = False
        self._storage_refresh_lock = threading.Lock()
        self._storage_cache_path = SETTINGS_DIR / "storage_snapshot.json"
        self._deploy_connection_cache_path = SETTINGS_DIR / "deployment_connection_summaries.json"
        self._deploy_connection_cache: dict[str, dict[str, object]] = {}
        self._load_persisted_storage_cache()
        self._load_persisted_deploy_connection_cache()
        try:
            from importlib.metadata import version

            self._version = version("llm-launchpad")
        except Exception:
            pass

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text via OSC 52, including tmux/screen passthrough variants."""
        super().copy_to_clipboard(text)
        driver = getattr(self, "_driver", None)
        if driver is not None:
            try:
                tmux_session = os.environ.get("TMUX")
                term = os.environ.get("TERM", "")
                if tmux_session:
                    driver.write(_tmux_passthrough_sequence(text))
                elif term.startswith("screen"):
                    driver.write(_screen_passthrough_sequence(text))
            except Exception:
                pass
        if sys.platform == "darwin":
            try:
                subprocess.run(
                    ["pbcopy"],
                    input=text.encode(),
                    check=True,
                    timeout=2,
                )
            except Exception:
                pass

    def _set_mouse_mode(self, enabled: bool) -> None:
        """Enable or disable driver mouse reporting at runtime."""
        driver = getattr(self, "_driver", None)
        if driver is not None:
            currently_enabled = bool(getattr(driver, "_mouse", self.mouse_enabled))
            if enabled and not currently_enabled:
                setattr(driver, "_mouse", True)
                enable = getattr(driver, "_enable_mouse_support", None)
                if callable(enable):
                    enable()
            elif not enabled and currently_enabled:
                disable = getattr(driver, "_disable_mouse_support", None)
                if callable(disable):
                    disable()
                setattr(driver, "_mouse", False)

        self.mouse_enabled = enabled
        try:
            screen = self.screen
        except Exception:
            return

        refresher = getattr(screen, "refresh_copy_help", None)
        if callable(refresher):
            refresher()
        screen.refresh()

    async def action_quit(self) -> None:
        """Terminate tracked subprocesses and workers before exiting.

        Without this, worker threads blocked on subprocess I/O prevent
        Python from shutting down cleanly (the atexit thread-join hangs).
        """
        ModalBackend.terminate_all()
        self.workers.cancel_all()
        import asyncio
        await asyncio.sleep(0.3)
        self.exit()

    async def action_request_quit(self) -> None:
        """Require a second Ctrl+C press before quitting the TUI."""
        now = time.monotonic()
        if (
            self._ctrl_c_last_requested_at > 0.0
            and now - self._ctrl_c_last_requested_at <= self._CTRL_C_CONFIRM_WINDOW_SECONDS
        ):
            self._ctrl_c_last_requested_at = 0.0
            await self.action_quit()
            return

        self._ctrl_c_last_requested_at = now
        self.notify(
            "Hit CTRL+C again to exit",
            title="Exit llm-launchpad?",
            severity="warning",
            timeout=self._CTRL_C_CONFIRM_WINDOW_SECONDS,
        )

    def action_toggle_mouse_mode(self) -> None:
        """Toggle between app mouse mode and terminal-native selection mode."""
        enabled = not self.mouse_enabled
        self._set_mouse_mode(enabled)
        if enabled:
            self.notify(
                "Mouse enabled. Clicks and drags go to llm-launchpad.",
                title="Mouse mode on",
                timeout=3,
            )
        else:
            self.notify(
                "Mouse disabled. Use your terminal selection and copy shortcuts.",
                title="Terminal copy mode",
                timeout=3,
            )

    def on_mount(self) -> None:
        """Launch the TUI if the Modal CLI is installed.

        Authentication is surfaced inside the main menu instead of gating startup.
        """
        if not ModalBackend.is_cli_available():
            self.notify(
                "Modal CLI not found. Reinstall llm-launchpad, then run: modal setup",
                severity="error",
                timeout=10,
            )
            self.exit(return_code=1)
            return
        self._username = ModalBackend.get_username() or ""
        self.push_screen(MainMenuScreen(username=self._username, version=self._version))
        if not self.mouse_enabled:
            self.notify(
                "Terminal copy mode active. Press Ctrl+T to enable mouse.",
                timeout=4,
            )

    # ------------------------------------------------------------------
    # Actions called by screens
    # ------------------------------------------------------------------

    def action_push_deploy(self) -> None:
        self.push_screen(BackendSelectScreen())

    def action_push_manage(self) -> None:
        self.push_screen(ManageScreen())

    def action_push_settings(self) -> None:
        self.push_screen(SettingsScreen())

    def action_push_storage(self, backend: BackendType | None = None) -> None:
        self.push_screen(StorageScreen(initial_backend=backend))

    def push_quick_deploy(self, profile: str | QuickDeployProfile) -> None:
        self.push_screen(QuickDeployScreen(profile_id=profile))

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def begin_deploy(self, config: DeploymentConfig) -> None:
        """Start a deploy operation via a threaded worker."""
        if not config.app_name:
            config.app_name = build_app_name(config.backend, config.instance_name)
        monitor = MonitorScreen(
            title="Deploy",
            deploy_backend=config.backend,
            summarize_backend_logs=True,
            show_debug_logs=config.show_debug_logs,
        )
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_deploy(config, monitor),
            name="deploy-worker",
            thread=True,
        )

    def _run_deploy(self, config: DeploymentConfig, monitor: MonitorScreen):  # type: ignore[return]
        """Generator consumed by run_worker in a thread."""
        deployed_web_url: Optional[str] = None
        deployed_web_url_priority = -1
        deploy_succeeded = False
        will_run_warmup = bool(config.do_warmup and config.do_deploy)

        def _emit_connection_summary(url: str) -> None:
            self._cache_deploy_connection_summary(config, url)
            for line in _deploy_connection_summary_lines(config, url):
                _dispatch_event(monitor, LogEvent(line=line))

        for event in self._orchestrator.deploy(config):
            if isinstance(event, LogEvent):
                line = event.line or ""
                maybe_url = ModalBackend.extract_modal_web_url(event.line)
                if maybe_url:
                    if "Created web function" in line:
                        # Prefer the Modal-emitted web function URL over backend-printed
                        # guidance URLs; within those, prefer the non-dev URL.
                        priority = 2 if not maybe_url.endswith("-dev.modal.run") else 1
                    else:
                        priority = 0
                    if priority >= deployed_web_url_priority:
                        deployed_web_url = maybe_url
                        deployed_web_url_priority = priority
            elif isinstance(event, OperationCompleteEvent):
                deploy_succeeded = event.success
                # When warmup immediately follows a successful deploy in the same
                # monitor session, suppress the intermediate completion footer
                # ("Operation complete... Press esc") to keep the summary cleaner.
                if will_run_warmup and event.success and event.operation == OperationType.DEPLOY:
                    continue
                if event.success and event.operation == OperationType.DEPLOY and config.do_deploy:
                    target_app_name = config.app_name or legacy_app_name(config.backend)
                    url = deployed_web_url or ModalBackend.default_server_url(
                        self._username,
                        app_name=target_app_name,
                        function_slug=config.function_slug,
                    )
                    _emit_connection_summary(url)
                    self._sync_opencode(
                        target_app_name=target_app_name,
                        target_url=url,
                        target_config=config,
                        current_rows=self._load_visible_launchpad_rows(),
                        monitor=monitor,
                        emit_skipped=True,
                    )
            _dispatch_event(monitor, event)

        # Warmup if requested and deploy was successful
        if config.do_warmup and config.do_deploy and deploy_succeeded:
            target_app_name = config.app_name or legacy_app_name(config.backend)
            url = deployed_web_url or ModalBackend.default_server_url(
                self._username,
                app_name=target_app_name,
                function_slug=config.function_slug,
            )
            for event in self._orchestrator.warmup(
                backend=config.backend,
                server_url=url,
                timeout=1800,
                tail_logs=True,
                app_name=target_app_name,
                served_model_name=config.served_model_name,
            ):
                if (
                    isinstance(event, OperationCompleteEvent)
                    and event.success
                    and event.operation == OperationType.WARMUP
                ):
                    completed_url = url
                    if isinstance(event.data, dict):
                        maybe_url = event.data.get("url")
                        if isinstance(maybe_url, str) and maybe_url.strip():
                            completed_url = maybe_url.strip()
                    _emit_connection_summary(completed_url)
                    self._sync_opencode(
                        target_app_name=target_app_name,
                        target_url=completed_url,
                        target_config=config,
                        current_rows=self._load_visible_launchpad_rows(),
                        monitor=monitor,
                        emit_skipped=True,
                    )
                _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: list
    # ------------------------------------------------------------------

    def begin_list(self) -> None:
        monitor = MonitorScreen(title="List Deployments")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_list(monitor),
            name="list-worker",
            thread=True,
        )

    def _run_list(self, monitor: MonitorScreen):  # type: ignore[return]
        for event in self._orchestrator.list_deployments():
            if (
                isinstance(event, OperationCompleteEvent)
                and event.success
                and isinstance(event.data, list)
            ):
                visible_rows = [row for row in event.data if isinstance(row, EndpointInfo)]
                self._sync_opencode(
                    current_rows=visible_rows,
                    monitor=monitor,
                    emit_skipped=True,
                )
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: status
    # ------------------------------------------------------------------

    def begin_status(
        self,
        backend: BackendType,
        server_url: Optional[str] = None,
        timeout: int = 60,
        app_name: Optional[str] = None,
        served_model_name: Optional[str] = None,
    ) -> None:
        target_app_name = app_name or legacy_app_name(backend)
        url = server_url or ModalBackend.default_server_url(self._username, app_name=target_app_name)
        monitor = MonitorScreen(title="Status Check")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_status(
                backend,
                url,
                timeout,
                target_app_name,
                served_model_name,
                monitor,
            ),
            name="status-worker",
            thread=True,
        )

    def _run_status(
        self,
        backend: BackendType,
        url: str,
        timeout: int,
        app_name: str,
        served_model_name: Optional[str],
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        _dispatch_event(monitor, LogEvent(line=f"Target app: {app_name}"))
        for event in self._orchestrator.check_status(
            backend,
            url,
            timeout,
            served_model_name=served_model_name,
        ):
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: benchmark
    # ------------------------------------------------------------------

    def begin_benchmark(
        self,
        row: EndpointInfo,
        *,
        concurrency: str = "1,2,4,8,16",
        request_count: int | None = None,
        input_tokens: int = 550,
        output_tokens: int = 256,
        tokenizer: str = "gpt2",
        output_dir: str | None = None,
    ) -> None:
        if row.backend is None:
            self.notify("Choose a Launchpad-managed app to benchmark.", severity="error", timeout=6)
            return
        try:
            concurrency_values = parse_concurrency_values(concurrency)
        except ValueError as exc:
            self.notify(str(exc), title="Invalid concurrency", severity="error", timeout=6)
            return
        config = benchmark_config_from_endpoint(
            row,
            backend=row.backend,
            username=self._username,
            app_name=row.name,
            instance_name=row.instance_name,
            concurrency=concurrency_values,
            request_count=request_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )
        monitor = MonitorScreen(title="Benchmark")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_benchmark(config, monitor),
            name="benchmark-worker",
            thread=True,
        )

    def _run_benchmark(
        self,
        config: BenchmarkConfig,
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        for event in self._orchestrator.benchmark(config):
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: logs
    # ------------------------------------------------------------------

    def begin_logs(
        self,
        backend: BackendType,
        follow: bool = True,
        app_name: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> None:
        monitor = MonitorScreen(title="Logs")
        self.push_screen(monitor)
        target_app_name = app_name or legacy_app_name(backend)
        target_ref = (app_id or target_app_name).strip()
        self.run_worker(
            lambda: self._run_logs(backend, follow, target_ref, target_app_name, monitor),
            name="logs-worker",
            thread=True,
        )

    def _run_logs(
        self,
        backend: BackendType,
        follow: bool,
        app_ref: str,
        app_name: str,
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        target_label = app_name if app_ref == app_name else f"{app_name} ({app_ref})"
        _dispatch_event(monitor, LogEvent(line=f"Target app: {target_label}"))
        for event in self._orchestrator.tail_logs(backend, follow, app_name=app_name, app_id=app_ref):
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: stop
    # ------------------------------------------------------------------

    def begin_stop(
        self,
        backend: BackendType,
        app_name: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> None:
        monitor = MonitorScreen(title="Stop")
        self.push_screen(monitor)
        target_app_name = app_name or legacy_app_name(backend)
        target_ref = (app_id or target_app_name).strip()
        self.run_worker(
            lambda: self._run_stop(backend, target_ref, target_app_name, monitor),
            name="stop-worker",
            thread=True,
        )

    def _run_stop(
        self,
        backend: BackendType,
        app_ref: str,
        app_name: str,
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        target_label = app_name if app_ref == app_name else f"{app_name} ({app_ref})"
        _dispatch_event(monitor, LogEvent(line=f"Target app: {target_label}"))
        for event in self._orchestrator.stop_app(backend, app_name=app_name, app_id=app_ref):
            if (
                isinstance(event, OperationCompleteEvent)
                and event.success
                and event.operation == OperationType.STOP
            ):
                self._sync_opencode(
                    current_rows=self._load_visible_launchpad_rows(),
                    remove_app_names=[app_name],
                    monitor=monitor,
                    emit_skipped=True,
                )
            _dispatch_event(monitor, event)

    def list_instances(self, backend: BackendType | None = None) -> list[EndpointInfo]:
        rows = ModalBackend.list_apps()
        if rows is None:
            rows = []
        else:
            self._merge_deploy_connection_cache(rows)
            self._sync_opencode(current_rows=visible_launchpad_rows(rows))
        if backend is None:
            return rows
        return [row for row in rows if row.backend == backend]

    def _load_visible_launchpad_rows(self) -> list[EndpointInfo] | None:
        rows = ModalBackend.list_apps()
        if rows is None:
            return None
        self._merge_deploy_connection_cache(rows)
        return visible_launchpad_rows(rows)

    def _sync_opencode(
        self,
        *,
        target_app_name: str | None = None,
        target_url: str | None = None,
        target_config: DeploymentConfig | None = None,
        current_rows: list[EndpointInfo] | None = None,
        remove_app_names: list[str] | None = None,
        monitor: MonitorScreen | None = None,
        emit_skipped: bool = False,
    ) -> None:
        target = None
        targets = None
        if target_app_name:
            target = resolve_connection_for_app(
                target_app_name,
                rows=current_rows,
                username=self._username,
                fallback_config=target_config,
                fallback_server_url=target_url,
            )
        elif current_rows is not None:
            targets = resolve_connections_for_rows(current_rows, username=self._username)
        try:
            result = sync_opencode_config(
                target=target,
                targets=targets,
                current_rows=current_rows,
                remove_app_names=remove_app_names,
            )
        except Exception as exc:
            if monitor is not None:
                _dispatch_event(monitor, LogEvent(line=f"OpenCode sync failed: {exc}"))
            return

        if monitor is None:
            return

        for line in result.messages:
            if result.detected or emit_skipped:
                _dispatch_event(monitor, LogEvent(line=line))

    # ------------------------------------------------------------------
    # Storage: list
    # ------------------------------------------------------------------

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        poster = getattr(receiver, "post_message", None)
        cache_age = time.time() - self._storage_snapshot_cached_at_epoch
        if not force and self._storage_snapshot_cache is not None and poster is not None:
            # Fast path: show the latest known snapshot immediately.
            poster(StorageLoaded(snapshot=self._storage_snapshot_cache))
            if cache_age <= self._STORAGE_CACHE_TTL_SECONDS:
                return

        with self._storage_refresh_lock:
            if self._storage_refresh_inflight:
                return
            self._storage_refresh_inflight = True

        self.run_worker(
            lambda: self._run_storage_refresh(receiver),
            name="storage-refresh-worker",
            thread=True,
        )

    def _run_storage_refresh(self, receiver: object) -> None:
        poster = getattr(receiver, "post_message", None)
        try:
            if poster is None:
                return
            result_snapshot: StorageSnapshot | None = None
            for event in self._orchestrator.list_storage():
                if isinstance(event, OperationCompleteEvent):
                    if event.success and isinstance(event.data, StorageSnapshot):
                        result_snapshot = event.data
                    elif not event.success:
                        poster(StorageFailed(error=event.detail or "Storage listing failed."))
                        return
                elif isinstance(event, ErrorEvent):
                    poster(StorageFailed(error=event.message))
                    return
            if result_snapshot is not None:
                self._cache_storage_snapshot(result_snapshot)
                poster(StorageLoaded(snapshot=result_snapshot))
            else:
                poster(StorageFailed(error="No storage data returned by backend."))
        finally:
            with self._storage_refresh_lock:
                self._storage_refresh_inflight = False

    # ------------------------------------------------------------------
    # Storage: predownload
    # ------------------------------------------------------------------

    def begin_storage_predownload(
        self,
        backend: BackendType,
        model_id: str,
        quant: str | None = None,
        revision: str | None = None,
    ) -> None:
        monitor = MonitorScreen(title="Pre-download")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_storage_predownload(
                backend=backend,
                model_id=model_id,
                quant=quant,
                revision=revision,
                monitor=monitor,
            ),
            name="storage-predownload-worker",
            thread=True,
        )

    def _run_storage_predownload(
        self,
        backend: BackendType,
        model_id: str,
        quant: str | None,
        revision: str | None,
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        for event in self._orchestrator.predownload_model(
            backend=backend,
            model_id=model_id,
            quant=quant,
            revision=revision,
        ):
            if isinstance(event, OperationCompleteEvent) and event.success:
                self._invalidate_storage_cache()
            _dispatch_event(monitor, event)

    def begin_storage_delete(self, model: StoredModelInfo) -> None:
        monitor = MonitorScreen(title="Delete model")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_storage_delete(model=model, monitor=monitor),
            name="storage-delete-worker",
            thread=True,
        )

    def _run_storage_delete(
        self,
        model: StoredModelInfo,
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        for event in self._orchestrator.delete_stored_model(model):
            if isinstance(event, OperationCompleteEvent) and event.success:
                self._invalidate_storage_cache()
            _dispatch_event(monitor, event)

    def _cache_storage_snapshot(self, snapshot: StorageSnapshot) -> None:
        with self._storage_refresh_lock:
            self._storage_snapshot_cache = snapshot
            self._storage_snapshot_cached_at_epoch = time.time()
        self._persist_storage_snapshot(snapshot)

    def _invalidate_storage_cache(self) -> None:
        with self._storage_refresh_lock:
            self._storage_snapshot_cache = None
            self._storage_snapshot_cached_at_epoch = 0.0
        try:
            self._storage_cache_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _persist_storage_snapshot(self, snapshot: StorageSnapshot) -> None:
        payload = {
            "cached_at_epoch": time.time(),
            "snapshot": self._snapshot_to_dict(snapshot),
        }
        try:
            self._storage_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_cache_path.write_text(json.dumps(payload))
        except Exception:
            pass

    def _cache_deploy_connection_summary(self, config: DeploymentConfig, server_url: str) -> None:
        app_name = (config.app_name or "").strip()
        if not app_name:
            return
        payload = _deploy_connection_summary_payload(config, server_url)
        self._deploy_connection_cache[app_name] = {
            "backend": config.backend.value,
            "base_url": payload["base_url"],
            "model_id": payload["model_id"],
            "display_name": payload["display_name"],
            "cached_at_epoch": time.time(),
        }
        self._persist_deploy_connection_cache()

    def _persist_deploy_connection_cache(self) -> None:
        payload = {"entries": self._deploy_connection_cache}
        try:
            self._deploy_connection_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._deploy_connection_cache_path.write_text(json.dumps(payload))
        except Exception:
            pass

    def _load_persisted_deploy_connection_cache(self) -> None:
        try:
            if not self._deploy_connection_cache_path.exists():
                return
            payload = json.loads(self._deploy_connection_cache_path.read_text())
            entries = payload.get("entries")
            if not isinstance(entries, dict):
                return
            normalized: dict[str, dict[str, object]] = {}
            for key, value in entries.items():
                app_name = str(key or "").strip()
                if not app_name or not isinstance(value, dict):
                    continue
                normalized[app_name] = value
            self._deploy_connection_cache = normalized
        except Exception:
            self._deploy_connection_cache = {}

    def _merge_deploy_connection_cache(self, rows: list[EndpointInfo]) -> None:
        if not self._deploy_connection_cache:
            return
        for row in rows:
            cached = self._deploy_connection_cache.get((row.name or "").strip())
            if not isinstance(cached, dict):
                continue
            if not (row.web_url or "").strip():
                base_url = str(cached.get("base_url", "") or "").strip()
                if base_url:
                    row.web_url = base_url
            if not (row.served_model_name or "").strip():
                model_id = str(cached.get("model_id", "") or "").strip()
                if model_id:
                    row.served_model_name = model_id
            if not (row.display_name or "").strip():
                display_name = str(cached.get("display_name", "") or "").strip()
                if display_name:
                    row.display_name = display_name

    def _load_persisted_storage_cache(self) -> None:
        try:
            if not self._storage_cache_path.exists():
                return
            payload = json.loads(self._storage_cache_path.read_text())
            snapshot_payload = payload.get("snapshot")
            if not isinstance(snapshot_payload, dict):
                return
            snapshot = self._snapshot_from_dict(snapshot_payload)
            if snapshot is None:
                return
            self._storage_snapshot_cache = snapshot
            self._storage_snapshot_cached_at_epoch = float(payload.get("cached_at_epoch", 0.0) or 0.0)
        except Exception:
            self._storage_snapshot_cache = None
            self._storage_snapshot_cached_at_epoch = 0.0

    @staticmethod
    def _model_to_dict(model: StoredModelInfo) -> dict[str, object]:
        return {
            "backend": model.backend.value,
            "model_id": model.model_id,
            "revision": model.revision,
            "quant": model.quant,
            "size_bytes": model.size_bytes,
            "file_count": model.file_count,
            "source_volume": model.source_volume,
            "paths": model.paths or [],
            "incomplete": model.incomplete,
        }

    @staticmethod
    def _model_from_dict(payload: dict[str, object]) -> StoredModelInfo | None:
        backend_raw = str(payload.get("backend", "")).strip().lower()
        if backend_raw not in {"llamacpp", "vllm"}:
            return None
        paths_value = payload.get("paths")
        if isinstance(paths_value, str):
            paths = [paths_value] if paths_value.strip() else []
        elif isinstance(paths_value, (list, tuple)):
            paths = [str(path) for path in paths_value if str(path).strip()]
        else:
            paths = []
        return StoredModelInfo(
            backend=BackendType(backend_raw),
            model_id=str(payload.get("model_id", "")).strip(),
            revision=str(payload.get("revision")) if payload.get("revision") not in {None, ""} else None,
            quant=str(payload.get("quant")) if payload.get("quant") not in {None, ""} else None,
            size_bytes=int(payload.get("size_bytes", 0) or 0),
            file_count=int(payload.get("file_count", 0) or 0),
            source_volume=str(payload.get("source_volume", "") or ""),
            paths=paths,
            incomplete=bool(payload.get("incomplete", False)),
        )

    @classmethod
    def _snapshot_to_dict(cls, snapshot: StorageSnapshot) -> dict[str, object]:
        return {
            "llamacpp_models": [cls._model_to_dict(model) for model in snapshot.llamacpp_models],
            "vllm_models": [cls._model_to_dict(model) for model in snapshot.vllm_models],
        }

    @classmethod
    def _snapshot_from_dict(cls, payload: dict[str, object]) -> StorageSnapshot | None:
        llamacpp_payload = payload.get("llamacpp_models")
        vllm_payload = payload.get("vllm_models")
        if not isinstance(llamacpp_payload, list) or not isinstance(vllm_payload, list):
            return None
        llamacpp_models: list[StoredModelInfo] = []
        for row in llamacpp_payload:
            if not isinstance(row, dict):
                continue
            model = cls._model_from_dict(row)
            if model is not None:
                llamacpp_models.append(model)
        vllm_models: list[StoredModelInfo] = []
        for row in vllm_payload:
            if not isinstance(row, dict):
                continue
            model = cls._model_from_dict(row)
            if model is not None:
                vllm_models.append(model)
        return StorageSnapshot(llamacpp_models=llamacpp_models, vllm_models=vllm_models)

    # ------------------------------------------------------------------
    # vLLM model discovery
    # ------------------------------------------------------------------

    def begin_fetch_vllm_models(self, mode: str, receiver: object) -> None:
        """Load ranked HF model candidates for vLLM without blocking the UI."""
        self.run_worker(
            lambda: self._run_fetch_vllm_models(mode, receiver),
            name=f"fetch-vllm-models-{mode}",
            thread=True,
        )

    def _run_fetch_vllm_models(self, mode: str, receiver: object) -> None:
        poster = getattr(receiver, "post_message", None)
        if poster is None:
            return
        try:
            models = list_vllm_candidates(mode=mode if mode in {"downloads", "trending"} else "downloads")
        except Exception as exc:
            poster(VllmModelsFailed(mode=mode, error=str(exc)))
            return
        poster(VllmModelsLoaded(mode=mode, models=models))

    # ------------------------------------------------------------------
    # llama.cpp model discovery
    # ------------------------------------------------------------------

    def begin_fetch_llamacpp_models(self, mode: str, receiver: object) -> None:
        """Load ranked HF model candidates for llama.cpp without blocking UI."""
        self.run_worker(
            lambda: self._run_fetch_llamacpp_models(mode, receiver),
            name=f"fetch-llamacpp-models-{mode}",
            thread=True,
        )

    def _run_fetch_llamacpp_models(self, mode: str, receiver: object) -> None:
        poster = getattr(receiver, "post_message", None)
        if poster is None:
            return
        try:
            models = list_llamacpp_candidates(mode=mode if mode in {"downloads", "trending"} else "downloads")
        except Exception as exc:
            poster(LlamaCppModelsFailed(mode=mode, error=str(exc)))
            return
        poster(LlamaCppModelsLoaded(mode=mode, models=models))

    def begin_fetch_llamacpp_quants(self, repo_id: str, revision: str | None, receiver: object) -> None:
        """Load GGUF quant variants for a llama.cpp model repo."""
        self.run_worker(
            lambda: self._run_fetch_llamacpp_quants(repo_id, revision, receiver),
            name=f"fetch-llamacpp-quants-{repo_id}",
            thread=True,
        )

    def _run_fetch_llamacpp_quants(self, repo_id: str, revision: str | None, receiver: object) -> None:
        poster = getattr(receiver, "post_message", None)
        if poster is None:
            return
        try:
            metadata = fetch_gguf_quant_metadata(repo_id=repo_id, revision=revision)
        except Exception as exc:
            poster(LlamaCppQuantsFailed(repo_id=repo_id, revision=revision, error=str(exc)))
            return
        poster(
            LlamaCppQuantsLoaded(
                repo_id=repo_id,
                revision=revision,
                quantizations=list(metadata.quantizations),
                vram_gb_by_quant=dict(metadata.vram_gb_by_quant),
            )
        )

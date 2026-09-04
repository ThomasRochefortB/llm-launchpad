"""Main Textual App: screen routing, keybindings, worker orchestration.

TuiApp is the entry point for the interactive ``llm-launchpad`` TUI. It owns
the screen stack and bridges user actions to Core via threaded workers.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding
from textual.filter import Monochrome
from textual.widgets import Input, TextArea

from ..core.backend import ModalBackend
from ..core.benchmark import benchmark_config_from_endpoint, parse_concurrency_values
from ..core.config import ConfigStore, SETTINGS_DIR
from ..core.deploy_journal import (
    InFlightDeployment,
    clear_in_flight,
    load_in_flight,
    record_in_flight,
)
from ..core.connection_store import (
    load_connection_entries,
    merge_connections,
    save_connection,
)
from ..core.diagnostics import log_exception
from ..core.hf_models import fetch_gguf_quant_metadata, list_llamacpp_candidates, list_vllm_candidates
from ..core.naming import (
    build_deployment_name,
    legacy_app_name,
)
from ..core.prime_auth import get_prime_auth_status
from ..core.prime_backend import PrimeBackend
from ..core.provider_options import prime_provider_options
from ..core.quick_deploy import QuickDeployProfile
from ..core.reasoning_profiles import discover_reasoning_capabilities
from ..core.runtime_support import evaluate_llamacpp_architecture
from ..core.opencode import (
    build_openai_connection_payload,
    resolve_connection_for_app,
    resolve_connections_for_rows,
    sync_opencode_config,
    visible_launchpad_rows,
)
from ..core.orchestrator import Orchestrator
from ..protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from ..protocol.events import (
    EndpointAvailableEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import BenchmarkConfig
from ..protocol.models import EndpointInfo
from ..protocol.models import DeploymentConfig
from ..protocol.models import InferencePlan
from ..protocol.models import StoredModelInfo
from ..protocol.models import StorageSnapshot

from .screens.main_menu import MainMenuScreen
from .screens.deploy import BackendSelectScreen
from .screens.fast_deploy import FastDeployScreen
from .screens.manage import ManageScreen
from .screens.monitor import MonitorScreen
from .screens.quick_deploy import QuickDeployScreen
from .screens.setup import SetupRequiredScreen
from .screens.storage import StorageScreen
from .screens.settings import SettingsScreen
from .clipboard import read_system_clipboard, write_system_clipboard
from .mouse import default_tui_mouse_enabled
from .visual import (
    LAUNCHPAD_THEMES,
    normalize_tui_density,
    normalize_tui_theme,
)
from .workers import (
    ConnectionSummaryReady,
    EndpointsFailed,
    EndpointsLoaded,
    LlamaCppModelsFailed,
    LlamaCppModelsLoaded,
    LlamaCppQuantsFailed,
    LlamaCppQuantsLoaded,
    ModalUsernameLoaded,
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
        (
            "API key: generated and stored locally"
            if config.endpoint_api_key
            else "API key: (leave blank; no auth by default)"
        ),
        "=========================",
    ]


def _deploy_connection_summary_payload(
    config: DeploymentConfig,
    server_url: str,
) -> dict[str, str]:
    """Structured OpenAI-compatible connection summary for a completed deploy."""
    return build_openai_connection_payload(config, server_url)


def _deploy_connection_card_payload(
    config: DeploymentConfig,
    server_url: str,
) -> dict[str, str]:
    """Connection card fields, including the generated endpoint API key."""
    payload = dict(_deploy_connection_summary_payload(config, server_url))
    payload["api_key"] = config.endpoint_api_key or ""
    return payload


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
    SUB_TITLE = "Modal + Prime LLM backends"

    CSS_PATH = Path(__file__).with_name("theme.tcss")

    BINDINGS = [
        Binding("ctrl+c", "request_quit", show=False, priority=True, system=True),
        Binding(
            "ctrl+v,ctrl+shift+v,super+v,meta+v,cmd+v,command+v",
            "paste_from_clipboard",
            show=False,
            priority=True,
        ),
        Binding("ctrl+t", "toggle_mouse_mode", "Mouse", show=True),
    ]
    _CTRL_C_CONFIRM_WINDOW_SECONDS = 10.0
    _ENDPOINT_CACHE_TTL_SECONDS = 20.0
    _STORAGE_CACHE_TTL_SECONDS = 20.0

    def __init__(self, *, mouse_enabled: bool | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        for theme in LAUNCHPAD_THEMES:
            self.register_theme(theme)
        visual_settings = ConfigStore().load()
        self.theme = normalize_tui_theme(visual_settings.tui_theme)
        self.tui_density = normalize_tui_density(visual_settings.tui_density)
        self._confirm_quit = visual_settings.confirm_quit
        if mouse_enabled is None:
            mouse_enabled = (
                visual_settings.tui_mouse
                if visual_settings.tui_mouse is not None
                else default_tui_mouse_enabled()
            )
        self._monochrome_filter = Monochrome(
            enabled=self.theme == "launchpad-monochrome"
        )
        self._filters.append(self._monochrome_filter)
        self.mouse_enabled = mouse_enabled
        self._ctrl_c_last_requested_at = 0.0
        self._orchestrator = Orchestrator()
        self._username: str = ""
        self._version: str = ""
        self._endpoint_snapshot_cache: list[EndpointInfo] | None = None
        self._endpoint_snapshot_cached_at_epoch: float = 0.0
        self._endpoint_refresh_inflight = False
        self._endpoint_refresh_lock = threading.Lock()
        self._endpoint_refresh_receivers: list[object] = []
        self._storage_snapshot_cache: StorageSnapshot | None = None
        self._storage_snapshot_cached_at_epoch: float = 0.0
        self._storage_refresh_inflight = False
        self._storage_refresh_lock = threading.Lock()
        self._storage_cache_path = SETTINGS_DIR / "storage_snapshot.json"
        self._deploy_connection_cache_path = SETTINGS_DIR / "deployment_connection_summaries.json"
        self._deploy_connection_cache: dict[str, dict[str, object]] = {}
        self._in_flight_deploy: InFlightDeployment | None = None
        self._load_persisted_storage_cache()
        self._load_persisted_deploy_connection_cache()
        try:
            from importlib.metadata import version

            self._version = version("llm-launchpad")
        except Exception:
            pass

    def apply_visual_preferences(self, theme: str, density: str) -> None:
        """Apply persisted visual preferences without restarting the TUI."""
        self.theme = normalize_tui_theme(theme)
        self.tui_density = normalize_tui_density(density)
        self._monochrome_filter.enabled = self.theme == "launchpad-monochrome"
        try:
            screen = self.screen
        except Exception:
            return
        refresher = getattr(screen, "refresh_visual_preferences", None)
        if callable(refresher):
            refresher()
        else:
            screen.refresh(repaint=True, layout=True)

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text via OSC 52 and the host clipboard when available."""
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
        # OSC 52 handles remote terminals; a local clipboard provider covers
        # terminals that don't implement OSC 52.
        write_system_clipboard(text)

    def paste_from_clipboard(self) -> str | None:
        """Refresh the app clipboard from the host clipboard when possible."""
        text = read_system_clipboard()
        if text is not None:
            self._clipboard = text
        return text

    def action_paste_from_clipboard(self) -> None:
        """Paste host clipboard text into the focused editable widget."""
        try:
            focused = self.focused
        except Exception:
            focused = None
        if not isinstance(focused, (Input, TextArea)):
            return
        self.paste_from_clipboard()
        action_paste = getattr(focused, "action_paste", None)
        if callable(action_paste):
            action_paste()

    def _copy_current_selection(self) -> bool:
        """Copy a real selection, returning whether one was available."""
        try:
            focused = self.focused
        except Exception:
            focused = None
        if isinstance(focused, (Input, TextArea)):
            selected_text = getattr(focused, "selected_text", "")
            if selected_text:
                self.copy_to_clipboard(selected_text.rstrip("\n"))
                return True

        try:
            selected_text = self.screen.get_selected_text()
        except Exception:
            selected_text = None
        if selected_text:
            normalized = selected_text.rstrip("\n")
            if normalized:
                self.copy_to_clipboard(normalized)
                return True
        return False

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
        import asyncio

        # Killing the local client does not stop what the provider already
        # created, so an in-flight deployment would keep billing with nothing
        # pointing at it. Stop it before the workers that own it are cancelled.
        pending = self._in_flight_deploy
        if pending is not None:
            self.notify(
                f"Stopping {pending.app_name} before exit...",
                severity="warning",
            )
            await asyncio.to_thread(self._stop_in_flight, pending)
        ModalBackend.terminate_all()
        self.workers.cancel_all()
        await asyncio.sleep(0.3)
        self.exit()

    def _stop_in_flight(self, pending: InFlightDeployment) -> bool:
        """Stop an abandoned deployment. Returns whether it is confirmed gone."""

        try:
            for event in self._orchestrator.stop_app(
                pending.backend_type,
                app_name=pending.app_name,
                app_id=pending.app_id,
                provider=pending.compute_provider,
            ):
                if isinstance(event, OperationCompleteEvent) and not event.success:
                    return False
        except Exception:
            log_exception(f"Failed to stop in-flight deployment {pending.app_name}")
            return False
        clear_in_flight(pending.app_name)
        return True

    def recover_abandoned_deployments(self) -> tuple[InFlightDeployment, ...]:
        """Return deployments a previous session started but never resolved.

        A SIGKILL runs no handler, so the journal is the only record that a
        deployment was left behind. Entries are reported rather than stopped
        automatically: the user may have deliberately left one running, and
        silently terminating someone's endpoint is worse than telling them.
        """

        return load_in_flight()

    async def action_request_quit(self) -> None:
        """Require a second Ctrl+C press before quitting the TUI."""
        if self._copy_current_selection():
            return
        if not self._confirm_quit:
            await self.action_quit()
            return
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
        """Launch the TUI when at least one compute provider is configured.

        Missing authentication is surfaced inside the app via a dedicated setup
        screen instead of exiting before the user can read the message.
        """
        if not self._provider_is_configured():
            self.push_screen(SetupRequiredScreen())
            return
        self._enter_main_menu()
        self._warn_about_abandoned_deployments()

    def _warn_about_abandoned_deployments(self) -> None:
        """Tell the user about deployments a previous session left running.

        Nothing is stopped automatically: the journal records that a deploy was
        interrupted, not that the resource is unwanted, and terminating an
        endpoint someone is using would be worse than the leak.
        """

        abandoned = self.recover_abandoned_deployments()
        if not abandoned:
            return
        for entry in abandoned:
            exposure = entry.exposure_usd()
            # Phrased as a ceiling, not a bill: the journal knows the deploy
            # never resolved, not whether the resource still exists.
            spend = (
                f" (up to ~${exposure:.2f} if it still is)"
                if exposure is not None and exposure >= 0.01
                else ""
            )
            self.notify(
                f"{entry.app_name} was still deploying when Launchpad last "
                f"exited and may still be running{spend}. Check Deployments.",
                severity="warning",
                timeout=15,
            )

    def _provider_is_configured(self) -> bool:
        modal_available = ModalBackend.is_cli_available()
        prime_available = get_prime_auth_status().authenticated
        return modal_available or prime_available

    def _enter_main_menu(self) -> None:
        self.push_screen(MainMenuScreen(username="", version=self._version))
        if ModalBackend.is_cli_available():
            self.run_worker(
                self._run_load_modal_username,
                name="modal-username-worker",
                thread=True,
            )
        if not self.mouse_enabled:
            self.notify(
                "Terminal copy mode active. Press Ctrl+T to enable mouse.",
                timeout=4,
            )

    def recheck_provider_setup(self) -> bool:
        """Re-run provider detection; enter the main menu when configured.

        Returns whether the app entered the main menu. Called by the setup
        screen's Re-check action after the user authenticates elsewhere.
        """
        if not self._provider_is_configured():
            return False
        try:
            if isinstance(self.screen, SetupRequiredScreen):
                self.pop_screen()
        except Exception:
            pass
        self._enter_main_menu()
        return True

    def _run_load_modal_username(self) -> None:
        self.post_message(ModalUsernameLoaded(ModalBackend.get_username() or ""))

    def on_modal_username_loaded(self, message: ModalUsernameLoaded) -> None:
        self._username = message.username
        for screen in self.screen_stack:
            setter = getattr(screen, "set_modal_username", None)
            if callable(setter):
                setter(message.username)

    # ------------------------------------------------------------------
    # Actions called by screens
    # ------------------------------------------------------------------

    def action_push_deploy(self) -> None:
        current_screen = self.screen
        if isinstance(current_screen, MainMenuScreen):
            current_screen.ensure_quick_deploy_catalog_refresh()
        self.push_screen(FastDeployScreen())

    def quick_deploy_catalog_updated(self) -> None:
        """Notify the active model picker that the shared catalog changed."""
        try:
            screen = self.screen
        except Exception:
            return
        refresher = getattr(screen, "refresh_quick_deploy_catalog", None)
        if callable(refresher):
            refresher()

    def action_push_custom_deploy(self) -> None:
        self.push_screen(BackendSelectScreen())

    def pop_to_main_menu(self) -> None:
        """Pop nested deploy/monitor screens until the home menu is current."""
        while len(self.screen_stack) > 1 and not isinstance(self.screen, MainMenuScreen):
            self.pop_screen()

    def action_push_manage(self) -> None:
        self.push_screen(ManageScreen())

    def action_push_settings(self) -> None:
        self.push_screen(SettingsScreen())

    def action_push_storage(self, backend: BackendType | None = None) -> None:
        self.push_screen(StorageScreen(initial_backend=backend))

    def push_quick_deploy(
        self,
        profile: str | QuickDeployProfile | InferencePlan,
        *,
        alternative_plans: tuple[InferencePlan, ...] | None = None,
    ) -> None:
        self.push_screen(
            QuickDeployScreen(
                profile_id=profile,
                alternative_plans=alternative_plans,
            )
        )

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def begin_deploy(self, config: DeploymentConfig) -> None:
        """Start a deploy operation via a threaded worker."""
        if not config.app_name:
            config.app_name = build_deployment_name(
                config.provider, config.backend, config.instance_name
            )
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

    def _run_deploy(self, config: DeploymentConfig, monitor: MonitorScreen) -> None:
        """Run one deployment to completion, journalling it for recovery."""
        # The provider creates the app before its command returns, so this is
        # journalled first: an entry that outlives the process is the only
        # evidence that a deployment was abandoned mid-flight.
        self._begin_in_flight(config)
        try:
            # Not a generator: run_worker calls this directly and relies on its
            # side effects, so the body must execute on call.
            self._run_deploy_inner(config, monitor)
        finally:
            self._finish_in_flight(config)

    def _begin_in_flight(self, config: DeploymentConfig) -> None:
        app_name = (config.app_name or legacy_app_name(config.backend)).strip()
        if not app_name:
            return
        requirements = config.serving_requirements
        entry = InFlightDeployment(
            app_name=app_name,
            provider=config.provider.value,
            backend=config.backend.value,
            instance_name=config.instance_name or None,
            gpu_type=config.gpu_type or None,
            gpu_count=max(1, int(config.gpu_count or 1)),
            price_per_hour_usd=(
                requirements.max_hourly_cost_usd if requirements is not None else None
            ),
            started_at_epoch=time.time(),
        )
        self._in_flight_deploy = entry
        record_in_flight(entry)

    def _finish_in_flight(self, config: DeploymentConfig) -> None:
        app_name = (config.app_name or legacy_app_name(config.backend)).strip()
        self._in_flight_deploy = None
        if app_name:
            clear_in_flight(app_name)

    def _run_deploy_inner(self, config: DeploymentConfig, monitor: MonitorScreen):  # type: ignore[return]
        """Consumed by run_worker in a thread."""
        deployed_web_url: Optional[str] = None
        deployed_endpoint: EndpointInfo | None = None
        deployed_web_url_priority = -1
        deploy_succeeded = False
        opencode_synced = False
        will_run_warmup = bool(config.do_warmup and config.do_deploy)

        def _emit_connection_summary(url: str) -> None:
            self._cache_deploy_connection_summary(config, url, deployed_endpoint)
            for line in _deploy_connection_summary_lines(config, url):
                _dispatch_event(monitor, LogEvent(line=line))
            poster = getattr(monitor, "post_message", None)
            if callable(poster):
                poster(ConnectionSummaryReady(_deploy_connection_card_payload(config, url)))

        def _sync_now(url: str) -> None:
            nonlocal opencode_synced
            target_app_name = config.app_name or legacy_app_name(config.backend)
            live_rows, prune_providers = self._visible_rows_and_prune_scope()
            self._sync_opencode(
                target_app_name=target_app_name,
                target_url=url,
                target_config=config,
                current_rows=live_rows,
                prune_providers=prune_providers,
                monitor=monitor,
                emit_skipped=True,
            )
            opencode_synced = True

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
            elif isinstance(event, EndpointAvailableEvent):
                deployed_endpoint = event.endpoint
                self._invalidate_endpoint_snapshot()
                if event.endpoint.web_url:
                    deployed_web_url = event.endpoint.web_url
                    deployed_web_url_priority = max(deployed_web_url_priority, 2)
                    if not will_run_warmup:
                        _emit_connection_summary(event.endpoint.web_url)
                        _sync_now(event.endpoint.web_url)
            elif isinstance(event, OperationCompleteEvent):
                deploy_succeeded = event.success
                if event.operation == OperationType.DEPLOY:
                    self._invalidate_endpoint_snapshot()
                if event.success and isinstance(event.data, EndpointInfo):
                    deployed_endpoint = event.data
                    deployed_web_url = event.data.web_url or deployed_web_url
                    if event.data.web_url and not will_run_warmup:
                        self._cache_deploy_connection_summary(
                            config, event.data.web_url, event.data
                        )
                elif (
                    not event.success
                    and event.operation == OperationType.DEPLOY
                    and opencode_synced
                ):
                    target_app_name = config.app_name or legacy_app_name(config.backend)
                    live_rows, prune_providers = self._visible_rows_and_prune_scope()
                    self._sync_opencode(
                        current_rows=live_rows,
                        remove_app_names=[target_app_name],
                        prune_providers=prune_providers,
                        monitor=monitor,
                    )
                    opencode_synced = False
                # When warmup immediately follows a successful deploy in the same
                # monitor session, suppress the intermediate completion footer
                # ("Operation complete... Press esc") to keep the summary cleaner.
                if will_run_warmup and event.success and event.operation == OperationType.DEPLOY:
                    continue
                if (
                    event.success
                    and event.operation == OperationType.DEPLOY
                    and config.do_deploy
                    and not opencode_synced
                ):
                    target_app_name = config.app_name or legacy_app_name(config.backend)
                    url = deployed_web_url
                    if not url and config.provider == ComputeProvider.MODAL:
                        url = ModalBackend.default_server_url(
                            self._username,
                            app_name=target_app_name,
                            function_slug=config.function_slug,
                        )
                    if url:
                        _emit_connection_summary(url)
                        _sync_now(url)
            _dispatch_event(monitor, event)

        if not deploy_succeeded and config.fallback_configs:
            fallbacks = list(config.fallback_configs)
            next_config = fallbacks.pop(0)
            next_config.fallback_configs = tuple(fallbacks)
            _dispatch_event(
                monitor,
                LogEvent(
                    line=(
                        "Deployment could not start; trying an equivalent placement "
                        "at or below the approved hourly price."
                    ),
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                ),
            )
            self._run_deploy(next_config, monitor)
            return

        # Warmup if requested and deploy was successful
        if config.do_warmup and config.do_deploy and deploy_succeeded:
            target_app_name = config.app_name or legacy_app_name(config.backend)
            url = deployed_web_url
            if not url and config.provider == ComputeProvider.MODAL:
                url = ModalBackend.default_server_url(
                    self._username,
                    app_name=target_app_name,
                    function_slug=config.function_slug,
                )
            if not url:
                _dispatch_event(monitor, ErrorEvent(message="Provider returned no endpoint URL."))
                return
            certification_kwargs = (
                {
                    "serving_requirements": config.serving_requirements,
                    "placement_assessment": config.placement_assessment,
                    "runtime_id": config.llamacpp_runtime_id,
                }
                if config.serving_requirements is not None
                else {}
            )
            warmup_events = (
                self._orchestrator.warmup(
                    backend=config.backend,
                    server_url=url,
                    timeout=1800,
                    tail_logs=True,
                    app_name=target_app_name,
                    served_model_name=config.served_model_name,
                    **certification_kwargs,
                )
                if config.provider == ComputeProvider.MODAL
                else self._orchestrator.warmup(
                    backend=config.backend,
                    server_url=url,
                    timeout=1800,
                    tail_logs=True,
                    app_name=target_app_name,
                    served_model_name=config.served_model_name,
                    provider=config.provider,
                    api_key=config.endpoint_api_key,
                    pod_id=deployed_endpoint.app_id if deployed_endpoint else None,
                    **certification_kwargs,
                )
            )
            for event in warmup_events:
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
                        attestation = event.data.get("attestation")
                        if attestation is not None:
                            config.runtime_attestation = attestation
                            if deployed_endpoint is not None:
                                deployed_endpoint.runtime_attestation = attestation
                    _dispatch_event(
                        monitor,
                        StateChangeEvent(
                            current=DeploymentState.PUBLISHING,
                            operation=OperationType.WARMUP,
                            detail="Publishing verified endpoint",
                        ),
                    )
                    _emit_connection_summary(completed_url)
                    if not opencode_synced or completed_url != url:
                        _sync_now(completed_url)
                elif (
                    isinstance(event, OperationCompleteEvent)
                    and not event.success
                    and event.operation == OperationType.WARMUP
                ):
                    keep_failed_prime = (
                        config.provider == ComputeProvider.PRIME
                        and prime_provider_options(config).keep_failed_resource
                    )
                    if not keep_failed_prime:
                        resource_label = (
                            f"Prime pod {deployed_endpoint.app_id}"
                            if config.provider == ComputeProvider.PRIME
                            and deployed_endpoint is not None
                            else f"failed {config.provider.display_name} deployment"
                        )
                        _dispatch_event(
                            monitor,
                            LogEvent(line=f"Certification failed; cleaning up {resource_label}."),
                        )
                        for cleanup_event in self._orchestrator.stop_app(
                            config.backend,
                            app_name=target_app_name,
                            app_id=(deployed_endpoint.app_id if deployed_endpoint else None),
                            provider=config.provider,
                        ):
                            _dispatch_event(monitor, cleanup_event)
                    fallbacks = list(config.fallback_configs)
                    if fallbacks:
                        next_config = fallbacks.pop(0)
                        next_config.fallback_configs = tuple(fallbacks)
                        _dispatch_event(
                            monitor,
                            LogEvent(
                                line=(
                                    "Trying an equivalent certified placement at or below "
                                    "the approved hourly price."
                                ),
                                operation=OperationType.DEPLOY,
                                is_milestone=True,
                            ),
                        )
                        self._run_deploy(next_config, monitor)
                        return
                _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: status
    # ------------------------------------------------------------------

    def begin_status(
        self,
        endpoint: EndpointInfo,
        url_override: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        """Probe one selected endpoint, resolving provider details centrally."""
        if endpoint.backend is None:
            self.notify("This endpoint has no recognized backend.", severity="error", timeout=6)
            return
        target_app_name = endpoint.name or legacy_app_name(endpoint.backend)
        url = url_override or endpoint.web_url
        if not url and endpoint.provider == ComputeProvider.MODAL:
            url = ModalBackend.default_server_url(self._username, app_name=target_app_name)
        if not url:
            self.notify("No endpoint URL is stored for this deployment.", severity="error", timeout=6)
            return
        monitor = MonitorScreen(title="Status Check")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_status(
                endpoint.backend,
                url,
                timeout,
                target_app_name,
                endpoint.served_model_name,
                endpoint.provider,
                endpoint.endpoint_api_key,
                endpoint.app_id or None,
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
        provider: ComputeProvider,
        api_key: Optional[str],
        pod_id: Optional[str],
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        _dispatch_event(monitor, LogEvent(line=f"Target app: {app_name}"))
        events = (
            self._orchestrator.check_status(
                backend,
                url,
                timeout,
                served_model_name=served_model_name,
            )
            if provider == ComputeProvider.MODAL
            else self._orchestrator.check_status(
                backend,
                url,
                timeout,
                served_model_name=served_model_name,
                provider=provider,
                api_key=api_key,
                pod_id=pod_id,
            )
        )
        for event in events:
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
        endpoint: EndpointInfo,
        follow: bool = True,
    ) -> None:
        """Tail logs for one selected endpoint."""
        if endpoint.backend is None:
            self.notify("This endpoint has no recognized backend.", severity="error", timeout=6)
            return
        monitor = MonitorScreen(title="Logs")
        self.push_screen(monitor)
        target_app_name = endpoint.name or legacy_app_name(endpoint.backend)
        target_ref = (endpoint.app_id or target_app_name).strip()
        self.run_worker(
            lambda: self._run_logs(
                endpoint.backend,
                follow,
                target_ref,
                target_app_name,
                monitor,
                provider=endpoint.provider,
            ),
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
        provider: ComputeProvider = ComputeProvider.MODAL,
    ):  # type: ignore[return]
        target_label = app_name if app_ref == app_name else f"{app_name} ({app_ref})"
        _dispatch_event(monitor, LogEvent(line=f"Target app: {target_label}"))
        events = (
            self._orchestrator.tail_logs(backend, follow, app_name=app_name, app_id=app_ref)
            if provider == ComputeProvider.MODAL
            else self._orchestrator.tail_logs(
                backend,
                follow,
                app_name=app_name,
                app_id=app_ref,
                provider=provider,
            )
        )
        for event in events:
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: stop
    # ------------------------------------------------------------------

    def begin_stop(
        self,
        endpoint: EndpointInfo,
    ) -> None:
        """Stop one explicitly confirmed endpoint."""
        if endpoint.backend is None:
            self.notify("This endpoint has no recognized backend.", severity="error", timeout=6)
            return
        monitor = MonitorScreen(title="Stop")
        self.push_screen(monitor)
        target_app_name = endpoint.name or legacy_app_name(endpoint.backend)
        target_ref = (endpoint.app_id or target_app_name).strip()
        self.run_worker(
            lambda: self._run_stop(
                endpoint.backend,
                target_ref,
                target_app_name,
                monitor,
                provider=endpoint.provider,
            ),
            name="stop-worker",
            thread=True,
        )

    def _run_stop(
        self,
        backend: BackendType,
        app_ref: str,
        app_name: str,
        monitor: MonitorScreen,
        provider: ComputeProvider = ComputeProvider.MODAL,
    ):  # type: ignore[return]
        target_label = app_name if app_ref == app_name else f"{app_name} ({app_ref})"
        _dispatch_event(monitor, LogEvent(line=f"Target app: {target_label}"))
        events = (
            self._orchestrator.stop_app(backend, app_name=app_name, app_id=app_ref)
            if provider == ComputeProvider.MODAL
            else self._orchestrator.stop_app(
                backend,
                app_name=app_name,
                app_id=app_ref,
                provider=provider,
            )
        )
        for event in events:
            if (
                isinstance(event, OperationCompleteEvent)
                and event.success
                and event.operation == OperationType.STOP
            ):
                self._deploy_connection_cache.pop(app_name, None)
                self._persist_deploy_connection_cache()
                live_rows, prune_providers = self._visible_rows_and_prune_scope()
                self._cache_endpoint_snapshot(live_rows)
                self._sync_opencode(
                    current_rows=live_rows,
                    prune_providers=prune_providers,
                    remove_app_names=[app_name],
                    monitor=monitor,
                    emit_skipped=True,
                )
            _dispatch_event(monitor, event)

    def _cache_endpoint_snapshot(self, rows: list[EndpointInfo]) -> None:
        with self._endpoint_refresh_lock:
            self._endpoint_snapshot_cache = [replace(row) for row in rows]
            self._endpoint_snapshot_cached_at_epoch = time.time()

    def _invalidate_endpoint_snapshot(self) -> None:
        with self._endpoint_refresh_lock:
            self._endpoint_snapshot_cached_at_epoch = 0.0

    def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
        """Load endpoints once and deliver the result to every waiting screen."""
        poster = getattr(receiver, "post_message", None)
        if poster is None:
            return

        now = time.time()
        cached_rows: list[EndpointInfo] | None = None
        cache_is_fresh = False
        start_worker = False
        with self._endpoint_refresh_lock:
            if self._endpoint_snapshot_cache is not None:
                cached_rows = [replace(row) for row in self._endpoint_snapshot_cache]
                cache_age = now - self._endpoint_snapshot_cached_at_epoch
                cache_is_fresh = not force and cache_age <= self._ENDPOINT_CACHE_TTL_SECONDS

            if not cache_is_fresh:
                if not any(
                    candidate is receiver for candidate in self._endpoint_refresh_receivers
                ):
                    self._endpoint_refresh_receivers.append(receiver)
                if not self._endpoint_refresh_inflight:
                    self._endpoint_refresh_inflight = True
                    start_worker = True

        if cached_rows is not None:
            poster(EndpointsLoaded(rows=cached_rows, is_stale=not cache_is_fresh))
        if cache_is_fresh:
            return
        if start_worker:
            self.run_worker(
                self._run_endpoint_refresh,
                name="endpoint-refresh-worker",
                thread=True,
            )

    def _run_endpoint_refresh(self) -> None:
        try:
            rows, _prune_providers = self._visible_rows_and_prune_scope()
        except Exception as exc:
            self._finish_endpoint_refresh(error=str(exc))
            return

        self._cache_endpoint_snapshot(rows)
        self._finish_endpoint_refresh(rows=rows)

    def _finish_endpoint_refresh(
        self,
        *,
        rows: list[EndpointInfo] | None = None,
        error: str | None = None,
    ) -> None:
        with self._endpoint_refresh_lock:
            receivers = self._endpoint_refresh_receivers
            self._endpoint_refresh_receivers = []
            self._endpoint_refresh_inflight = False

        for receiver in receivers:
            poster = getattr(receiver, "post_message", None)
            if poster is None:
                continue
            if error is not None:
                poster(EndpointsFailed(error=error))
            else:
                poster(EndpointsLoaded(rows=[replace(row) for row in rows or []]))

    def list_instances(self, backend: BackendType | None = None) -> list[EndpointInfo]:
        """Synchronously discover endpoints without unrelated configuration writes."""
        rows, _prune_providers = self._visible_rows_and_prune_scope()
        if backend is None:
            return list(rows)
        return [row for row in rows if row.backend == backend]

    def _visible_rows_and_prune_scope(
        self,
    ) -> tuple[list[EndpointInfo], tuple[ComputeProvider, ...]]:
        rows: list[EndpointInfo] = []
        prune_providers: list[ComputeProvider] = []
        prime_enabled = get_prime_auth_status().authenticated
        with ThreadPoolExecutor(max_workers=2 if prime_enabled else 1) as executor:
            modal_future = executor.submit(ModalBackend.list_apps)
            prime_future = (
                executor.submit(PrimeBackend().list_deployments)
                if prime_enabled
                else None
            )
            try:
                modal_rows = modal_future.result()
            except Exception:
                modal_rows = None
            try:
                prime_rows = prime_future.result() if prime_future is not None else None
            except Exception:
                prime_rows = None

        if modal_rows is not None:
            rows.extend(modal_rows)
            prune_providers.append(ComputeProvider.MODAL)
        if prime_rows is not None:
            rows.extend(prime_rows)
            prune_providers.append(ComputeProvider.PRIME)
        self._merge_deploy_connection_cache(rows)
        return visible_launchpad_rows(rows), tuple(prune_providers)

    def _sync_opencode(
        self,
        *,
        target_app_name: str | None = None,
        target_url: str | None = None,
        target_config: DeploymentConfig | None = None,
        current_rows: list[EndpointInfo] | None = None,
        remove_app_names: list[str] | None = None,
        prune_providers: tuple[ComputeProvider, ...] | None = None,
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
                prune_providers=prune_providers,
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

    def cached_storage_snapshot(self) -> StorageSnapshot | None:
        """Return the latest cached storage snapshot, if one is available."""
        with self._storage_refresh_lock:
            return self._storage_snapshot_cache

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
            log_exception("Failed to remove cached storage snapshot")

    def _persist_storage_snapshot(self, snapshot: StorageSnapshot) -> None:
        payload = {
            "cached_at_epoch": time.time(),
            "snapshot": self._snapshot_to_dict(snapshot),
        }
        try:
            self._storage_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_cache_path.write_text(json.dumps(payload))
        except Exception:
            log_exception("Failed to persist storage snapshot cache")

    def _cache_deploy_connection_summary(
        self,
        config: DeploymentConfig,
        server_url: str,
        endpoint: EndpointInfo | None = None,
    ) -> None:
        app_name = (config.app_name or "").strip()
        if not app_name:
            return
        cached_endpoint = replace(endpoint) if endpoint is not None else EndpointInfo()
        cached_endpoint.name = cached_endpoint.name or app_name
        cached_endpoint.web_url = server_url
        try:
            save_connection(config, cached_endpoint, self._deploy_connection_cache_path)
            self._deploy_connection_cache = load_connection_entries(
                self._deploy_connection_cache_path
            )
        except Exception:
            # Connection summaries are a convenience cache; a read-only home
            # directory must not turn an otherwise successful deploy into a
            # failed operation.
            log_exception("Failed to cache deploy connection summary")
            return

    def _persist_deploy_connection_cache(self) -> None:
        payload = {"entries": self._deploy_connection_cache}
        try:
            self._deploy_connection_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._deploy_connection_cache_path.write_text(json.dumps(payload))
            os.chmod(self._deploy_connection_cache_path, 0o600)
        except Exception:
            log_exception("Failed to persist deploy connection cache")

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
        merge_connections(
            rows,
            self._deploy_connection_cache_path,
            self._storage_cache_path,
        )
        self._deploy_connection_cache = load_connection_entries(
            self._deploy_connection_cache_path
        )

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
        compatibility = evaluate_llamacpp_architecture(metadata.architecture)
        poster(
            LlamaCppQuantsLoaded(
                repo_id=repo_id,
                revision=revision,
                quantizations=list(metadata.quantizations),
                vram_gb_by_quant=dict(metadata.vram_gb_by_quant),
                architecture=metadata.architecture,
                compatibility_status=compatibility.status.value,
                compatibility_message=compatibility.message,
                llamacpp_runtime_id=compatibility.runtime_id,
            )
        )
        try:
            discover_reasoning_capabilities(
                BackendType.LLAMACPP,
                repo_id,
                revision,
            )
        except Exception:
            pass

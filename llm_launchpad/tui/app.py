"""Main Textual App: screen routing, keybindings, worker orchestration.

WizardApp is the entry point for ``llm-launchpad wizard``.  It owns the
screen stack and bridges user actions to Core via threaded workers.
"""

from __future__ import annotations

from typing import Optional

from textual.app import App
from textual.binding import Binding

from ..core.backend import ModalBackend
from ..core.hf_models import fetch_gguf_quantizations, list_llamacpp_candidates, list_vllm_candidates
from ..core.naming import build_app_name, legacy_app_name
from ..core.orchestrator import Orchestrator
from ..protocol.enums import BackendType
from ..protocol.events import LogEvent
from ..protocol.models import EndpointInfo
from ..protocol.models import DeploymentConfig

from .screens.main_menu import MainMenuScreen
from .screens.deploy import BackendSelectScreen
from .screens.manage import ManageScreen
from .screens.monitor import MonitorScreen
from .screens.settings import SettingsScreen
from .workers import (
    LlamaCppModelsFailed,
    LlamaCppModelsLoaded,
    LlamaCppQuantsFailed,
    LlamaCppQuantsLoaded,
    VllmModelsFailed,
    VllmModelsLoaded,
    _dispatch_event,
)


class WizardApp(App):
    """llm-launchpad interactive wizard."""

    TITLE = "llm-launchpad"
    SUB_TITLE = "Modal LLM backends"

    CSS_PATH = "theme.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._orchestrator = Orchestrator()
        self._username: str = ""
        self._version: str = ""
        try:
            from importlib.metadata import version

            self._version = version("llm-launchpad")
        except Exception:
            pass

    def action_quit(self) -> None:
        """Terminate tracked subprocesses before exiting.

        Without this, worker threads blocked on subprocess I/O prevent
        Python from shutting down cleanly (the atexit thread-join hangs).
        """
        ModalBackend.terminate_all()
        self.exit()

    def on_mount(self) -> None:
        """Run pre-flight checks and push main menu."""
        ok, username, err = self._orchestrator.preflight()
        if not ok:
            self.notify(err, severity="error", timeout=10)
            self.exit(return_code=1)
            return
        self._username = username
        self.push_screen(MainMenuScreen(username=self._username, version=self._version))

    # ------------------------------------------------------------------
    # Actions called by screens
    # ------------------------------------------------------------------

    def action_push_deploy(self) -> None:
        self.push_screen(BackendSelectScreen())

    def action_push_manage(self) -> None:
        self.push_screen(ManageScreen())

    def action_push_settings(self) -> None:
        self.push_screen(SettingsScreen())

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def begin_deploy(self, config: DeploymentConfig) -> None:
        """Start a deploy operation via a threaded worker."""
        if not config.app_name:
            config.app_name = build_app_name(config.backend, config.instance_name)
        monitor = MonitorScreen(title="Deploy")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_deploy(config, monitor),
            name="deploy-worker",
            thread=True,
        )

    def _run_deploy(self, config: DeploymentConfig, monitor: MonitorScreen):  # type: ignore[return]
        """Generator consumed by run_worker in a thread."""
        for event in self._orchestrator.deploy(config):
            _dispatch_event(monitor, event)

        # Warmup if requested and deploy was successful
        if config.do_warmup and config.do_deploy:
            target_app_name = config.app_name or legacy_app_name(config.backend)
            url = ModalBackend.default_server_url(self._username, app_name=target_app_name)
            for event in self._orchestrator.warmup(
                backend=config.backend,
                server_url=url,
                timeout=1800,
                tail_logs=True,
                app_name=target_app_name,
                served_model_name=config.served_model_name,
            ):
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
    ) -> None:
        target_app_name = app_name or legacy_app_name(backend)
        url = server_url or ModalBackend.default_server_url(self._username, app_name=target_app_name)
        monitor = MonitorScreen(title="Status Check")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_status(backend, url, timeout, target_app_name, monitor),
            name="status-worker",
            thread=True,
        )

    def _run_status(
        self,
        backend: BackendType,
        url: str,
        timeout: int,
        app_name: str,
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        _dispatch_event(monitor, LogEvent(line=f"Target app: {app_name}"))
        for event in self._orchestrator.check_status(backend, url, timeout):
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: logs
    # ------------------------------------------------------------------

    def begin_logs(
        self,
        backend: BackendType,
        follow: bool = True,
        app_name: Optional[str] = None,
    ) -> None:
        monitor = MonitorScreen(title="Logs")
        self.push_screen(monitor)
        target_app_name = app_name or legacy_app_name(backend)
        self.run_worker(
            lambda: self._run_logs(backend, follow, target_app_name, monitor),
            name="logs-worker",
            thread=True,
        )

    def _run_logs(
        self,
        backend: BackendType,
        follow: bool,
        app_name: str,
        monitor: MonitorScreen,
    ):  # type: ignore[return]
        _dispatch_event(monitor, LogEvent(line=f"Target app: {app_name}"))
        for event in self._orchestrator.tail_logs(backend, follow, app_name=app_name):
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: stop
    # ------------------------------------------------------------------

    def begin_stop(self, backend: BackendType, app_name: Optional[str] = None) -> None:
        monitor = MonitorScreen(title="Stop")
        self.push_screen(monitor)
        target_app_name = app_name or legacy_app_name(backend)
        self.run_worker(
            lambda: self._run_stop(backend, target_app_name, monitor),
            name="stop-worker",
            thread=True,
        )

    def _run_stop(
        self, backend: BackendType, app_name: str, monitor: MonitorScreen
    ):  # type: ignore[return]
        _dispatch_event(monitor, LogEvent(line=f"Target app: {app_name}"))
        for event in self._orchestrator.stop_app(backend, app_name=app_name):
            _dispatch_event(monitor, event)

    def list_instances(self, backend: BackendType) -> list[EndpointInfo]:
        rows = ModalBackend.list_apps() or []
        return [row for row in rows if row.backend == backend]

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
            quantizations = fetch_gguf_quantizations(repo_id=repo_id, revision=revision)
        except Exception as exc:
            poster(LlamaCppQuantsFailed(repo_id=repo_id, revision=revision, error=str(exc)))
            return
        poster(LlamaCppQuantsLoaded(repo_id=repo_id, revision=revision, quantizations=quantizations))

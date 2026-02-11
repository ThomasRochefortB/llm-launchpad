"""Main Textual App: screen routing, keybindings, worker orchestration.

WizardApp is the entry point for ``llm-launchpad wizard``.  It owns the
screen stack and bridges user actions to Core via threaded workers.
"""

from __future__ import annotations

import sys
from typing import Optional

from textual.app import App
from textual.binding import Binding
from textual.worker import Worker, WorkerState

from ..core.backend import ModalBackend
from ..core.config import ConfigStore
from ..core.orchestrator import Orchestrator
from ..protocol.enums import BackendType, OperationType
from ..protocol.models import DeploymentConfig

from .screens.main_menu import MainMenuScreen
from .screens.deploy import BackendSelectScreen
from .screens.manage import ManageScreen
from .screens.monitor import MonitorScreen
from .screens.settings import SettingsScreen
from .workers import _dispatch_event


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
            url = ModalBackend.default_server_url(self._username, config.backend)
            for event in self._orchestrator.warmup(
                backend=config.backend,
                server_url=url,
                timeout=1800,
                tail_logs=True,
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
    ) -> None:
        url = server_url or ModalBackend.default_server_url(self._username, backend)
        monitor = MonitorScreen(title="Status Check")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_status(backend, url, timeout, monitor),
            name="status-worker",
            thread=True,
        )

    def _run_status(
        self, backend: BackendType, url: str, timeout: int, monitor: MonitorScreen
    ):  # type: ignore[return]
        for event in self._orchestrator.check_status(backend, url, timeout):
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: logs
    # ------------------------------------------------------------------

    def begin_logs(self, backend: BackendType, follow: bool = True) -> None:
        monitor = MonitorScreen(title="Logs")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_logs(backend, follow, monitor),
            name="logs-worker",
            thread=True,
        )

    def _run_logs(
        self, backend: BackendType, follow: bool, monitor: MonitorScreen
    ):  # type: ignore[return]
        for event in self._orchestrator.tail_logs(backend, follow):
            _dispatch_event(monitor, event)

    # ------------------------------------------------------------------
    # Manage: stop
    # ------------------------------------------------------------------

    def begin_stop(self, backend: BackendType) -> None:
        monitor = MonitorScreen(title="Stop")
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_stop(backend, monitor),
            name="stop-worker",
            thread=True,
        )

    def _run_stop(self, backend: BackendType, monitor: MonitorScreen):  # type: ignore[return]
        for event in self._orchestrator.stop_app(backend):
            _dispatch_event(monitor, event)

"""Main Textual App: screen routing, keybindings, worker orchestration.

WizardApp is the entry point for ``llm-launchpad wizard``.  It owns the
screen stack and bridges user actions to Core via threaded workers.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from ..core.backend import ModalBackend
from ..core.config import SETTINGS_DIR
from ..core.hf_models import fetch_gguf_quant_metadata, list_llamacpp_candidates, list_vllm_candidates
from ..core.naming import build_app_name, legacy_app_name
from ..core.orchestrator import Orchestrator
from ..protocol.enums import BackendType
from ..protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from ..protocol.models import EndpointInfo
from ..protocol.models import DeploymentConfig
from ..protocol.models import StoredModelInfo
from ..protocol.models import StorageSnapshot

from .screens.main_menu import MainMenuScreen
from .screens.deploy import BackendSelectScreen
from .screens.manage import ManageScreen
from .screens.monitor import MonitorScreen
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


class WizardApp(App):
    """llm-launchpad interactive wizard."""

    TITLE = "llm-launchpad"
    SUB_TITLE = "Modal LLM backends"

    CSS_PATH = "theme.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
    ]
    _STORAGE_CACHE_TTL_SECONDS = 20.0

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._orchestrator = Orchestrator()
        self._username: str = ""
        self._version: str = ""
        self._storage_snapshot_cache: StorageSnapshot | None = None
        self._storage_snapshot_cached_at_epoch: float = 0.0
        self._storage_refresh_inflight = False
        self._storage_refresh_lock = threading.Lock()
        self._storage_cache_path = SETTINGS_DIR / "storage_snapshot.json"
        self._load_persisted_storage_cache()
        try:
            from importlib.metadata import version

            self._version = version("llm-launchpad")
        except Exception:
            pass

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text via OSC 52 and use pbcopy fallback on macOS terminals."""
        super().copy_to_clipboard(text)
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

    def action_push_storage(self, backend: BackendType | None = None) -> None:
        self.push_screen(StorageScreen(initial_backend=backend))

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
        deployed_web_url: Optional[str] = None
        deploy_succeeded = False
        for event in self._orchestrator.deploy(config):
            if isinstance(event, LogEvent):
                maybe_url = ModalBackend.extract_modal_web_url(event.line)
                if maybe_url:
                    deployed_web_url = maybe_url
            elif isinstance(event, OperationCompleteEvent):
                deploy_succeeded = event.success
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

    def list_instances(self, backend: BackendType | None = None) -> list[EndpointInfo]:
        rows = ModalBackend.list_apps() or []
        if backend is None:
            return rows
        return [row for row in rows if row.backend == backend]

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

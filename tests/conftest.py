"""Shared pytest fixtures that keep the unit suite hermetic and fast."""

from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

import pytest

from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.hf_models import GgufQuantMetadata


@pytest.fixture(autouse=True)
def _disable_isolated_asyncio_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run async UI tests without asyncio's expensive debug instrumentation."""

    def setup_runner(case: unittest.IsolatedAsyncioTestCase) -> None:
        assert case._asyncioRunner is None
        case._asyncioRunner = asyncio.Runner(debug=False)

    monkeypatch.setattr(
        unittest.IsolatedAsyncioTestCase,
        "_setupAsyncioRunner",
        setup_runner,
    )


@pytest.fixture(autouse=True)
def _shutdown_asyncio_executors_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid the hanging helper-thread executor shutdown path in tests.

    Textual and its Log widget genuinely need their ``thread=True`` workers to
    execute off the app thread. At loop teardown, however, asyncio normally
    launches one more helper thread just to join the default executor. That
    helper path can stall isolated unittest loops even when all queued work is
    complete. Joining directly preserves real worker behavior and makes
    teardown immediate and deterministic.
    """

    async def shutdown_directly(
        loop: asyncio.BaseEventLoop,
        timeout: float | None = None,
    ) -> None:
        del timeout
        loop._executor_shutdown_called = True
        executor = loop._default_executor
        if executor is not None:
            executor.shutdown(wait=True)

    monkeypatch.setattr(
        asyncio.BaseEventLoop,
        "shutdown_default_executor",
        shutdown_directly,
    )


@pytest.fixture(autouse=True)
def _stub_deploy_screen_gpu_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent deploy-screen mounts from fetching live Modal documentation.

    The Modal GPU parser and HTTP behavior have focused tests of their own. UI
    tests only need a deterministic catalog result so Textual's worker can
    finish before ``Pilot.pause`` waits for the app to become idle.
    """
    monkeypatch.setattr(
        "llm_launchpad.tui.screens.deploy.fetch_modal_gpu_catalog",
        lambda: [ModalGpuSpec("A100-80GB", price_per_hour_usd=2.50)],
    )


@pytest.fixture(autouse=True)
def _stub_orchestrator_llamacpp_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deployment unit tests from making Hugging Face preflight requests."""

    monkeypatch.setattr(
        "llm_launchpad.core.orchestrator.fetch_gguf_quant_metadata",
        lambda _repo_id, revision=None, **_kwargs: GgufQuantMetadata(
            quantizations=[],
            vram_gb_by_quant={},
            architecture="llama",
        ),
    )


@pytest.fixture(autouse=True)
def _stub_orchestrator_reasoning_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deployment tests from inspecting live Hugging Face repositories."""

    monkeypatch.setattr(
        "llm_launchpad.core.orchestrator.discover_selected_model_reasoning",
        lambda config: config.reasoning,
    )
    monkeypatch.setattr(
        "llm_launchpad.tui.app.discover_reasoning_capabilities",
        lambda _backend, _repo_id, _revision=None: None,
    )
    monkeypatch.setattr(
        "llm_launchpad.tui.screens.deploy.discover_reasoning_capabilities",
        lambda _backend, _repo_id, _revision=None: None,
    )
    monkeypatch.setattr(
        "llm_launchpad.core.connection_store.discover_reasoning_capabilities",
        lambda _backend, _repo_id, _revision=None: None,
    )


@pytest.fixture(autouse=True)
def _isolate_quick_deploy_catalog_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep catalog warm-start snapshots out of the real user settings dir.

    ``quick_deploy._load_quick_deploy_catalog`` imports
    ``quick_deploy_refresh.load_cached_quick_deploy_catalog`` lazily, so
    stub the loader at its definition site plus the already-imported
    reference inside ``quick_deploy``'s module namespace.
    """

    from llm_launchpad.core import quick_deploy as quick_deploy_module
    from llm_launchpad.core import quick_deploy_refresh as quick_deploy_refresh_module

    quick_deploy_module._reset_quick_deploy_catalog_cache()

    # Catalog retention compares a rebuild against the snapshot on disk, so the
    # snapshot path itself has to be isolated or tests read the developer's own
    # catalog and inherit whatever state it happens to be in.
    monkeypatch.setattr(
        quick_deploy_refresh_module,
        "_quick_deploy_catalog_cache_path",
        lambda: tmp_path_factory.mktemp("quick-deploy-catalog") / "catalog.json",
    )

    def _no_warm_catalog(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        quick_deploy_refresh_module,
        "load_cached_quick_deploy_catalog",
        _no_warm_catalog,
    )
    monkeypatch.setattr(
        quick_deploy_module,
        "load_cached_quick_deploy_catalog",
        _no_warm_catalog,
        raising=False,
    )
    yield
    quick_deploy_module._reset_quick_deploy_catalog_cache()


def _discover_settings_paths() -> tuple[tuple[str, str, Path], ...]:
    """Find every module attribute that resolves into the real settings dir.

    Constants like ``CONNECTIONS_PATH`` are computed from ``SETTINGS_DIR`` at
    import time, so redirecting ``SETTINGS_DIR`` alone comes too late. Rather
    than maintain a hand-written list -- which was already incomplete the first
    time it was written -- the modules are walked and every offending path is
    rebased. New caches are covered automatically.
    """

    import importlib
    import pkgutil

    import llm_launchpad
    from llm_launchpad.core.config import SETTINGS_DIR

    found: list[tuple[str, str, Path]] = []
    for info in pkgutil.walk_packages(
        llm_launchpad.__path__, prefix=f"{llm_launchpad.__name__}."
    ):
        try:
            module = importlib.import_module(info.name)
        except Exception:
            continue
        for name, value in vars(module).items():
            if not isinstance(value, Path):
                continue
            if value == SETTINGS_DIR:
                found.append((info.name, name, Path(".")))
            elif SETTINGS_DIR in value.parents:
                found.append((info.name, name, value.relative_to(SETTINGS_DIR)))
    return tuple(found)


_SETTINGS_PATHS = _discover_settings_paths()


@pytest.fixture(autouse=True)
def _isolate_user_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Redirect every ~/.llm_launchpad path the suite can reach.

    These caches hold real user state: OpenCode registrations, deployment
    connection summaries, Artificial Analysis credentials, saved settings, and
    Prime SSH key material. A test that writes one edits the installed product,
    and a test that reads one inherits whatever the developer's machine happens
    to contain. Both have already happened, so isolation is enforced globally
    and verified by ``test_settings_isolation``.
    """

    root = tmp_path_factory.mktemp("launchpad-settings")
    for module_name, attribute, relative in _SETTINGS_PATHS:
        monkeypatch.setattr(f"{module_name}.{attribute}", root / relative)
    return root

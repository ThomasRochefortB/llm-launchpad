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
def _isolate_tui_local_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep TUI connection/storage tests out of the real user settings directory."""

    monkeypatch.setattr("llm_launchpad.tui.app.SETTINGS_DIR", tmp_path / "tui-settings")


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


@pytest.fixture(autouse=True)
def _isolate_prime_local_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep Prime disk/frpc caches out of the real ~/.llm_launchpad directory."""

    root = tmp_path_factory.mktemp("prime-launchpad")
    monkeypatch.setattr(
        "llm_launchpad.core.prime_disks.PRIME_CACHE_DISKS_PATH",
        root / "disks.json",
    )
    monkeypatch.setattr(
        "llm_launchpad.core.prime_frpc.PRIME_FRPC_CACHE_DIR",
        root / "frpc",
    )


@pytest.fixture(autouse=True)
def _isolate_serving_certificate_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep planner placement assessments independent of real attestations.

    ``assess_memory_placement`` consults the certificate cache to promote or
    reject a topology. Without isolation, a developer machine that has deployed
    before would resolve genuine certificates and change placement outcomes.
    """

    root = tmp_path_factory.mktemp("serving-certificates")
    monkeypatch.setattr(
        "llm_launchpad.core.llamacpp_planner.CERTIFICATE_CACHE_PATH",
        root / "serving_certificates.json",
    )

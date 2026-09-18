"""Shared pytest fixtures that keep the unit suite hermetic and fast."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import unittest

import pytest

from llm_launchpad.core.artificial_analysis import ArtificialAnalysisAuthStatus
from llm_launchpad.core.hf_auth import HuggingFaceAuthStatus
from llm_launchpad.core.modal_auth import ModalAuthStatus
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.prime_auth import PrimeAuthStatus
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
def _stub_main_menu_catalog_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the live catalog build out of unrelated UI tests.

    ``MainMenuScreen.on_mount`` starts a worker running
    ``build_live_quick_deploy_catalog``: rankings parse plus up to 24
    concurrent HF lookups. Any UI test that mounts ``MainMenuScreen`` (or
    boots the full ``TuiApp``, whose ``on_mount`` enters the main menu)
    otherwise pays for that build and races with it. The catalog lifecycle
    has focused tests of its own, so stub the worker body here; the entry
    point (warm-snapshot activation + worker dispatch) stays real.
    """
    from llm_launchpad.tui.screens.main_menu import MainMenuScreen

    monkeypatch.setattr(
        MainMenuScreen,
        "_run_refresh_quick_deploy_catalog",
        lambda self: None,
    )


@pytest.fixture(autouse=True)
def _stub_provider_status_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep home-screen auth probes off the network in UI tests.

    ``MainMenuScreen`` fans out to four auth CLIs (Modal subprocess, HF
    ``whoami`` HTTPS, Prime config, AAI key check). Auth status has focused
    unit tests of its own; UI tests that mount the menu without patching all
    four paths otherwise serialize on real subprocess/HTTPS latency on every
    ``run_test()`` boot. Stub at the narrow helper boundary; tests that
    exercise a probe override the stub locally.
    """
    monkeypatch.setattr(
        "llm_launchpad.tui.screens.main_menu.get_modal_auth_status",
        lambda: ModalAuthStatus(authenticated=False),
    )
    monkeypatch.setattr(
        "llm_launchpad.tui.screens.main_menu.get_prime_auth_status",
        lambda: PrimeAuthStatus(authenticated=False),
    )
    monkeypatch.setattr(
        "llm_launchpad.tui.screens.main_menu.get_huggingface_auth_status",
        lambda: HuggingFaceAuthStatus(authenticated=False),
    )
    monkeypatch.setattr(
        "llm_launchpad.tui.screens.main_menu.get_artificial_analysis_auth_status",
        lambda: ArtificialAnalysisAuthStatus(authenticated=False),
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
def _stub_deploy_screen_vllm_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep typing in vLLM UI tests from starting live Hugging Face requests."""
    from llm_launchpad.core.hf_models import VllmMemoryBreakdown

    monkeypatch.setattr(
        "llm_launchpad.tui.screens.deploy.fetch_vllm_memory_breakdown",
        lambda **_kwargs: VllmMemoryBreakdown(
            total_gb=8.0,
            weights_gb=4.0,
            kv_cache_gb=2.0,
            overhead_gb=2.0,
            context_tokens=8192,
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


@pytest.fixture(autouse=True)
def _reset_serving_metrics_tracker() -> None:
    """Keep one test's serving readings out of the next test's totals.

    The fleet probe accumulates into a process-wide tracker that caches the
    previous reading and the on-disk totals in memory, so redirecting
    ``USAGE_PATH`` alone leaves the earlier case's numbers loaded.
    """
    from llm_launchpad.core.serving_metrics import default_tracker

    default_tracker().reset()


@pytest.fixture(autouse=True)
def _stub_vision_hub_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing deployment tests hermetic; vision tests override this boundary."""
    from llm_launchpad.protocol.models import VisionCapabilities
    monkeypatch.setattr(
        "llm_launchpad.core.vision.inspect_model_vision",
        lambda repo_id, revision=None: (VisionCapabilities(), {}),
    )


# The real implementations, captured before any fixture swaps them out so the
# interception wrappers below can delegate without recursing into themselves.
_REAL_SUBPROCESS_RUN = subprocess.run
_REAL_SUBPROCESS_POPEN = subprocess.Popen

_PROVIDER_CLI_NAMES = frozenset({"modal", "prime"})


def _provider_cli_command(command: object) -> list[str] | None:
    """Return ``command`` as argv when it invokes a provider CLI, else ``None``."""

    argv = command if isinstance(command, (list, tuple)) else [command]
    if not argv:
        return None
    try:
        decoded = [os.fsdecode(argument) for argument in argv]
    except TypeError:
        return None
    if Path(decoded[0]).name not in _PROVIDER_CLI_NAMES:
        return None
    return decoded


@pytest.fixture(autouse=True)
def _offline_provider_clis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer provider CLI calls offline instead of spawning the real binary.

    Every Modal call in the product shells out to the ``modal`` binary, and one
    ``modal volume ls`` costs a Python interpreter start plus a round trip to
    Modal. UI tests reach that path without asking for it -- the home screen
    alone fans out to storage, fleet, billing and profile -- so the suite spent
    most of its wall clock inside the Modal CLI, against the developer's own
    workspace, and a single storage worker joined at loop teardown could stall
    one test for over a minute.

    Only a command whose argv[0] *is* a provider CLI is answered here; the
    ``sh`` snippets the SSH tests verify, and every other local subprocess,
    still run for real. ``--json`` calls get an empty JSON document and the
    rest get empty output, which is what an account with nothing in it looks
    like. Tests that drive the CLI wrapper itself patch ``subprocess.run`` at
    this same seam, so their stub replaces this one.
    """

    def run(command: object, *args: object, **kwargs: object) -> object:
        argv = _provider_cli_command(command)
        if argv is None:
            return _REAL_SUBPROCESS_RUN(command, *args, **kwargs)
        output: str | bytes = "[]" if "--json" in argv[1:] else ""
        if not (kwargs.get("text") or kwargs.get("universal_newlines")):
            output = output.encode()
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=output,
            stderr=output[:0],
        )

    def popen(command: object, *args: object, **kwargs: object) -> object:
        argv = _provider_cli_command(command)
        if argv is None:
            return _REAL_SUBPROCESS_POPEN(command, *args, **kwargs)
        raise AssertionError(
            "A test started the provider CLI for real: "
            f"{' '.join(argv)}. Streaming CLI calls (deploy, logs) have to be "
            "stubbed by the test; they cost a live deployment otherwise."
        )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)


@pytest.fixture(autouse=True)
def _isolate_prime_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep Prime's own CLI config -- real API credentials -- out of the suite.

    ``load_prime_config`` reads ``~/.prime/config.json`` and the ``PRIME_*``
    environment, neither of which ``_isolate_user_settings`` covers. With a key
    resolved from either, a fleet refresh in a UI test calls the live Prime API
    as the developer. Pointing the loader at an empty directory leaves
    ``api_key`` blank, and ``PrimeBackend._request`` then fails before it opens
    a socket.
    """

    from llm_launchpad.core import prime_auth

    root = tmp_path_factory.mktemp("prime-config")
    monkeypatch.setattr(prime_auth, "PRIME_CONFIG_DIR", root)
    monkeypatch.setattr(prime_auth, "PRIME_CONFIG_PATH", root / "config.json")
    for name in (
        "PRIME_API_KEY",
        "PRIME_TEAM_ID",
        "PRIME_USER_ID",
        "PRIME_API_BASE_URL",
        "PRIME_BASE_URL",
        "PRIME_CONTEXT",
        "PRIME_SSH_KEY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _stub_modal_gpu_catalog_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep catalog and availability builds off Modal's documentation site.

    ``fetch_modal_gpu_catalog`` scrapes two modal.com pages. The parser and its
    caching have focused tests that patch ``requests`` inside ``modal_gpu``, so
    stub the imported reference in each consumer instead of the fetcher itself,
    leaving those tests their seam.
    """

    catalog = [ModalGpuSpec("A100-80GB", price_per_hour_usd=2.50)]
    for module_name in (
        "llm_launchpad.core.compute_availability",
        "llm_launchpad.core.quick_deploy_refresh",
    ):
        monkeypatch.setattr(f"{module_name}.fetch_modal_gpu_catalog", lambda: list(catalog))

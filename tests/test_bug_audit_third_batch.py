"""Behavioral reproductions for the third batch of ten bugs.

Each test states the observable behavior the fix restores, so a regression
reads as the symptom a user would report rather than as an internal detail.
"""

from __future__ import annotations

import importlib
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_launchpad.core import prime_disks
from llm_launchpad.core.job_store import HEARTBEAT_TIMEOUT_SECONDS, JobStore, JobStatus
from llm_launchpad.core.llamacpp_planner import (
    ATTENTION_SCRATCH_BUDGET_GB,
    ubatch_for_attention_scratch,
)
from llm_launchpad.core.prime_backend import PrimeBackend, PrimeDiskOffer
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    EndpointInfo,
    MemoryEstimate,
    PlacementAssessment,
    RuntimeTuning,
)
from llm_launchpad.tui.format import format_token_count, format_token_rate


# 1. Token counts shed digits at every decade except the first.


def test_a_count_that_rounds_up_into_the_next_unit_is_promoted() -> None:
    # The rule the formatter documents for 999,999 -> 1.0M, applied at the
    # boundary where it used to print a five-character "1,000" instead.
    assert format_token_count(999.4) == "999"
    assert format_token_count(999.6) == "1.0K"
    assert format_token_count(999_999) == "1.0M"
    assert format_token_rate(999.6) == "1.0K tok/s"
    assert format_token_rate(999.4) == "999 tok/s"


# 3. A physical batch that already fits the budget was shrunk anyway.


def test_a_batch_within_budget_is_left_at_the_size_it_was_asked_for() -> None:
    per_token = 0.001  # 2048 tokens is ~2 GB, far inside the budget.
    assert per_token * 2048 < ATTENTION_SCRATCH_BUDGET_GB

    assert ubatch_for_attention_scratch(per_token, max_ubatch=2048) == 2048
    # A cost of exactly nothing and a cost of nearly nothing now agree.
    assert ubatch_for_attention_scratch(0.0, max_ubatch=2048) == 2048
    # A batch that genuinely does not fit is still stepped down the ladder.
    assert ubatch_for_attention_scratch(0.5, max_ubatch=2048) == 64


# 4. The documented "skip" hydration mode could never be selected.


def test_an_empty_hydration_mode_selects_the_documented_skip_mode() -> None:
    module_name = "llm_launchpad.backends.modal_llamacpp_app"
    with patch.dict(os.environ, {"LLAMACPP_HYDRATION_MODE": ""}):
        module = importlib.reload(importlib.import_module(module_name))
        assert module.LLAMACPP_HYDRATION_MODE == ""
        assert module._should_hydrate_volume(None, []) is False
    # Restore the module to the ambient environment for the rest of the suite.
    importlib.reload(importlib.import_module(module_name))


def test_an_unset_hydration_mode_still_defaults_to_marker() -> None:
    module_name = "llm_launchpad.backends.modal_llamacpp_app"
    environment = {k: v for k, v in os.environ.items() if k != "LLAMACPP_HYDRATION_MODE"}
    with patch.dict(os.environ, environment, clear=True):
        module = importlib.reload(importlib.import_module(module_name))
        assert module.LLAMACPP_HYDRATION_MODE == "marker"
    importlib.reload(importlib.import_module(module_name))


# 9. A remembered Prime disk was reused without the headroom a new one gets.


def _config_with_weights(weights_gb: float) -> DeploymentConfig:
    config = DeploymentConfig()
    config.placement_assessment = PlacementAssessment(
        fingerprint="fingerprint",
        memory=MemoryEstimate(
            weights_gb=weights_gb,
            kv_cache_gb=0.0,
            compute_gb=0.0,
            speculative_gb=0.0,
            reserve_gb=0.0,
            total_gb=weights_gb,
        ),
        tuning=RuntimeTuning(),
    )
    return config


def _disk_offer() -> PrimeDiskOffer:
    return PrimeDiskOffer(
        cloud_id="cloud",
        provider_name="provider",
        data_center="dc",
        country="country",
        region="region",
        stock_status="Available",
        price_per_gb_hour=0.0,
        minimum_size_gb=None,
        maximum_size_gb=None,
        raw={},
    )


@pytest.mark.parametrize("weights_gb", [95.0, 99.0, 100.0, 240.0])
def test_a_disk_is_reused_only_at_the_size_a_new_one_would_be_created_at(
    weights_gb: float,
) -> None:
    required = prime_disks.cache_disk_size_gb(_disk_offer(), _config_with_weights(weights_gb))
    config = _config_with_weights(weights_gb)

    just_short = prime_disks.StoredPrimeDisk(id="disk", size_gb=required - 1)
    exact = prime_disks.StoredPrimeDisk(id="disk", size_gb=required)

    assert prime_disks._disk_fits_model(just_short, config) is False
    assert prime_disks._disk_fits_model(exact, config) is True


def test_a_deploy_with_no_weight_estimate_still_reuses_its_disk() -> None:
    disk = prime_disks.StoredPrimeDisk(id="disk", size_gb=prime_disks.PRIME_CACHE_DISK_SIZE_GB)
    assert prime_disks._disk_fits_model(disk, DeploymentConfig()) is True


# 10. Listing Prime disks stopped after one page, or crashed on a null total.


class _FakePrimeApi:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, object] | None] = []

    def _request(self, method: str, path: str, params: dict[str, object] | None = None):
        self.calls.append(params)
        index = len(self.calls) - 1
        return self.pages[index] if index < len(self.pages) else {"data": []}


def test_every_disk_page_is_read_when_the_account_reports_no_total() -> None:
    page = [{"id": f"disk-{index}"} for index in range(100)]
    api = _FakePrimeApi([{"data": page}, {"data": page}, {"data": page[:50]}])

    rows = PrimeBackend.list_disks(api)  # type: ignore[arg-type]

    assert len(rows) == 250, "a disk omitted here is a disk Launchpad forgets it is paying for"
    assert len(api.calls) == 3


def test_a_null_total_count_does_not_crash_the_disk_listing() -> None:
    page = [{"id": f"disk-{index}"} for index in range(100)]
    api = _FakePrimeApi([{"data": page, "total_count": None}, {"data": page[:10]}])

    rows = PrimeBackend.list_disks(api)  # type: ignore[arg-type]

    assert len(rows) == 110


# 7. A live worker that stopped reporting was never flagged.


def _pending_config(app_name: str) -> DeploymentConfig:
    config = DeploymentConfig()
    config.app_name = app_name
    return config


def test_a_worker_that_stops_heartbeating_is_flagged_without_being_ended(
    tmp_path,
) -> None:
    store = JobStore(tmp_path / "jobs.db")
    record = store.create_job(_pending_config("llamacpp-stalled"))
    assert store.claim_job(record.id, os.getpid())

    # Our own pid is alive; only the heartbeat has gone quiet.
    with patch.object(
        time, "time", return_value=time.time() + HEARTBEAT_TIMEOUT_SECONDS + 10
    ):
        store.reconcile_workers()

    updated = store.get_job(record.id)
    assert updated is not None
    assert updated.status == JobStatus.RUNNING, "a live worker still owns its job"
    assert "not reported in" in updated.outcome


def test_a_job_that_was_never_claimed_is_not_called_stalled(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    record = store.create_job(_pending_config("llamacpp-unclaimed"))

    with patch.object(
        time, "time", return_value=time.time() + HEARTBEAT_TIMEOUT_SECONDS + 10
    ):
        store.reconcile_workers()

    updated = store.get_job(record.id)
    assert updated is not None
    assert updated.status == JobStatus.PENDING
    assert "not reported in" not in updated.outcome


# 5, 6. Fast Deploy's model list explained the wrong absence, and its search
# box could not be put away again.


def _catalog_model():
    from llm_launchpad.core.quick_deploy import (
        QuickDeployProfile,
        list_quick_deploy_recipes,
    )
    from llm_launchpad.tui.screens.fast_deploy import QuickDeployModel

    profile = QuickDeployProfile(
        id="test-model",
        display_name="Test Model",
        repo_id="acme/test-model-GGUF",
        quant="Q4_K_M",
        gpu_type="H100",
        gpu_count=1,
        profile_label="Test",
        approx_cost_per_hour_usd=4.0,
        max_context_tokens=32768,
        instance_slug_hint="test-model",
        summary="Summary for test-model.",
        server_args=(),
        required_vram_gb=100.0,
        backend=BackendType.LLAMACPP,
    )
    return QuickDeployModel(
        id=profile.id,
        display_name=profile.display_name,
        recipes=list_quick_deploy_recipes((profile,)),
        profiles=(profile,),
        max_context_tokens=profile.max_context_tokens,
        quality_score=12.5,
    )


class FastDeployModelSearchTests(unittest.IsolatedAsyncioTestCase):
    """Drive the real screen: both defects are only visible once it is rendered."""

    @staticmethod
    def _app():
        from textual.app import App

        class _App(App[None]):
            _username = "alice"

            def begin_quick_deploy_catalog_refresh(self, *args: object, **kwargs: object) -> None:
                return None

        return _App()

    async def test_a_search_with_no_matches_names_the_search_not_the_catalog(self) -> None:
        from textual.widgets import Input, OptionList

        from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen

        app = self._app()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(_catalog_model(),),
        ):
            async with app.run_test(size=(140, 40)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()
                screen = app.screen
                search = screen.query_one("#fast-deploy-model-search", Input)
                search.value = "zzzz-no-such-model"
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                prompts = " ".join(
                    str(option_list.get_option_at_index(index).prompt)
                    for index in range(option_list.option_count)
                )
                # "No models in the quick-deploy catalog" sends the reader off
                # to rebuild a catalog that is fine.
                self.assertNotIn("No models in the quick-deploy catalog", prompts)
                self.assertIn("zzzz-no-such-model", prompts)

    async def test_an_unmatched_gpu_filter_still_names_the_gpu(self) -> None:
        from textual.widgets import OptionList

        from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen

        app = self._app()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(_catalog_model(),),
        ):
            async with app.run_test(size=(140, 40)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()
                screen = app.screen
                screen._gpu_filter = "NOT-A-GPU"
                screen._render_model_list()
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                prompts = " ".join(
                    str(option_list.get_option_at_index(index).prompt)
                    for index in range(option_list.option_count)
                )
                self.assertIn("NOT-A-GPU", prompts)

    async def test_search_box_stays_visible_at_every_supported_size(self) -> None:
        from textual.widgets import Input

        from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen

        for size in ((140, 40), (80, 24), (60, 20)):
            with self.subTest(size=size):
                app = self._app()
                with patch(
                    "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
                    return_value=(_catalog_model(),),
                ):
                    async with app.run_test(size=size) as pilot:
                        app.push_screen(FastDeployScreen())
                        await pilot.pause()
                        screen = app.screen
                        search = screen.query_one("#fast-deploy-model-search", Input)
                        self.assertTrue(search.display)
                        self.assertGreater(search.region.height, 0)

    async def test_search_shortcut_focuses_the_visible_box(self) -> None:
        from textual.widgets import Input

        from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen

        app = self._app()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(_catalog_model(),),
        ):
            async with app.run_test(size=(80, 24)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()
                screen = app.screen
                screen.action_focus_model_search()
                await pilot.pause()
                self.assertTrue(
                    screen.query_one("#fast-deploy-model-search", Input).has_focus
                )


# 8. The Operations screen reconciled durable jobs on its repaint timer.


class OperationsReconcileCadenceTests(unittest.TestCase):
    def test_a_repaint_does_not_run_job_reconciliation(self) -> None:
        from llm_launchpad.tui.screens.operations import DeploymentJobsPanel

        calls: list[str] = []

        class _Store:
            def reconcile_workers(self) -> list[object]:
                calls.append("reconcile")
                return []

            def import_journal_entries(self) -> list[object]:
                calls.append("import")
                return []

            def list_jobs(self, *args: object, **kwargs: object) -> list[object]:
                return []

        store = _Store()
        # ``_durable_jobs`` reads only ``self.app``; a stand-in keeps this off
        # the event loop, where the cost being measured does not apply.
        stub = SimpleNamespace(
            app=SimpleNamespace(deployment_jobs={}, _get_job_store=lambda: store)
        )

        DeploymentJobsPanel._durable_jobs(stub)
        self.assertEqual(calls, [], "repainting twice a second must not write to SQLite")

        DeploymentJobsPanel._durable_jobs(stub, reconcile=True)
        self.assertEqual(calls, ["reconcile", "import"])


# 11, 12, 13. Cancelling a durable deployment left resources running.


class _CancelOrchestrator:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.calls: list[tuple[str | None, str | None]] = []

    def stop_app(self, backend, app_name=None, app_id=None, provider=None):  # type: ignore[no-untyped-def]
        from llm_launchpad.protocol.enums import OperationType as _OperationType
        from llm_launchpad.protocol.events import OperationCompleteEvent as _Complete

        self.calls.append((app_name, app_id))
        yield _Complete(operation=_OperationType.STOP, success=self.success)


def _cancel_job(tmp_path, provider, *, resource_id=None, allocated=True, banked=None):
    from llm_launchpad.core import job_runner
    from llm_launchpad.core.job_store import JobStore as _JobStore

    store = _JobStore(tmp_path / f"jobs-{provider.value}-{allocated}-{banked}.db")
    config = DeploymentConfig(backend=BackendType.LLAMACPP, provider=provider)
    config.app_name = f"llamacpp-{provider.value}"
    record = store.create_job(config)
    if banked:
        store.set_resource(record.id, banked)
    orchestrator = _CancelOrchestrator()
    with patch.object(job_runner, "_orchestrator", lambda: orchestrator):
        job_runner._cancel_with_cleanup(
            record.id, store, config, resource_id,
            "Cancelled; cleaning up allocated resource.", allocated=allocated,
        )
    return orchestrator, store.get_job(record.id), store.get_events(record.id)


def test_cancelling_a_modal_deploy_stops_the_app_it_published(tmp_path) -> None:
    # Modal emits no allocation event -- its app is addressed by name -- so
    # requiring an id skipped teardown and reported nothing had been allocated
    # while the app stayed published.
    orchestrator, record, _ = _cancel_job(tmp_path, ComputeProvider.MODAL)

    assert orchestrator.calls == [("llamacpp-modal", None)]
    assert record is not None and record.outcome == "Cancelled; resource stopped."


def test_cancelling_a_vast_deploy_destroys_the_rental_by_name(tmp_path) -> None:
    orchestrator, record, _ = _cancel_job(tmp_path, ComputeProvider.VAST)

    assert orchestrator.calls == [("llamacpp-vast", None)]
    assert record is not None and record.outcome == "Cancelled; resource stopped."


def test_cancelling_before_the_deploy_starts_stops_nothing(tmp_path) -> None:
    orchestrator, record, _ = _cancel_job(tmp_path, ComputeProvider.MODAL, allocated=False)

    assert orchestrator.calls == []
    assert record is not None
    assert record.outcome == "Cancelled before allocating a resource."


def test_prime_without_a_pod_id_still_reports_nothing_allocated(tmp_path) -> None:
    # Prime termination genuinely needs a pod id, and Prime emits one the
    # moment the pod exists -- so no id here really does mean no pod.
    orchestrator, record, _ = _cancel_job(tmp_path, ComputeProvider.PRIME)

    assert orchestrator.calls == []
    assert record is not None
    assert record.outcome == "Cancelled before allocating a resource."


def test_a_warmup_cancellation_recovers_the_id_the_deploy_banked(tmp_path) -> None:
    # The warmup call site passes whatever endpoint it was handed, which
    # carries no app_id when the pod arrived via ResourceAllocatedEvent alone.
    # Throwing the banked id away left a live Prime pod billing.
    orchestrator, record, _ = _cancel_job(
        tmp_path, ComputeProvider.PRIME, resource_id=None, banked="pod-4242"
    )

    assert orchestrator.calls == [("llamacpp-prime", "pod-4242")]
    assert record is not None and record.outcome == "Cancelled; resource stopped."


def test_the_cancellation_says_which_wait_it_interrupted(tmp_path) -> None:
    from llm_launchpad.protocol.events import LogEvent as _LogEvent

    _, _, events = _cancel_job(tmp_path, ComputeProvider.MODAL)

    lines = [row.event.line for row in events if isinstance(row.event, _LogEvent)]
    # The caller's phrase was accepted and discarded, so every cancellation
    # read the same whenever it happened.
    assert "Cancelled; cleaning up allocated resource." in lines


# 14. An installed-but-unauthenticated Modal CLI opened the setup gate.


def _readiness_env(*, modal_authed: bool, prime_key: str = "", vast_key: str = ""):
    from llm_launchpad.core import provider_readiness as readiness
    from llm_launchpad.core.modal_auth import ModalAuthStatus

    readiness.clear_provider_readiness_cache()
    status = ModalAuthStatus(
        authenticated=modal_authed,
        error=None if modal_authed else "not authenticated",
    )
    return readiness, (
        patch.object(readiness.ModalBackend, "is_cli_available", return_value=True),
        patch.object(readiness, "get_modal_auth_status", return_value=status),
        patch.object(
            readiness, "load_prime_config",
            return_value=SimpleNamespace(api_key=prime_key),
        ),
        patch.object(
            readiness, "resolve_vast_credentials",
            return_value=SimpleNamespace(api_key=vast_key, source="env"),
        ),
    )


def test_an_unauthenticated_modal_cli_is_not_a_credential() -> None:
    # The module's own premise is that installation and credential presence
    # are different states; collapsing them let an executable on PATH stand in
    # for a login.
    readiness, patches = _readiness_env(modal_authed=False)
    with patches[0], patches[1], patches[2], patches[3]:
        stage = readiness.check_modal_readiness(verify=False).stage
        assert stage is readiness.ProviderReadinessStage.MISSING_CREDENTIALS
        readiness.clear_provider_readiness_cache()
        assert readiness.has_provider_credentials() is False
    readiness.clear_provider_readiness_cache()


def test_an_authenticated_modal_cli_still_opens_the_gate() -> None:
    readiness, patches = _readiness_env(modal_authed=True)
    with patches[0], patches[1], patches[2], patches[3]:
        stage = readiness.check_modal_readiness(verify=False).stage
        assert stage is readiness.ProviderReadinessStage.CREDENTIALS_PRESENT
        readiness.clear_provider_readiness_cache()
        assert readiness.has_provider_credentials() is True
    readiness.clear_provider_readiness_cache()


@pytest.mark.parametrize("prime_key,vast_key", [("pk-1", ""), ("", "vk-1")])
def test_another_providers_stored_key_still_opens_the_gate(
    prime_key: str, vast_key: str
) -> None:
    readiness, patches = _readiness_env(
        modal_authed=False, prime_key=prime_key, vast_key=vast_key
    )
    with patches[0], patches[1], patches[2], patches[3]:
        assert readiness.has_provider_credentials() is True
    readiness.clear_provider_readiness_cache()


# 15. The durable worker recorded no startup-phase timings.


def test_a_background_deploy_records_its_startup_phases(tmp_path) -> None:
    from llm_launchpad.core import job_runner
    from llm_launchpad.core.deploy_log_summary import parse_startup_phase_line
    from llm_launchpad.core.job_store import JobStore as _JobStore
    from llm_launchpad.protocol.enums import OperationType
    from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent

    store = _JobStore(tmp_path / "jobs.db")
    config = DeploymentConfig(backend=BackendType.LLAMACPP, provider=ComputeProvider.MODAL)
    config.app_name = "llamacpp-phases"
    config.do_deploy = True
    config.do_warmup = True
    record = store.create_job(config)

    class _Orchestrator:
        def deploy(self, _config):  # type: ignore[no-untyped-def]
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY,
                success=True,
                data=EndpointInfo(name="llamacpp-phases", web_url="https://x.modal.run"),
            )

        def warmup(self, backend, url, timeout, tail_logs, **kwargs):  # type: ignore[no-untyped-def]
            timer = kwargs.get("phase_timer")
            assert timer is not None, "the worker must hand the runner its timer"
            timer.ready()
            yield from timer.finish_events()
            yield OperationCompleteEvent(
                operation=OperationType.WARMUP, success=True, data={"url": url}
            )

    with patch.object(job_runner, "_save_connection", lambda *a, **k: None):
        job_runner._run_attempt_phases(record.id, store, _Orchestrator(), config, None)

    phases = [
        parsed
        for row in store.get_events(record.id)
        if isinstance(row.event, LogEvent)
        and (parsed := parse_startup_phase_line(row.event.line)) is not None
    ]
    # Deploys run in this worker by default, so it was the one path that
    # measured nothing while the CLI and the in-session fallback both did.
    assert [name for name, _ in phases] == ["deploy", "warmup-wait", "total"]


# 16. A timed-out provider fetch was not abandoned; it held the process open.


def test_a_provider_that_never_answers_is_dropped_not_deferred() -> None:
    import threading as _threading

    import llm_launchpad.core.compute_availability as availability

    release = _threading.Event()
    running = _threading.Event()

    def _hung_catalog():  # type: ignore[no-untyped-def]
        running.set()
        release.wait(timeout=30)
        return []

    try:
        with (
            patch.object(availability, "COMPUTE_AVAILABILITY_TIMEOUT_SECONDS", 0.2),
            patch.object(availability, "resolve_modal_cli_path", return_value="/usr/bin/modal"),
            patch.object(availability, "fetch_modal_gpu_catalog", _hung_catalog),
            patch.object(availability, "get_prime_auth_status", side_effect=Exception("skip")),
            patch.object(availability, "resolve_vast_credentials", side_effect=ValueError("skip")),
        ):
            started = time.monotonic()
            snapshot = availability.load_compute_availability()
            elapsed = time.monotonic() - started

        assert running.is_set(), "the fetch must actually have been started"
        assert elapsed < 5.0, "the shared budget must bound the wait"
        assert any("did not answer" in error for error in snapshot.errors)

        # The thread carrying the abandoned fetch must be a daemon: a pooled
        # worker is joined by the interpreter at exit however it was shut down,
        # so the wait was not removed, only moved to quitting time.
        carriers = [
            thread
            for thread in _threading.enumerate()
            if thread.is_alive() and not thread.daemon and thread is not _threading.main_thread()
        ]
        assert carriers == [], f"non-daemon threads still holding the process: {carriers}"
    finally:
        release.set()


# 17. A mixture-of-experts repository was sized by one expert.


def test_an_expert_by_size_repo_id_is_not_read_as_one_expert() -> None:
    from llm_launchpad.core.artificial_analysis import _parameter_count_from_name
    from llm_launchpad.core.hf_models import _parse_parameter_count_from_repo_id

    # "Mixtral-8x7B" is not a 7B model. The benchmark feed's own parser has
    # always read this form; the repository-id fallback did not, so the two
    # disagreed by a factor of eight on the best-known MoE family.
    assert _parse_parameter_count_from_repo_id("mistralai/Mixtral-8x7B-Instruct-v0.1") == 56e9
    assert _parse_parameter_count_from_repo_id("mistralai/Mixtral-8x22B-v0.1") == 176e9
    assert _parameter_count_from_name("Mixtral 8x7B") == 56.0


@pytest.mark.parametrize(
    "repo_id,expected_b",
    [
        ("unsloth/Qwen3-8B-GGUF", 8.0),
        ("org/Llama-3.1-405B-Instruct", 405.0),
        ("org/GLM-4.6-355B-A32B", 355.0),
        ("org/Qwen3-30B-A3B", 30.0),
        ("org/Llama-4-Scout-17B-16E", 17.0),
        ("org/model-1.5b-gguf", 1.5),
    ],
)
def test_ordinary_repo_ids_are_unchanged(repo_id: str, expected_b: float) -> None:
    from llm_launchpad.core.hf_models import _parse_parameter_count_from_repo_id

    assert _parse_parameter_count_from_repo_id(repo_id) == pytest.approx(expected_b * 1e9)


# 18. Ternary quantizations were ranked as full quality.


def test_a_ternary_quantization_is_recognised_as_reduced() -> None:
    from llm_launchpad.core.quant_quality import (
        is_reduced_quality,
        quant_bits,
        quant_quality_label,
        serving_quality_bits,
    )

    # TQ1_0 is llama.cpp ternary at about 1.69 bits per weight. Unrecognised
    # labels are scored at the quality floor -- deliberately, so an unknown one
    # is not buried -- which ranked this level with Q4_K_M and disclosed
    # nothing, the exact degradation the module exists to surface.
    assert quant_bits("TQ1_0") == 1
    assert quant_bits("TQ2_0") == 2
    assert quant_bits("UD-TQ1_0") == 1
    assert is_reduced_quality("TQ1_0") is True
    assert quant_quality_label("TQ1_0") == "1-bit"
    assert serving_quality_bits("TQ1_0") < serving_quality_bits("Q4_K_M")


@pytest.mark.parametrize(
    "quant,bits",
    [("Q4_K_M", 4), ("UD-Q4_K_XL", 4), ("IQ4_XS", 4), ("Q2_K", 2), ("Q8_0", 8), ("BF16", 16)],
)
def test_known_quantizations_keep_their_width(quant: str, bits: int) -> None:
    from llm_launchpad.core.quant_quality import quant_bits

    assert quant_bits(quant) == bits


def test_an_unrecognised_label_still_gets_the_benefit_of_the_doubt() -> None:
    from llm_launchpad.core.quant_quality import (
        QUALITY_FLOOR_BITS,
        is_reduced_quality,
        quant_bits,
        serving_quality_bits,
    )

    assert quant_bits("some-new-format") is None
    assert serving_quality_bits("some-new-format") == QUALITY_FLOOR_BITS
    assert is_reduced_quality("some-new-format") is False


# 19. The placement list claimed a cost ordering it did not have.


class InfraSortBasisTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_status_does_not_claim_an_ordering_the_list_lacks(self) -> None:
        from textual.widgets import Static

        from llm_launchpad.core.compute_availability import aggregate_compute_availability
        from llm_launchpad.core.modal_gpu import ModalGpuSpec
        from llm_launchpad.tui.screens.fast_deploy import (
            FastDeployAvailabilityLoaded,
            FastDeployScreen,
        )

        model = _catalog_model()
        snapshot = aggregate_compute_availability(
            modal_catalog=[
                ModalGpuSpec("H100", price_per_hour_usd=4.0),
                ModalGpuSpec("L40S", price_per_hour_usd=2.0),
            ]
        )
        app = FastDeployModelSearchTests._app()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test(size=(140, 40)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()
                screen = app.screen
                screen._selected_model = model
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot)
                )
                await pilot.pause()
                status = str(screen.query_one("#fast-deploy-status", Static).renderable)

        # Placements sort on their assessment, which the status already calls
        # "best full-context throughput first". A second line asserting they
        # were sorted by a monthly cost contradicted it, and the rows visibly
        # disagreed with whichever line named a price.
        assert "Sorted by" not in status
        # The scenario the per-row monthly figure is computed under is still
        # named, because that number means nothing without its assumptions.
        assert "Workday 8h" in status
        assert "storage separate" in status


def test_no_helper_claims_to_name_a_cost_ordering() -> None:
    from llm_launchpad.core import inference_options

    # The retired helper's contract was "name the cost basis placement
    # ordering uses"; no list is ordered on a cost basis, so the contract
    # could not be met by any implementation.
    assert not hasattr(inference_options, "cost_sort_basis_label")

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
from llm_launchpad.tui.screens.main_menu import _runtime_display


# 1. The fleet panel said "checked 12m ago ago".


def test_the_health_age_is_not_followed_by_a_second_ago() -> None:
    row = EndpointInfo(
        name="vllm-qwen",
        app_id="ap-1",
        state="deployed",
        backend=BackendType.VLLM,
        provider=ComputeProvider.MODAL,
    )
    row.runtime_status = "healthy"
    row.runtime_checked_at = 1_000.0

    minutes = _runtime_display(row, now=1_000.0 + 750)
    moments = _runtime_display(row, now=1_000.0 + 10)

    assert "12m ago ago" not in minutes
    assert "(checked 12m ago)" in minutes
    assert "just now ago" not in moments
    assert "(checked just now)" in moments


# 2. Token counts shed digits at every decade except the first.


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

    async def test_leaving_an_empty_search_gives_its_rows_back(self) -> None:
        from textual.widgets import Input

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
                screen.action_focus_model_search()
                await pilot.pause()
                self.assertTrue(screen.has_class("search-open"))

                search = screen.query_one("#fast-deploy-model-search", Input)
                screen.on_input_submitted(Input.Submitted(search, ""))
                await pilot.pause()

                # The CSS calls this a temporary control; nothing used to clear
                # it, so one press of "/" shortened the model list for the rest
                # of the step.
                self.assertFalse(screen.has_class("search-open"))

    async def test_a_search_with_matches_keeps_the_box_open(self) -> None:
        from textual.widgets import Input

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
                screen.action_focus_model_search()
                search = screen.query_one("#fast-deploy-model-search", Input)
                search.value = "test"
                await pilot.pause()

                screen.on_input_submitted(Input.Submitted(search, "test"))
                await pilot.pause()

                self.assertTrue(screen.has_class("search-open"))


# 8. The Operations screen reconciled durable jobs on its repaint timer.


class OperationsReconcileCadenceTests(unittest.TestCase):
    def test_a_repaint_does_not_run_job_reconciliation(self) -> None:
        from llm_launchpad.tui.screens.operations import OperationsScreen

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

        OperationsScreen._durable_jobs(stub)
        self.assertEqual(calls, [], "repainting twice a second must not write to SQLite")

        OperationsScreen._durable_jobs(stub, reconcile=True)
        self.assertEqual(calls, ["reconcile", "import"])

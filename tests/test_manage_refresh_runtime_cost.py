"""Stale refresh status, passive labels, runclock, and cumulative cost."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.coordinate import Coordinate
from textual.widgets import Static

from llm_launchpad.core.endpoint_runtime import EndpointRuntimeTracker
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, ServingSnapshot, ServingStats
from llm_launchpad.tui.screens.manage import (
    _ENDPOINT_COLUMNS,
    _endpoint_cost,
    _endpoint_run_time,
    _serving_detail_lines,
    _serving_observed_age_label,
)


def _serving(
    *,
    total_prompt: float = 0.0,
    total_generation: float = 0.0,
    observed_at: float | None = None,
    runtime_started_at: float | None = None,
    tokens_per_second: float | None = None,
) -> ServingSnapshot:
    return ServingSnapshot(
        stats=ServingStats(captured_at=1.0, runtime_started_at=runtime_started_at),
        total_prompt_tokens=total_prompt,
        total_generation_tokens=total_generation,
        tokens_per_second=tokens_per_second,
        observed_at=observed_at,
    )


def _row(**overrides: object) -> EndpointInfo:
    fields: dict[str, object] = {
        "name": "vllm-qwen",
        "app_id": "ap-1",
        "state": "deployed",
        "backend": BackendType.VLLM,
        "provider": ComputeProvider.MODAL,
    }
    fields.update(overrides)
    return EndpointInfo(**fields)  # type: ignore[arg-type]


class StaleRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_cache_renders_refreshing_not_refreshed(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.workers import EndpointsLoaded

        row = _row()

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[row], is_stale=True))  # type: ignore[attr-defined]

        app = _App()
        async with app.run_test(size=(140, 24)) as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            await screen.workers.wait_for_complete()
            await pilot.pause()
            status = screen.query_one("#manage-status", Static)
            self.assertIn("Refreshing", str(status.renderable))
            self.assertNotIn("Fleet refreshed", str(status.renderable))

            # The live pass clears the refreshing state.
            screen.post_message(EndpointsLoaded(rows=[row], is_stale=False))
            await pilot.pause()
            await screen.workers.wait_for_complete()
            await pilot.pause()
            status = screen.query_one("#manage-status", Static)
            self.assertIn("Fleet refreshed", str(status.renderable))

    async def test_failed_refresh_clears_refreshing_state(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.workers import EndpointsFailed, EndpointsLoaded

        row = _row()

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[row], is_stale=True))  # type: ignore[attr-defined]

        app = _App()
        async with app.run_test(size=(140, 24)) as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            await screen.workers.wait_for_complete()
            await pilot.pause()
            self.assertTrue(screen._fleet_refresh_inflight)
            screen.post_message(EndpointsFailed(error="boom"))
            await pilot.pause()
            self.assertFalse(screen._fleet_refresh_inflight)
            status = screen.query_one("#manage-status", Static)
            self.assertIn("failed", str(status.renderable).lower())


class PassiveLabelTests(unittest.TestCase):
    def test_banked_total_carries_its_observation_age(self) -> None:
        row = _row(serving=_serving(total_generation=1200.0, observed_at=1000.0))
        self.assertEqual(_serving_observed_age_label(row, now=1005.0), "just now")
        self.assertEqual(_serving_observed_age_label(row, now=1600.0), "10m ago")
        detail = _serving_detail_lines(row, now=1600.0)
        self.assertIn("1.2K tokens served", detail)
        self.assertIn("observed 10m ago", detail)

    def test_never_collected_total_says_so(self) -> None:
        self.assertIsNone(_serving_observed_age_label(_row()))
        detail = _serving_detail_lines(_row())
        self.assertIn("not collected yet", detail)

    def test_run_and_cost_columns_exist_on_wide(self) -> None:
        by_key = {column.key: column for column in _ENDPOINT_COLUMNS}
        self.assertIn("run", by_key)
        self.assertIn("cost", by_key)


class ExplicitLiveFetchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from llm_launchpad.core.runtime_health import reset as reset_health
        from llm_launchpad.core.serving_metrics import default_tracker

        default_tracker().reset()
        reset_health()

    async def test_explicit_fetch_updates_modal_totals(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
        from llm_launchpad.tui.workers import EndpointsLoaded

        cached = _row(web_url="https://example.modal.run")
        metrics = "llamacpp:prompt_tokens_total 1200\nllamacpp:tokens_predicted_total 3400\n"

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[cached]))  # type: ignore[attr-defined]

        # Background passes stay passive: no totals yet.
        app = _App()
        async with app.run_test(size=(140, 24)) as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            await screen.workers.wait_for_complete()
            await pilot.pause()
            table = screen.query_one("#manage-endpoint-table", AdaptiveDataTable)
            served_col = table.visible_column_keys.index("served")
            self.assertEqual(table.get_cell_at(Coordinate(0, served_col)), "-")

            # An explicit fetch is allowed to wake the container.
            fake_requests = types.SimpleNamespace(
                get=lambda *a, **k: types.SimpleNamespace(status_code=200, text=metrics)
            )
            # The cached row is llama.cpp-shaped for this probe.
            cached.backend = BackendType.LLAMACPP
            with patch.dict(sys.modules, {"requests": fake_requests}):
                screen.action_fetch_live_selected()
                await screen.workers.wait_for_complete()
                await pilot.pause()
            served = table.get_cell_at(Coordinate(0, served_col))
            self.assertEqual(served, "4.6K")
            self.assertIsNone(cached.live_metrics_error)
            self.assertIsNotNone(cached.live_metrics_checked_at)

    async def test_failed_fetch_keeps_banked_totals_and_reports_error(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.workers import EndpointsLoaded

        cached = _row(web_url="https://example.modal.run")

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[cached]))  # type: ignore[attr-defined]

        fake_requests = types.SimpleNamespace(
            get=lambda *a, **k: types.SimpleNamespace(status_code=500, text="")
        )
        app = _App()
        with patch.dict(sys.modules, {"requests": fake_requests}):
            async with app.run_test(size=(140, 24)) as pilot:
                app.push_screen(ManageScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ManageScreen)
                await screen.workers.wait_for_complete()
                await pilot.pause()
                # Seed banked totals after the passive background pass (which
                # re-attaches the tracker's snapshot, initially empty).
                cached.serving = _serving(total_generation=1200.0, observed_at=1000.0)
                total_before = cached.serving.total_tokens
                screen.action_fetch_live_selected()
                await screen.workers.wait_for_complete()
                await pilot.pause()
                assert cached.serving is not None
                self.assertEqual(cached.serving.total_tokens, total_before)
                assert cached.live_metrics_error is not None
                detail = screen.query_one("#manage-selection-detail", Static)
                self.assertIn("did not return metrics", str(detail.renderable))


class RuntimeTrackerTests(unittest.TestCase):
    def _tracker(self) -> tuple[EndpointRuntimeTracker, Path]:
        tmp = Path(tempfile.mkdtemp())
        return EndpointRuntimeTracker(path=tmp / "runtime.json"), tmp

    def test_current_run_resets_but_endpoint_total_survives(self) -> None:
        tracker, _ = self._tracker()
        row = _row(
            provider=ComputeProvider.PRIME,
            hourly_cost_usd=2.0,
            serving=_serving(runtime_started_at=900.0),
        )
        tracker.observe_rows([row], now=1000.0, explicit=True)
        self.assertEqual(row.run_started_at_epoch, 900.0)
        first_total = row.cumulative_cost_usd

        # A restart changes the process gauge: new run, same endpoint ledger.
        row2 = _row(
            provider=ComputeProvider.PRIME,
            hourly_cost_usd=2.0,
            serving=_serving(runtime_started_at=1500.0),
        )
        tracker.observe_rows([row2], now=1600.0, explicit=True)
        self.assertEqual(row2.run_started_at_epoch, 1500.0)
        assert row2.cumulative_cost_usd is not None and first_total is not None
        self.assertGreaterEqual(row2.cumulative_cost_usd, first_total)
        self.assertEqual(_endpoint_run_time(row2, now=1660.0), "2m40s")

    def test_unknown_run_start_never_invents_uptime(self) -> None:
        tracker, _ = self._tracker()
        row = _row(provider=ComputeProvider.MODAL)
        tracker.observe_rows([row], now=1000.0, explicit=False)
        self.assertIsNone(row.run_started_at_epoch)
        self.assertEqual(_endpoint_run_time(row, now=2000.0), "-")
        detail = _serving_detail_lines(row, now=2000.0)
        self.assertIn("unknown", detail)
        self.assertNotIn("33m", detail)

    def test_modal_passive_never_accrues_but_explicit_does(self) -> None:
        tracker, _ = self._tracker()
        row = _row(provider=ComputeProvider.MODAL, hourly_cost_usd=2.0)
        tracker.observe_rows([row], now=1000.0, explicit=False)
        tracker.observe_rows([row], now=2000.0, explicit=False)
        self.assertIsNone(row.cumulative_cost_usd)

        live = _row(
            provider=ComputeProvider.MODAL,
            hourly_cost_usd=2.0,
            serving=_serving(runtime_started_at=1950.0),
        )
        tracker.observe_rows([live], now=2000.0, explicit=True)
        tracker.observe_rows([live], now=2060.0, explicit=True)
        assert live.cumulative_cost_usd is not None
        self.assertGreater(live.cumulative_cost_usd, 0)
        self.assertIn("~" if live.cost_estimated else "$", _endpoint_cost(live))

    def test_provisioned_cost_accumulates_and_persists(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "runtime.json"
        tracker = EndpointRuntimeTracker(path=path)
        row = _row(provider=ComputeProvider.PRIME, hourly_cost_usd=3.6)
        tracker.observe_rows([row], now=1000.0, explicit=False)
        tracker.observe_rows([row], now=4600.0, explicit=False)
        assert row.cumulative_cost_usd is not None
        self.assertAlmostEqual(row.cumulative_cost_usd, 3.6, places=2)
        self.assertTrue(row.cost_estimated)  # hour-wide gap is inferred continuity

        reloaded = EndpointRuntimeTracker(path=path)
        row2 = _row(provider=ComputeProvider.PRIME, hourly_cost_usd=3.6)
        reloaded.observe_rows([row2], now=4660.0, explicit=False)
        assert row2.cumulative_cost_usd is not None
        assert row.cumulative_cost_usd is not None
        self.assertGreater(row2.cumulative_cost_usd, row.cumulative_cost_usd)

    def test_missing_rate_keeps_cost_unknown_not_free(self) -> None:
        tracker, _ = self._tracker()
        row = _row(provider=ComputeProvider.PRIME)
        tracker.observe_rows([row], now=1000.0, explicit=False)
        tracker.observe_rows([row], now=2000.0, explicit=False)
        self.assertIsNone(row.cumulative_cost_usd)
        self.assertEqual(_endpoint_cost(row), "-")
        detail = _serving_detail_lines(row, now=2000.0)
        self.assertIn("hourly rate unavailable", detail)

    def test_rate_change_marks_total_estimated(self) -> None:
        tracker, _ = self._tracker()
        row = _row(provider=ComputeProvider.PRIME, hourly_cost_usd=1.0)
        tracker.observe_rows([row], now=1000.0, explicit=False)
        tracker.observe_rows([row], now=2000.0, explicit=False)
        row.hourly_cost_usd = 2.0
        tracker.observe_rows([row], now=2600.0, explicit=False)
        self.assertTrue(row.cost_estimated)


if __name__ == "__main__":
    unittest.main()

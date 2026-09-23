"""Cover how live serving traffic reaches the fleet panel and manage screen."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock, patch

from textual.coordinate import Coordinate
from textual.widgets import Static

from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, ServingSnapshot, ServingStats
from llm_launchpad.tui.format import format_token_count, format_token_rate
from llm_launchpad.tui.screens.main_menu import (
    _render_deployment_status,
    _serving_panel_line,
)
from llm_launchpad.tui.screens.manage import (
    _ENDPOINT_COLUMNS,
    _endpoint_tokens_served,
    _endpoint_throughput,
    _serving_detail_lines,
)
from llm_launchpad.tui.responsive import WidthMode


def _serving(**kwargs: object) -> ServingSnapshot:
    stats_fields = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key not in {"total_prompt_tokens", "total_generation_tokens", "tokens_per_second"}
    }
    return ServingSnapshot(
        stats=ServingStats(captured_at=1.0, **stats_fields),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _row(serving: ServingSnapshot | None = None, **overrides: object) -> EndpointInfo:
    fields: dict[str, object] = {
        "name": "vllm-qwen",
        "app_id": "ap-1",
        "state": "deployed",
        "backend": BackendType.VLLM,
        "provider": ComputeProvider.MODAL,
        "serving": serving,
    }
    fields.update(overrides)
    return EndpointInfo(**fields)  # type: ignore[arg-type]


class TokenFormattingTests(unittest.TestCase):
    def test_counts_shed_digits_as_they_grow(self) -> None:
        self.assertEqual(format_token_count(0), "0")
        self.assertEqual(format_token_count(812), "812")
        self.assertEqual(format_token_count(1_200), "1.2K")
        self.assertEqual(format_token_count(12_345), "12K")
        self.assertEqual(format_token_count(1_240_000), "1.2M")
        self.assertEqual(format_token_count(3_200_000_000), "3.2B")

    def test_a_count_that_rounds_up_promotes_to_the_next_unit(self) -> None:
        # "1,000K" is a unit that stopped doing its job.
        self.assertEqual(format_token_count(999_999), "1.0M")
        self.assertEqual(format_token_count(999_499), "999K")

    def test_rates_keep_a_decimal_only_where_it_means_something(self) -> None:
        self.assertEqual(format_token_rate(0), "0 tok/s")
        self.assertEqual(format_token_rate(4.73), "4.7 tok/s")
        self.assertEqual(format_token_rate(47.2), "47 tok/s")
        self.assertEqual(format_token_rate(1250), "1.2K tok/s")

    def test_negative_readings_never_reach_the_screen(self) -> None:
        self.assertEqual(format_token_count(-5), "0")
        self.assertEqual(format_token_rate(-5), "0 tok/s")


class ManageTrafficColumnTests(unittest.TestCase):
    def test_traffic_columns_are_offered_only_where_there_is_room(self) -> None:
        by_key = {column.key: column for column in _ENDPOINT_COLUMNS}
        self.assertEqual(by_key["throughput"].modes, frozenset({WidthMode.WIDE}))
        self.assertEqual(by_key["served"].modes, frozenset({WidthMode.WIDE}))

    def test_cells_render_the_measured_rate_and_lifetime_total(self) -> None:
        row = _row(
            _serving(
                total_prompt_tokens=940_000,
                total_generation_tokens=302_400,
                tokens_per_second=47.2,
            ),
            provider=ComputeProvider.PRIME,
        )
        self.assertEqual(_endpoint_throughput(row), "47 tok/s")
        self.assertEqual(_endpoint_tokens_served(row), "1.2M")

    def test_modal_passive_hides_the_rate_but_keeps_the_total(self) -> None:
        # A passively monitored Modal row holds banked totals without a live
        # rate: showing "-" is honest, showing a stale rate would imply a
        # probe that never happened.
        row = _row(
            _serving(
                total_prompt_tokens=940_000,
                total_generation_tokens=302_400,
                tokens_per_second=47.2,
            ),
            provider=ComputeProvider.MODAL,
        )
        self.assertEqual(_endpoint_throughput(row), "-")
        self.assertEqual(_endpoint_tokens_served(row), "1.2M")

    def test_an_endpoint_with_no_reading_yet_shows_a_dash(self) -> None:
        self.assertEqual(_endpoint_throughput(_row()), "-")
        self.assertEqual(_endpoint_tokens_served(_row()), "-")

    def test_a_stopped_endpoint_keeps_its_total_but_loses_its_rate(self) -> None:
        row = _row(
            ServingSnapshot(total_prompt_tokens=120_000, total_generation_tokens=60_000),
            state="stopped",
        )
        self.assertEqual(_endpoint_throughput(row), "-")
        self.assertEqual(_endpoint_tokens_served(row), "180K")

    def test_an_idle_endpoint_reads_as_idle_rather_than_unknown(self) -> None:
        row = _row(
            _serving(total_generation_tokens=500.0, tokens_per_second=0.0),
            provider=ComputeProvider.PRIME,
        )
        self.assertEqual(_endpoint_throughput(row), "0 tok/s")


class ManageDetailTests(unittest.TestCase):
    def test_detail_splits_the_total_and_names_the_live_gauges(self) -> None:
        row = _row(
            _serving(
                total_prompt_tokens=940_000,
                total_generation_tokens=302_400,
                tokens_per_second=47.2,
                requests_running=3.0,
                requests_waiting=1.0,
                kv_cache_usage=0.31,
                avg_ttft_seconds=0.28,
            ),
            provider=ComputeProvider.PRIME,
        )
        detail = _serving_detail_lines(row)
        self.assertIn("1.2M tokens served", detail)
        self.assertIn("940K in / 302K out", detail)
        self.assertIn("47 tok/s", detail)
        self.assertIn("3 running", detail)
        self.assertIn("1 queued", detail)
        self.assertIn("KV 31%", detail)
        self.assertIn("TTFT 0.28s", detail)

    def test_modal_passive_detail_shows_history_not_live_gauges(self) -> None:
        row = _row(
            _serving(
                total_prompt_tokens=940_000,
                total_generation_tokens=302_400,
                tokens_per_second=47.2,
                requests_running=3.0,
            ),
            provider=ComputeProvider.MODAL,
        )
        detail = _serving_detail_lines(row)
        self.assertIn("1.2M tokens served", detail)
        self.assertNotIn("47 tok/s", detail)
        self.assertNotIn("3 running", detail)
        self.assertIn("Live metrics paused", detail)

    def test_an_empty_queue_is_not_worth_a_word(self) -> None:
        row = _row(
            _serving(requests_running=1.0, requests_waiting=0.0, tokens_per_second=5.0),
            provider=ComputeProvider.PRIME,
        )
        self.assertNotIn("queued", _serving_detail_lines(row))

    def test_llamacpps_own_gauge_is_labelled_as_the_last_request(self) -> None:
        # It is not a windowed rate, so it must not be shown as one.
        row = _row(
            _serving(reported_tokens_per_second=53.5, requests_running=0.0),
            provider=ComputeProvider.PRIME,
        )
        detail = _serving_detail_lines(row)
        self.assertIn("54 tok/s last request", detail)

    def test_a_measured_rate_wins_over_the_runtimes_own_gauge(self) -> None:
        row = _row(
            _serving(reported_tokens_per_second=53.5, tokens_per_second=12.0),
            provider=ComputeProvider.PRIME,
        )
        detail = _serving_detail_lines(row)
        self.assertIn("12 tok/s", detail)
        self.assertNotIn("last request", detail)

    def test_an_endpoint_without_traffic_reports_no_reading_yet(self) -> None:
        # Cached totals are labelled, including the empty state: a blank detail
        # once hid that Refresh never contacts a scaled-to-zero runtime.
        detail = _serving_detail_lines(_row())
        self.assertIn("not collected yet", detail)
        self.assertIn("Run time:", detail)
        self.assertIn("Compute cost:", detail)
        empty = _serving_detail_lines(_row(ServingSnapshot()))
        self.assertNotIn("tokens served", empty)
        self.assertIn("Run time:", empty)
        self.assertIn("Compute cost:", empty)


class FleetPanelTrafficTests(unittest.TestCase):
    def test_the_panel_line_leads_with_the_rate_then_the_total(self) -> None:
        line = _serving_panel_line(
            _row(
                _serving(
                    requests_running=3.0,
                    total_prompt_tokens=940_000,
                    total_generation_tokens=302_400,
                    tokens_per_second=47.2,
                ),
                provider=ComputeProvider.PRIME,
            )
        )
        self.assertEqual(line, "[dim]Traffic:[/dim] 47 tok/s · 3 running · 1.2M served")

    def test_modal_passive_panel_shows_totals_without_live_gauges(self) -> None:
        line = _serving_panel_line(
            _row(
                _serving(
                    requests_running=3.0,
                    total_prompt_tokens=940_000,
                    total_generation_tokens=302_400,
                    tokens_per_second=47.2,
                ),
                provider=ComputeProvider.MODAL,
            )
        )
        self.assertEqual(line, "[dim]Traffic:[/dim] 1.2M served")

    def test_a_stopped_endpoint_still_reports_what_it_served(self) -> None:
        line = _serving_panel_line(
            _row(ServingSnapshot(total_prompt_tokens=120_000, total_generation_tokens=60_000))
        )
        self.assertEqual(line, "[dim]Traffic:[/dim] 180K served")

    def test_nothing_to_report_adds_no_line(self) -> None:
        self.assertIsNone(_serving_panel_line(_row()))
        self.assertIsNone(_serving_panel_line(_row(ServingSnapshot())))

    def test_the_traffic_line_reaches_the_rendered_panel(self) -> None:
        rows = [
            _row(
                _serving(total_generation_tokens=302_400, tokens_per_second=47.2),
                state="deployed",
                provider=ComputeProvider.PRIME,
            )
        ]
        rendered = _render_deployment_status(rows, username="alice")
        self.assertIn("47 tok/s", rendered)
        self.assertIn("302K served", rendered)

    def test_modal_panel_explains_paused_live_metrics(self) -> None:
        rows = [
            _row(
                _serving(total_generation_tokens=302_400, tokens_per_second=47.2),
                state="deployed",
                provider=ComputeProvider.MODAL,
            )
        ]
        rendered = _render_deployment_status(rows, username="alice")
        self.assertNotIn("47 tok/s", rendered)
        self.assertIn("302K served", rendered)
        self.assertIn("Live metrics paused", rendered)
        self.assertIn("Not checked", rendered)


class ManageScreenReadsItsOwnTrafficTests(unittest.IsolatedAsyncioTestCase):
    """The fleet listing carries no traffic, so Manage has to go and read it.

    Rows reach this screen straight from the shared endpoint cache, which the
    home screen deliberately copies before annotating so it does not mutate
    what every other screen is holding. Manage therefore receives rows whose
    ``serving`` is None and must fill them in itself -- it shipped without
    that and both columns read "-" against a live endpoint.

    Prime rows are still read live. Modal rows stay passive so an open Manage
    screen can never wake a scaled-to-zero container.
    """

    async def asyncSetUp(self) -> None:
        from llm_launchpad.core.runtime_health import reset as reset_health
        from llm_launchpad.core.serving_metrics import default_tracker

        default_tracker().reset()
        reset_health()

    async def test_live_rows_get_their_traffic_read_and_painted(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
        from llm_launchpad.tui.workers import EndpointsLoaded

        # Exactly what the shared cache hands over: no traffic attached.
        cached = EndpointInfo(
            name="llamacpp-glm-5-3-flash",
            app_id="ap-1",
            state="deployed",
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            web_url="https://example.prime.run",
        )
        self.assertIsNone(cached.serving)

        metrics = "llamacpp:prompt_tokens_total 1200\nllamacpp:tokens_predicted_total 3400\n"

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[cached]))  # type: ignore[attr-defined]

        fake_requests = types.SimpleNamespace(
            get=lambda *a, **k: types.SimpleNamespace(status_code=200, text=metrics)
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

                table = screen.query_one("#manage-endpoint-table", AdaptiveDataTable)
                self.assertIn("served", table.visible_column_keys)
                served = table.get_cell_at(Coordinate(0, table.visible_column_keys.index("served")))
                self.assertEqual(served, "4.6K")
                detail = screen.query_one("#manage-selection-detail", Static)
                self.assertIn("4.6K tokens served", str(detail.renderable))

    async def test_modal_deployed_rows_are_never_probed(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
        from llm_launchpad.tui.workers import EndpointsLoaded

        cached = EndpointInfo(
            name="llamacpp-modal-passive",
            app_id="ap-modal-1",
            state="deployed",
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            web_url="https://example.modal.run",
        )

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[cached]))  # type: ignore[attr-defined]

        def _never_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("a Modal deployed row must not be woken for metrics")

        app = _App()
        with patch.dict(sys.modules, {"requests": types.SimpleNamespace(get=_never_called)}):
            async with app.run_test(size=(140, 24)) as pilot:
                app.push_screen(ManageScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ManageScreen)
                await screen.workers.wait_for_complete()
                await pilot.pause()

                table = screen.query_one("#manage-endpoint-table", AdaptiveDataTable)
                # No banked totals yet, so nothing to show -- the column is left
                # out rather than filled with dashes -- but crucially no request
                # was sent to learn that.
                self.assertNotIn("served", table.visible_column_keys)

    async def test_a_second_pass_turns_counters_into_a_rate(self) -> None:
        # Throughput is a delta, so one reading can never produce it. The
        # screen refreshes on its own rather than leaving the column empty
        # until someone presses a key.
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
        from llm_launchpad.tui.workers import EndpointsLoaded

        row = EndpointInfo(
            name="vllm-qwen",
            app_id="ap-1",
            state="deployed",
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            web_url="https://example.prime.run",
        )
        counters = iter(
            [
                "vllm:generation_tokens_total 1000\n",
                "vllm:generation_tokens_total 1000\n",
            ]
        )

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[row]))  # type: ignore[attr-defined]

        fake_requests = types.SimpleNamespace(
            get=lambda *a, **k: types.SimpleNamespace(
                status_code=200, text=next(counters, "vllm:generation_tokens_total 1000\n")
            )
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

                table = screen.query_one("#manage-endpoint-table", AdaptiveDataTable)
                # One reading has no rate yet, so the column is not shown.
                self.assertNotIn("throughput", table.visible_column_keys)

                screen._refresh_serving_stats()
                await screen.workers.wait_for_complete()
                await pilot.pause()
                column = table.visible_column_keys.index("throughput")
                self.assertEqual(table.get_cell_at(Coordinate(0, column)), "0 tok/s")

    async def test_the_traffic_timer_stops_while_the_screen_is_away(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.workers import EndpointsLoaded

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[]))  # type: ignore[attr-defined]

        app = _App()
        async with app.run_test(size=(140, 24)) as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            self.assertIsNotNone(screen._traffic_refresh_timer)

            timer = Mock()
            screen._traffic_refresh_timer = timer
            screen.on_screen_suspend(None)  # type: ignore[arg-type]
            timer.pause.assert_called_once_with()
            screen.on_screen_resume(None)  # type: ignore[arg-type]
            timer.resume.assert_called_once_with()

    async def test_a_stopped_row_is_never_probed(self) -> None:
        from textual.app import App

        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.workers import EndpointsLoaded

        stopped = EndpointInfo(
            name="llamacpp-old",
            app_id="ap-2",
            state="stopped",
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            web_url="https://example.modal.run",
        )

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[stopped]))  # type: ignore[attr-defined]

        def _never_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("a stopped endpoint must not be woken for metrics")

        app = _App()
        with patch.dict(sys.modules, {"requests": types.SimpleNamespace(get=_never_called)}):
            async with app.run_test(size=(140, 24)) as pilot:
                app.push_screen(ManageScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ManageScreen)
                await screen.workers.wait_for_complete()
                await pilot.pause()


if __name__ == "__main__":
    unittest.main()

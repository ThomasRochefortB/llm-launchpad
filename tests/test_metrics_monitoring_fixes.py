"""Regression tests for metrics monitoring reliability and Manage UI upgrades."""

from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.app import App
from textual.content import Content
from textual.widgets import Static

from llm_launchpad.core.endpoint_runtime import EndpointRuntimeTracker
from llm_launchpad.core.runtime_health import get_health, reset as reset_health
from llm_launchpad.core.serving_metrics import default_tracker, stats_from_metrics, usage_key
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, ServingSnapshot, ServingStats
from llm_launchpad.tui.fleet_status import (
    _EXPLICIT_METRICS_TIMEOUT_SECONDS,
    annotate_serving_stats,
    attach_cached_serving_stats,
    deployment_and_health_line,
    fetch_serving_snapshot_detailed,
    fleet_summary_line,
    has_explicit_live_metrics,
)
from llm_launchpad.tui.screens.main_menu import _should_show_in_panel, _state_bucket as home_state_bucket
from llm_launchpad.tui.screens.manage import (
    ManageScreen,
    _endpoint_throughput,
    _serving_detail_lines,
)
from llm_launchpad.tui.workers import EndpointsLoaded


class ActivePrimeStateMonitoringTests(unittest.TestCase):
    def test_prime_active_state_is_treated_as_running_on_home_and_manage(self) -> None:
        row = EndpointInfo(
            name="llp-prime-vllm-qwen",
            app_id="pod-1",
            state="active",
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            web_url="https://prime.example.test",
        )
        self.assertEqual(home_state_bucket(row.state), "healthy")
        self.assertTrue(_should_show_in_panel(row.state))
        plain = Content.from_markup(deployment_and_health_line(row)).plain
        self.assertIn("Deployment: Running", plain)
        self.assertIn("Health: Healthy", plain)
        summary = Content.from_markup(fleet_summary_line([row])).plain
        self.assertIn("1 running", summary)


class MetricsFetchReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        default_tracker().reset()
        reset_health()

    def tearDown(self) -> None:
        default_tracker().reset()
        reset_health()

    def test_explicit_fetch_bypasses_unsupported_cooldown_and_uses_longer_timeout(self) -> None:
        row = EndpointInfo(
            name="vllm-modal-1",
            app_id="ap-1",
            state="deployed",
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            web_url="https://example.modal.run",
        )
        default_tracker().mark_unsupported(usage_key(row))
        self.assertFalse(default_tracker().metrics_supported(usage_key(row)))

        seen_timeouts: list[float] = []

        def _fake_get(url: str, *, headers: object = None, timeout: float = 0.0) -> object:
            _ = (url, headers)
            seen_timeouts.append(timeout)
            return types.SimpleNamespace(
                status_code=200,
                text="vllm:prompt_tokens 100\nvllm:generation_tokens 50\n",
            )

        with patch.dict(sys.modules, {"requests": types.SimpleNamespace(get=_fake_get)}):
            result = fetch_serving_snapshot_detailed(row, explicit=True)

        self.assertIsNotNone(result.snapshot)
        self.assertIsNone(result.error)
        self.assertEqual(seen_timeouts, [_EXPLICIT_METRICS_TIMEOUT_SECONDS])
        self.assertTrue(default_tracker().metrics_supported(usage_key(row)))

    def test_failed_explicit_fetch_reports_specific_reason(self) -> None:
        row = EndpointInfo(
            name="llamacpp-prime-1",
            app_id="pod-1",
            state="active",
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            web_url="https://prime.example.test",
        )

        with patch.dict(
            sys.modules,
            {
                "requests": types.SimpleNamespace(
                    get=lambda *a, **k: types.SimpleNamespace(status_code=501, text="")
                )
            },
        ):
            result = fetch_serving_snapshot_detailed(row, explicit=True)

        self.assertIsNone(result.snapshot)
        assert result.error is not None
        self.assertIn("did not return metrics", result.error)
        self.assertIn("HTTP 501", result.error)

    def test_loopback_vast_metrics_bypasses_environment_proxies(self) -> None:
        row = EndpointInfo(
            name="vast-llamacpp-1",
            app_id="v-1",
            state="running",
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.VAST,
            web_url="http://127.0.0.1:41234",
            endpoint_api_key="vast-secret",
        )
        trust_env_values: list[bool] = []

        class _FakeSession:
            def __init__(self) -> None:
                self.trust_env = True

            def __enter__(self) -> _FakeSession:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str, *, headers: object = None, timeout: float = 0.0) -> object:
                _ = (url, headers, timeout)
                trust_env_values.append(self.trust_env)
                return types.SimpleNamespace(
                    status_code=200,
                    text="llamacpp:tokens_predicted_total 42\nllamacpp:kv_cache_usage_ratio 0.25\n",
                )

        with patch.dict(sys.modules, {"requests": types.SimpleNamespace(Session=_FakeSession)}):
            result = fetch_serving_snapshot_detailed(row, explicit=False)

        self.assertEqual(trust_env_values, [False])
        assert result.snapshot is not None
        self.assertEqual(result.snapshot.total_generation_tokens, 42.0)
        self.assertAlmostEqual(result.snapshot.stats.kv_cache_usage or 0.0, 0.25)

    def test_openmetrics_names_and_llamacpp_kv_ratio_parse(self) -> None:
        vllm_stats = stats_from_metrics(
            "vllm:prompt_tokens 320\nvllm:generation_tokens 180\n",
            BackendType.VLLM,
        )
        self.assertTrue(vllm_stats.has_readings)
        self.assertEqual(vllm_stats.prompt_tokens, 320.0)
        self.assertEqual(vllm_stats.generation_tokens, 180.0)

        llama_stats = stats_from_metrics(
            "llamacpp:predicted_tokens_seconds 71.5\nllamacpp:kv_cache_usage_ratio 0.4\n",
            BackendType.LLAMACPP,
        )
        self.assertTrue(llama_stats.has_readings)
        self.assertAlmostEqual(llama_stats.reported_tokens_per_second or 0.0, 71.5)
        self.assertAlmostEqual(llama_stats.kv_cache_usage or 0.0, 0.4)

    def test_runtime_tracker_preserves_provider_reported_start_time(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        tracker = EndpointRuntimeTracker(path=tmp / "runtime.json")
        row = EndpointInfo(
            name="llp-prime-vllm-qwen",
            app_id="pod-1",
            state="active",
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            run_started_at_epoch=1000.0,
            hourly_cost_usd=2.0,
        )
        tracker.observe_rows([row], now=2800.0, explicit=False)
        self.assertEqual(row.run_started_at_epoch, 1000.0)

    def test_explicit_modal_metrics_stay_visible_and_survive_passive_refresh(self) -> None:
        now = time.time()
        row = EndpointInfo(
            name="llamacpp-modal-1",
            app_id="ap-1",
            state="deployed",
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            live_metrics_checked_at=now,
            serving=ServingSnapshot(
                stats=ServingStats(
                    captured_at=10.0,
                    prompt_tokens=200.0,
                    generation_tokens=100.0,
                    requests_running=1.0,
                    reported_tokens_per_second=55.0,
                ),
                total_prompt_tokens=200.0,
                total_generation_tokens=100.0,
                tokens_per_second=55.0,
                observed_at=now,
            ),
        )
        self.assertTrue(has_explicit_live_metrics(row))
        self.assertEqual(_endpoint_throughput(row), "55 tok/s")
        detail = _serving_detail_lines(row, now=now)
        self.assertIn("55 tok/s", detail)
        self.assertIn("1 running", detail)

        # Background passive attach does not wipe explicit live readings.
        attach_cached_serving_stats([row])
        self.assertEqual(_endpoint_throughput(row), "55 tok/s")


class ManageLiveMetricsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        default_tracker().reset()
        reset_health()

    async def test_explicit_live_fetch_updates_modal_health_and_displays_live_gauges(self) -> None:
        cached = EndpointInfo(
            name="vllm-modal-live",
            app_id="ap-live",
            state="deployed",
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            web_url="https://example.modal.run",
        )
        metrics = (
            "vllm:prompt_tokens_total 500\n"
            "vllm:generation_tokens_total 250\n"
            "vllm:num_requests_running 2\n"
            "vllm:kv_cache_usage_perc 0.42\n"
        )

        class _App(App[None]):
            _username = "alice"

            def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(EndpointsLoaded(rows=[cached]))  # type: ignore[attr-defined]

        fake_requests = types.SimpleNamespace(
            get=lambda *a, **k: types.SimpleNamespace(status_code=200, text=metrics)
        )
        app = _App()
        with patch.dict(sys.modules, {"requests": fake_requests}):
            async with app.run_test(size=(140, 30)) as pilot:
                app.push_screen(ManageScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ManageScreen)
                await screen.workers.wait_for_complete()
                await pilot.pause()

                screen.action_fetch_live_selected()
                await screen.workers.wait_for_complete()
                await pilot.pause()

                self.assertEqual(cached.runtime_status, "healthy")
                stored = get_health(cached)
                self.assertIsNotNone(stored)
                assert stored is not None
                self.assertEqual(stored.status, "healthy")
                detail = str(screen.query_one("#manage-selection-detail", Static).renderable)
                self.assertIn("Healthy", detail)
                self.assertIn("2 running", detail)
                self.assertIn("KV 42%", detail)

    async def test_background_prime_metrics_probe_marks_endpoint_healthy(self) -> None:
        row = EndpointInfo(
            name="prime-vllm-1",
            app_id="pod-1",
            state="active",
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            web_url="https://prime.example.test",
            runtime_status="in_progress",
        )
        fake_requests = types.SimpleNamespace(
            get=lambda *a, **k: types.SimpleNamespace(
                status_code=200,
                text="vllm:prompt_tokens_total 100\nvllm:generation_tokens_total 50\n",
            )
        )
        with patch.dict(sys.modules, {"requests": fake_requests}):
            annotate_serving_stats([row], "alice")
        self.assertEqual(row.runtime_status, "healthy")
        self.assertIsNone(row.live_metrics_error)


if __name__ == "__main__":
    unittest.main()

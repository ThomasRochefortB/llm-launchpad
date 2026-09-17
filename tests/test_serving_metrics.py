"""Cover Prometheus parsing and lifetime token accounting for live endpoints."""

from __future__ import annotations

from pathlib import Path
import unittest

from llm_launchpad.core.serving_metrics import (
    MAX_RATE_WINDOW_SECONDS,
    UNSUPPORTED_RETRY_SECONDS,
    ServingMetricsTracker,
    derive_tokens_per_second,
    parse_prometheus_text,
    stats_from_metrics,
    usage_key,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, ServingStats


VLLM_SAMPLE = """
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:prompt_tokens_total{model_name="qwen"} 1200.0
vllm:generation_tokens_total{model_name="qwen"} 3400.0
vllm:num_requests_running{model_name="qwen"} 2.0
vllm:num_requests_waiting{model_name="qwen"} 1.0
vllm:kv_cache_usage_perc{model_name="qwen"} 0.31
vllm:request_success_total{finished_reason="stop"} 30.0
vllm:request_success_total{finished_reason="length"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="qwen"} 12.0
vllm:time_to_first_token_seconds_count{model_name="qwen"} 40.0
process_start_time_seconds 1.7e9
"""

LLAMACPP_SAMPLE = """
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 880
llamacpp:tokens_predicted_total 2100
llamacpp:predicted_tokens_seconds 53.5
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
"""


class ParsePrometheusTextTests(unittest.TestCase):
    def test_comments_and_blank_lines_are_skipped(self) -> None:
        samples = parse_prometheus_text("# HELP x y\n\n# TYPE x counter\nx 3\n")
        self.assertEqual(samples, {"x": 3.0})

    def test_label_sets_are_summed_into_one_value(self) -> None:
        samples = parse_prometheus_text(
            'r_total{reason="stop"} 30\nr_total{reason="length"} 10\n'
        )
        self.assertEqual(samples["r_total"], 40.0)

    def test_brace_inside_a_label_value_is_not_the_terminator(self) -> None:
        samples = parse_prometheus_text('x{template="{a}"} 7\n')
        self.assertEqual(samples, {"x": 7.0})

    def test_trailing_timestamp_is_ignored(self) -> None:
        samples = parse_prometheus_text("x 5 1395066363000\n")
        self.assertEqual(samples, {"x": 5.0})

    def test_non_finite_and_unparseable_values_are_dropped(self) -> None:
        samples = parse_prometheus_text("a NaN\nb +Inf\nc oops\nd 1\n")
        self.assertEqual(samples, {"d": 1.0})

    def test_empty_text_yields_no_samples(self) -> None:
        self.assertEqual(parse_prometheus_text(""), {})


class StatsFromMetricsTests(unittest.TestCase):
    def test_vllm_reading_covers_tokens_queue_and_cache(self) -> None:
        stats = stats_from_metrics(VLLM_SAMPLE, BackendType.VLLM, captured_at=10.0)
        self.assertEqual(stats.prompt_tokens, 1200.0)
        self.assertEqual(stats.generation_tokens, 3400.0)
        self.assertEqual(stats.requests_running, 2.0)
        self.assertEqual(stats.requests_waiting, 1.0)
        self.assertEqual(stats.kv_cache_usage, 0.31)
        self.assertEqual(stats.requests_finished, 40.0)
        self.assertEqual(stats.avg_ttft_seconds, 0.3)
        self.assertEqual(stats.runtime_started_at, 1.7e9)
        self.assertTrue(stats.has_readings)

    def test_vllm_falls_back_to_the_older_cache_metric_name(self) -> None:
        text = 'vllm:gpu_cache_usage_perc{model_name="qwen"} 0.5\n'
        stats = stats_from_metrics(text, BackendType.VLLM)
        self.assertEqual(stats.kv_cache_usage, 0.5)

    def test_llamacpp_reading_uses_its_own_metric_names(self) -> None:
        stats = stats_from_metrics(LLAMACPP_SAMPLE, BackendType.LLAMACPP, captured_at=5.0)
        self.assertEqual(stats.prompt_tokens, 880.0)
        self.assertEqual(stats.generation_tokens, 2100.0)
        self.assertEqual(stats.reported_tokens_per_second, 53.5)
        self.assertEqual(stats.requests_running, 1.0)
        # llama.cpp publishes no TTFT histogram and does not date its process.
        self.assertIsNone(stats.avg_ttft_seconds)
        self.assertIsNone(stats.runtime_started_at)

    def test_a_backend_reading_the_other_runtime_finds_nothing(self) -> None:
        stats = stats_from_metrics(LLAMACPP_SAMPLE, BackendType.VLLM)
        self.assertFalse(stats.has_readings)

    def test_unknown_backend_yields_no_readings(self) -> None:
        self.assertFalse(stats_from_metrics(VLLM_SAMPLE, None).has_readings)

    def test_a_page_that_is_not_metrics_yields_no_readings(self) -> None:
        stats = stats_from_metrics("<html>gateway timeout</html>", BackendType.VLLM)
        self.assertFalse(stats.has_readings)


class DeriveTokensPerSecondTests(unittest.TestCase):
    def test_rate_is_the_delta_over_the_window(self) -> None:
        first = ServingStats(captured_at=0.0, generation_tokens=100.0)
        second = ServingStats(captured_at=20.0, generation_tokens=500.0)
        self.assertEqual(derive_tokens_per_second(first, second, restarted=False), 20.0)

    def test_no_rate_across_a_restart(self) -> None:
        first = ServingStats(captured_at=0.0, generation_tokens=100.0)
        second = ServingStats(captured_at=20.0, generation_tokens=5.0)
        self.assertIsNone(derive_tokens_per_second(first, second, restarted=True))

    def test_no_rate_when_the_counter_went_backwards(self) -> None:
        first = ServingStats(captured_at=0.0, generation_tokens=100.0)
        second = ServingStats(captured_at=20.0, generation_tokens=5.0)
        self.assertIsNone(derive_tokens_per_second(first, second, restarted=False))

    def test_no_rate_across_a_window_too_wide_to_mean_anything(self) -> None:
        first = ServingStats(captured_at=0.0, generation_tokens=100.0)
        second = ServingStats(
            captured_at=MAX_RATE_WINDOW_SECONDS + 1.0, generation_tokens=500.0
        )
        self.assertIsNone(derive_tokens_per_second(first, second, restarted=False))

    def test_no_rate_without_two_token_readings(self) -> None:
        first = ServingStats(captured_at=0.0)
        second = ServingStats(captured_at=20.0, generation_tokens=500.0)
        self.assertIsNone(derive_tokens_per_second(first, second, restarted=False))


class ServingMetricsTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(self._tmp_dir()) / "serving_usage.json"
        self.tracker = ServingMetricsTracker(path=self.path)

    def _tmp_dir(self) -> str:
        import tempfile

        directory = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, directory, ignore_errors=True)
        return directory

    def _stats(self, **kwargs: float) -> ServingStats:
        return ServingStats(**kwargs)  # type: ignore[arg-type]

    def test_first_reading_banks_the_whole_counter(self) -> None:
        # The container has already served this much; nothing of it is banked.
        snapshot = self.tracker.record(
            "modal:a", self._stats(captured_at=1.0, prompt_tokens=100.0, generation_tokens=400.0)
        )
        self.assertEqual(snapshot.total_prompt_tokens, 100.0)
        self.assertEqual(snapshot.total_generation_tokens, 400.0)
        self.assertEqual(snapshot.total_tokens, 500.0)
        self.assertIsNone(snapshot.tokens_per_second)

    def test_steady_traffic_banks_only_the_delta(self) -> None:
        self.tracker.record("modal:a", self._stats(captured_at=1.0, generation_tokens=400.0))
        snapshot = self.tracker.record(
            "modal:a", self._stats(captured_at=21.0, generation_tokens=800.0)
        )
        self.assertEqual(snapshot.total_generation_tokens, 800.0)
        self.assertEqual(snapshot.tokens_per_second, 20.0)

    def test_a_reset_counter_banks_the_new_container_in_full(self) -> None:
        self.tracker.record("modal:a", self._stats(captured_at=1.0, generation_tokens=400.0))
        snapshot = self.tracker.record(
            "modal:a", self._stats(captured_at=21.0, generation_tokens=50.0)
        )
        self.assertEqual(snapshot.total_generation_tokens, 450.0)
        self.assertIsNone(snapshot.tokens_per_second)

    def test_a_restarted_process_banks_in_full_even_when_the_counter_climbed(self) -> None:
        # The give-away is the process start time, not the counter: a new
        # container that has already served more than the old one reported
        # would otherwise look like ordinary traffic and be undercounted.
        self.tracker.record(
            "modal:a",
            self._stats(captured_at=1.0, generation_tokens=400.0, runtime_started_at=1000.0),
        )
        snapshot = self.tracker.record(
            "modal:a",
            self._stats(captured_at=21.0, generation_tokens=900.0, runtime_started_at=2000.0),
        )
        self.assertEqual(snapshot.total_generation_tokens, 1300.0)
        self.assertIsNone(snapshot.tokens_per_second)

    def test_same_process_keeps_counting_deltas(self) -> None:
        self.tracker.record(
            "modal:a",
            self._stats(captured_at=1.0, generation_tokens=400.0, runtime_started_at=1000.0),
        )
        snapshot = self.tracker.record(
            "modal:a",
            self._stats(captured_at=21.0, generation_tokens=900.0, runtime_started_at=1000.0),
        )
        self.assertEqual(snapshot.total_generation_tokens, 900.0)
        self.assertEqual(snapshot.tokens_per_second, 25.0)

    def test_totals_survive_a_new_tracker_reading_the_same_store(self) -> None:
        self.tracker.record(
            "modal:a",
            self._stats(captured_at=1.0, prompt_tokens=100.0, generation_tokens=400.0),
        )
        reopened = ServingMetricsTracker(path=self.path)
        snapshot = reopened.record(
            "modal:a",
            self._stats(captured_at=99.0, prompt_tokens=150.0, generation_tokens=600.0),
        )
        self.assertEqual(snapshot.total_prompt_tokens, 150.0)
        self.assertEqual(snapshot.total_generation_tokens, 600.0)
        # The old process's monotonic clock is gone, so no rate is invented
        # across the restart of the TUI itself.
        self.assertIsNone(snapshot.tokens_per_second)

    def test_zero_counter_reset_survives_reopening_the_tracker(self) -> None:
        self.tracker.record(
            "modal:a", self._stats(prompt_tokens=100.0, generation_tokens=400.0)
        )
        self.tracker.record(
            "modal:a", self._stats(prompt_tokens=0.0, generation_tokens=0.0)
        )

        reopened = ServingMetricsTracker(path=self.path)
        snapshot = reopened.record(
            "modal:a", self._stats(prompt_tokens=150.0, generation_tokens=600.0)
        )

        self.assertEqual(snapshot.total_prompt_tokens, 250.0)
        self.assertEqual(snapshot.total_generation_tokens, 1000.0)

    def test_runtime_identity_changes_are_saved_without_new_tokens(self) -> None:
        from unittest.mock import patch

        self.tracker.record(
            "modal:a", self._stats(generation_tokens=0.0, runtime_started_at=1000.0)
        )
        with patch.object(self.tracker, "_save", wraps=self.tracker._save) as save:
            self.tracker.record(
                "modal:a", self._stats(generation_tokens=0.0, runtime_started_at=2000.0)
            )
            save.assert_called_once()

    def test_unchanged_counters_and_runtime_do_not_rewrite_the_store(self) -> None:
        from unittest.mock import patch

        self.tracker.record(
            "modal:a",
            self._stats(captured_at=1.0, generation_tokens=400.0, runtime_started_at=1000.0),
        )
        with patch.object(self.tracker, "_save", wraps=self.tracker._save) as save:
            self.tracker.record(
                "modal:a",
                self._stats(captured_at=21.0, generation_tokens=400.0, runtime_started_at=1000.0),
            )
            save.assert_not_called()

    def test_snapshot_reports_banked_totals_without_a_reading(self) -> None:
        self.tracker.record(
            "modal:a",
            self._stats(captured_at=1.0, prompt_tokens=100.0, generation_tokens=400.0),
        )
        reopened = ServingMetricsTracker(path=self.path)
        snapshot = reopened.snapshot("modal:a")
        assert snapshot is not None
        self.assertEqual(snapshot.total_tokens, 500.0)
        self.assertIsNone(snapshot.tokens_per_second)

    def test_snapshot_of_an_unknown_endpoint_is_none(self) -> None:
        self.assertIsNone(self.tracker.snapshot("modal:never-seen"))

    def test_a_corrupt_total_is_treated_as_missing_rather_than_raising(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            '{"entries": {"modal:a": {"total_generation_tokens": "lots"}}}', encoding="utf-8"
        )
        snapshot = self.tracker.record("modal:a", self._stats(captured_at=1.0, generation_tokens=7.0))
        self.assertEqual(snapshot.total_generation_tokens, 7.0)

    def test_an_unreadable_store_does_not_lose_the_next_reading(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json", encoding="utf-8")
        snapshot = self.tracker.record("modal:a", self._stats(captured_at=1.0, generation_tokens=7.0))
        self.assertEqual(snapshot.total_generation_tokens, 7.0)

    def test_unsupported_endpoints_are_skipped_until_the_retry_window(self) -> None:
        self.assertTrue(self.tracker.metrics_supported("modal:a"))
        self.tracker.mark_unsupported("modal:a")
        self.assertFalse(self.tracker.metrics_supported("modal:a"))
        # Ageing the mark past the window lets the probe ask once more, which
        # is how a redeploy that added --metrics gets noticed.
        self.tracker._unsupported["modal:a"] -= UNSUPPORTED_RETRY_SECONDS + 1
        self.assertTrue(self.tracker.metrics_supported("modal:a"))

    def test_a_successful_reading_clears_an_earlier_verdict(self) -> None:
        self.tracker.mark_unsupported("modal:a")
        self.tracker.mark_supported("modal:a")
        self.assertTrue(self.tracker.metrics_supported("modal:a"))

    def test_reset_drops_cached_readings(self) -> None:
        self.tracker.record("modal:a", self._stats(captured_at=1.0, generation_tokens=400.0))
        self.tracker.mark_unsupported("modal:b")
        self.tracker.reset()
        self.assertTrue(self.tracker.metrics_supported("modal:b"))
        # Totals live on disk, so resetting in-memory state keeps them.
        snapshot = self.tracker.snapshot("modal:a")
        assert snapshot is not None
        self.assertEqual(snapshot.total_generation_tokens, 400.0)


class UsageKeyTests(unittest.TestCase):
    def test_name_and_provider_identify_an_endpoint(self) -> None:
        row = EndpointInfo(name="vllm-qwen", app_id="ap-1", provider=ComputeProvider.MODAL)
        self.assertEqual(usage_key(row), "modal:vllm-qwen")

    def test_same_name_on_two_providers_does_not_share_a_total(self) -> None:
        modal = EndpointInfo(name="qwen", provider=ComputeProvider.MODAL)
        prime = EndpointInfo(name="qwen", provider=ComputeProvider.PRIME)
        self.assertNotEqual(usage_key(modal), usage_key(prime))

    def test_a_nameless_row_falls_back_to_its_resource_id(self) -> None:
        row = EndpointInfo(name="", app_id="ap-9", provider=ComputeProvider.MODAL)
        self.assertEqual(usage_key(row), "modal:id:ap-9")


if __name__ == "__main__":
    unittest.main()

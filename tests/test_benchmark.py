from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from llm_launchpad.core.benchmark import (
    aiperf_metrics_have_successful_requests,
    benchmark_config_from_endpoint,
    build_aiperf_command,
    default_benchmark_run_dir,
    expected_export_paths,
    merge_cached_benchmark_connections,
    parse_aiperf_summary,
    parse_concurrency_values,
    request_count_for_concurrency,
)
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import BenchmarkConfig, EndpointInfo


class BenchmarkCoreTests(unittest.TestCase):
    def test_parse_concurrency_values_accepts_comma_and_space_separated_values(self) -> None:
        self.assertEqual(parse_concurrency_values("1, 2 4,8"), [1, 2, 4, 8])

    def test_parse_concurrency_values_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_concurrency_values("1,zero")
        with self.assertRaises(ValueError):
            parse_concurrency_values("0")

    def test_request_count_defaults_to_balanced_formula(self) -> None:
        self.assertEqual(request_count_for_concurrency(1), 24)
        self.assertEqual(request_count_for_concurrency(8), 32)
        self.assertEqual(request_count_for_concurrency(8, override=99), 99)

    def test_default_benchmark_run_dir_uses_app_name_and_utc_timestamp(self) -> None:
        now = datetime(2026, 4, 24, 12, 30, 5, tzinfo=timezone.utc)

        run_dir = default_benchmark_run_dir("llamacpp/My App", now=now)

        self.assertEqual(run_dir.name, "20260424T123005Z")
        self.assertEqual(run_dir.parent.name, "llamacpp-my-app")

    def test_build_aiperf_command_maps_default_synthetic_chat_options(self) -> None:
        config = BenchmarkConfig(
            backend=BackendType.VLLM,
            app_name="vllm-qwen",
            server_url="https://alice--vllm-qwen-serve.modal.run/v1",
            model_name="Qwen3-4B",
            concurrency=[4],
            aiperf_args=["--warmup-request-count", "2"],
        )

        cmd = build_aiperf_command(
            config,
            concurrency=4,
            artifact_dir=Path("/tmp/aiperf-c4"),
            executable="aiperf",
        )

        self.assertEqual(cmd[:2], ["aiperf", "profile"])
        self.assertIn("--endpoint-type", cmd)
        self.assertIn("chat", cmd)
        self.assertIn("--streaming", cmd)
        self.assertIn("--use-legacy-max-tokens", cmd)
        self.assertIn("--tokenizer", cmd)
        self.assertIn("gpt2", cmd)
        self.assertIn("--ui", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--no-server-metrics", cmd)
        self.assertIn("--export-level", cmd)
        self.assertIn("summary", cmd)
        self.assertIn("--prompt-input-tokens-mean", cmd)
        self.assertNotIn("--input-tokens-mean", cmd)
        self.assertIn("550", cmd)
        self.assertIn("--output-tokens-mean", cmd)
        self.assertIn("256", cmd)
        self.assertIn("--random-seed", cmd)
        self.assertIn("42", cmd)
        self.assertIn("--request-timeout-seconds", cmd)
        self.assertIn("300", cmd)
        self.assertIn("--concurrency", cmd)
        self.assertIn("4", cmd)
        self.assertIn("--request-count", cmd)
        self.assertIn("24", cmd)
        self.assertIn("--artifact-dir", cmd)
        self.assertIn("/tmp/aiperf-c4", cmd)
        self.assertIn("--warmup-request-count", cmd)
        self.assertNotIn("https://alice--vllm-qwen-serve.modal.run/v1", cmd)
        self.assertIn("https://alice--vllm-qwen-serve.modal.run", cmd)

    def test_benchmark_config_from_endpoint_prefers_endpoint_metadata(self) -> None:
        row = EndpointInfo(
            name="llamacpp-qwen",
            state="running",
            backend=BackendType.LLAMACPP,
            instance_name="qwen",
            web_url="https://alice--llamacpp-qwen-serve-fn.modal.run/v1",
            repo_id="unsloth/Qwen-GGUF",
            quant="Q4_K_M",
        )

        config = benchmark_config_from_endpoint(
            row,
            backend=BackendType.LLAMACPP,
            username="alice",
            app_name=row.name,
            concurrency=[1],
        )

        self.assertEqual(config.server_url, "https://alice--llamacpp-qwen-serve-fn.modal.run")
        self.assertEqual(config.model_name, "Qwen-GGUF-Q4_K_M")

    def test_parse_aiperf_summary_prefers_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            json_path, csv_path = expected_export_paths(artifact_dir)
            json_path.write_text(
                """
                {
                  "metrics": {
                    "output_token_throughput": {"value": 42.5},
                    "request_throughput": {"avg": 2.5},
                    "request_count": {"value": 24},
                    "time_to_first_token": {"avg": 100, "p50": 90, "p90": 140, "p99": 180},
                    "inter_token_latency": {"avg": 8, "p90": 12},
                    "request_latency": {"avg": 1000, "p90": 1300, "p99": 1600}
                  }
                }
                """
            )

            metrics, source = parse_aiperf_summary(json_path, csv_path)

        self.assertTrue(source.endswith("profile_export_aiperf.json"))
        self.assertEqual(metrics["output_token_throughput"], 42.5)
        self.assertEqual(metrics["time_to_first_token_p90"], 140.0)
        self.assertEqual(metrics["inter_token_latency_p90"], 12.0)
        self.assertEqual(metrics["request_latency_p99"], 1600.0)

    def test_parse_aiperf_summary_falls_back_to_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            json_path, csv_path = expected_export_paths(artifact_dir)
            csv_path.write_text(
                "Metric,avg,p50,p90,p99\n"
                "Output Token Throughput (tokens/sec),12.5,,,\n"
                "Request Throughput (requests/sec),1.25,,,\n"
                "Time to First Token (ms),100,90,140,180\n"
                "Request Latency (ms),1000,,1300,1600\n"
            )

            metrics, source = parse_aiperf_summary(json_path, csv_path)

        self.assertTrue(source.endswith("profile_export_aiperf.csv"))
        self.assertEqual(metrics["output_token_throughput"], 12.5)
        self.assertEqual(metrics["request_throughput"], 1.25)
        self.assertEqual(metrics["time_to_first_token_p50"], 90.0)
        self.assertEqual(metrics["request_latency_p90"], 1300.0)

    def test_aiperf_metrics_have_successful_requests_rejects_empty_exports(self) -> None:
        self.assertFalse(aiperf_metrics_have_successful_requests({"request_count": 0.0}))
        self.assertFalse(aiperf_metrics_have_successful_requests({"request_count": None}))
        self.assertTrue(
            aiperf_metrics_have_successful_requests(
                {
                    "request_count": 24.0,
                    "output_token_throughput": 12.5,
                }
            )
        )

    def test_merge_cached_benchmark_connections_adds_missing_url_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "deployment_connection_summaries.json"
            cache_path.write_text(
                """
                {
                  "entries": {
                    "vllm-qwen": {
                      "base_url": "https://alice--vllm-qwen-serve.modal.run/v1",
                      "model_id": "Qwen3-4B",
                      "display_name": "Qwen"
                    }
                  }
                }
                """
            )
            row = EndpointInfo(name="vllm-qwen", backend=BackendType.VLLM, state="running")
            from llm_launchpad.core import benchmark as benchmark_module

            original_settings_dir = benchmark_module.SETTINGS_DIR
            benchmark_module.SETTINGS_DIR = Path(tmp)
            try:
                merge_cached_benchmark_connections([row])
            finally:
                benchmark_module.SETTINGS_DIR = original_settings_dir

        self.assertEqual(row.web_url, "https://alice--vllm-qwen-serve.modal.run")
        self.assertEqual(row.served_model_name, "Qwen3-4B")
        self.assertEqual(row.display_name, "Qwen")


if __name__ == "__main__":
    unittest.main()

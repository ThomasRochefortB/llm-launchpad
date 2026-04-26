from __future__ import annotations

import unittest

from llm_launchpad.core.paths import MODAL_LLAMACPP_SCRIPT, MODAL_VLLM_SCRIPT
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.models import (
    BenchmarkConcurrencyResult,
    BenchmarkConfig,
    BenchmarkRunSummary,
    LaunchpadSettings,
    StorageSnapshot,
    StoredModelInfo,
)


class LaunchpadSettingsTests(unittest.TestCase):
    def test_to_env_defaults(self) -> None:
        env = LaunchpadSettings().to_env()
        self.assertEqual(env["SCALEDOWN_WINDOW"], "1800")

    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = LaunchpadSettings(scaledown_window=600)
        serialized = original.to_dict()
        restored = LaunchpadSettings.from_dict(serialized)
        self.assertEqual(restored.scaledown_window, 600)

    def test_from_dict_accepts_legacy_lowercase_key(self) -> None:
        restored = LaunchpadSettings.from_dict({"scaledown_window": 900})
        self.assertEqual(restored.scaledown_window, 900)

    def test_to_env_omits_non_positive_scaledown(self) -> None:
        env = LaunchpadSettings(scaledown_window=0).to_env()
        self.assertNotIn("SCALEDOWN_WINDOW", env)


class ProtocolEnumsTests(unittest.TestCase):
    def test_backend_type_display_name_and_script(self) -> None:
        self.assertEqual(BackendType.VLLM.display_name, "vLLM (OpenAI-compatible)")
        self.assertEqual(BackendType.LLAMACPP.display_name, "llama.cpp (GGUF)")
        self.assertEqual(BackendType.VLLM.script, MODAL_VLLM_SCRIPT)
        self.assertEqual(BackendType.LLAMACPP.script, MODAL_LLAMACPP_SCRIPT)

    def test_storage_operation_types_exist(self) -> None:
        self.assertEqual(OperationType.STORAGE_LIST.value, "storage_list")
        self.assertEqual(OperationType.STORAGE_PREDOWNLOAD.value, "storage_predownload")
        self.assertEqual(OperationType.STORAGE_DELETE.value, "storage_delete")

    def test_benchmark_operation_type_exists(self) -> None:
        self.assertEqual(OperationType.BENCHMARK.value, "benchmark")


class BenchmarkModelsTests(unittest.TestCase):
    def test_benchmark_config_defaults_to_balanced_synthetic_workload(self) -> None:
        config = BenchmarkConfig()

        self.assertEqual(config.concurrency, [1, 2, 4, 8, 16])
        self.assertEqual(config.input_tokens, 550)
        self.assertEqual(config.output_tokens, 256)
        self.assertEqual(config.tokenizer, "gpt2")
        self.assertEqual(config.request_timeout_seconds, 300)
        self.assertEqual(config.aiperf_args, [])

    def test_benchmark_run_summary_stores_results(self) -> None:
        config = BenchmarkConfig(app_name="vllm-qwen")
        result = BenchmarkConcurrencyResult(
            concurrency=1,
            command=["aiperf", "profile"],
            artifact_dir="/tmp/c1",
            metrics={"output_token_throughput": 12.5},
        )
        summary = BenchmarkRunSummary(
            config=config,
            run_dir="/tmp/run",
            results=[result],
            best_concurrency=1,
            best_output_token_throughput=12.5,
        )

        self.assertEqual(summary.results[0].concurrency, 1)
        self.assertEqual(summary.best_concurrency, 1)


class StorageModelsTests(unittest.TestCase):
    def test_storage_snapshot_totals(self) -> None:
        llamacpp = StoredModelInfo(
            backend=BackendType.LLAMACPP,
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            size_bytes=2_000,
            file_count=2,
            source_volume="huggingface-cache",
        )
        vllm = StoredModelInfo(
            backend=BackendType.VLLM,
            model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
            size_bytes=3_000,
            file_count=5,
            source_volume="huggingface-cache",
        )
        snapshot = StorageSnapshot(llamacpp_models=[llamacpp], vllm_models=[vllm])
        self.assertEqual(snapshot.total_models, 2)
        self.assertEqual(snapshot.total_size_bytes, 5_000)
        self.assertFalse(llamacpp.incomplete)
        self.assertFalse(vllm.incomplete)


if __name__ == "__main__":
    unittest.main()

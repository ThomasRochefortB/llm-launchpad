from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import BenchmarkConfig, BenchmarkRunSummary


class BenchmarkOrchestratorTests(unittest.TestCase):
    def test_benchmark_runs_each_concurrency_and_writes_summary(self) -> None:
        seen_commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            config = BenchmarkConfig(
                backend=BackendType.VLLM,
                app_name="vllm-qwen",
                server_url="https://example.modal.run",
                model_name="Qwen3-4B",
                concurrency=[1, 2],
                output_dir=tmp,
            )

            def _run_streaming(cmd):  # type: ignore[no-untyped-def]
                seen_commands.append(list(cmd))
                artifact_dir = Path(cmd[cmd.index("--artifact-dir") + 1])
                artifact_dir.mkdir(parents=True, exist_ok=True)
                throughput = 10 * int(cmd[cmd.index("--concurrency") + 1])
                (artifact_dir / "profile_export_aiperf.json").write_text(
                    json.dumps({"metrics": {"output_token_throughput": {"value": throughput}}})
                )
                return iter(
                    [
                        LogEvent(line="aiperf output"),
                        OperationCompleteEvent(success=True, exit_code=0),
                    ]
                )

            with (
                patch("llm_launchpad.core.orchestrator.aiperf_cli_path", return_value="aiperf"),
                patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming", side_effect=_run_streaming),
            ):
                events = list(Orchestrator().benchmark(config))

            completions = [event for event in events if isinstance(event, OperationCompleteEvent)]
            self.assertEqual(completions[-1].operation, OperationType.BENCHMARK)
            self.assertTrue(completions[-1].success)
            self.assertIsInstance(completions[-1].data, BenchmarkRunSummary)
            summary = completions[-1].data
            assert isinstance(summary, BenchmarkRunSummary)
            self.assertEqual(summary.best_concurrency, 2)
            self.assertEqual(summary.best_output_token_throughput, 20.0)
            self.assertEqual(len(seen_commands), 2)
            self.assertTrue((Path(tmp) / "benchmark_summary.json").exists())

    def test_benchmark_continues_after_failed_concurrency(self) -> None:
        seen_concurrency: list[int] = []
        with tempfile.TemporaryDirectory() as tmp:
            config = BenchmarkConfig(
                backend=BackendType.VLLM,
                app_name="vllm-qwen",
                server_url="https://example.modal.run",
                model_name="Qwen3-4B",
                concurrency=[1, 2],
                output_dir=tmp,
            )

            def _run_streaming(cmd):  # type: ignore[no-untyped-def]
                concurrency = int(cmd[cmd.index("--concurrency") + 1])
                seen_concurrency.append(concurrency)
                if concurrency == 1:
                    artifact_dir = Path(cmd[cmd.index("--artifact-dir") + 1])
                    (artifact_dir / "profile_export_aiperf.json").write_text(
                        '{"metrics":{"output_token_throughput":{"value":10}}}'
                    )
                    return iter([OperationCompleteEvent(success=True, exit_code=0)])
                return iter([OperationCompleteEvent(success=False, exit_code=9, detail="failed")])

            with (
                patch("llm_launchpad.core.orchestrator.aiperf_cli_path", return_value="aiperf"),
                patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming", side_effect=_run_streaming),
            ):
                events = list(Orchestrator().benchmark(config))

        completion = [event for event in events if isinstance(event, OperationCompleteEvent)][-1]
        self.assertFalse(completion.success)
        self.assertEqual(seen_concurrency, [1, 2])

    def test_benchmark_fails_fast_when_aiperf_is_missing(self) -> None:
        config = BenchmarkConfig(
            backend=BackendType.VLLM,
            app_name="vllm-qwen",
            server_url="https://example.modal.run",
            model_name="Qwen3-4B",
            concurrency=[1],
        )

        with patch("llm_launchpad.core.orchestrator.aiperf_cli_path", return_value=None):
            events = list(Orchestrator().benchmark(config))

        self.assertTrue(any(isinstance(event, ErrorEvent) for event in events))
        completion = [event for event in events if isinstance(event, OperationCompleteEvent)][-1]
        self.assertFalse(completion.success)
        self.assertIn("llm-launchpad[benchmark]", completion.detail)

    def test_benchmark_marks_export_parse_failure_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = BenchmarkConfig(
                backend=BackendType.VLLM,
                app_name="vllm-qwen",
                server_url="https://example.modal.run",
                model_name="Qwen3-4B",
                concurrency=[1],
                output_dir=tmp,
            )

            with (
                patch("llm_launchpad.core.orchestrator.aiperf_cli_path", return_value="aiperf"),
                patch(
                    "llm_launchpad.core.orchestrator.ModalBackend.run_streaming",
                    return_value=iter([OperationCompleteEvent(success=True, exit_code=0)]),
                ),
            ):
                events = list(Orchestrator().benchmark(config))

        completion = [event for event in events if isinstance(event, OperationCompleteEvent)][-1]
        self.assertFalse(completion.success)
        self.assertIn("One or more benchmark runs failed", completion.detail)


if __name__ == "__main__":
    unittest.main()

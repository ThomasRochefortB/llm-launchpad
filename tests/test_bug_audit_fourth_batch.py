"""Regression tests for the codebase-wide bug audit fixes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_launchpad.backends import modal_llamacpp_app
from llm_launchpad.core.backend import ModalBackend, ModalCliError
from llm_launchpad.core.connection_store import load_connection_entries, save_connection
from llm_launchpad.core.deployment_preflight import validate_request
from llm_launchpad.core.endpoint_runtime import EndpointRuntimeTracker
from llm_launchpad.core.estimate_evaluation import _observed_peak_per_device_gb, score_record
from llm_launchpad.core.evaluation_store import write_evaluation_record
from llm_launchpad.core.hf_download import classify_download_failure
from llm_launchpad.core.hf_models import _extract_parameter_count
from llm_launchpad.core.job_store import JobStore
from llm_launchpad.core.llamacpp_planner import memory_for_cache_type
from llm_launchpad.core.local_watchdog import run as run_local_watchdog
from llm_launchpad.core.modal_deployment import _deploy_vllm
from llm_launchpad.core.modal_storage import walk_volume_files
from llm_launchpad.core.prime_backend import PrimeDiskOffer
from llm_launchpad.core.prime_disks import matching_disk_offer
from llm_launchpad.core.prime_runtime import (
    PrimeLaunchSpec,
    build_bootstrap_docker_command,
)
from llm_launchpad.core.runtime_evidence import parse_offload_report
from llm_launchpad.core.serving_metrics import ServingMetricsTracker, parse_prometheus_text
from llm_launchpad.core.serving_runtime import GGUF_FILE_PLACEHOLDER, gguf_exec_command
from llm_launchpad.core.tool_call_probe import classify_tool_call_response
from llm_launchpad.core.vast_deployment import VastDeploymentBackend
from llm_launchpad.core.warmup import WarmupRequest, WarmupRunner, _ModalLogTail
from llm_launchpad.gguf_files import auxiliary_gguf_ignore_patterns, is_target_gguf
from llm_launchpad.protocol.enums import (
    BackendType,
    ComputeProvider,
    OperationType,
    OutcomeKind,
)
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import (
    ComputeOffer,
    DeploymentConfig,
    DeploymentRequest,
    EndpointInfo,
    EngramTable,
    EvaluationRecord,
    MemoryEstimate,
    MemoryObservation,
    PerformancePoint,
    PlacementAssessment,
    PredictionSnapshot,
    RuntimeIdentity,
    RuntimeTuning,
    ServingStats,
    WorkloadScenario,
)


class ModalAndGgufAuditTests(unittest.TestCase):
    def test_llamacpp_engram_table_excluded_from_target_gguf(self) -> None:
        self.assertFalse(is_target_gguf("model-engram-table.gguf"))
        self.assertFalse(is_target_gguf("engram_Q4_K_M.gguf"))
        self.assertTrue(is_target_gguf("model-Q4_K_M.gguf"))
        patterns = auxiliary_gguf_ignore_patterns()
        self.assertTrue(any("e][n" in p.lower() for p in patterns))

    def test_llamacpp_config_path_isolated_per_app(self) -> None:
        self.assertIn("/serve_configs/", modal_llamacpp_app.CONFIG_PATH)
        self.assertTrue(modal_llamacpp_app.CONFIG_PATH.endswith(".json"))
        self.assertNotEqual(modal_llamacpp_app.CONFIG_PATH, modal_llamacpp_app._LEGACY_CONFIG_PATH)

    def test_llamacpp_dspark_fit_args_rejects_malformed_fit_target(self) -> None:
        with self.assertRaises(ValueError):
            modal_llamacpp_app._dspark_fit_args(
                ["--fit-target", "not-a-number", "--ctx-size", "8192"],
                size_bytes=512 * 1024 * 1024,
            )

    def test_llamacpp_and_hf_download_status_regex_word_boundary(self) -> None:
        # Must not misclassify filenames or byte counts containing 401/403/404/429
        self.assertIsNone(
            modal_llamacpp_app._classify_download_failure(
                "downloaded 140429 bytes from shard-0403.gguf: connection reset"
            )
        )
        self.assertIsNone(
            classify_download_failure(
                "downloaded 140429 bytes from shard-0403.gguf: connection reset"
            )
        )
        self.assertEqual(classify_download_failure("HTTP 401 Unauthorized"), "auth")
        self.assertEqual(
            modal_llamacpp_app._classify_download_failure("404 Client Error: Not Found"),
            "not_found",
        )

    def test_modal_vllm_deploy_redacts_api_key_in_logs(self) -> None:
        secret = "super-secret-vllm-key-12345"
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            model_name="Qwen/Qwen3-8B",
            gpu_type="H100",
            gpu_count=2,
            endpoint_api_key=secret,
            do_deploy=True,
        )
        env = {
            "MODEL_NAME": "Qwen/Qwen3-8B",
            "VLLM_API_KEY": secret,
            "HF_TOKEN": "hf_secret_token",
        }
        with patch.object(
            ModalBackend,
            "run_streaming",
            return_value=iter(
                [OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)]
            ),
        ):
            events = list(_deploy_vllm(config, env))
        joined = "\n".join(e.line for e in events if isinstance(e, LogEvent))
        self.assertNotIn(secret, joined)
        self.assertNotIn("hf_secret_token", joined)
        self.assertIn("VLLM_API_KEY=[redacted]", joined)

    def test_backend_vllm_env_defaults_n_gpu_to_gpu_count(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            model_name="Qwen/Qwen3-8B",
            gpu_type="H100",
            gpu_count=4,
            n_gpu=None,
        )
        env = ModalBackend.env_for_backend(config)
        self.assertEqual(env["N_GPU"], "4")

    def test_walk_volume_files_raises_on_backend_error(self) -> None:
        with (
            patch.object(ModalBackend, "list_volume", return_value=None),
            patch.object(
                ModalBackend,
                "last_volume_error",
                return_value=ModalCliError(
                    command="modal volume ls",
                    exit_code=1,
                    message="modal authentication expired",
                ),
            ),
            self.assertRaises(RuntimeError),
        ):
            walk_volume_files("vol-name", "/")


class LifecycleWarmupAndJobStoreAuditTests(unittest.TestCase):
    def test_modal_log_tail_preserves_chronological_order(self) -> None:
        tail = _ModalLogTail("test-app")
        tail.record_line("llm_load_tensors: offloaded 40/40 layers to GPU")
        tail.record_line("llm_load_tensors: offloaded 20/40 layers to GPU")
        self.assertEqual(
            tail.ordered_lines,
            [
                "llm_load_tensors: offloaded 40/40 layers to GPU",
                "llm_load_tensors: offloaded 20/40 layers to GPU",
            ],
        )
        # Re-observing an earlier line still records chronological arrival order
        self.assertFalse(tail.record_line("llm_load_tensors: offloaded 40/40 layers to GPU"))
        self.assertEqual(
            tail.ordered_lines[-1],
            "llm_load_tensors: offloaded 40/40 layers to GPU",
        )

    def test_parse_offload_report_resets_total_when_later_line_omits_total(self) -> None:
        logs = (
            "load_tensors: offloaded 40/40 layers to GPU\n"
            "load_tensors: offloading 24 layers to GPU\n"
        )
        gpu_layers, total_layers = parse_offload_report(logs)
        self.assertEqual(gpu_layers, 24)
        self.assertIsNone(total_layers)

    def test_warmup_stops_log_tail_on_generator_close(self) -> None:
        req = WarmupRequest(
            backend=BackendType.VLLM,
            server_url="https://example.modal.run",
            app_name="vllm-test",
            tail_logs=True,
        )
        mock_tail = MagicMock()
        mock_tail.attach.return_value = iter([LogEvent(line="attached")])
        mock_tail.may_attach.return_value = False
        mock_tail.drain.return_value = iter(())
        with patch("llm_launchpad.core.warmup._ModalLogTail", return_value=mock_tail):
            gen = WarmupRunner().run_request(req)
            next(gen)  # StateChangeEvent
            next(gen)  # Probing log
            next(gen)  # inside try: attach() log
            gen.close()
        mock_tail.stop.assert_called_once()

    def test_preflight_rejects_zero_gpu_count(self) -> None:
        request = DeploymentRequest(
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            gpu_type="H100",
            gpu_count=0,
        )
        findings = validate_request(request)
        self.assertTrue(any(f.code == "gpu-count-invalid" for f in findings))

    def test_job_store_clear_resource_and_terminal_cancel_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs.db")
            config = DeploymentConfig(
                backend=BackendType.VLLM,
                provider=ComputeProvider.MODAL,
                model_name="Qwen/Qwen3-8B",
                app_name="vllm-a",
            )
            job = store.create_job(config)
            store.set_resource(job.id, "res-1")
            store.clear_resource(job.id)
            updated = store.get_job(job.id)
            assert updated is not None
            self.assertIsNone(updated.resource_app_id)

            store.finish_job(job.id, status="succeeded", outcome="deployed")
            self.assertFalse(store.request_cancel(job.id))
            final = store.get_job(job.id)
            assert final is not None
            self.assertEqual(final.status, "succeeded")

    def test_local_watchdog_retries_when_stop_returns_false(self) -> None:
        now = [100.0]
        stop_calls = [0]

        def fake_stop() -> bool:
            stop_calls[0] += 1
            return stop_calls[0] >= 2

        def fake_sleep(seconds: float) -> None:
            now[0] += seconds

        fake_resp = MagicMock(ok=True, text="vllm:generation_tokens_total 10\n")
        with patch("requests.get", return_value=fake_resp):
            reason = run_local_watchdog(
                metrics_url="http://127.0.0.1:8000/metrics",
                endpoint_api_key="key",
                idle_seconds=60,
                pod_exists=lambda: True,
                stop=fake_stop,
                poll_seconds=60.0,
                clock=lambda: now[0],
                sleep=fake_sleep,
            )
        self.assertEqual(reason, "stopped after idle window")
        self.assertEqual(stop_calls[0], 2)


class PrimeAndVastAuditTests(unittest.TestCase):
    def test_vast_failure_log_redacts_secrets(self) -> None:
        fake_ssh = MagicMock()
        fake_ssh.run.return_value = "token=hf_abc123 key=exact-endpoint-key Bearer supersecret\n"
        backend = VastDeploymentBackend(api=MagicMock(), state=MagicMock())
        record = MagicMock(name="vast-test", endpoint_api_key="exact-endpoint-key")
        instance = MagicMock(ssh_host="1.2.3.4", ssh_port=22)
        with patch("llm_launchpad.core.vast_deployment.VastSsh", return_value=fake_ssh):
            lines = backend._failure_log(record, instance, fallback=("fallback",))
        joined = "\n".join(lines)
        self.assertNotIn("hf_abc123", joined)
        self.assertNotIn("exact-endpoint-key", joined)
        self.assertNotIn("supersecret", joined)

    def test_prime_disk_selection_rejects_too_small_maximum_size(self) -> None:
        gpu_offer = ComputeOffer(
            id="gpu-1",
            cloud_id="cloud-1",
            provider_name="prov",
            data_center="us-east-1",
            gpu_type="H100_80GB",
            gpu_count=1,
            gpu_memory_gb=80,
            price_per_hour=2.0,
            security="secure_cloud",
            stock_status="Available",
            images=("ubuntu_22_cuda_12",),
        )
        too_small = PrimeDiskOffer(
            cloud_id="cloud-1",
            provider_name="prov",
            data_center="us-east-1",
            country="US",
            region="us-east",
            price_per_gb_hour=0.001,
            minimum_size_gb=10,
            maximum_size_gb=100,
            stock_status="Available",
            raw={},
        )
        big_enough = PrimeDiskOffer(
            cloud_id="cloud-1",
            provider_name="prov",
            data_center="us-east-1",
            country="US",
            region="us-east",
            price_per_gb_hour=0.002,
            minimum_size_gb=10,
            maximum_size_gb=500,
            stock_status="Available",
            raw={},
        )
        self.assertIsNone(matching_disk_offer([too_small], gpu_offer, required_gb=250))
        selected = matching_disk_offer([too_small, big_enough], gpu_offer, required_gb=250)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.maximum_size_gb, 500)

    def test_prime_runtime_engram_table_included_in_fit_params(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            repo_id="org/model-GGUF",
            quant="Q4_K_M",
            gpu_type="H100_80GB",
            gpu_count=1,
            engram_table=EngramTable(
                repo_id="org/model-GGUF",
                revision="main",
                filename="engram.gguf",
                size_bytes=1024,
            ),
            placement_assessment=PlacementAssessment(
                fits=True,
                fingerprint="fp-1",
                tuning=RuntimeTuning(),
                memory=MemoryEstimate(
                    weights_gb=15.0,
                    kv_cache_gb=2.0,
                    compute_gb=1.0,
                    speculative_gb=0.0,
                    reserve_gb=2.0,
                    total_gb=20.0,
                    per_device_required_gb=(20.0,),
                ),
            ),
        )
        launch = PrimeLaunchSpec(
            offer_image="ubuntu_22_cuda_12",
            container_image="ghcr.io/ggml-org/llama.cpp:server-cuda",
        )
        cmd = build_bootstrap_docker_command(config, launch)
        inner = cmd[-1]
        fit_idx = inner.find("llama-fit-params")
        self.assertGreater(fit_idx, -1)
        fit_segment = inner[fit_idx : inner.find("LLM_LAUNCHPAD_ATTESTATION", fit_idx)]
        self.assertIn("--ple", fit_segment)


class CorePlanningAndUtilsAuditTests(unittest.TestCase):
    def test_estimate_evaluation_peak_memory_is_elementwise_per_device(self) -> None:
        record = EvaluationRecord(
            record_id="rec-1",
            scenario=WorkloadScenario(
                id="sc-1",
                display_name="s",
                prompt_tokens=100,
                output_tokens=100,
                concurrency=1,
                request_count=1,
            ),
            runtime=RuntimeIdentity(model_id="org/model", gpu_type="H100", gpu_count=2),
            prediction=PredictionSnapshot(
                predicted_fits=True,
                calibrated_per_device_required_gb=(30.0, 30.0),
                predicted_single_tps=50.0,
                predicted_aggregate_tps=100.0,
            ),
            outcome=OutcomeKind.SUCCESS,
            memory_observations=(
                MemoryObservation(per_device_used_gb=(20.0, 28.0), phase="load"),
                MemoryObservation(per_device_used_gb=(25.0, 22.0), phase="runtime"),
            ),
            performance=(
                PerformancePoint(
                    prompt_tokens=100,
                    output_tokens=100,
                    concurrency=1,
                    output_tokens_per_second=None,
                    aggregate_output_tokens_per_second=None,
                ),
            ),
        )
        self.assertEqual(_observed_peak_per_device_gb(record), (25.0, 28.0))
        scored = score_record(record)
        self.assertIsNone(scored.single_relative_error)
        self.assertIsNone(scored.aggregate_relative_error)

    def test_hf_extract_parameter_count_handles_safetensors_object(self) -> None:
        class _FakeSafeTensors:
            total = 8_000_000_000
            parameters = {"BF16": 8_000_000_000}

        self.assertEqual(
            _extract_parameter_count("org/model", None, _FakeSafeTensors()),
            8_000_000_000.0,
        )

    def test_gguf_exec_command_handles_trailing_hf_file(self) -> None:
        cmd = gguf_exec_command(
            ["llama-server", "--ctx-size", "8192", "--hf-file", GGUF_FILE_PLACEHOLDER],
            weights_args_path="/tmp/weights.args",
        )
        else_branch = cmd.split("\nelse\n", 1)[1]
        self.assertNotIn("--hf-file", else_branch)
        self.assertNotIn(GGUF_FILE_PLACEHOLDER, cmd)
        self.assertIn("exec llama-server --ctx-size 8192", else_branch)

    def test_connection_store_atomic_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "connections.json"
            config = DeploymentConfig(
                backend=BackendType.VLLM,
                provider=ComputeProvider.VAST,
                model_name="Qwen/Qwen3-8B",
                endpoint_api_key="secret",
            )
            endpoint = EndpointInfo(
                name="vast-app",
                app_id="1234",
                state="running",
                backend=BackendType.VLLM,
                provider=ComputeProvider.VAST,
                web_url="http://1.2.3.4:8000",
                endpoint_api_key="secret",
            )
            save_connection(config, endpoint, path=store_path)
            entries = load_connection_entries(path=store_path)
            self.assertIn("vast-app", entries)
            self.assertEqual(entries["vast-app"]["provider"], "vast")

    def test_tool_call_probe_handles_malformed_shapes(self) -> None:
        self.assertFalse(classify_tool_call_response({"choices": ["not-a-dict"]}).passed)
        self.assertFalse(
            classify_tool_call_response(
                {"choices": [{"message": {"tool_calls": ["not-a-dict"]}}]}
            ).passed
        )
        self.assertFalse(
            classify_tool_call_response(
                {"choices": [{"message": {"tool_calls": [{"function": "bad"}]}}]}
            ).passed
        )

    def test_prometheus_parser_handles_tabs_and_openmetrics_exemplars(self) -> None:
        text = (
            'vllm:prompt_tokens_total{model="a#b"}\t42.0 # {trace_id="xyz"} 1.0 1700000000\n'
            "llamacpp:tokens_predicted_total\t15\n"
        )
        parsed = parse_prometheus_text(text)
        self.assertEqual(parsed["vllm:prompt_tokens_total"], 42.0)
        self.assertEqual(parsed["llamacpp:tokens_predicted_total"], 15.0)

    def test_serving_metrics_preserves_counter_when_scrape_omits_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ServingMetricsTracker(Path(tmp) / "serving_usage.json")
            first = tracker.record(
                "modal:vllm-a",
                ServingStats(
                    captured_at=2000.0,
                    prompt_tokens=100.0,
                    generation_tokens=50.0,
                    runtime_started_at=1000.0,
                ),
            )
            self.assertEqual(first.total_prompt_tokens, 100.0)
            second = tracker.record(
                "modal:vllm-a",
                ServingStats(
                    captured_at=2020.0,
                    prompt_tokens=None,
                    generation_tokens=60.0,
                    runtime_started_at=None,
                ),
            )
            self.assertEqual(second.total_prompt_tokens, 100.0)
            self.assertEqual(second.total_generation_tokens, 60.0)
            third = tracker.record(
                "modal:vllm-a",
                ServingStats(
                    captured_at=2040.0,
                    prompt_tokens=125.0,
                    generation_tokens=70.0,
                    runtime_started_at=1000.0,
                ),
            )
            self.assertEqual(third.total_prompt_tokens, 125.0)

    def test_evaluation_store_sanitizes_record_id_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evals"
            record = EvaluationRecord(
                record_id="../../escape-me",
                scenario=WorkloadScenario(
                    id="sc-1",
                    display_name="s",
                    prompt_tokens=100,
                    output_tokens=100,
                    concurrency=1,
                    request_count=1,
                ),
                runtime=RuntimeIdentity(model_id="org/model"),
                prediction=PredictionSnapshot(predicted_fits=True),
                outcome=OutcomeKind.SUCCESS,
            )
            written = write_evaluation_record(record, root)
            self.assertEqual(written.parent.resolve(), root.resolve())
            self.assertTrue(written.exists())

    def test_llamacpp_planner_rejects_unknown_cache_type(self) -> None:
        memory = MemoryEstimate(
            weights_gb=8.0,
            kv_cache_gb=2.0,
            compute_gb=0.5,
            speculative_gb=0.0,
            reserve_gb=1.0,
            total_gb=11.5,
            per_device_required_gb=(11.5,),
            source="gguf-metadata",
        )
        with self.assertRaises(ValueError):
            memory_for_cache_type(memory, from_cache_type="f16", to_cache_type="bogus_quant")

    def test_endpoint_runtime_ignores_nan_and_corrupt_floats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_path = Path(tmp) / "endpoint_runtime.json"
            runtime_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            "modal:vllm-nan": {
                                "hourly_cost_usd": "NaN",
                                "run_started_at_epoch": "corrupt",
                                "cumulative_cost_usd": "inf",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            tracker = EndpointRuntimeTracker(runtime_path)
            row = EndpointInfo(
                name="vllm-nan",
                app_id="ap-1",
                state="running",
                backend=BackendType.VLLM,
                provider=ComputeProvider.MODAL,
            )
            tracker.observe_rows([row], now=1000.0, explicit=False)
            self.assertIsNone(row.run_started_at_epoch)
            self.assertIsNone(row.cumulative_cost_usd)


if __name__ == "__main__":
    unittest.main()

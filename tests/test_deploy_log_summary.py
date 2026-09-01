from __future__ import annotations

import unittest

from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.tui.deploy_log_summary import (
    DeployLogSummarizer,
    beautify_summary_line,
    classify_summary_kind,
    percent_in_text,
    summary_progress_parts,
)


class DeployLogSummarizerTests(unittest.TestCase):
    def test_llamacpp_cache_hit_maps_to_loading_cached_model(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform("🦙 cache hit: using cached GGUF for repo/model", OperationType.WARMUP)
        self.assertEqual(out, ["Loading cached model"])

    def test_llamacpp_offload_maps_to_loading_weights_on_gpu(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform("load_tensors: offloaded 33/33 layers to GPU", OperationType.WARMUP)
        self.assertEqual(out, ["Loading weights on GPU"])

    def test_llamacpp_metadata_dump_line_is_suppressed(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform(
            "llama_model_loader: - kv   0: general.architecture str = llama",
            OperationType.WARMUP,
        )
        self.assertEqual(out, [])

    def test_llamacpp_preload_downloading_banner_is_not_treated_as_download(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform(
            "🦙 downloading repo/model (patterns: ['*Q4*.gguf'], revision: None) into /root/.cache/huggingface/hub",
            OperationType.DEPLOY,
        )
        self.assertEqual(out, [])

    def test_llamacpp_download_progress_maps_to_downloading_model(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform(
            "🦙 download in progress... elapsed=20s files=1 size=2.28GiB complete=0 inflight=1 avg_rate=12.34MiB/s",
            OperationType.DEPLOY,
        )
        self.assertEqual(out, ["Downloading model"])

    def test_llamacpp_download_progress_maps_to_percent_update_when_available(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform(
            "🦙 download in progress... elapsed=20s files=1 size=2.28GiB/10.00GiB pct=22% complete=0 inflight=1 avg_rate=12.34MiB/s",
            OperationType.DEPLOY,
        )
        self.assertEqual(out, ["Downloading model (22%)"])

    def test_llamacpp_download_progress_emits_new_line_when_percent_changes(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        first = s.transform(
            "🦙 download in progress... elapsed=20s files=1 size=2.28GiB/10.00GiB pct=22% complete=0 inflight=1 avg_rate=12.34MiB/s",
            OperationType.DEPLOY,
        )
        second = s.transform(
            "🦙 download in progress... elapsed=40s files=2 size=3.28GiB/10.00GiB pct=32% complete=1 inflight=1 avg_rate=12.34MiB/s",
            OperationType.DEPLOY,
        )
        self.assertEqual(first, ["Downloading model (22%)"])
        self.assertEqual(second, ["Downloading model (32%)"])

    def test_llamacpp_download_progress_cache_hit_is_suppressed(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform(
            "🦙 download in progress... elapsed=0s files=1 size=2.28GiB complete=1 inflight=0 avg_rate=0.00MiB/s",
            OperationType.DEPLOY,
        )
        self.assertEqual(out, [])

    def test_vllm_resolved_architecture_maps_to_loading_model_metadata(self) -> None:
        s = DeployLogSummarizer(BackendType.VLLM)
        out = s.transform(
            "(APIServer pid=4) INFO 02-25 03:08:40 [model.py:514] Resolved architecture: LlamaForCausalLM",
            OperationType.WARMUP,
        )
        self.assertEqual(out, ["Loading model metadata"])

    def test_vllm_cuda_graph_progress_spam_dedupes_to_single_milestone(self) -> None:
        s = DeployLogSummarizer(BackendType.VLLM)
        first = s.transform(
            "Capturing CUDA graphs (decode, FULL): 3%| | 1/35 [00:00<00:05, 5.76it/s]",
            OperationType.WARMUP,
        )
        second = s.transform(
            "Capturing CUDA graphs (decode, FULL): 6%| | 2/35 [00:00<00:04, 6.87it/s]",
            OperationType.WARMUP,
        )
        self.assertEqual(first, ["Capturing CUDA graphs"])
        self.assertEqual(second, [])

    def test_vllm_compile_lines_map_to_compiling_kernels(self) -> None:
        s = DeployLogSummarizer(BackendType.VLLM)
        out = s.transform(
            "(EngineCore_DP0 pid=38) INFO [backends.py:703] Dynamo bytecode transform time: 6.38 s",
            OperationType.WARMUP,
        )
        self.assertEqual(out, ["Compiling kernels"])

    def test_error_line_passes_through(self) -> None:
        s = DeployLogSummarizer(BackendType.VLLM)
        line = "RuntimeError: CUDA out of memory"
        self.assertEqual(s.transform(line, OperationType.WARMUP), [line])

    def test_llamacpp_numeric_500_series_values_do_not_trigger_http_error_passthrough(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(s.transform("print_info: n_embd_k_gqa          = 512", OperationType.WARMUP), [])
        self.assertEqual(
            s.transform(
                "system_info: ... CUDA : ARCHS = 500,610,700,750,800,860,890 | ...",
                OperationType.WARMUP,
            ),
            [],
        )

    def test_llamacpp_dev_web_function_url_is_hidden_in_summary(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform(
            "├── 🔨 Created web function serve => https://alice--llamacpp-test-serve-dev.modal.run",
            OperationType.DEPLOY,
        )
        self.assertEqual(out, [])

    def test_llamacpp_next_steps_guidance_is_hidden_in_summary(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform(
                "   Use the exact URL from the `Created web function serve => ...` line above.",
                OperationType.DEPLOY,
            ),
            [],
        )

    def test_dedupe_resets_on_operation_change(self) -> None:
        s = DeployLogSummarizer(BackendType.VLLM)
        line = "Starting vLLM command:"
        first = s.transform(line, OperationType.DEPLOY)
        second = s.transform(line, OperationType.DEPLOY)
        third = s.transform(line, OperationType.WARMUP)
        self.assertEqual(first, ["Starting server"])
        self.assertEqual(second, [])
        self.assertEqual(third, ["Starting server"])

    def test_modal_commands_and_env_dumps_are_hidden_or_mapped(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform(
                "Running: modal run -m llm_launchpad.backends.modal_llamacpp_app::main --preload",
                OperationType.DEPLOY,
            ),
            ["Preparing model cache"],
        )
        self.assertEqual(
            s.transform("  env: SCALEDOWN_WINDOW=1800, GPU_CONFIG=T4:1", OperationType.DEPLOY),
            [],
        )
        self.assertEqual(
            s.transform(
                "Running: modal deploy -m llm_launchpad.backends.modal_llamacpp_app --name llamacpp-logbeauty",
                OperationType.DEPLOY,
            ),
            ["Publishing endpoint"],
        )
        self.assertEqual(
            s.transform_state(
                "modal run -m llm_launchpad.backends.modal_llamacpp_app::main --preload",
                OperationType.DEPLOY,
            ),
            [],
        )

    def test_huggingface_fetch_progress_maps_to_download_percent(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform("Fetching 1 files:   0%|          | 0/1 [00:00<?, ?it/s]", OperationType.DEPLOY),
            ["Downloading model"],
        )
        self.assertEqual(
            s.transform(
                "Fetching 1 files: 100%|██████████| 1/1 [00:12<00:00, 12.15s/it]",
                OperationType.DEPLOY,
            ),
            ["Downloading model (100%)"],
        )

    def test_llamacpp_zero_percent_with_known_total_is_a_real_download(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform(
                "🦙 download in progress... elapsed=0s files=0 size=0.00GiB/0.37GiB pct=0% complete=0 inflight=0 avg_rate=0.00MiB/s",
                OperationType.DEPLOY,
            ),
            ["Downloading model"],
        )

    def test_found_gguf_entries_are_not_called_cache_hits(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform(
                "🦙 found GGUF entries: ['hub/models--bartowski--Qwen2.5-0.5B-Instruct-GGUF/snapshots/abc/model.gguf']",
                OperationType.DEPLOY,
            ),
            [],
        )

    def test_weights_cached_maps_to_model_cached(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform("✅ Weights cached in Modal Volume (1 GGUF file(s)).", OperationType.DEPLOY),
            ["Model cached"],
        )

    def test_cuda_init_maps_to_initializing_gpu(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform("ggml_cuda_init: found 1 CUDA devices:", OperationType.WARMUP),
            ["Initializing GPU"],
        )

    def test_connection_summary_and_test_command_are_hidden(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(s.transform("=== OpenAI-compatible ===", OperationType.DEPLOY), [])
        self.assertEqual(s.transform("Base URL: https://example.modal.run/v1", OperationType.DEPLOY), [])
        self.assertEqual(
            s.transform(
                "Test command:\ncurl -s -X POST https://example.modal.run/v1/completions "
                "-H 'Authorization: Bearer super-secret'",
                OperationType.WARMUP,
            ),
            [],
        )

    def test_bearer_tokens_are_redacted_in_passthrough_errors(self) -> None:
        s = DeployLogSummarizer(BackendType.VLLM)
        out = s.transform(
            "RuntimeError: Authorization: Bearer super-secret was rejected",
            OperationType.WARMUP,
        )
        self.assertEqual(
            out,
            ["RuntimeError: Authorization: Bearer $LLM_LAUNCHPAD_API_KEY was rejected"],
        )
        self.assertNotIn("super-secret", out[0])

    def test_prime_milestones_are_short_and_hide_internal_ids(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform(
                "Selected Prime offer b1fae8: 1x A6000_48GB via massedcompute (US, $0.54/hr)",
                OperationType.DEPLOY,
            ),
            ["GPU ready: 1× A6000 48GB · US · $0.54/hr"],
        )
        self.assertEqual(
            s.transform("Prime runtime: portable bootstrap on ubuntu_22_cuda_12", OperationType.DEPLOY),
            [],
        )
        self.assertEqual(
            s.transform("Prime pod created: 4d907acda6124ed98d817a26eb233225", OperationType.DEPLOY),
            ["Provisioning machine"],
        )
        self.assertEqual(
            s.transform("Prime pod state: PROVISIONING/FINISHED", OperationType.DEPLOY),
            [],
        )
        self.assertEqual(
            s.transform("Prime pod state: ACTIVE/FINISHED", OperationType.DEPLOY),
            ["Machine ready"],
        )
        self.assertEqual(
            s.transform(
                "Prime networking: secure tunnel t-2-25790e93174a990b; registration expires 2026-09-06T12:54:28",
                OperationType.DEPLOY,
            ),
            ["Opening secure endpoint"],
        )
        self.assertEqual(
            s.transform(
                "Prime runtime: runtime container is loading the model (network 302MB / 2.86MB)",
                OperationType.DEPLOY,
            ),
            ["Loading model"],
        )
        self.assertEqual(
            s.transform(
                "Prime runtime: runtime container is downloading the model (74%)",
                OperationType.DEPLOY,
            ),
            ["Downloading model (74%)"],
        )
        self.assertEqual(
            s.transform(
                "Prime runtime: runtime container is loading the model (40%)",
                OperationType.DEPLOY,
            ),
            ["Loading model (40%)"],
        )
        self.assertEqual(
            s.transform("Prime runtime: OpenAI-compatible endpoint is ready", OperationType.DEPLOY),
            ["Runtime ready"],
        )
        self.assertEqual(
            s.transform(
                "Prime endpoint URL ready: https://t-2-25790e93174a990b.tunnel.pinfra.io",
                OperationType.DEPLOY,
            ),
            [],
        )

    def test_probing_url_collapses_to_waiting_for_readiness(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform(
                "Probing readiness at: https://example.modal.run/v1/completions",
                OperationType.DEPLOY,
            ),
            ["Waiting for readiness"],
        )
        self.assertEqual(
            s.transform_state("Probing https://example.modal.run", OperationType.WARMUP),
            [],
        )

    def test_live_modal_sample_collapses_to_friendly_stages(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        lines = [
            "Running: modal run -m llm_launchpad.backends.modal_llamacpp_app::main --preload",
            "  env: SCALEDOWN_WINDOW=1800, GPU_CONFIG=T4:1, MODAL_APP_NAME=llamacpp-logbeauty",
            "✓ Created objects.",
            "├── 🔨 Created mount PythonPackage:llm_launchpad",
            "🦙 acquired download lease for bartowski/Qwen2.5-0.5B-Instruct-GGUF (revision: main)",
            "Fetching 1 files: 100%|██████████| 1/1 [00:12<00:00, 12.15s/it]",
            "✅ Weights cached in Modal Volume (1 GGUF file(s)).",
            "Next steps:",
            "Running: modal deploy -m llm_launchpad.backends.modal_llamacpp_app --name llamacpp-logbeauty",
            "✓ App deployed in 1.494s! 🎉",
            "Probing readiness at: https://example.modal.run/v1/completions",
            "🦙 cache hit: using cached GGUF for bartowski/Qwen2.5-0.5B-Instruct-GGUF",
            "ggml_cuda_init: found 1 CUDA devices:",
            "print_info: n_embd                = 896",
            "load_tensors: offloaded 25/25 layers to GPU",
            "Server is ready!",
            "Test command:\ncurl -s -X POST https://example.modal.run/v1/completions",
        ]
        out: list[str] = []
        op = OperationType.DEPLOY
        for line in lines:
            if line.startswith("Probing "):
                op = OperationType.WARMUP
            out.extend(s.transform(line, op))
        self.assertEqual(
            out,
            [
                "Preparing model cache",
                "Downloading model",
                "Downloading model (100%)",
                "Model cached",
                "Publishing endpoint",
                "Endpoint published",
                "Waiting for readiness",
                "Loading cached model",
                "Initializing GPU",
                "Loading weights on GPU",
                "Server is ready!",
            ],
        )

    def test_vllm_shard_progress_maps_to_percent_update(self) -> None:
        s = DeployLogSummarizer(BackendType.VLLM)
        first = s.transform(
            "Loading safetensors checkpoint shards:  3%| | 1/35 [00:00<00:05, 5.76it/s]",
            OperationType.WARMUP,
        )
        second = s.transform(
            "Loading safetensors checkpoint shards: 12%| | 4/35 [00:00<00:04, 6.87it/s]",
            OperationType.WARMUP,
        )
        self.assertEqual(first, ["Loading model weights (3%)"])
        self.assertEqual(second, ["Loading model weights (12%)"])

    def test_summary_progress_helpers_split_percent_and_classify_kind(self) -> None:
        self.assertEqual(summary_progress_parts("Downloading model (22%)"), ("Downloading model", 22))
        self.assertEqual(summary_progress_parts("· Loading model"), ("Loading model", None))
        self.assertEqual(percent_in_text("runtime container is downloading the model (74%)"), 74)
        self.assertIsNone(percent_in_text("runtime container is loading the model (network 1.2GB / 8GB)"))
        self.assertEqual(classify_summary_kind("Downloading model (22%)"), "step")
        self.assertEqual(classify_summary_kind("Machine ready"), "done")
        self.assertEqual(classify_summary_kind("Prime cache disk unavailable; model weights will not persist"), "info")
        self.assertEqual(classify_summary_kind("RuntimeError: CUDA out of memory"), "error")

    def test_beautify_summary_line_adds_status_markers(self) -> None:
        self.assertEqual(beautify_summary_line("Server is ready!"), "✓ Server is ready!")
        self.assertEqual(beautify_summary_line("Downloading model (22%)"), "· Downloading model (22%)")
        self.assertEqual(
            beautify_summary_line("Downloading model (22%)", spinner_frame="⠋"),
            "⠋ Downloading model (22%)",
        )
        self.assertEqual(beautify_summary_line("RuntimeError: CUDA out of memory"), "✗ RuntimeError: CUDA out of memory")
        self.assertEqual(beautify_summary_line("✓ already formatted"), "✓ already formatted")


if __name__ == "__main__":
    unittest.main()

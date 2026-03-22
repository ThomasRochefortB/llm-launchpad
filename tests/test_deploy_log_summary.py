from __future__ import annotations

import unittest

from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.tui.deploy_log_summary import DeployLogSummarizer


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


if __name__ == "__main__":
    unittest.main()

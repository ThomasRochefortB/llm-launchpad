from __future__ import annotations

import unittest

from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.tui.deploy_log_summary import (
    DeployLogSummarizer,
    beautify_summary_line,
    classify_summary_kind,
    is_startup_phase_line,
    parse_startup_phase_line,
    percent_in_text,
    startup_phase_summary_line,
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

    def test_fit_planner_arithmetic_survives_the_noise_filter(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        line = (
            "llama_params_fit_impl: projected to use 205000 MiB of device memory "
            "vs. 160000 MiB of free device memory"
        )

        # The only statement of how far a rejected plan overflowed.
        self.assertEqual(s.transform(line, OperationType.WARMUP), [line])

    def test_fit_planner_progress_stays_hidden(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)

        self.assertEqual(
            s.transform(
                "llama_params_fit: fitting params to free memory took 0.28 seconds",
                OperationType.WARMUP,
            ),
            [],
        )

    def test_startup_phase_timing_lines_survive_the_summarizer(self) -> None:
        # The phase timer's lines are the payload callers parse for
        # before/after comparisons; the headless CLI prints every warmup
        # event through here, so a dropped line is a measurement lost.
        for backend in (BackendType.LLAMACPP, BackendType.VLLM):
            s = DeployLogSummarizer(backend)
            for line in (
                "startup-phase deploy 12.3s",
                "startup-phase warmup-wait 300.0s",
                "startup-phase calibration 45.1s",
                "startup-phase total 357.4s",
            ):
                self.assertEqual(s.transform(line, OperationType.WARMUP), [line], line)

    def test_an_unknown_phase_name_is_not_a_timing_summary(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            s.transform("startup-phase frobnicate 1.0s", OperationType.WARMUP),
            [],
        )
        self.assertEqual(
            s.transform("startup-phase deploy nope", OperationType.WARMUP),
            [],
        )

    def test_a_phase_line_round_trips_through_the_parser(self) -> None:
        line = startup_phase_summary_line("warmup-wait", 300.04)
        self.assertEqual(parse_startup_phase_line(line), ("warmup-wait", 300.0))
        self.assertTrue(is_startup_phase_line(line))
        # The CLI prints milestones beautified; the parser reads what it prints.
        self.assertEqual(
            parse_startup_phase_line(beautify_summary_line(line)),
            ("warmup-wait", 300.0),
        )
        self.assertFalse(is_startup_phase_line("Server is ready!"))

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

    def test_modal_image_build_steps_map_to_building_runtime_image(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        for line in (
            "=> Step 2: COPY --from=build /src/build/bin/ /app/",
            "=> Step 5: ENTRYPOINT [\"/app/llama-server\"]",
            "Saving image...",
            "Image saved, took 8.12s",
            "Built image im-N4h4A1AiXcKoJJ8FpACJIf in 603.75s",
            "(Reading database ... 30%",
            "Preparing to unpack .../ca-certificates_20260601~22.04.1_all.deb ...",
            "Unpacking libgomp1:amd64 (12.3.0-1ubuntu1~22.04.3) ...",
            "Setting up ca-certificates (20260601~22.04.1) ...",
            "debconf: unable to initialize frontend: Dialog",
        ):
            out = s.transform(line, OperationType.DEPLOY)
            self.assertIn(out, (["Building runtime image"], []), line)

    def test_modal_cmake_progress_maps_to_building_runtime_image_percent(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        out = s.transform(
            "[ 45%] Building CXX object CMakeFiles/llama.dir/common.cpp.o",
            OperationType.DEPLOY,
        )
        self.assertEqual(out, ["Building runtime image (45%)"])

    def test_modal_image_build_milestone_updates_in_place(self) -> None:
        s = DeployLogSummarizer(BackendType.LLAMACPP)
        first = s.transform("=> Step 2: COPY --from=build /src/build/bin/ /app/", OperationType.DEPLOY)
        second = s.transform(
            "[ 45%] Building CXX object CMakeFiles/llama.dir/common.cpp.o",
            OperationType.DEPLOY,
        )
        self.assertEqual(first, ["Building runtime image"])
        self.assertEqual(second, ["Building runtime image (45%)"])

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
        # Prime reports bytes rather than a percentage for a plain
        # snapshot download; a 55 GB model with no number on screen is
        # twenty minutes of an unchanging row.
        self.assertEqual(
            s.transform(
                "Prime runtime: runtime container is downloading the model, 46.6 GB so far",
                OperationType.DEPLOY,
            ),
            ["Downloading model — 46.6 GB so far"],
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


class PreflightDecisionTests(unittest.TestCase):
    """Preflight decisions are the only report that MTP actually engaged.

    The confirm screen states what was planned; only these lines state what
    the deployment was launched with, so the summary view has to carry them.
    """

    def test_preflight_decisions_survive_the_summary_view(self) -> None:
        for line in (
            "MTP preflight: enabled native draft-mtp with up to 3 draft tokens.",
            "MTP preflight: The selected target GGUF has no embedded MTP heads. "
            "Using normal decoding.",
            "MTP preflight warning: metadata lookup failed. Disabling MTP and "
            "continuing with normal decoding.",
            "Speculative decoding disabled: unsupported method requested.",
            "Compatibility preflight: Architecture 'qwen35' is supported.",
            "Serving plan: full 262,144-token context, 4 parallel slot(s), "
            "GPU-only placement.",
        ):
            with self.subTest(line=line):
                summarizer = DeployLogSummarizer(BackendType.LLAMACPP)
                self.assertEqual(
                    summarizer.transform(line, OperationType.DEPLOY),
                    [line],
                )

    def test_runtime_chatter_is_still_dropped(self) -> None:
        summarizer = DeployLogSummarizer(BackendType.LLAMACPP)
        self.assertEqual(
            summarizer.transform("srv  update_slots: all slots are idle", OperationType.DEPLOY),
            [],
        )


class StickyMilestoneTests(unittest.TestCase):
    def test_server_ready_is_announced_once_across_deploy_and_warmup(self) -> None:
        """"Server is ready!" is sticky, but was emitted without being recorded.

        It bypassed ``_emit_once``, so it never entered the seen set that
        ``_STICKY_MILESTONES`` is filtered against, and the deploy summary
        showed the milestone twice.
        """
        summarizer = DeployLogSummarizer(BackendType.LLAMACPP)
        deploy = summarizer.transform("Server is ready!", OperationType.DEPLOY)
        warmup = summarizer.transform("Server is ready!", OperationType.WARMUP)
        self.assertEqual(deploy, ["Server is ready!"])
        self.assertEqual(warmup, [])


if __name__ == "__main__":
    unittest.main()


class PrimeWaitHeartbeatTests(unittest.TestCase):
    """A long silent step has to prove it is still a step, not a hang.

    Prime lines are only emitted when their detail text changes, and the
    detail for a model download is a constant string whenever the pod cannot
    report an expected size. A 109 GB download therefore produced one log line
    and then minutes of nothing on a screen that bills by the hour.
    """

    def _milestone(self, line: str) -> str | None:
        return DeployLogSummarizer(BackendType.LLAMACPP)._map_prime_line(line)

    def test_a_heartbeat_carries_its_wait_onto_the_milestone(self) -> None:
        self.assertEqual(
            self._milestone(
                "Prime runtime: runtime container is loading the model (waiting 5m20s)"
            ),
            "Loading model (5m20s)",
        )

    def test_the_ticking_row_replaces_itself_rather_than_stacking(self) -> None:
        # The monitor replaces the last step when the label matches, so the
        # label has to survive the clock changing.
        labels = {
            summary_progress_parts(
                self._milestone(
                    f"Prime runtime: runtime container is loading the model (waiting {elapsed})"
                )
                or ""
            )[0]
            for elapsed in ("30s", "5m20s", "1h04m20s")
        }

        self.assertEqual(labels, {"Loading model"})

    def test_a_real_percentage_is_left_to_speak_for_itself(self) -> None:
        # A percentage already shows movement; a clock beside it is noise, and
        # it would also push the percentage into the collapse label.
        milestone = self._milestone(
            "Prime runtime: runtime container is downloading the model (42%) "
            "(waiting 2m00s)"
        )

        self.assertEqual(milestone, "Downloading model (42%)")
        self.assertEqual(summary_progress_parts(milestone or ""), ("Downloading model", 42))

    def test_percentages_still_collapse_onto_one_row(self) -> None:
        labels = {
            summary_progress_parts(
                self._milestone(
                    f"Prime runtime: runtime container is downloading the model ({pct}%)"
                )
                or ""
            )[0]
            for pct in (7, 42, 99)
        }

        self.assertEqual(labels, {"Downloading model"})

    def test_a_hidden_line_stays_hidden_when_it_carries_a_wait(self) -> None:
        self.assertEqual(
            self._milestone("Prime runtime: portable bootstrap (waiting 45s)"), ""
        )


class VastProvisioningSummaryTests(unittest.TestCase):
    """A Vast rental's provisioning has to show on screen while it runs.

    Nothing a Vast host does before SSH matched a summary rule, so a deploy
    that spent six minutes pulling and unpacking its image showed the rental
    line and then an error. That reads as a hung client rather than a host
    that was still working.
    """

    def _transform(self, line: str) -> list[str]:
        return DeployLogSummarizer(BackendType.LLAMACPP).transform(
            line, OperationType.DEPLOY
        )

    def test_percentage_and_transfer_details_share_a_stable_row(self) -> None:
        result = self._transform(
            "Vast model starting: downloading weights (60%), 9.0 / 15.0 GB at 200 MB/s, ~30s remaining (waiting 1m00s)"
        )
        self.assertEqual(result, [
            "Downloading model (60%) — 9.0 / 15.0 GB at 200 MB/s, ~30s remaining",
        ])
        self.assertEqual(summary_progress_parts(result[0]), ("Downloading model", 60))

    def test_ssh_readiness_is_only_reported_after_a_successful_connection(self) -> None:
        self.assertEqual(self._transform("Vast waiting for SSH: success, running image (waiting 3m00s)"), ["Waiting for SSH (3m00s)"])
        self.assertEqual(self._transform("Vast SSH ready"), ["Machine ready"])
        self.assertEqual(self._transform("Vast opening secure endpoint"), ["Opening secure endpoint"])

    def test_image_pull_is_distinguished_when_reported(self) -> None:
        self.assertEqual(self._transform("Vast rental preparing: pulling image layers (waiting 30s)"), ["Pulling runtime image (30s)"])

    def test_the_provisioning_wait_reaches_the_summary(self) -> None:
        self.assertEqual(
            self._transform("Vast instance state: loading"), ["Provisioning machine"]
        )
        self.assertEqual(
            self._transform(
                "Vast rental preparing: #6 63.10 Get:9 noble/main Packages (waiting 30s)"
            ),
            ["Provisioning machine (30s)"],
        )

    def test_the_model_startup_wait_reaches_the_summary(self) -> None:
        self.assertEqual(
            self._transform(
                "Vast model starting: downloading weights, 6.6 GB fetched at 35 MB/s"
                " (waiting 4m10s)"
            ),
            ["Downloading model — 6.6 GB fetched at 35 MB/s (4m10s)"],
        )
        self.assertEqual(
            self._transform(
                "Vast model starting: load_tensors: offloaded 63/63 layers (waiting 9m00s)"
            ),
            ["Loading weights on GPU (9m00s)"],
        )

    def test_the_download_row_replaces_itself_rather_than_stacking(self) -> None:
        summarizer = DeployLogSummarizer(BackendType.LLAMACPP)
        labels = {
            summary_progress_parts(milestone)[0]
            for gigabytes, elapsed in ((1.2, "30s"), (6.6, "3m00s"), (14.9, "8m30s"))
            for milestone in summarizer.transform(
                f"Vast model starting: downloading weights, {gigabytes} GB fetched"
                f" (waiting {elapsed})",
                OperationType.DEPLOY,
            )
        }

        self.assertEqual(labels, {"Downloading model"})

    def test_the_ticking_row_replaces_itself_rather_than_stacking(self) -> None:
        summarizer = DeployLogSummarizer(BackendType.LLAMACPP)
        labels = {
            summary_progress_parts(milestone)[0]
            for elapsed in ("30s", "1m00s", "6m30s")
            for milestone in summarizer.transform(
                f"Vast rental preparing: no progress reported yet (waiting {elapsed})",
                OperationType.DEPLOY,
            )
        }

        self.assertEqual(labels, {"Provisioning machine"})

    def test_the_rest_of_the_rental_reads_as_milestones(self) -> None:
        self.assertEqual(self._transform("Vast instance state: running"), ["Waiting for SSH"])
        self.assertEqual(
            self._transform("Rented GPUs: 0:NVIDIA GeForce RTX 4090 23.5 GiB free"),
            ["GPU ready: 0:NVIDIA GeForce RTX 4090 23.5 GiB free"],
        )
        self.assertEqual(
            self._transform("Vast host is running but not accepting SSH yet: refused"),
            ["Waiting for SSH"],
        )
        self.assertEqual(
            self._transform(
                "Vast streaming chat verified. The endpoint is local to this computer; "
                "stop destroys the rental and its disk."
            ),
            ["Server is ready!"],
        )


class ElapsedFormatTests(unittest.TestCase):
    def test_waits_read_as_durations(self) -> None:
        from llm_launchpad.core.orchestrator import _format_elapsed

        self.assertEqual(_format_elapsed(45), "45s")
        self.assertEqual(_format_elapsed(320), "5m20s")
        self.assertEqual(_format_elapsed(3860), "1h04m20s")

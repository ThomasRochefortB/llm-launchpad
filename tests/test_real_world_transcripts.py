"""Golden regression fixtures derived from real Modal and Hugging Face outputs.

Each test replays an anonymized transcript or repository layout from an actual
production incident so formatting quirks (regional `.us-east.modal.run` URLs,
multi-line URL wrapping, auxiliary `dspark`/`engram` GGUFs, noisy 503 bursts)
cannot silently regress behind idealized unit mocks.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.backends import modal_llamacpp_app
from llm_launchpad.core.deploy_events import EndpointUrlResolver
from llm_launchpad.core.deploy_log_summary import DeployLogSummarizer
from llm_launchpad.core.deployment_preflight import resolve_engram_table
from llm_launchpad.core.gguf_metadata import (
    GGML_TYPE_F8_E4M3_ID,
    PLE_TENSOR_NAME,
    GgufPleTensorInfo,
    select_target_gguf_file,
)
from llm_launchpad.core.runtime_evidence import runtime_log_evidence
from llm_launchpad.core.warmup import WarmupRunner
from llm_launchpad.gguf_files import is_engram_table_file, is_target_gguf
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationIntent, OperationType
from llm_launchpad.protocol.events import DeploymentAttemptStartedEvent, LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig
from llm_launchpad.tui.deployment_progress import DeploymentProgress, Stage

# Anonymized real output from `modal run` + `modal deploy` on regional Modal URLs.
_MODAL_DEPLOY_TRANSCRIPT = (
    "✓ Initialized. View run at ",
    "https://modal.com/apps/user/main/ap-ephemeral123",
    "✓ Created objects.",
    "├── ✓ Uploaded PythonPackage:llm_launchpad",
    "├── ✓ Created Function serve-practical-scorpion",
    "│   └── https://user--llp-lc-683f7cb3380d-dev.us-east.modal.run ",
    "│       (unauthenticated)",
    "🦙 acquired download lease for unsloth/DeepSeek-V4.1-Flash-1017-GGUF (revision: main)",
    "🦙 download in progress... transport=xet-high-performance elapsed=61s files=5 size=45.32GiB/233.21GiB pct=19% complete=1 inflight=4 avg_rate=760.80MiB/s",
    "🦙 download in progress... transport=xet-high-performance elapsed=1067s files=5 size=233.21GiB/233.21GiB pct=100% complete=4 inflight=1 avg_rate=223.81MiB/s",
    "✅ Weights cached in Modal Volume (5 GGUF file(s)).",
    "Stopping app - local entrypoint completed.",
    "[modal-client] 2026-11-24T16:03:13-0500 Timed out waiting for final app logs.",
    "✓ App completed. View run at ",
    "https://modal.com/apps/user/main/ap-ephemeral123",
    "├── ✓ Created Function serve-practical-scorpion",
    "│   └── https://user--llp-lc-683f7cb3380d.us-east.modal.run ",
    "│       (unauthenticated)",
    "✓ App deployed in 1.909s! 🎉",
)

# Anonymized real llama.cpp b12464 startup output on 4x B200.
_LLAMACPP_STARTUP_TRANSCRIPT = (
    "🦙 cache hit: using cached GGUF for unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
    "🦙 using model file: /root/.cache/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/dfcee0ec/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf",
    "🦙 volume already hydrated for this snapshot; skipping warm read",
    "🦙 starting llama-server:",
    "   POST /v1/completions -> 503 Service Unavailable  (duration: 6.45 s, execution: 133.2 ms)",
    "load_tensors: offloaded 40/40 layers to GPU",
    "2.23.610.292 W resolve_fused_ops: layer 11 is assigned to device CUDA1 but fused DeepSeek V4 HC pre is assigned to device CUDA0 (usually due to missing support)",
    "   POST /v1/completions -> 503 Service Unavailable  (duration: 198.3 ms, execution: 94.5 ms)",
    "3.35.717.112 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 1048576, kv_unified = 'true'",
    "3.35.723.010 I srv  server_run_i: model loaded",
    "3.35.723.041 I srv  server_on_li: listening on http://0.0.0.0:8080",
)

# Anonymized real crash transcript when an auxiliary dspark GGUF was loaded as main model.
_DSPARK_CRASH_TRANSCRIPT = (
    "0.00.187.638 I srv    load_model: loading model '/root/.cache/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/dfcee0ec/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'",
    "   POST /v1/completions -> 503 Service Unavailable  (duration: 3.17 s, execution: 129.1 ms)",
    "0.00.879.321 E llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be set (this warning is normal during memory fitting)",
    "0.04.502.407 E srv    load_model: failed to create_context with model '/root/.cache/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/dfcee0ec/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'",
    "0.04.503.385 E srv  server_run_i: exiting due to model loading error",
)

# Realistic Hugging Face sibling layout for Unsloth's DeepSeek-V4.1-Flash-1017-GGUF.
_REAL_REPO_SIBLINGS = (
    ("README.md", 12_000),
    ("mmproj-BF16.gguf", 850_000_000),
    ("dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf", 7_980_000_000),
    ("engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file", 202_800_000_000),
    ("engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file", 110_600_000_000),
    ("engram/DeepSeek-V4.1-Flash-1017-engram-Q6_K.gguf_file", 161_300_000_000),
    ("UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00003.gguf", 48_000_000_000),
    ("UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00002-of-00003.gguf", 202_800_000_000),
    ("UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00003-of-00003.gguf", 44_000_000_000),
)


class ModalTranscriptRegressionTests(unittest.TestCase):
    def test_regional_deployed_url_beats_ephemeral_dev_url_and_drives_progress(self) -> None:
        resolver = EndpointUrlResolver()
        summarizer = DeployLogSummarizer(BackendType.LLAMACPP)
        progress = DeploymentProgress()
        progress.start(OperationType.DEPLOY, "Deploy")
        progress.on_attempt_started(
            DeploymentAttemptStartedEvent(
                attempt=1,
                provider=ComputeProvider.MODAL,
                backend=BackendType.LLAMACPP,
                intent=OperationIntent.SERVE,
                warmup=True,
                gpu_type="H200",
                gpu_count=4,
            )
        )

        milestones: list[str] = []
        for line in _MODAL_DEPLOY_TRANSCRIPT:
            event = LogEvent(line=line, operation=OperationType.DEPLOY)
            resolver.observe(event)
            for milestone in summarizer.transform(line, OperationType.DEPLOY):
                milestones.append(milestone)
                progress.on_milestone(milestone)

        self.assertEqual(resolver.url, "https://user--llp-lc-683f7cb3380d.us-east.modal.run")
        self.assertIn("Downloading model (19%)", milestones)
        self.assertIn("Downloading model (100%)", milestones)
        self.assertIn("Model cached", milestones)
        self.assertIn("Endpoint published", milestones)
        self.assertTrue(progress.endpoint_known)
        self.assertEqual(progress.active_stage(), Stage.GPU)

        # Replay real startup logs and verify 503s stay filtered while GPU residency is captured.
        warmup_milestones: list[str] = []
        for line in _LLAMACPP_STARTUP_TRANSCRIPT:
            for milestone in summarizer.transform(line, OperationType.WARMUP):
                warmup_milestones.append(milestone)
                progress.on_milestone(milestone)

        self.assertFalse(any("503 Service Unavailable" in item for item in warmup_milestones))
        self.assertIn("Starting server", warmup_milestones)
        self.assertEqual(progress.active_stage(), Stage.START)

        evidence = runtime_log_evidence("\n".join(_LLAMACPP_STARTUP_TRANSCRIPT))
        self.assertEqual((evidence.gpu_layers, evidence.total_layers), (40, 40))
        self.assertTrue(evidence.gpu_resident)

    def test_real_dspark_crash_transcript_aborts_warmup_immediately(self) -> None:
        class _CrashTail:
            def __init__(self, app_name: str) -> None:
                self.app_name = app_name
                self.seen_lines: set[str] = set()
                self.stopped: str | None = None
                self._emitted = False

            def may_attach(self) -> bool:
                return False

            def attach(self):  # type: ignore[no-untyped-def]
                yield from ()

            def drain(self):  # type: ignore[no-untyped-def]
                if not self._emitted:
                    self._emitted = True
                    for line in _DSPARK_CRASH_TRANSCRIPT:
                        yield LogEvent(line=line, operation=OperationType.WARMUP)

            def stop(self, reason: str) -> None:
                self.stopped = reason

        with (
            patch.dict(
                "sys.modules",
                {"requests": types.SimpleNamespace(post=lambda *_a, **_k: SimpleNamespace(status_code=503, text=""))},
            ),
            patch("llm_launchpad.core.warmup._ModalLogTail", _CrashTail),
        ):
            events = list(
                WarmupRunner().run(
                    backend=BackendType.LLAMACPP,
                    server_url="https://user--llp-lc-deadbeef0000.us-east.modal.run",
                    timeout=1800,
                    tail_logs=True,
                    app_name="llamacpp-ds41",
                )
            )

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("Choose a main-model quantization", completion.detail)


class RealHfRepoSnapshotTests(unittest.TestCase):
    def test_auxiliary_artifacts_never_hijack_main_model_or_engram_selection(self) -> None:
        siblings = [SimpleNamespace(rfilename=name, size=size) for name, size in _REAL_REPO_SIBLINGS]

        # Auxiliary artifacts (mmproj, dspark, engram) are rejected as target GGUFs.
        self.assertFalse(is_target_gguf("dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"))
        self.assertFalse(is_target_gguf("mmproj-BF16.gguf"))
        self.assertFalse(is_target_gguf("engram/DeepSeek-V4.1-Flash-1017-engram-Q6_K.gguf_file"))
        self.assertTrue(is_engram_table_file("engram/DeepSeek-V4.1-Flash-1017-engram-Q6_K.gguf_file"))

        # Requesting Q8_0 (which only exists as dspark) must not select the draft file.
        self.assertEqual(
            select_target_gguf_file(siblings, quant="UD-Q4_K_XL"),
            "UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00003.gguf",
        )
        self.assertNotIn("dspark", select_target_gguf_file(siblings, quant="Q8_0") or "")

        # MXFP8 type-43 shard is substituted with the highest-precision non-MXFP8 table.
        broken = "UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00002-of-00003.gguf"
        q6k = "engram/DeepSeek-V4.1-Flash-1017-engram-Q6_K.gguf_file"
        info = SimpleNamespace(sha="dfcee0ec8d6b874c6663eca0fc55d90ab32534c8", siblings=siblings)
        ple_by_file = {
            broken: GgufPleTensorInfo(PLE_TENSOR_NAME, (256, 768_022_850), GGML_TYPE_F8_E4M3_ID, broken),
            q6k: GgufPleTensorInfo(PLE_TENSOR_NAME, (256, 768_022_850), 14, q6k),
        }
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            repo_id="unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
            quant="UD-Q4_K_XL",
            gguf_architecture="deepseek41",
        )
        with (
            patch("huggingface_hub.HfApi.model_info", return_value=info),
            patch(
                "llm_launchpad.core.gguf_metadata.fetch_gguf_ple_tensor_info",
                side_effect=lambda _repo, filename, *, revision=None: ple_by_file.get(filename),
            ),
        ):
            table, _message = resolve_engram_table(config)

        assert table is not None
        self.assertEqual(table.filename, q6k)
        self.assertEqual(table.replaced_shard, broken)

        # Excluding the replaced shard still treats the remaining shards as a complete split set.
        remaining_paths = [
            Path("/cache/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00003.gguf"),
            Path("/cache/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00003-of-00003.gguf"),
        ]
        complete, incomplete = modal_llamacpp_app._partition_complete_gguf_candidates(
            remaining_paths,
            exclude_files=[table.replaced_shard],
        )
        self.assertEqual(sorted(complete), remaining_paths)
        self.assertEqual(incomplete, [])


if __name__ == "__main__":
    unittest.main()

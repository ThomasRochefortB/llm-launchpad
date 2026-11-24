"""External Engram (--ple) table detection and serving for DeepSeek-V4.1 GGUFs."""

from __future__ import annotations

import json
import struct
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_launchpad.backends import modal_llamacpp_app
from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.deployment_lifecycle import LifecycleAttempt, effective_config_for_attempt
from llm_launchpad.core.deployment_preflight import preflight_config, resolve_engram_table
from llm_launchpad.core.gguf_metadata import (
    GGML_TYPE_F8_E4M3_ID,
    PLE_TENSOR_NAME,
    GgufPleTensorInfo,
    parse_gguf_ple_tensor_info,
)
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.prime_backend import PrimeBackend, PrimeLaunchSpec
from llm_launchpad.core.vast_runtime import VAST_RUNTIME_DIR, vast_runtime_script
from llm_launchpad.gguf_files import is_engram_table_file
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    EngramTable,
    MemoryEstimate,
    PlacementAssessment,
    PrimeProviderOptions,
    RuntimeTuning,
    VastProviderOptions,
)

_REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
_SHA = "dfcee0ec8d6b874c6663eca0fc55d90ab32534c8"
_ENGRAM_ELEMENTS = 256 * 768_022_850
_MXFP8_SHARD = "UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00008-of-00009.gguf"
_Q6K_TABLE = "engram/DeepSeek-V4.1-Flash-1017-engram-Q6_K.gguf_file"


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _ple_header(tensor_name: str, type_id: int, *, n_tensors: int = 1) -> bytes:
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", 3, n_tensors, 1))
    payload.extend(_gguf_string("split.no"))
    payload.extend(struct.pack("<I", 2))  # uint16
    payload.extend(struct.pack("<H", 7))
    payload.extend(_gguf_string(tensor_name))
    payload.extend(struct.pack("<I", 2))
    payload.extend(struct.pack("<QQ", 256, 768_022_850))
    payload.extend(struct.pack("<IQ", type_id, 0))
    return bytes(payload)


def _sibling(name: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=size)


def _deepseek_config(**overrides: object) -> DeploymentConfig:
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.MODAL,
        repo_id=_REPO,
        quant="UD-Q4_K_XL",
        gguf_architecture="deepseek41",
        app_name="llamacpp-ds41",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _table(**overrides: object) -> EngramTable:
    base = EngramTable(
        repo_id=_REPO,
        revision=_SHA,
        filename=_Q6K_TABLE,
        size_bytes=161_300_000_000,
        replaced_shard=_MXFP8_SHARD,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class PleTensorParsingTests(unittest.TestCase):
    def test_parses_single_ple_tensor_type_and_dims(self) -> None:
        info = parse_gguf_ple_tensor_info(_ple_header(PLE_TENSOR_NAME, 43), source_file="x.gguf")

        assert info is not None
        self.assertEqual(info.dims, (256, 768_022_850))
        self.assertEqual(info.type_id, GGML_TYPE_F8_E4M3_ID)
        self.assertEqual(info.source_file, "x.gguf")

    def test_non_ple_and_multi_tensor_shards_are_ignored(self) -> None:
        self.assertIsNone(parse_gguf_ple_tensor_info(_ple_header("blk.0.attn_q.weight", 43)))
        self.assertIsNone(parse_gguf_ple_tensor_info(_ple_header(PLE_TENSOR_NAME, 43, n_tensors=2)))

    def test_mxfp8_collision_is_distinguished_from_real_f8_e4m3_by_size(self) -> None:
        info = GgufPleTensorInfo(PLE_TENSOR_NAME, (256, 768_022_850), GGML_TYPE_F8_E4M3_ID)

        self.assertTrue(info.is_mxfp8_type43_collision(202_800_000_000))
        self.assertTrue(info.is_mxfp8_type43_collision(None))
        # One byte per element is genuine F8_E4M3 with no MXFP8 scales.
        self.assertFalse(info.is_mxfp8_type43_collision(_ENGRAM_ELEMENTS))

    def test_k_quant_tables_never_collide(self) -> None:
        info = GgufPleTensorInfo(PLE_TENSOR_NAME, (256, 768_022_850), 14)
        self.assertFalse(info.is_mxfp8_type43_collision(161_300_000_000))

    def test_engram_table_filenames(self) -> None:
        self.assertTrue(is_engram_table_file(_Q6K_TABLE))
        self.assertFalse(is_engram_table_file(_MXFP8_SHARD))
        self.assertFalse(is_engram_table_file("engram/README.md"))


class ResolveEngramTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.info = SimpleNamespace(sha=_SHA, siblings=[
            _sibling("UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf", 40_000_000_000),
            _sibling(_MXFP8_SHARD, 202_800_000_000),
            _sibling("UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00009-of-00009.gguf", 30_000_000_000),
            _sibling("engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file", 202_800_000_000),
            _sibling("engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file", 110_600_000_000),
            _sibling(_Q6K_TABLE, 161_300_000_000),
        ])
        self.types = {
            _MXFP8_SHARD: 43,
            _Q6K_TABLE: 14,
            "engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file": 20,
        }

    def _fetch(self, _repo: str, filename: str, *, revision: str | None = None) -> GgufPleTensorInfo | None:
        self.assertEqual(revision, _SHA)
        type_id = self.types.get(filename)
        if type_id is None:
            return None
        return GgufPleTensorInfo(PLE_TENSOR_NAME, (256, 768_022_850), type_id, filename)

    def _resolve(self, config: DeploymentConfig) -> tuple[EngramTable | None, str | None]:
        with (
            patch("huggingface_hub.HfApi.model_info", return_value=self.info),
            patch("llm_launchpad.core.gguf_metadata.fetch_gguf_ple_tensor_info", side_effect=self._fetch),
        ):
            return resolve_engram_table(config)

    def test_substitutes_highest_precision_external_table_for_mxfp8_shard(self) -> None:
        table, message = self._resolve(_deepseek_config())

        self.assertEqual(table, _table())
        assert message is not None
        self.assertIn("F8_E4M3", message)
        self.assertIn("--ple", message)

    def test_fails_when_no_compatible_external_table_exists(self) -> None:
        self.info.siblings = [s for s in self.info.siblings if "engram/" not in s.rfilename]

        with self.assertRaisesRegex(ValueError, "MXFP8"):
            self._resolve(_deepseek_config())

    def test_compatible_quant_needs_no_external_table(self) -> None:
        self.types[_MXFP8_SHARD] = 13

        self.assertEqual(self._resolve(_deepseek_config()), (None, None))

    def test_other_architectures_and_manual_ple_are_left_alone(self) -> None:
        with patch("huggingface_hub.HfApi.model_info") as info:
            self.assertEqual(resolve_engram_table(_deepseek_config(gguf_architecture="deepseek4")), (None, None))
            self.assertEqual(
                resolve_engram_table(_deepseek_config(server_args="--ple /models/table.gguf_file")),
                (None, None),
            )
        info.assert_not_called()


class EngramServingTests(unittest.TestCase):
    def test_modal_run_command_carries_the_pinned_table(self) -> None:
        command = ModalBackend.build_run_command(_deepseek_config(engram_table=_table()))

        payload = json.loads(command[command.index("--engram-json") + 1])
        self.assertEqual(payload["filename"], _Q6K_TABLE)
        self.assertEqual(payload["replaced_shard"], _MXFP8_SHARD)

    def test_planned_attempt_keeps_the_resolved_table(self) -> None:
        config = _deepseek_config(engram_table=_table())
        plan = preflight_config(config).plan
        assert plan is not None

        effective = effective_config_for_attempt(LifecycleAttempt(config=config, plan=plan))

        self.assertIsNot(effective, config)
        self.assertEqual(effective.engram_table, _table())

    def test_split_set_is_complete_without_the_replaced_shard(self) -> None:
        shards = [Path(f"/m/X-UD-Q4_K_XL-{idx:05d}-of-00003.gguf") for idx in (1, 2)]
        excluded = ["UD-Q4_K_XL/X-UD-Q4_K_XL-00003-of-00003.gguf"]

        complete, incomplete = modal_llamacpp_app._partition_complete_gguf_candidates(
            shards, exclude_files=excluded,
        )
        self.assertEqual(sorted(complete), shards)
        self.assertEqual(incomplete, [])
        _, still_missing = modal_llamacpp_app._partition_complete_gguf_candidates(shards)
        self.assertEqual(still_missing[0]["missing"], [3])

    def test_modal_serve_passes_ple_and_skips_warming_the_replaced_shard(self) -> None:
        shards = [Path(f"/m/X-UD-Q4_K_XL-{idx:05d}-of-00003.gguf") for idx in (1, 2, 3)]
        config = {
            "repo_id": _REPO,
            "quant": "UD-Q4_K_XL",
            "revision": _SHA,
            "served_model_name": "ds41",
            "server_args": ["--ctx-size", "4096"],
            "host": "0.0.0.0",
            "port": 8080,
            "n_gpu_layers": None,
            "engram": {
                "repo_id": _REPO,
                "revision": _SHA,
                "filename": _Q6K_TABLE,
                "size_bytes": 161_300_000_000,
                "replaced_shard": "UD-Q4_K_XL/X-UD-Q4_K_XL-00003-of-00003.gguf",
            },
        }
        warmed: list[list[object]] = []
        with (
            patch.object(modal_llamacpp_app, "_load_config", return_value=config),
            patch.object(
                modal_llamacpp_app, "_resolve_or_download_model_entrypoint", return_value=shards[0],
            ) as resolve,
            patch.object(modal_llamacpp_app, "_gguf_weight_paths", return_value=shards),
            patch.object(modal_llamacpp_app, "WARM_VOLUME", True),
            patch.object(modal_llamacpp_app, "_should_hydrate_volume", return_value=True),
            patch.object(
                modal_llamacpp_app, "_warm_volume_paths",
                side_effect=lambda paths, **_kwargs: warmed.append(list(paths)),
            ),
            patch.object(modal_llamacpp_app.download_engram, "remote", return_value="/m/engram-Q6_K.gguf_file"),
            patch.object(modal_llamacpp_app.model_cache, "reload"),
            patch.object(modal_llamacpp_app, "_resolve_llama_server_binary", return_value="llama-server"),
            patch.object(modal_llamacpp_app, "_llama_server_runtime_env", return_value=({}, None)),
            patch.object(modal_llamacpp_app.subprocess, "Popen", return_value=MagicMock()) as popen,
        ):
            modal_llamacpp_app.serve.local()

        self.assertEqual(
            resolve.call_args.kwargs["exclude_files"],
            ["UD-Q4_K_XL/X-UD-Q4_K_XL-00003-of-00003.gguf"],
        )
        command = list(popen.call_args.args[0])
        self.assertEqual(command[command.index("--ple") + 1], "/m/engram-Q6_K.gguf_file")
        self.assertEqual(warmed, [[shards[0], shards[1], "/m/engram-Q6_K.gguf_file"]])

    def test_vast_script_stages_table_and_excludes_replaced_shard(self) -> None:
        config = _deepseek_config(
            provider=ComputeProvider.VAST,
            app_name="llp-vast-llamacpp-ds41",
            gpu_type="RTX 4090",
            gpu_count=1,
            provider_options=VastProviderOptions("1001", 100, 0.42, "42"),
            endpoint_api_key="private-key",
            engram_table=_table(),
        )

        script = vast_runtime_script(config)

        self.assertIn("--ple", script)
        self.assertIn(f"{VAST_RUNTIME_DIR}/engrams/", script)
        self.assertIn("LLM_LAUNCHPAD_GGUF_EXCLUDE=", script)
        self.assertIn("00008-of-00009.gguf", script)

    def test_prime_docker_command_stages_table_passes_ple_and_excludes_replaced_shard(self) -> None:
        config = _deepseek_config(
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-ds41",
            gpu_type="B200",
            gpu_count=4,
            provider_options=PrimeProviderOptions(disk_id="disk-1"),
            endpoint_api_key="prime-key",
            engram_table=_table(),
            placement_assessment=PlacementAssessment(
                fingerprint="fp-engram",
                memory=MemoryEstimate(
                    weights_gb=460.0,
                    kv_cache_gb=86.0,
                    compute_gb=16.0,
                    speculative_gb=0.0,
                    reserve_gb=36.0,
                    total_gb=598.0,
                ),
                tuning=RuntimeTuning(),
                fits=True,
                gpu_resident=True,
            ),
        )
        launch = PrimeLaunchSpec(
            offer_image="ubuntu_22_cuda_12",
            container_image="ghcr.io/ggml-org/llama.cpp:server-cuda-b12464",
        )

        command = PrimeBackend._bootstrap_docker_command(config, launch)
        inner = command[-1]

        self.assertIn("/data/llama.cpp/engrams/", inner)
        self.assertIn("LLM_LAUNCHPAD_GGUF_EXCLUDE=", inner)
        self.assertIn("00008-of-00009.gguf", inner)
        self.assertGreaterEqual(inner.count("--ple"), 2)

    def test_orchestrator_deploy_fails_preflight_on_unresolvable_mxfp8_table(self) -> None:
        orch = Orchestrator()
        config = _deepseek_config(do_deploy=True, preload=False)
        with patch(
            "llm_launchpad.core.deployment_preflight.resolve_engram_table",
            side_effect=ValueError("MXFP8 collision without fallback"),
        ):
            events = list(orch.deploy(config))

        completions = [e for e in events if isinstance(e, OperationCompleteEvent)]
        self.assertEqual(len(completions), 1)
        self.assertFalse(completions[0].success)
        self.assertEqual(completions[0].exit_code, 2)
        self.assertIn("Engram preflight failed", completions[0].detail)

    def test_orchestrator_deploy_attaches_resolved_engram_table(self) -> None:
        orch = Orchestrator()
        config = _deepseek_config(do_deploy=True, preload=False)
        resolved = _table()
        with (
            patch(
                "llm_launchpad.core.deployment_preflight.resolve_engram_table",
                return_value=(resolved, "Engram table preflight: substituting Q6_K via --ple."),
            ),
            patch.object(orch, "_deploy_llamacpp", return_value=iter([
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True),
            ])) as deploy_llamacpp,
        ):
            events = list(orch.deploy(config))

        self.assertEqual(config.engram_table, resolved)
        deploy_llamacpp.assert_called_once()
        self.assertTrue(
            any(isinstance(e, LogEvent) and "--ple" in e.line for e in events)
        )


if __name__ == "__main__":
    unittest.main()

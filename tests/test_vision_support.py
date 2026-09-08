"""Hermetic coverage for image capability detection, wiring, and verification."""

from __future__ import annotations

import base64
import json
import struct
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_launchpad.core import connection_store, hf_models, opencode, vision
from llm_launchpad.core.artificial_analysis import AAModelCandidate
from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.prime_backend import PrimeBackend, resolve_prime_launch_spec
from llm_launchpad.core.quick_deploy_refresh import _build_resolved_aa_model
from llm_launchpad.tui.screens import manage
from llm_launchpad.core.vision import (
    image_input_verified,
    inspect_model_vision,
    prepare_vision,
    select_projector,
    validate_vision_options,
    vision_from_dict,
    vision_to_dict,
    vllm_vision_limits,
)
from llm_launchpad.core.vision_probe import (
    ImageProbeCancelled,
    assistant_text,
    image_probe_payload,
    image_test_command,
    is_vision_probe_failure,
    verify_image_request,
)
from llm_launchpad.protocol.enums import (
    BackendType,
    ComputeProvider,
    VisionMode,
    VisionVerification,
)
from llm_launchpad.protocol.events import OperationCompleteEvent
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    EndpointInfo,
    ProjectorArtifact,
    VisionCapabilities,
)


SHA = "a" * 40


def _sibling(name: str, size: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=size)


def _fake_hub(info: SimpleNamespace, calls: list[tuple[str, str | None]] | None = None):
    """Patch huggingface_hub with a model_info stub for the vision module."""

    class FakeApi:
        def model_info(self, repo_id, revision=None, files_metadata=False, timeout=None):
            if calls is not None:
                calls.append((repo_id, revision))
            return info

    return patch.dict(
        "sys.modules",
        {"huggingface_hub": types.SimpleNamespace(HfApi=FakeApi, hf_hub_url=_fake_hub_url)},
    )


def _fake_hub_url(repo_id: str, filename: str, revision: str | None = None) -> str:
    return f"https://hf.test/{repo_id}/resolve/{revision}/{filename}"


def _info(files: list[SimpleNamespace], sha: str = SHA) -> SimpleNamespace:
    return SimpleNamespace(sha=sha, siblings=files)


class VisionCapabilityDetectionTests(unittest.TestCase):
    """Capability must come from pinned metadata, never a model's name."""

    def _inspect(self, files, config=None, processor=None) -> VisionCapabilities:
        payloads = {"config.json": config, "preprocessor_config.json": processor}
        with (
            _fake_hub(_info(files)),
            patch.object(vision, "_load_repo_json_file", lambda repo, sha, name: payloads.get(name)),
        ):
            capabilities, _files = inspect_model_vision("acme/Model")
        return capabilities

    def test_vision_config_marks_the_model_as_supported(self) -> None:
        capabilities = self._inspect(
            [_sibling("config.json")],
            config={"architectures": ["Qwen3VLForConditionalGeneration"], "vision_config": {"depth": 32}},
        )
        self.assertIs(capabilities.supported, True)
        self.assertEqual(capabilities.model_revision, SHA)

    def test_projector_sibling_marks_the_model_as_supported(self) -> None:
        capabilities = self._inspect([_sibling("mmproj-F16.gguf", 900)])
        self.assertIs(capabilities.supported, True)

    def test_image_processor_metadata_marks_the_model_as_supported(self) -> None:
        capabilities = self._inspect(
            [_sibling("preprocessor_config.json")],
            processor={"image_processor_type": "Qwen2VLImageProcessor"},
        )
        self.assertIs(capabilities.supported, True)

    def test_plain_decoder_without_an_image_processor_is_text_only(self) -> None:
        capabilities = self._inspect(
            [_sibling("config.json")],
            config={"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
        )
        self.assertIs(capabilities.supported, False)

    def test_unrecognized_architecture_stays_unknown(self) -> None:
        capabilities = self._inspect(
            [_sibling("config.json")],
            config={"architectures": ["FutureForCausalLM"], "model_type": "future_arch"},
        )
        self.assertIsNone(capabilities.supported)

    def test_unresolved_revision_is_rejected(self) -> None:
        with _fake_hub(_info([], sha="")), self.assertRaises(ValueError):
            inspect_model_vision("acme/Model")


class ProjectorSelectionTests(unittest.TestCase):
    """A projector is either unambiguous or explicitly chosen."""

    def test_single_projector_is_selected_automatically(self) -> None:
        files = {"model-Q4_K_M.gguf": 10, "mmproj-F16.gguf": 20}
        self.assertEqual(select_projector(files), "mmproj-F16.gguf")

    def test_ambiguous_projectors_require_an_explicit_choice(self) -> None:
        files = {"mmproj-F16.gguf": 20, "mmproj-Q8_0.gguf": 10}
        with self.assertRaises(ValueError) as ctx:
            select_projector(files)
        self.assertIn("mmproj-F16.gguf", str(ctx.exception))
        self.assertIn("mmproj-Q8_0.gguf", str(ctx.exception))

    def test_missing_projector_reports_that_none_were_found(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            select_projector({"model-Q4_K_M.gguf": 10})
        self.assertIn("none found", str(ctx.exception))

    def test_override_must_name_a_file_in_the_revision(self) -> None:
        with self.assertRaises(ValueError):
            select_projector({"mmproj-F16.gguf": 20}, "mmproj-Q8_0.gguf")

    def test_override_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            select_projector({"../etc/passwd": 1}, "../etc/passwd")

    def test_override_must_be_a_gguf_file(self) -> None:
        with self.assertRaises(ValueError):
            select_projector({"mmproj.safetensors": 20}, "mmproj.safetensors")


class VisionOptionValidationTests(unittest.TestCase):
    """Conflicting inputs must fail before any compute is allocated."""

    def test_raw_projector_arguments_are_rejected(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, server_args="--mmproj /tmp/p.gguf")
        with self.assertRaises(ValueError) as ctx:
            validate_vision_options(config)
        self.assertIn("--vision", str(ctx.exception))

    def test_raw_projector_arguments_are_rejected_in_equals_form(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, server_args="--no-mmproj=1")
        with self.assertRaises(ValueError):
            validate_vision_options(config)

    def test_projector_overrides_are_rejected_for_vllm(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, projector_file="mmproj-F16.gguf")
        with self.assertRaises(ValueError):
            validate_vision_options(config)

    def test_image_limits_are_rejected_for_llamacpp(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, image_limit=4)
        with self.assertRaises(ValueError):
            validate_vision_options(config)

    def test_processor_kwargs_must_be_a_json_object(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, mm_processor_kwargs="[1, 2]")
        with self.assertRaises(ValueError):
            validate_vision_options(config)

    def test_processor_kwargs_must_parse(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, mm_processor_kwargs="{not json")
        with self.assertRaises(ValueError):
            validate_vision_options(config)

    def test_projector_overrides_conflict_with_text_only_mode(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            vision_mode=VisionMode.OFF,
            projector_repo="acme/Projectors",
        )
        with self.assertRaises(ValueError):
            validate_vision_options(config)

    def test_valid_vllm_options_are_accepted(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM, image_limit=2, mm_processor_kwargs='{"max_pixels": 100}'
        )
        validate_vision_options(config)
        self.assertEqual(config.vision_mode, VisionMode.AUTO)


class PrepareVisionTests(unittest.TestCase):
    """Effective image state is resolved once, before provisioning."""

    def _prepare(self, config: DeploymentConfig, files, config_json=None) -> VisionCapabilities:
        payloads = {"config.json": config_json}
        with (
            _fake_hub(_info(files)),
            # conftest stubs inspection for every other suite; restore the real
            # detection path and fake only the Hub boundary beneath it.
            patch.object(vision, "inspect_model_vision", inspect_model_vision),
            patch.object(vision, "_load_repo_json_file", lambda repo, sha, name: payloads.get(name)),
        ):
            return prepare_vision(config)

    def test_auto_enables_a_detected_vision_model_and_pins_the_projector(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF", quant="Q4_K_M")
        state = self._prepare(config, [_sibling("model-Q4_K_M.gguf", 10), _sibling("mmproj-F16.gguf", 900)])
        self.assertTrue(state.enabled)
        self.assertEqual(state.projector, ProjectorArtifact("acme/VL-GGUF", SHA, "mmproj-F16.gguf", 900))
        self.assertTrue(state.fingerprint)
        self.assertEqual(state.verification, VisionVerification.UNTESTED)
        self.assertIs(config.vision, state)

    def test_text_only_mode_disables_images_without_inspecting_the_repo(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF", vision_mode=VisionMode.OFF
        )
        with patch.object(vision, "inspect_model_vision", side_effect=AssertionError("inspected")):
            state = prepare_vision(config)
        self.assertFalse(state.enabled)
        self.assertIsNone(state.projector)
        self.assertIn("text-only", state.message)

    def test_requiring_images_on_a_text_only_model_fails(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM, model_name="acme/Llama", vision_mode=VisionMode.ON
        )
        with self.assertRaises(ValueError):
            self._prepare(
                config,
                [_sibling("config.json")],
                config_json={"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            )

    def test_auto_leaves_unknown_models_text_only(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, model_name="acme/Future")
        state = self._prepare(
            config,
            [_sibling("config.json")],
            config_json={"architectures": ["FutureForCausalLM"], "model_type": "future_arch"},
        )
        self.assertFalse(state.enabled)

    def test_requiring_images_on_an_unknown_model_is_permitted(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM, model_name="acme/Future", vision_mode=VisionMode.ON
        )
        state = self._prepare(
            config,
            [_sibling("config.json")],
            config_json={"architectures": ["FutureForCausalLM"], "model_type": "future_arch"},
        )
        self.assertTrue(state.enabled)
        self.assertEqual(state.verification, VisionVerification.UNTESTED)

    def test_ambiguous_projectors_fail_instead_of_serving_text_only(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF")
        with self.assertRaises(ValueError):
            self._prepare(
                config,
                [_sibling("mmproj-F16.gguf", 900), _sibling("mmproj-Q8_0.gguf", 400)],
            )

    def test_fingerprint_changes_when_the_runtime_changes(self) -> None:
        files = [_sibling("model-Q4_K_M.gguf", 10), _sibling("mmproj-F16.gguf", 900)]
        first = self._prepare(
            DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF", llamacpp_runtime_id="b6100"),
            files,
        )
        second = self._prepare(
            DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF", llamacpp_runtime_id="b6200"),
            files,
        )
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_fingerprint_changes_when_the_projector_changes(self) -> None:
        base = self._prepare(
            DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF"),
            [_sibling("mmproj-F16.gguf", 900)],
        )
        other = self._prepare(
            DeploymentConfig(
                backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF", projector_file="mmproj-Q8_0.gguf"
            ),
            [_sibling("mmproj-F16.gguf", 900), _sibling("mmproj-Q8_0.gguf", 400)],
        )
        self.assertNotEqual(base.fingerprint, other.fingerprint)

    def test_inspection_failure_leaves_capability_unknown(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, model_name="acme/Model")
        with patch.object(vision, "inspect_model_vision", side_effect=RuntimeError("hub down")):
            state = prepare_vision(config)
        self.assertIsNone(state.supported)
        self.assertFalse(state.enabled)
        self.assertIn("hub down", state.message)


class VisionSerializationTests(unittest.TestCase):
    """Saved records must never overstate what was verified."""

    def _verified(self) -> VisionCapabilities:
        return VisionCapabilities(
            supported=True,
            enabled=True,
            model_revision=SHA,
            runtime_id="b6100",
            projector=ProjectorArtifact("acme/VL-GGUF", SHA, "mmproj-F16.gguf", 900),
            verification=VisionVerification.PASSED,
            message="Image request verified.",
            fingerprint="f" * 64,
        )

    def test_round_trip_preserves_the_projector_and_verification(self) -> None:
        restored = vision_from_dict(vision_to_dict(self._verified()))
        self.assertEqual(restored, self._verified())

    def test_records_without_vision_load_as_unknown(self) -> None:
        self.assertIsNone(vision_from_dict(None))
        self.assertIsNone(vision_from_dict("enabled"))

    def test_corrupt_projector_records_load_as_unknown(self) -> None:
        payload = vision_to_dict(self._verified())
        del payload["projector"]["revision"]
        self.assertIsNone(vision_from_dict(payload))

    def test_verification_resets_without_a_fingerprint(self) -> None:
        payload = vision_to_dict(self._verified())
        payload["fingerprint"] = ""
        restored = vision_from_dict(payload)
        self.assertEqual(restored.verification, VisionVerification.UNTESTED)

    def test_verification_resets_when_images_are_disabled(self) -> None:
        payload = vision_to_dict(self._verified())
        payload["enabled"] = False
        restored = vision_from_dict(payload)
        self.assertEqual(restored.verification, VisionVerification.UNTESTED)

    def test_image_input_is_only_verified_after_a_passing_request(self) -> None:
        verified = self._verified()
        self.assertTrue(image_input_verified(verified))
        self.assertFalse(image_input_verified(None))
        for field, value in (("verification", VisionVerification.UNTESTED), ("enabled", False)):
            with self.subTest(field=field):
                state = self._verified()
                setattr(state, field, value)
                self.assertFalse(image_input_verified(state))


class ImageProbeTests(unittest.TestCase):
    """The probe proves transport and text output, nothing more."""

    def _response(self, payload, status_error: Exception | None = None) -> SimpleNamespace:
        def raise_for_status() -> None:
            if status_error is not None:
                raise status_error

        return SimpleNamespace(raise_for_status=raise_for_status, json=lambda: payload)

    def test_probe_payload_carries_a_decodable_png_data_url(self) -> None:
        payload = image_probe_payload("Qwen3-VL")
        url = payload["messages"][0]["content"][1]["image_url"]["url"]
        prefix = "data:image/png;base64,"
        self.assertTrue(url.startswith(prefix))
        png = base64.b64decode(url[len(prefix):])
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (64, 64))
        # The IDAT payload must inflate, or the server rejects the image.
        self.assertEqual(len(zlib.decompress(png[41:-12])), 64 * (1 + 64 * 3))
        self.assertEqual(payload["model"], "Qwen3-VL")

    def test_assistant_text_reads_every_supported_response_shape(self) -> None:
        cases = [
            ({"content": "hello"}, "hello"),
            ({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, "a b"),
            ({"content": "", "reasoning_content": "thought"}, "thought"),
            ({"content": "   ", "reasoning_content": "thought"}, "thought"),
        ]
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(assistant_text(message).strip(), expected)
        for empty in ({}, {"content": ""}, {"content": []}, {"content": [{"type": "image"}]},
                      {"content": None, "reasoning_content": None}):
            with self.subTest(empty=empty):
                self.assertEqual(assistant_text(empty), "")

    def test_successful_request_records_a_pass(self) -> None:
        state = VisionCapabilities(enabled=True, fingerprint="f")
        payload = {"choices": [{"message": {"content": "A red square."}}]}
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=self._response(payload)):
            verify_image_request("https://host/v1", "Qwen3-VL", "secret", state)
        self.assertEqual(state.verification, VisionVerification.PASSED)
        self.assertIn("accuracy", state.message)

    def test_authorization_header_and_endpoint_are_normalized(self) -> None:
        state = VisionCapabilities(enabled=True, fingerprint="f")
        payload = {"choices": [{"message": {"content": "ok"}}]}
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=self._response(payload)) as post:
            verify_image_request("https://host/v1/", "Qwen3-VL", "secret", state)
        self.assertEqual(post.call_args.args[0], "https://host/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer secret")

    def test_http_failure_records_a_failure_and_propagates(self) -> None:
        state = VisionCapabilities(enabled=True, fingerprint="f")
        response = self._response({}, status_error=RuntimeError("400 Bad Request"))
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=response):
            with self.assertRaises(RuntimeError):
                verify_image_request("https://host/v1", "Qwen3-VL", None, state)
        self.assertEqual(state.verification, VisionVerification.FAILED)

    def test_empty_assistant_text_records_a_failure(self) -> None:
        state = VisionCapabilities(enabled=True, fingerprint="f")
        payload = {"choices": [{"message": {"content": "   "}}]}
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=self._response(payload)):
            with self.assertRaises(ValueError):
                verify_image_request("https://host/v1", "Qwen3-VL", None, state)
        self.assertEqual(state.verification, VisionVerification.FAILED)

    def test_malformed_response_records_a_failure(self) -> None:
        state = VisionCapabilities(enabled=True, fingerprint="f")
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=self._response({"choices": []})):
            with self.assertRaises(IndexError):
                verify_image_request("https://host/v1", "Qwen3-VL", None, state)
        self.assertEqual(state.verification, VisionVerification.FAILED)

    def test_cancellation_leaves_verification_untested(self) -> None:
        # Shutting down teaches nothing about the model, so a cancelled probe
        # must not be recorded as a failure.
        state = VisionCapabilities(
            enabled=True, fingerprint="f", verification=VisionVerification.PASSED
        )
        with (
            patch("llm_launchpad.core.vision_probe.is_shutting_down", return_value=True),
            patch("llm_launchpad.core.vision_probe.requests.post") as post,
        ):
            with self.assertRaises(ImageProbeCancelled):
                verify_image_request("https://host/v1", "Qwen3-VL", None, state)
        post.assert_not_called()
        self.assertEqual(state.verification, VisionVerification.UNTESTED)

    def test_reasoning_only_response_counts_as_a_pass(self) -> None:
        # A thinking model can leave content empty; the server still accepted
        # the image, which is all this probe claims to prove.
        state = VisionCapabilities(enabled=True, fingerprint="f")
        payload = {"choices": [{"message": {"content": "", "reasoning_content": "A red square."}}]}
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=self._response(payload)):
            verify_image_request("https://host/v1", "Qwen3-VL", None, state)
        self.assertEqual(state.verification, VisionVerification.PASSED)

    def test_list_content_parts_count_as_a_pass(self) -> None:
        state = VisionCapabilities(enabled=True, fingerprint="f")
        payload = {"choices": [{"message": {"content": [{"type": "text", "text": "A red square."}]}}]}
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=self._response(payload)):
            verify_image_request("https://host/v1", "Qwen3-VL", None, state)
        self.assertEqual(state.verification, VisionVerification.PASSED)

    def test_wholly_empty_response_still_fails(self) -> None:
        state = VisionCapabilities(enabled=True, fingerprint="f")
        payload = {"choices": [{"message": {"content": "", "reasoning_content": "  "}}]}
        with patch("llm_launchpad.core.vision_probe.requests.post", return_value=self._response(payload)):
            with self.assertRaises(ValueError):
                verify_image_request("https://host/v1", "Qwen3-VL", None, state)
        self.assertEqual(state.verification, VisionVerification.FAILED)

    def test_probe_budget_allows_a_thinking_model_to_answer(self) -> None:
        self.assertGreaterEqual(image_probe_payload("VL")["max_tokens"], 512)

    def test_copyable_command_keeps_the_key_as_an_environment_reference(self) -> None:
        command = image_test_command("https://host/v1", "Qwen3-VL")
        self.assertIn("https://host/v1/chat/completions", command)
        self.assertIn("$LLM_LAUNCHPAD_API_KEY", command)
        self.assertIn("data:image/png;base64,", command)


class VisionCommandPropagationTests(unittest.TestCase):
    """Every backend and provider must serve exactly what was planned."""

    def _llamacpp_config(self, enabled: bool) -> DeploymentConfig:
        projector = ProjectorArtifact("acme/VL-GGUF", SHA, "mmproj-F16.gguf", 900) if enabled else None
        return DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="acme/VL-GGUF",
            quant="Q4_K_M",
            served_model_name="vl",
            vision=VisionCapabilities(
                supported=True, enabled=enabled, projector=projector, fingerprint="f" * 64
            ),
        )

    def test_modal_llamacpp_run_command_carries_the_resolved_projector(self) -> None:
        args = ModalBackend.build_run_command(self._llamacpp_config(True))
        payload = json.loads(args[args.index("--vision-json") + 1])
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["projector"]["filename"], "mmproj-F16.gguf")
        self.assertEqual(payload["projector"]["revision"], SHA)

    def test_modal_llamacpp_run_command_states_text_only_explicitly(self) -> None:
        args = ModalBackend.build_run_command(self._llamacpp_config(False))
        payload = json.loads(args[args.index("--vision-json") + 1])
        self.assertFalse(payload["enabled"])
        self.assertIsNone(payload["projector"])

    def test_modal_llamacpp_run_command_omits_vision_when_never_resolved(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/VL-GGUF")
        self.assertNotIn("--vision-json", ModalBackend.build_run_command(config))

    def test_modal_vllm_env_limits_modalities_to_images(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            model_name="acme/VL",
            image_limit=3,
            mm_processor_kwargs='{"max_pixels": 100}',
            vision=VisionCapabilities(supported=True, enabled=True, fingerprint="f"),
        )
        env = ModalBackend.env_for_backend(config)
        self.assertEqual(json.loads(env["LIMIT_MM_PER_PROMPT"]), {"image": 3, "video": 0, "audio": 0})
        self.assertEqual(env["MM_PROCESSOR_KWARGS"], '{"max_pixels": 100}')

    def test_modal_vllm_env_disables_images_for_text_only_deployments(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            model_name="acme/VL",
            vision_mode=VisionMode.OFF,
            vision=VisionCapabilities(supported=True, enabled=False, fingerprint="f"),
        )
        env = ModalBackend.env_for_backend(config)
        self.assertEqual(json.loads(env["LIMIT_MM_PER_PROMPT"]), {"image": 0, "video": 0, "audio": 0})
        self.assertNotIn("MM_PROCESSOR_KWARGS", env)

    def test_prime_llamacpp_stages_the_projector_and_passes_it(self) -> None:
        config = self._llamacpp_config(True)
        config.provider = ComputeProvider.PRIME
        with _fake_hub(_info([])):
            command = PrimeBackend._bootstrap_docker_command(config, resolve_prime_launch_spec(config))
        script = command[-1]
        self.assertIn("--mmproj", script)
        self.assertIn("/projectors/", script)
        self.assertIn("https://hf.test/acme/VL-GGUF/resolve/", script)
        self.assertNotIn("--no-mmproj", script)

    def test_prime_llamacpp_disables_the_projector_explicitly(self) -> None:
        config = self._llamacpp_config(False)
        config.provider = ComputeProvider.PRIME
        command = PrimeBackend._bootstrap_docker_command(config, resolve_prime_launch_spec(config))
        self.assertIn("--no-mmproj", command[-1])

    def test_prime_vllm_passes_image_limits_and_processor_kwargs(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            model_name="acme/VL",
            image_limit=2,
            mm_processor_kwargs='{"max_pixels": 100}',
            vision=VisionCapabilities(supported=True, enabled=True, fingerprint="f"),
        )
        script = PrimeBackend._bootstrap_docker_command(config, resolve_prime_launch_spec(config))[-1]
        self.assertIn("--limit-mm-per-prompt", script)
        self.assertIn('{"image": 2, "video": 0, "audio": 0}', script)
        self.assertIn("--mm-processor-kwargs", script)

    def test_vllm_limits_default_to_one_image_before_resolution(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, model_name="acme/VL")
        self.assertEqual(json.loads(vllm_vision_limits(config)), {"image": 1, "video": 0, "audio": 0})


class OpenCodeVisionAdvertisingTests(unittest.TestCase):
    """OpenCode learns about images only once a request has succeeded."""

    def _connection(self, state: VisionCapabilities | None) -> opencode.OpenCodeConnection:
        return opencode.OpenCodeConnection(
            app_name="vllm-vl",
            instance_name="vl",
            provider_id=opencode.provider_id_for_app("vllm-vl"),
            provider_name="llm-launchpad",
            base_url="https://host/v1",
            model_id="VL",
            display_name="VL",
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            vision=state,
        )

    def _modalities(self, state: VisionCapabilities | None) -> dict:
        payload = opencode._provider_payload(self._connection(state))
        return payload["models"]["VL"]["modalities"]

    def test_verified_deployments_advertise_image_input(self) -> None:
        state = VisionCapabilities(
            supported=True, enabled=True, verification=VisionVerification.PASSED, fingerprint="f"
        )
        self.assertEqual(self._modalities(state)["input"], ["text", "image"])

    def test_unverified_and_failed_deployments_advertise_text_only(self) -> None:
        for verification in (VisionVerification.UNTESTED, VisionVerification.FAILED):
            with self.subTest(verification=verification):
                state = VisionCapabilities(
                    supported=True, enabled=True, verification=verification, fingerprint="f"
                )
                self.assertEqual(self._modalities(state)["input"], ["text"])

    def test_older_records_advertise_text_only(self) -> None:
        self.assertEqual(self._modalities(None)["input"], ["text"])
        self.assertEqual(self._modalities(None)["output"], ["text"])


class VisionVerificationPersistenceTests(unittest.TestCase):
    """Verification belongs to the endpoint that was actually tested."""

    def _state(self, verification: VisionVerification) -> VisionCapabilities:
        return VisionCapabilities(
            supported=True, enabled=True, verification=verification, fingerprint="f" * 64
        )

    def _store(self, tmp: Path, base_url: str) -> Path:
        path = tmp / "connections.json"
        path.write_text(json.dumps({"entries": {"vl-app": {"base_url": base_url}}}), encoding="utf-8")
        return path

    def test_verification_is_written_back_for_the_tested_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._store(Path(tmp), "https://host/v1")
            connection_store.update_vision_verification(
                "vl-app", "https://host/v1", self._state(VisionVerification.PASSED), path
            )
            entry = json.loads(path.read_text())["entries"]["vl-app"]
        self.assertEqual(entry["vision"]["verification"], "passed")

    def test_a_different_endpoint_url_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._store(Path(tmp), "https://other-host/v1")
            connection_store.update_vision_verification(
                "vl-app", "https://host/v1", self._state(VisionVerification.PASSED), path
            )
            entry = json.loads(path.read_text())["entries"]["vl-app"]
        self.assertNotIn("vision", entry)


class _ReadyResponse:
    """A vLLM readiness probe that always reports the server is up."""

    status_code = 200
    text = '{"data": []}'

    def json(self) -> dict:
        return {"data": []}


class WarmupImageVerificationTests(unittest.TestCase):
    """A failed image request must fail deployment validation."""

    def _warmup(self, state: VisionCapabilities | None, probe_error: Exception | None = None):
        fake_requests = types.SimpleNamespace(
            get=lambda *_args, **_kwargs: _ReadyResponse(),
            post=lambda *_args, **_kwargs: _ReadyResponse(),
        )
        probe = patch(
            "llm_launchpad.core.warmup.verify_image_request",
            side_effect=probe_error or (lambda *_args, **_kwargs: None),
        )
        with (
            patch.dict("sys.modules", {"requests": fake_requests}),
            patch(
                "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                return_value="curl ok",
            ),
            probe as verify,
        ):
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.VLLM,
                    server_url="https://example.modal.run/v1",
                    timeout=10,
                    tail_logs=False,
                    served_model_name="VL",
                    vision=state,
                )
            )
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        return completion, verify

    def _enabled(self) -> VisionCapabilities:
        return VisionCapabilities(supported=True, enabled=True, fingerprint="f" * 64)

    def test_enabled_deployments_are_verified_before_being_reported_healthy(self) -> None:
        completion, verify = self._warmup(self._enabled())
        self.assertTrue(completion.success)
        verify.assert_called_once()

    def test_a_failed_image_request_fails_warmup(self) -> None:
        completion, _verify = self._warmup(self._enabled(), RuntimeError("400 Bad Request"))
        self.assertFalse(completion.success)

    def test_text_only_deployments_are_never_probed_for_images(self) -> None:
        disabled = VisionCapabilities(supported=True, enabled=False, fingerprint="f" * 64)
        completion, verify = self._warmup(disabled)
        self.assertTrue(completion.success)
        verify.assert_not_called()

    def test_deployments_without_resolved_vision_are_never_probed(self) -> None:
        completion, verify = self._warmup(None)
        self.assertTrue(completion.success)
        verify.assert_not_called()


class ConnectionInfoVisionDisplayTests(unittest.TestCase):
    """Deployment details must distinguish enabled from verified."""

    def test_each_state_reads_differently(self) -> None:
        enabled = VisionCapabilities(supported=True, enabled=True, fingerprint="f")
        passed = VisionCapabilities(
            supported=True, enabled=True, verification=VisionVerification.PASSED, fingerprint="f"
        )
        failed = VisionCapabilities(
            supported=True, enabled=True, verification=VisionVerification.FAILED, fingerprint="f"
        )
        self.assertIn("not verified", manage._vision_summary(enabled))
        self.assertIn("verified", manage._vision_summary(passed))
        self.assertIn("failed", manage._vision_summary(failed))
        self.assertIn("disabled", manage._vision_summary(VisionCapabilities()))
        self.assertIn("unknown", manage._vision_summary(None))

    def test_connection_fields_include_an_images_row(self) -> None:
        payload = {"base_url": "https://host/v1", "model_id": "VL", "display_name": "VL"}
        markup = manage.ConnectionInfoScreen._fields_markup(
            payload, VisionCapabilities(supported=True, enabled=True, fingerprint="f")
        )
        self.assertIn("Images", markup)
        self.assertIn("not verified", markup)


class ProjectorDiscoveryTests(unittest.TestCase):
    """Auxiliary GGUFs must never be mistaken for the model's own weights."""

    def test_projector_and_auxiliary_files_are_not_offered_as_quantizations(self) -> None:
        siblings = [
            _sibling("model-Q4_K_M.gguf"),
            _sibling("mmproj-F16.gguf"),
            _sibling("mmproj-Q8_0.gguf"),
            _sibling("imatrix-Q4_K_M.gguf"),
            _sibling("draft-Q4_0.gguf"),
        ]
        self.assertEqual(hf_models._extract_gguf_quantizations(siblings), ["Q4_K_M"])

    def test_projector_siblings_are_detected(self) -> None:
        self.assertTrue(hf_models._has_projector_sibling([_sibling("mmproj-F16.gguf")]))
        self.assertFalse(hf_models._has_projector_sibling([_sibling("model-Q4_K_M.gguf")]))
        # A non-GGUF file with a matching name is not a llama.cpp projector.
        self.assertFalse(hf_models._has_projector_sibling([_sibling("mmproj.safetensors")]))
        self.assertFalse(hf_models._has_projector_sibling(None))


class FastDeployVisionExclusionTests(unittest.TestCase):
    """Fast Deploy only recommends configurations whose memory it models."""

    def _candidate(self) -> AAModelCandidate:
        return AAModelCandidate(
            aa_model_id="vl", name="VL", slug="vl", creator_name="",
            coding_score=70.0, intelligence_score=50.0, rank=1,
            parameter_count_b=8.0, max_context_tokens=None,
        )

    def _metadata(self, has_projector: bool) -> GgufQuantMetadata:
        return GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": 6.0},
            architecture="llama",
            has_projector=has_projector,
        )

    def _resolve(self, has_projector: bool):
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._fetch_serving_metadata",
            return_value=self._metadata(has_projector),
        ):
            return _build_resolved_aa_model(
                self._candidate(),
                [ModalGpuSpec("A100-80GB", price_per_hour_usd=2.5)],
                "unsloth/VL-GGUF",
            )

    def test_vision_models_are_not_offered_as_guaranteed_fits(self) -> None:
        self.assertIsNone(self._resolve(True))

    def test_text_only_models_are_still_recommended(self) -> None:
        self.assertIsNotNone(self._resolve(False))


class ProbeFailureIsolationTests(unittest.TestCase):
    """A failed image probe must not read as a failed deployment."""

    def test_warmup_marks_probe_failure_in_the_completion_event(self) -> None:
        state = VisionCapabilities(supported=True, enabled=True, fingerprint="f" * 64)
        fake_requests = types.SimpleNamespace(
            get=lambda *_a, **_k: _ReadyResponse(), post=lambda *_a, **_k: _ReadyResponse()
        )
        with (
            patch.dict("sys.modules", {"requests": fake_requests}),
            patch("llm_launchpad.core.warmup.ModalBackend.test_curl_command", return_value="curl ok"),
            patch(
                "llm_launchpad.core.warmup.verify_image_request",
                side_effect=RuntimeError("400 Bad Request"),
            ),
        ):
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.VLLM,
                    server_url="https://example.modal.run/v1",
                    timeout=10, tail_logs=False, served_model_name="VL", vision=state,
                )
            )
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertTrue(is_vision_probe_failure(completion))

    def test_other_warmup_failures_are_not_marked_as_probe_failures(self) -> None:
        # Teardown must still happen for a genuinely broken deployment.
        from llm_launchpad.core.operation_events import fail_operation
        from llm_launchpad.protocol.enums import OperationType

        events = list(fail_operation(OperationType.WARMUP, "Timed out after 1800s"))
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(is_vision_probe_failure(completion))

    def test_predicate_ignores_unrelated_event_payloads(self) -> None:
        self.assertFalse(is_vision_probe_failure(None))
        self.assertFalse(is_vision_probe_failure(OperationCompleteEvent(data={"url": "x"})))
        self.assertFalse(is_vision_probe_failure(OperationCompleteEvent(data="not-a-dict")))

    def test_cancelled_probe_does_not_overwrite_stored_verification(self) -> None:
        from llm_launchpad.core.vision_probe import ImageProbeCancelled

        state = VisionCapabilities(supported=True, enabled=True, fingerprint="f" * 64)
        fake_requests = types.SimpleNamespace(
            get=lambda *_a, **_k: _ReadyResponse(), post=lambda *_a, **_k: _ReadyResponse()
        )
        with (
            patch.dict("sys.modules", {"requests": fake_requests}),
            patch("llm_launchpad.core.warmup.ModalBackend.test_curl_command", return_value="curl ok"),
            patch(
                "llm_launchpad.core.warmup.verify_image_request",
                side_effect=ImageProbeCancelled("cancelled"),
            ),
            patch("llm_launchpad.core.connection_store.update_vision_verification") as persist,
        ):
            list(
                Orchestrator().warmup(
                    backend=BackendType.VLLM,
                    server_url="https://example.modal.run/v1",
                    timeout=10, tail_logs=False, app_name="vl-app",
                    served_model_name="VL", vision=state,
                )
            )
        persist.assert_not_called()


class ImageTestCommandGuardTests(unittest.TestCase):
    """`--image-test` must never invent an image capability."""

    def _run(self, vision: VisionCapabilities | None):
        from typer.testing import CliRunner
        from llm_launchpad.cli import main as cli_main

        target = EndpointInfo(
            name="vl-app", backend=BackendType.VLLM, web_url="https://host/v1", vision=vision
        )
        with (
            patch.object(cli_main, "_preflight", return_value=(Mock(), "alice")),
            patch.object(cli_main, "_resolve_manage_target", return_value=target),
            patch.object(cli_main, "_print_banner"),
        ):
            return CliRunner().invoke(cli_main.app, ["warmup", "--app-name", "vl-app", "--image-test"])

    def test_successful_image_test_republishes_the_capability(self) -> None:
        """Verification is what unlocks image input for clients, so a passing
        probe must reach OpenCode without waiting for the next deploy."""
        from typer.testing import CliRunner
        from llm_launchpad.cli import main as cli_main
        from llm_launchpad.protocol.enums import OperationType

        state = VisionCapabilities(supported=True, enabled=True, fingerprint="f" * 64)
        target = EndpointInfo(
            name="vl-app", backend=BackendType.VLLM, web_url="https://host/v1", vision=state
        )
        orch = Mock()
        orch.warmup.return_value = [
            OperationCompleteEvent(operation=OperationType.WARMUP, success=True)
        ]
        with (
            patch.object(cli_main, "_preflight", return_value=(orch, "alice")),
            patch.object(cli_main, "_resolve_manage_target", return_value=target),
            patch.object(cli_main, "_print_banner"),
            patch.object(cli_main, "_load_visible_launchpad_rows", return_value=[]),
            patch.object(cli_main, "_sync_opencode_cli") as sync,
        ):
            result = CliRunner().invoke(
                cli_main.app, ["warmup", "--app-name", "vl-app", "--image-test"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        sync.assert_called_once()
        self.assertEqual(sync.call_args.kwargs["target_app_name"], "vl-app")

    def test_refuses_when_the_deployment_never_enabled_images(self) -> None:
        for state in (
            None,
            VisionCapabilities(supported=True, enabled=False, fingerprint="f" * 64),
            VisionCapabilities(supported=True, enabled=True, fingerprint=""),
        ):
            with self.subTest(state=state):
                result = self._run(state)
                self.assertEqual(result.exit_code, 2)
                self.assertIn("image input enabled", result.output)


class VllmRuntimeImageTests(unittest.TestCase):
    """The official vLLM image must not be overwritten from PyPI."""

    def test_vllm_is_not_reinstalled_over_the_official_image(self) -> None:
        source = Path("llm_launchpad/backends/modal_vllm_app.py").read_text(encoding="utf-8")
        # Modal needs its own interpreter; the image only ships `python3`.
        self.assertIn('"vllm/vllm-openai:v0.19.1", add_python="3.12"', source)
        # Reinstalling vLLM from PyPI would replace the image's CUDA build.
        self.assertNotIn("vllm==", source)


class PrimeProjectorAuthTests(unittest.TestCase):
    """Public projectors must download without a token."""

    def test_authorization_header_is_omitted_when_no_token_is_set(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP, provider=ComputeProvider.PRIME,
            repo_id="acme/VL-GGUF", quant="Q4_K_M", served_model_name="vl",
            vision=VisionCapabilities(
                supported=True, enabled=True, fingerprint="f" * 64,
                projector=ProjectorArtifact("acme/VL-GGUF", SHA, "mmproj-F16.gguf", 900),
            ),
        )
        with _fake_hub(_info([])):
            script = PrimeBackend._bootstrap_docker_command(
                config, resolve_prime_launch_spec(config)
            )[-1]
        # Expanded by the container shell, and dropped entirely when unset.
        self.assertIn('${HF_TOKEN:+--header "Authorization: Bearer $HF_TOKEN"}', script)
        self.assertNotIn('--header "Authorization: Bearer $HF_TOKEN" http', script)


if __name__ == "__main__":
    unittest.main()
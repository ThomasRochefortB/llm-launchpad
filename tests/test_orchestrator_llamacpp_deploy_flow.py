from __future__ import annotations

import unittest
import shlex
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.gguf_metadata import GgufMtpCapability, GgufMtpStatus
from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.prime_backend import PrimeBackend
from llm_launchpad.protocol.enums import (
    BackendType,
    ComputeProvider,
    OperationType,
    SpeculativeDecodingMethod,
)
from llm_launchpad.protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    LaunchpadSettings,
    ReasoningCapabilities,
    SpeculativeDecodingConfig,
)


class _FakeConfigStore:
    def load(self) -> LaunchpadSettings:
        return LaunchpadSettings()


class OrchestratorLlamaCppDeployFlowTests(unittest.TestCase):
    def test_mtp_preflight_adds_exact_flags_for_modal_and_prime(self) -> None:
        metadata = GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": 20.0},
            architecture="qwen35",
            mtp=GgufMtpCapability(
                status=GgufMtpStatus.SUPPORTED,
                nextn_predict_layers=1,
                source_file="model.gguf",
            ),
        )
        orchestrator = Orchestrator(config_store=_FakeConfigStore())
        for provider in (ComputeProvider.MODAL, ComputeProvider.PRIME):
            config = DeploymentConfig(
                backend=BackendType.LLAMACPP,
                provider=provider,
                repo_id="unsloth/Qwen3.8-27B-GGUF",
                quant="Q4_K_M",
                server_args="--ctx-size 131072 --spec-type stale --spec-draft-n-max 9",
                speculative_decoding=SpeculativeDecodingConfig(
                    method=SpeculativeDecodingMethod.MTP,
                    num_speculative_tokens=3,
                    nextn_predict_layers=1,
                ),
            )
            with patch(
                "llm_launchpad.core.orchestrator.fetch_gguf_quant_metadata",
                return_value=metadata,
            ) as fetch:
                events = list(orchestrator._prepare_llamacpp_speculative_decoding(config))

            fetch.assert_called_once_with(
                config.repo_id,
                revision=None,
                inspect_mtp=True,
                mtp_quant="Q4_K_M",
                force_refresh=True,
            )
            args = shlex.split(config.server_args or "")
            self.assertEqual(args.count("--spec-type"), 1)
            self.assertEqual(args.count("--spec-draft-n-max"), 1)
            self.assertIn("draft-mtp", args)
            self.assertIn("3", args)
            self.assertTrue(any("enabled native draft-mtp" in event.line for event in events))
            if provider == ComputeProvider.MODAL:
                command = ModalBackend.build_run_command(config)
                self.assertIn("draft-mtp", " ".join(command))
            else:
                runtime_args = PrimeBackend.runtime_env(config)["LLAMACPP_SERVER_ARGS"]
                self.assertIn("draft-mtp", runtime_args)

    def test_mtp_preflight_disables_and_warns_when_metadata_is_unknown(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            quant="Q4_K_M",
            server_args="--ctx-size 131072",
            speculative_decoding=SpeculativeDecodingConfig(
                method=SpeculativeDecodingMethod.MTP,
                num_speculative_tokens=3,
                nextn_predict_layers=1,
            ),
        )
        metadata = GgufQuantMetadata(
            quantizations=[],
            vram_gb_by_quant={},
            architecture="qwen35",
            mtp=GgufMtpCapability.unknown("range request failed"),
        )
        with patch(
            "llm_launchpad.core.orchestrator.fetch_gguf_quant_metadata",
            return_value=metadata,
        ):
            events = list(
                Orchestrator(config_store=_FakeConfigStore())._prepare_llamacpp_speculative_decoding(
                    config
                )
            )

        self.assertIsNone(config.speculative_decoding)
        self.assertNotIn("--spec-type", config.server_args or "")
        self.assertTrue(any("continuing with normal decoding" in event.line for event in events))

    def test_deploy_inspects_selected_model_before_allocating_compute(self) -> None:
        capabilities = ReasoningCapabilities(
            profile_id="hf-test-profile",
            canonical_model_id="acme/Future-Reasoner-GGUF",
            model_revision="a" * 40,
            efforts=("brief", "deep"),
            default_effort="deep",
            source_repo="acme/Future-Reasoner-GGUF",
            source_revision="a" * 40,
            source_path="gguf.chat_template",
            request_option_path="chat_template_kwargs.reasoning_effort",
        )
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="acme/Future-Reasoner-GGUF",
            quant="Q4_K_M",
            gguf_architecture="unsupported-test-architecture",
            do_warmup=False,
        )

        with (
            patch(
                "llm_launchpad.core.orchestrator.discover_selected_model_reasoning",
                return_value=capabilities,
            ) as discover,
            patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming") as run,
        ):
            events = list(Orchestrator(config_store=_FakeConfigStore()).deploy(config))

        discover.assert_called_once_with(config)
        run.assert_not_called()
        self.assertEqual(config.reasoning, capabilities)
        self.assertTrue(
            any(
                isinstance(event, LogEvent)
                and "brief, deep" in event.line
                and "verified from" in event.line
                for event in events
            )
        )

    def test_llamacpp_deploy_blocks_unsupported_architecture_before_modal(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="unsloth/GLM-5.3-Flash-GGUF",
            quant="UD-Q2_K_XL",
            gguf_architecture="glm5next",
            do_warmup=False,
        )

        with patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming") as run:
            events = list(Orchestrator(config_store=_FakeConfigStore()).deploy(config))

        run.assert_not_called()
        errors = [event for event in events if isinstance(event, ErrorEvent)]
        self.assertEqual(len(errors), 1)
        self.assertIn("glm5next", errors[0].message)
        completions = [event for event in events if isinstance(event, OperationCompleteEvent)]
        self.assertEqual(len(completions), 1)
        self.assertFalse(completions[0].success)
        self.assertEqual(completions[0].exit_code, 2)

    def test_llamacpp_deploy_splits_prepare_and_deploy_commands(self) -> None:
        seen_commands: list[list[str]] = []

        def _run_streaming(cmd, env=None):  # type: ignore[no-untyped-def]
            _ = env
            seen_commands.append(list(cmd))
            if cmd[:2] == ["modal", "run"]:
                return iter(
                    [
                        LogEvent(line="prep"),
                        OperationCompleteEvent(success=True, exit_code=0),
                    ]
                )
            if cmd[:2] == ["modal", "deploy"]:
                return iter(
                    [
                        LogEvent(line="deploy"),
                        OperationCompleteEvent(success=True, exit_code=0),
                    ]
                )
            return iter([])

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
            quant="Q4_K_M",
            preload=True,
            do_deploy=True,
            do_warmup=False,
            app_name="llamacpp-test",
            function_slug="slug-1",
        )

        with patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming", side_effect=_run_streaming):
            events = list(Orchestrator(config_store=_FakeConfigStore()).deploy(config))

        self.assertEqual(len(seen_commands), 2)
        prep_cmd, deploy_cmd = seen_commands

        self.assertEqual(
            prep_cmd[:4],
            ["modal", "run", "-m", "llm_launchpad.backends.modal_llamacpp_app::main"],
        )
        self.assertIn("--preload", prep_cmd)
        self.assertNotIn("--deploy", prep_cmd)

        self.assertEqual(deploy_cmd[:3], ["modal", "deploy", "-m"])
        self.assertIn("llm_launchpad.backends.modal_llamacpp_app", deploy_cmd)
        self.assertIn("--name", deploy_cmd)
        self.assertIn("llamacpp-test", deploy_cmd)

        completions = [e for e in events if isinstance(e, OperationCompleteEvent)]
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].operation, OperationType.DEPLOY)
        self.assertTrue(completions[0].success)

    def test_llamacpp_deploy_does_not_run_modal_deploy_when_prepare_fails(self) -> None:
        seen_commands: list[list[str]] = []

        def _run_streaming(cmd, env=None):  # type: ignore[no-untyped-def]
            _ = env
            seen_commands.append(list(cmd))
            return iter(
                [
                    LogEvent(line="prep failed"),
                    OperationCompleteEvent(success=False, exit_code=7, detail="prep failed"),
                ]
            )

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="repo/model",
            quant="Q4_K_M",
            preload=True,
            do_deploy=True,
            do_warmup=False,
            app_name="llamacpp-test",
            function_slug="slug-1",
        )

        with patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming", side_effect=_run_streaming):
            events = list(Orchestrator(config_store=_FakeConfigStore()).deploy(config))

        self.assertEqual(len(seen_commands), 1)
        self.assertEqual(seen_commands[0][:2], ["modal", "run"])

        completions = [e for e in events if isinstance(e, OperationCompleteEvent)]
        self.assertEqual(len(completions), 1)
        self.assertFalse(completions[0].success)
        self.assertEqual(completions[0].exit_code, 7)


if __name__ == "__main__":
    unittest.main()

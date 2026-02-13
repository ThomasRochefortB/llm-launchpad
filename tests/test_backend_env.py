from __future__ import annotations

import unittest

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import DeploymentConfig, LaunchpadSettings


class BackendEnvTests(unittest.TestCase):
    def test_vllm_reasoning_fields_are_forwarded(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            model_name="Qwen/Qwen3-8B",
            reasoning_parser="qwen3",
            default_chat_template_kwargs='{"enable_thinking": false}',
        )

        env = ModalBackend.env_for_backend(config)
        self.assertEqual(env["MODEL_NAME"], "Qwen/Qwen3-8B")
        self.assertEqual(env["REASONING_PARSER"], "qwen3")
        self.assertEqual(env["DEFAULT_CHAT_TEMPLATE_KWARGS"], '{"enable_thinking": false}')

    def test_vllm_reasoning_fields_are_omitted_when_empty(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            model_name="Qwen/Qwen3-8B",
            reasoning_parser=None,
            default_chat_template_kwargs="",
        )

        env = ModalBackend.env_for_backend(config)
        self.assertNotIn("REASONING_PARSER", env)
        self.assertNotIn("DEFAULT_CHAT_TEMPLATE_KWARGS", env)

    def test_build_full_env_uses_deployment_gpu_config(self) -> None:
        settings = LaunchpadSettings(scaledown_window=900)
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            gpu_type="H100",
            gpu_count=2,
        )
        env = ModalBackend.build_full_env(settings, config)
        self.assertEqual(env["GPU_CONFIG"], "H100:2")
        self.assertEqual(env["SCALEDOWN_WINDOW"], "900")

    def test_vllm_n_gpu_is_independent_from_deployment_gpu_count(self) -> None:
        settings = LaunchpadSettings()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            gpu_type="A100-80GB",
            gpu_count=2,
            n_gpu=4,
        )
        env = ModalBackend.build_full_env(settings, config)
        self.assertEqual(env["GPU_CONFIG"], "A100-80GB:2")
        self.assertEqual(env["N_GPU"], "4")


if __name__ == "__main__":
    unittest.main()

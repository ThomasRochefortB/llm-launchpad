from __future__ import annotations

import shlex
import unittest

from llm_launchpad.core.quick_deploy import (
    build_quick_deploy_config,
    format_context_length,
    get_quick_deploy_profile,
    list_quick_deploy_profiles,
)
from llm_launchpad.protocol.enums import BackendType


class QuickDeployConfigTests(unittest.TestCase):
    def test_format_context_length_uses_grouped_tokens(self) -> None:
        self.assertEqual(format_context_length(262144), "262,144 ctx")

    def test_catalog_contains_expected_profiles(self) -> None:
        profiles = list_quick_deploy_profiles()
        self.assertEqual(
            [profile.id for profile in profiles],
            [
                "qwen35-397b-rtxpro",
                "glm5-rtxpro",
                "kimi25-rtxpro",
            ],
        )

    def test_build_quick_deploy_config_maps_profile_defaults(self) -> None:
        profile = get_quick_deploy_profile("qwen35-397b-rtxpro")

        config = build_quick_deploy_config(profile)

        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "unsloth/Qwen3.5-397B-A17B-GGUF")
        self.assertEqual(config.quant, "UD-Q3_K_XL")
        self.assertEqual(config.gpu_type, "RTX-PRO-6000")
        self.assertEqual(config.gpu_count, 3)
        self.assertEqual(profile.max_context_tokens, 262144)
        self.assertTrue(config.preload)
        self.assertTrue(config.do_deploy)
        self.assertTrue(config.do_warmup)
        self.assertFalse(config.show_debug_logs)
        self.assertEqual(config.instance_name, "qwen35-397b-rtxpro")
        self.assertEqual(config.app_name, "llamacpp-qwen35-397b-rtxpro")
        self.assertEqual(
            shlex.split(config.server_args or ""),
            ["--ctx-size", "262144", "--threads", "16", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.00"],
        )

    def test_build_quick_deploy_config_maps_glm_profile_defaults(self) -> None:
        profile = get_quick_deploy_profile("glm5-rtxpro")

        config = build_quick_deploy_config(profile)

        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "unsloth/GLM-5-GGUF")
        self.assertEqual(config.quant, "UD-Q2_K_XL")
        self.assertEqual(config.gpu_type, "RTX-PRO-6000")
        self.assertEqual(config.gpu_count, 4)
        self.assertEqual(profile.max_context_tokens, 202752)
        self.assertEqual(config.instance_name, "glm5-rtxpro")
        self.assertEqual(config.app_name, "llamacpp-glm5-rtxpro")
        self.assertEqual(
            shlex.split(config.server_args or ""),
            ["--ctx-size", "202752", "--flash-attn", "on", "--temp", "0.7", "--top-p", "1.0", "--min-p", "0.01"],
        )

    def test_build_quick_deploy_config_maps_kimi_profile_defaults(self) -> None:
        profile = get_quick_deploy_profile("kimi25-rtxpro")

        config = build_quick_deploy_config(profile)

        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "unsloth/Kimi-K2.5-GGUF")
        self.assertEqual(config.quant, "UD-Q2_K_XL")
        self.assertEqual(config.gpu_type, "RTX-PRO-6000")
        self.assertEqual(config.gpu_count, 5)
        self.assertEqual(profile.max_context_tokens, 262144)
        self.assertEqual(config.instance_name, "kimi25-rtxpro")
        self.assertEqual(config.app_name, "llamacpp-kimi25-rtxpro")
        self.assertEqual(
            shlex.split(config.server_args or ""),
            ["--special", "--kv-unified", "--ctx-size", "98304", "--temp", "1.0", "--top-p", "0.95", "--min-p", "0.01"],
        )

    def test_build_quick_deploy_config_applies_overrides(self) -> None:
        profile = get_quick_deploy_profile("qwen35-397b-rtxpro")

        config = build_quick_deploy_config(
            profile,
            instance_name="My Qwen Prod",
            app_name="llamacpp-custom-prod",
            do_warmup=False,
            show_debug_logs=True,
        )

        self.assertEqual(config.instance_name, "my-qwen-prod")
        self.assertEqual(config.app_name, "llamacpp-custom-prod")
        self.assertFalse(config.do_warmup)
        self.assertTrue(config.show_debug_logs)
        self.assertEqual(config.gpu_type, "RTX-PRO-6000")
        self.assertEqual(config.gpu_count, 3)


if __name__ == "__main__":
    unittest.main()

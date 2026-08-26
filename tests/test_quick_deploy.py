from __future__ import annotations

import json
import shlex
import unittest
from unittest.mock import patch

from llm_launchpad.core import quick_deploy
from llm_launchpad.core.quick_deploy import (
    build_quick_deploy_config,
    format_context_length,
    get_quick_deploy_profile,
    get_quick_deploy_catalog_info,
    list_quick_deploy_profiles,
    quick_deploy_model_label_parts,
)
from llm_launchpad.protocol.enums import BackendType


class QuickDeployConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        quick_deploy._reset_quick_deploy_catalog_cache()

    def tearDown(self) -> None:
        quick_deploy._reset_quick_deploy_catalog_cache()

    def test_format_context_length_uses_grouped_tokens(self) -> None:
        self.assertEqual(format_context_length(262144), "262,144 ctx")

    def test_catalog_contains_expected_profiles(self) -> None:
        profiles = list_quick_deploy_profiles()
        self.assertGreaterEqual(len(profiles), 1)
        self.assertIn("UD-Q4_K_XL", {profile.quant for profile in profiles})
        self.assertIn("UD-Q2_K_XL", {profile.quant for profile in profiles})
        self.assertFalse(get_quick_deploy_catalog_info().is_fallback)
        self.assertEqual(
            get_quick_deploy_catalog_info().source_label,
            "Curated popular open-weight models",
        )

    def test_catalog_loader_accepts_generated_json(self) -> None:
        payload = {
            "schema_version": 1,
            "generated_at": "2026-05-01T12:00:00Z",
            "source": "Artificial Analysis coding rankings",
            "attribution": "Artificial Analysis",
            "profiles": [
                {
                    "id": "test-model",
                    "display_name": "Test Model",
                    "repo_id": "unsloth/Test-Model-GGUF",
                    "quant": "Q4_K_M",
                    "gpu_type": "L40S",
                    "gpu_count": 2,
                    "profile_label": "AA Coding",
                    "resource_tier": "rtx-pro",
                    "resource_tier_label": "$$",
                    "approx_cost_per_hour_usd": 3.9,
                    "required_vram_gb": 88.5,
                    "max_context_tokens": 65536,
                    "instance_slug_hint": "test-model",
                    "summary": "Generated profile.",
                    "server_args": ["--ctx-size", "65536"],
                    "source_label": "Artificial Analysis",
                    "aa_model_id": "aa-1",
                    "aa_model_name": "Test Model",
                    "aa_model_slug": "test-model",
                    "aa_coding_score": 42.5,
                    "aa_rank": 1,
                }
            ],
        }

        with patch(
            "llm_launchpad.core.quick_deploy._read_bundled_catalog_text",
            return_value=json.dumps(payload),
        ):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profiles = list_quick_deploy_profiles()

        self.assertEqual([profile.id for profile in profiles], ["test-model"])
        self.assertEqual(profiles[0].aa_coding_score, 42.5)
        self.assertEqual(profiles[0].required_vram_gb, 88.5)
        self.assertEqual(profiles[0].resource_tier, "rtx-pro")
        self.assertEqual(profiles[0].resource_tier_label, "$$")
        self.assertEqual(profiles[0].server_args, ("--ctx-size", "65536"))
        self.assertEqual(get_quick_deploy_catalog_info().generated_at, "2026-05-01T12:00:00Z")

    def test_catalog_loader_accepts_vllm_recipe_without_quant(self) -> None:
        payload = {
            "schema_version": 1,
            "source": "Curated models",
            "profiles": [
                {
                    "id": "test-vllm",
                    "display_name": "Test vLLM Model",
                    "repo_id": "",
                    "model_name": "org/Test-Model",
                    "backend": "vllm",
                    "quant": "",
                    "gpu_type": "H100_80GB",
                    "gpu_count": 1,
                    "profile_label": "Fast",
                    "approx_cost_per_hour_usd": 2.0,
                    "max_context_tokens": 32768,
                    "instance_slug_hint": "test-vllm",
                    "summary": "Generated vLLM recipe.",
                    "server_args": [],
                }
            ],
        }

        with patch(
            "llm_launchpad.core.quick_deploy._read_bundled_catalog_text",
            return_value=json.dumps(payload),
        ):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profile = list_quick_deploy_profiles()[0]

        self.assertEqual(profile.backend, BackendType.VLLM)
        self.assertEqual(profile.model_name, "org/Test-Model")
        self.assertEqual(profile.quant, "")

    def test_catalog_loader_falls_back_when_file_missing(self) -> None:
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value=None):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profiles = list_quick_deploy_profiles()

        self.assertEqual([profile.id for profile in profiles], ["qwen35-397b-rtxpro", "glm5-rtxpro", "kimi25-rtxpro"])
        self.assertTrue(get_quick_deploy_catalog_info().is_fallback)

    def test_catalog_loader_falls_back_when_json_invalid(self) -> None:
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value="{not-json"):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profiles = list_quick_deploy_profiles()

        self.assertEqual([profile.id for profile in profiles], ["qwen35-397b-rtxpro", "glm5-rtxpro", "kimi25-rtxpro"])
        self.assertTrue(get_quick_deploy_catalog_info().is_fallback)

    def test_quick_deploy_model_label_parts_split_quant_suffix(self) -> None:
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value=None):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profile = get_quick_deploy_profile("kimi25-rtxpro")

        self.assertEqual(
            quick_deploy_model_label_parts(profile),
            ("Kimi K2.5", "(UD-Q4_K_XL)"),
        )

    def test_build_quick_deploy_config_maps_profile_defaults(self) -> None:
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value=None):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profile = get_quick_deploy_profile("qwen35-397b-rtxpro")

        config = build_quick_deploy_config(profile)

        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "unsloth/Qwen3.5-397B-A17B-GGUF")
        self.assertEqual(config.quant, "UD-Q4_K_XL")
        self.assertEqual(config.gpu_type, "RTX-PRO-6000")
        self.assertEqual(config.gpu_count, 3)
        self.assertEqual(config.required_vram_gb, profile.required_vram_gb)
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
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value=None):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profile = get_quick_deploy_profile("glm5-rtxpro")

        config = build_quick_deploy_config(profile)

        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "unsloth/GLM-5-GGUF")
        self.assertEqual(config.quant, "UD-Q4_K_XL")
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
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value=None):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profile = get_quick_deploy_profile("kimi25-rtxpro")

        config = build_quick_deploy_config(profile)

        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "unsloth/Kimi-K2.5-GGUF")
        self.assertEqual(config.quant, "UD-Q4_K_XL")
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
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value=None):
            quick_deploy._reset_quick_deploy_catalog_cache()
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

    def test_build_quick_deploy_config_infers_instance_from_prefixed_app_name(self) -> None:
        with patch("llm_launchpad.core.quick_deploy._read_bundled_catalog_text", return_value=None):
            quick_deploy._reset_quick_deploy_catalog_cache()
            profile = get_quick_deploy_profile("qwen35-397b-rtxpro")

        config = build_quick_deploy_config(profile, app_name="llamacpp-custom-prod")

        self.assertEqual(config.app_name, "llamacpp-custom-prod")
        self.assertEqual(config.instance_name, "custom-prod")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import shlex
import unittest

from llm_launchpad.core import quick_deploy
from llm_launchpad.core.quick_deploy import (
    QuickDeployProfile,
    build_quick_deploy_config,
    format_context_length,
    get_quick_deploy_catalog_info,
    get_quick_deploy_profile,
    list_quick_deploy_profiles,
    quick_deploy_model_label_parts,
    record_quick_deploy_catalog_failure,
)
from llm_launchpad.protocol.enums import BackendType, SpeculativeDecodingMethod
from llm_launchpad.protocol.models import SpeculativeDecodingConfig

from tests.catalog_fixtures import activate_static_like_catalog


class QuickDeployConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        quick_deploy._reset_quick_deploy_catalog_cache()

    def tearDown(self) -> None:
        quick_deploy._reset_quick_deploy_catalog_cache()

    def test_format_context_length_uses_grouped_tokens(self) -> None:
        self.assertEqual(format_context_length(262144), "262,144 ctx")

    def test_empty_cache_reports_pending_catalog(self) -> None:
        info = get_quick_deploy_catalog_info()
        profiles = list_quick_deploy_profiles()
        self.assertEqual(profiles, ())
        self.assertFalse(info.ready)
        self.assertIsNone(info.error)
        self.assertEqual(info.source_label, "Loading live catalog")

    def test_record_failure_marks_catalog_unavailable_when_empty(self) -> None:
        recorded = record_quick_deploy_catalog_failure("network unreachable")
        self.assertTrue(recorded)
        info = get_quick_deploy_catalog_info()
        self.assertFalse(info.ready)
        self.assertEqual(info.error, "network unreachable")
        self.assertEqual(info.source_label, "Live catalog unavailable")
        self.assertEqual(list_quick_deploy_profiles(), ())

    def test_record_failure_keeps_last_good_catalog(self) -> None:
        activate_static_like_catalog()
        recorded = record_quick_deploy_catalog_failure("late failure")
        self.assertFalse(recorded)
        self.assertEqual([p.id for p in list_quick_deploy_profiles()], ["qwen35-397b-rtxpro", "glm5-rtxpro", "kimi25-rtxpro"])
        self.assertTrue(get_quick_deploy_catalog_info().ready)

    def test_activate_populates_profiles(self) -> None:
        activate_static_like_catalog()
        profiles = list_quick_deploy_profiles()
        self.assertEqual([p.id for p in profiles], ["qwen35-397b-rtxpro", "glm5-rtxpro", "kimi25-rtxpro"])
        info = get_quick_deploy_catalog_info()
        self.assertTrue(info.ready)
        self.assertTrue(info.is_live)
        self.assertIsNone(info.error)

    def test_build_config_can_disable_speculative_decoding(self) -> None:
        profile = QuickDeployProfile(
            id="test-mtp",
            display_name="Test MTP Model",
            repo_id="unsloth/Test-MTP-GGUF",
            quant="UD-Q4_K_XL",
            gpu_type="L40S",
            gpu_count=1,
            profile_label="Test",
            approx_cost_per_hour_usd=2.0,
            max_context_tokens=32768,
            instance_slug_hint="test-mtp",
            summary="Test profile with MTP.",
            server_args=("--ctx-size", "32768"),
            speculative_decoding=SpeculativeDecodingConfig(
                method=SpeculativeDecodingMethod.MTP,
                num_speculative_tokens=3,
                nextn_predict_layers=1,
            ),
        )
        enabled = build_quick_deploy_config(profile)
        disabled = build_quick_deploy_config(profile, enable_speculative_decoding=False)
        self.assertIsNotNone(enabled.speculative_decoding)
        self.assertIsNone(disabled.speculative_decoding)

    def test_quick_deploy_model_label_parts_split_quant_suffix(self) -> None:
        activate_static_like_catalog()
        profile = get_quick_deploy_profile("kimi25-rtxpro")
        self.assertEqual(
            quick_deploy_model_label_parts(profile),
            ("Kimi K2.5", "(UD-Q4_K_XL)"),
        )

    def test_build_quick_deploy_config_maps_profile_defaults(self) -> None:
        activate_static_like_catalog()
        profile = get_quick_deploy_profile("qwen35-397b-rtxpro")
        config = build_quick_deploy_config(profile)
        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "unsloth/Qwen3.5-397B-A17B-GGUF")
        self.assertEqual(config.quant, "UD-Q4_K_XL")
        self.assertEqual(config.gpu_type, "RTX-PRO-6000")
        self.assertEqual(config.gpu_count, 3)
        self.assertEqual(config.required_vram_gb, profile.required_vram_gb)
        self.assertEqual(profile.max_context_tokens, 262144)
        self.assertEqual(config.max_context_tokens, 262144)
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
        activate_static_like_catalog()
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
        activate_static_like_catalog()
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
        activate_static_like_catalog()
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
        activate_static_like_catalog()
        profile = get_quick_deploy_profile("qwen35-397b-rtxpro")
        config = build_quick_deploy_config(profile, app_name="llamacpp-custom-prod")
        self.assertEqual(config.app_name, "llamacpp-custom-prod")
        self.assertEqual(config.instance_name, "custom-prod")


if __name__ == "__main__":
    unittest.main()

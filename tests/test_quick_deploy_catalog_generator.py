from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.modal_gpu import ModalGpuSpec


def _load_generator_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "refresh_quick_deploy_catalog.py"
    spec = importlib.util.spec_from_file_location("refresh_quick_deploy_catalog", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QuickDeployCatalogGeneratorTests(unittest.TestCase):
    def test_build_profile_rows_excludes_unsupported_gguf_architecture(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-glm53",
            name="GLM-5.3 Flash",
            slug="glm-5-3-flash",
            creator_name="Z.ai",
            coding_score=90.0,
            rank=1,
        )

        rows = self.generator.build_profile_rows(
            candidate,
            "unsloth/GLM-5.3-Flash-GGUF",
            metadata=GgufQuantMetadata(
                quantizations=["UD-Q2_K_XL"],
                vram_gb_by_quant={"UD-Q2_K_XL": 100.0},
                architecture="glm5next",
            ),
            modal_gpu_catalog=[ModalGpuSpec(value="B200", price_per_hour_usd=6.25)],
        )

        self.assertEqual(rows, [])

    def setUp(self) -> None:
        self.generator = _load_generator_module()

    def test_normalize_aa_candidates_ranks_and_filters_known_proprietary_rows(self) -> None:
        payload = {
            "data": [
                {
                    "id": "closed",
                    "name": "Closed Code",
                    "slug": "closed-code",
                    "open_weights": False,
                    "evaluations": {"artificial_analysis_intelligence_index": 99.0},
                },
                {
                    "id": "open",
                    "name": "Open Code",
                    "slug": "open-code",
                    "open_weights": True,
                    "evaluations": {"artificial_analysis_intelligence_index": 50.0},
                },
                {
                    "id": "unknown",
                    "name": "Unknown Code",
                    "slug": "unknown-code",
                    "evaluations": {"artificial_analysis_intelligence_index": 75.0},
                },
                {
                    "id": "hosted-open",
                    "name": "Hosted Open Code",
                    "slug": "hosted-open-code",
                    "provider": {"availability": "API only"},
                    "license": "MIT",
                    "evaluations": {"artificial_analysis_intelligence_index": 90.0},
                },
            ]
        }

        candidates = self.generator.normalize_aa_candidates(payload)

        self.assertEqual([candidate.aa_model_id for candidate in candidates], ["hosted-open", "unknown", "open"])
        self.assertEqual([candidate.rank for candidate in candidates], [1, 2, 3])

    def test_unsloth_match_accepts_unique_exact_match(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-1",
            name="Alpha Code",
            slug="alpha-code",
            creator_name="Alpha",
            coding_score=77.0,
            rank=1,
        )

        class FakeApi:
            def list_models(self, **_kwargs):  # type: ignore[no-untyped-def]
                return [SimpleNamespace(id="unsloth/Alpha-Code-GGUF")]

        match = self.generator.find_unsloth_gguf_match(candidate, FakeApi())

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.repo_id, "unsloth/Alpha-Code-GGUF")

    def test_unsloth_match_rejects_ambiguous_exact_matches(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-1",
            name="Alpha Code",
            slug="alpha-code",
            creator_name="Alpha",
            coding_score=77.0,
            rank=1,
        )

        class FakeApi:
            def list_models(self, **_kwargs):  # type: ignore[no-untyped-def]
                return [
                    SimpleNamespace(id="unsloth/Alpha-Code-GGUF"),
                    SimpleNamespace(id="unsloth/Alpha.Code-GGUF"),
                ]

        self.assertIsNone(self.generator.find_unsloth_gguf_match(candidate, FakeApi()))

    def test_unsloth_match_tries_creator_stripped_search_aliases(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-glm51",
            name="Z.ai GLM-5.1",
            slug="glm-5-1",
            creator_name="Z.ai",
            coding_score=86.0,
            rank=1,
        )
        calls: list[str] = []

        class FakeApi:
            def list_models(self, **kwargs):  # type: ignore[no-untyped-def]
                calls.append(str(kwargs.get("search")))
                if kwargs.get("search") == "GLM-5.1":
                    return [SimpleNamespace(id="unsloth/GLM-5.1-GGUF")]
                return []

        match = self.generator.find_unsloth_gguf_match(candidate, FakeApi())

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.repo_id, "unsloth/GLM-5.1-GGUF")
        self.assertIn("Z.ai GLM-5.1", calls)
        self.assertIn("GLM-5.1", calls)

    def test_build_catalog_payload_matches_hf_and_selects_gpu_shape(self) -> None:
        aa_payload = {
            "data": [
                {
                    "id": "aa-alpha",
                    "name": "Alpha Code",
                    "slug": "alpha-code",
                    "open_weights": True,
                    "context_window": "128K",
                    "evaluations": {
                        "artificial_analysis_coding_index": 77.0,
                        "artificial_analysis_intelligence_index": 88.0,
                    },
                }
            ]
        }

        class FakeApi:
            def list_models(self, **_kwargs):  # type: ignore[no-untyped-def]
                return [SimpleNamespace(id="unsloth/Alpha-Code-GGUF")]

        with (
            patch.object(
                self.generator,
                "fetch_gguf_quant_metadata",
                return_value=GgufQuantMetadata(
                    quantizations=["Q4_K_M", "Q8_0"],
                    vram_gb_by_quant={"Q4_K_M": 40.0, "Q8_0": 80.0},
                    architecture="llama",
                ),
            ),
            patch.object(self.generator, "fetch_model_max_context", return_value=None),
        ):
            payload = self.generator.build_catalog_payload(
                aa_payload,
                hf_api=FakeApi(),
                modal_gpu_catalog=[
                    ModalGpuSpec(value="L40S", price_per_hour_usd=2.0),
                    ModalGpuSpec(value="A100-80GB", price_per_hour_usd=3.0),
                ],
                generated_at="2026-05-01T00:00:00Z",
            )

        self.assertEqual(payload["generated_at"], "2026-05-01T00:00:00Z")
        self.assertEqual(len(payload["profiles"]), 1)
        profile = payload["profiles"][0]
        self.assertEqual(profile["repo_id"], "unsloth/Alpha-Code-GGUF")
        self.assertEqual(profile["quant"], "Q4_K_M")
        self.assertEqual(profile["gpu_type"], "L40S")
        self.assertEqual(profile["gpu_count"], 1)
        self.assertEqual(profile["profile_label"], "Slow but cheap")
        self.assertEqual(profile["resource_tier"], "cheap")
        self.assertEqual(profile["resource_tier_label"], "$")
        self.assertEqual(profile["approx_cost_per_hour_usd"], 2.0)
        self.assertEqual(profile["aa_coding_score"], 77.0)
        self.assertEqual(profile["aa_intelligence_score"], 88.0)
        self.assertEqual(profile["required_vram_gb"], 40.0)
        self.assertEqual(profile["max_context_tokens"], 128000)
        self.assertEqual(profile["server_args"], ["--ctx-size", "128000"])

    def test_build_catalog_payload_falls_back_to_hf_context(self) -> None:
        aa_payload = {
            "data": [
                {
                    "id": "aa-beta",
                    "name": "Beta Code",
                    "slug": "beta-code",
                    "open_weights": True,
                    "evaluations": {"artificial_analysis_intelligence_index": 80.0},
                }
            ]
        }

        class FakeApi:
            def list_models(self, **_kwargs):  # type: ignore[no-untyped-def]
                return [SimpleNamespace(id="unsloth/Beta-Code-GGUF")]

        with (
            patch.object(
                self.generator,
                "fetch_gguf_quant_metadata",
                return_value=GgufQuantMetadata(
                    quantizations=["Q4_K_M"],
                    vram_gb_by_quant={"Q4_K_M": 40.0},
                    architecture="llama",
                ),
            ),
            patch.object(self.generator, "fetch_model_max_context", return_value=32768),
        ):
            payload = self.generator.build_catalog_payload(
                aa_payload,
                hf_api=FakeApi(),
                modal_gpu_catalog=[ModalGpuSpec(value="L40S", price_per_hour_usd=2.0)],
                generated_at="2026-05-01T00:00:00Z",
            )

        profile = payload["profiles"][0]
        self.assertEqual(profile["max_context_tokens"], 32768)
        self.assertEqual(profile["server_args"], ["--ctx-size", "32768"])

    def test_build_popular_catalog_payload_uses_explicit_repos_without_aa_metadata(self) -> None:
        models = (
            self.generator.PopularModelCandidate(
                name="Popular 27B",
                slug="popular-27b",
                creator_name="Example",
                repo_id="unsloth/Popular-27B-GGUF",
                max_context_tokens=131072,
            ),
        )
        with (
            patch.object(
                self.generator,
                "fetch_gguf_quant_metadata",
                return_value=GgufQuantMetadata(
                    quantizations=["UD-Q4_K_XL"],
                    vram_gb_by_quant={"UD-Q4_K_XL": 17.0},
                    architecture="llama",
                ),
            ),
            patch.object(self.generator, "fetch_model_max_context") as context_mock,
        ):
            payload = self.generator.build_popular_catalog_payload(
                modal_gpu_catalog=[ModalGpuSpec(value="L4", price_per_hour_usd=0.8)],
                generated_at="2026-08-23T00:00:00Z",
                models=models,
            )

        context_mock.assert_not_called()
        self.assertEqual(payload["source"], "Curated popular open-weight models")
        self.assertEqual(len(payload["profiles"]), 1)
        profile = payload["profiles"][0]
        self.assertEqual(profile["display_name"], "Popular 27B")
        self.assertEqual(profile["repo_id"], "unsloth/Popular-27B-GGUF")
        self.assertEqual(profile["gpu_type"], "L4")
        self.assertEqual(profile["gpu_count"], 1)
        self.assertEqual(profile["required_vram_gb"], 17.0)
        self.assertEqual(profile["max_context_tokens"], 131072)
        self.assertEqual(profile["server_args"], ["--ctx-size", "131072"])
        self.assertEqual(profile["source_label"], "Hugging Face")
        self.assertNotIn("aa_coding_score", profile)
        self.assertNotIn("aa_intelligence_score", profile)
        self.assertNotIn("aa_rank", profile)

    def test_popular_catalog_applies_deployment_measured_vram_floors(self) -> None:
        models = (
            self.generator.PopularModelCandidate(
                name="Qwen3.8 27B",
                slug="qwen3-8-27b",
                creator_name="Alibaba",
                repo_id="unsloth/Qwen3.8-27B-GGUF",
                max_context_tokens=131072,
                required_vram_floor_gb_by_quant=(
                    ("UD-Q2_K_XL", 21.0),
                    ("UD-Q4_K_XL", 29.0),
                ),
            ),
        )
        with patch.object(
            self.generator,
            "fetch_gguf_quant_metadata",
            return_value=GgufQuantMetadata(
                quantizations=["UD-Q4_K_XL", "UD-Q2_K_XL"],
                vram_gb_by_quant={"UD-Q4_K_XL": 17.6, "UD-Q2_K_XL": 9.8},
                architecture="llama",
            ),
        ):
            payload = self.generator.build_popular_catalog_payload(
                modal_gpu_catalog=[ModalGpuSpec(value="L4", price_per_hour_usd=0.8)],
                generated_at="2026-08-24T00:00:00Z",
                models=models,
            )

        cheap_rows = [
            row for row in payload["profiles"] if row["resource_tier"] == "cheap"
        ]
        self.assertEqual(
            [
                (row["quant"], row["required_vram_gb"], row["gpu_type"], row["gpu_count"])
                for row in cheap_rows
            ],
            [
                ("UD-Q2_K_XL", 21.0, "L4", 1),
                ("UD-Q4_K_XL", 29.0, "L4", 2),
            ],
        )

    def test_build_catalog_payload_keeps_newest_model_per_family_before_limit(self) -> None:
        aa_payload = {
            "data": [
                {
                    "id": "aa-kimi",
                    "name": "Kimi K2.6",
                    "slug": "kimi-k2-6",
                    "open_weights": True,
                    "model_creator": {"name": "Moonshot AI"},
                    "evaluations": {"artificial_analysis_intelligence_index": 99.0},
                },
                {
                    "id": "aa-glm5",
                    "name": "GLM-5",
                    "slug": "glm-5",
                    "open_weights": True,
                    "model_creator": {"name": "Z.ai"},
                    "evaluations": {"artificial_analysis_intelligence_index": 95.0},
                },
                {
                    "id": "aa-glm51",
                    "name": "GLM-5.1",
                    "slug": "glm-5-1",
                    "open_weights": True,
                    "model_creator": {"name": "Z.ai"},
                    "evaluations": {"artificial_analysis_intelligence_index": 94.0},
                },
                {
                    "id": "aa-minimax",
                    "name": "MiniMax-M2.7",
                    "slug": "minimax-m2-7",
                    "open_weights": True,
                    "model_creator": {"name": "MiniMax"},
                    "evaluations": {"artificial_analysis_intelligence_index": 90.0},
                },
            ]
        }

        class FakeApi:
            def list_models(self, **kwargs):  # type: ignore[no-untyped-def]
                repo_by_search = {
                    "Kimi K2.6": "unsloth/Kimi-K2.6-GGUF",
                    "GLM-5": "unsloth/GLM-5-GGUF",
                    "GLM-5.1": "unsloth/GLM-5.1-GGUF",
                    "MiniMax-M2.7": "unsloth/MiniMax-M2.7-GGUF",
                }
                repo_id = repo_by_search.get(kwargs.get("search"))
                return [SimpleNamespace(id=repo_id)] if repo_id else []

        with (
            patch.object(
                self.generator,
                "fetch_gguf_quant_metadata",
                return_value=GgufQuantMetadata(
                    quantizations=["UD-Q4_K_XL"],
                    vram_gb_by_quant={"UD-Q4_K_XL": 40.0},
                    architecture="llama",
                ),
            ),
            patch.object(self.generator, "fetch_model_max_context", return_value=65536),
        ):
            payload = self.generator.build_catalog_payload(
                aa_payload,
                hf_api=FakeApi(),
                modal_gpu_catalog=[ModalGpuSpec(value="L40S", price_per_hour_usd=2.0)],
                max_profiles=3,
            )

        self.assertEqual(
            [profile["aa_model_name"] for profile in payload["profiles"]],
            ["Kimi K2.6", "GLM-5.1", "MiniMax-M2.7"],
        )
        self.assertNotIn("GLM-5", [profile["aa_model_name"] for profile in payload["profiles"]])

    def test_context_resolution_uses_known_base_repo_override(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-minimax",
            name="MiniMax-M2.7",
            slug="minimax-m2-7",
            creator_name="MiniMax",
            coding_score=70.0,
            rank=1,
        )

        def fake_context(repo_id: str):
            if repo_id == "MiniMaxAI/MiniMax-M2.7":
                return 196608
            return None

        with patch.object(self.generator, "fetch_model_max_context", side_effect=fake_context):
            context = self.generator._resolve_max_context_tokens(
                candidate,
                "unsloth/MiniMax-M2.7-GGUF",
                fallback=65536,
            )

        self.assertEqual(context, 196608)

    def test_manual_override_applies_known_qwen_profile(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-qwen",
            name="Qwen3.5 397B A17B",
            slug="qwen3-5-397b-a17b",
            creator_name="Alibaba",
            coding_score=91.0,
            rank=1,
        )

        with patch.object(self.generator, "fetch_model_max_context", return_value=262144):
            profiles = self.generator.build_profile_rows(
                candidate,
                "unsloth/Qwen3.5-397B-A17B-GGUF",
                metadata=GgufQuantMetadata(
                    quantizations=[],
                    vram_gb_by_quant={},
                    architecture="llama",
                ),
                modal_gpu_catalog=[ModalGpuSpec(value="RTX-PRO-6000", price_per_hour_usd=3.0312)],
            )

        profile = profiles[0]
        self.assertEqual(profile["id"], "qwen35-397b-rtxpro-q4xl-cheap-rtx-pro-6000")
        self.assertEqual(profile["quant"], "UD-Q4_K_XL")
        self.assertEqual(profile["gpu_count"], 3)
        self.assertEqual(profile["max_context_tokens"], 262144)
        self.assertEqual(profile["server_args"][0:2], ["--ctx-size", "262144"])

    def test_manual_override_uses_hf_vram_for_gpu_shape(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-qwen",
            name="Qwen3.5 397B A17B",
            slug="qwen3-5-397b-a17b",
            creator_name="Alibaba",
            coding_score=91.0,
            rank=1,
        )

        with patch.object(self.generator, "fetch_model_max_context", return_value=262144):
            profiles = self.generator.build_profile_rows(
                candidate,
                "unsloth/Qwen3.5-397B-A17B-GGUF",
                metadata=GgufQuantMetadata(
                    quantizations=["UD-Q4_K_XL"],
                    vram_gb_by_quant={"UD-Q4_K_XL": 141.0},
                    architecture="llama",
                ),
                modal_gpu_catalog=[
                    ModalGpuSpec(value="A100-80GB", price_per_hour_usd=2.5),
                    ModalGpuSpec(value="RTX-PRO-6000", price_per_hour_usd=3.0),
                ],
            )

        profile = profiles[0]
        self.assertEqual(profile["id"], "qwen3-5-397b-a17b-q4xl-cheap-a100-80gb")
        self.assertEqual(profile["gpu_type"], "A100-80GB")
        self.assertEqual(profile["gpu_count"], 2)
        self.assertEqual(profile["approx_cost_per_hour_usd"], 5.0)
        self.assertEqual(profile["required_vram_gb"], 141.0)
        self.assertEqual(profile["instance_slug_hint"], "qwen3-5-397b-a17b-q4xl-cheap")
        self.assertEqual(profile["server_args"][0:2], ["--ctx-size", "262144"])

    def test_manual_override_applies_glm51_server_args_and_context(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-glm51",
            name="GLM-5.1",
            slug="glm-5-1",
            creator_name="Z.ai",
            coding_score=85.0,
            rank=1,
        )

        with patch.object(self.generator, "fetch_model_max_context", return_value=202752):
            rows = self.generator.build_profile_rows(
                candidate,
                "unsloth/GLM-5.1-GGUF",
                metadata=GgufQuantMetadata(
                    quantizations=["UD-Q4_K_XL"],
                    vram_gb_by_quant={"UD-Q4_K_XL": 466.0},
                    architecture="llama",
                ),
                modal_gpu_catalog=[
                    ModalGpuSpec(value="A100-80GB", price_per_hour_usd=2.4984),
                    ModalGpuSpec(value="RTX-PRO-6000", price_per_hour_usd=3.0312),
                    ModalGpuSpec(value="B200", price_per_hour_usd=6.2496),
                ],
            )

        self.assertEqual([row["resource_tier"] for row in rows], ["cheap", "rtx-pro", "b200"])
        self.assertEqual(rows[0]["display_name"], "GLM-5.1")
        self.assertEqual(rows[0]["gpu_type"], "A100-80GB")
        self.assertEqual(rows[0]["gpu_count"], 7)
        self.assertEqual(rows[0]["max_context_tokens"], 202752)
        self.assertEqual(rows[0]["server_args"][0:4], ["--ctx-size", "202752", "--flash-attn", "on"])
        self.assertEqual(rows[1]["gpu_type"], "RTX-PRO-6000")
        self.assertEqual(rows[1]["gpu_count"], 6)
        self.assertEqual(rows[2]["gpu_type"], "B200")
        self.assertEqual(rows[2]["gpu_count"], 3)

    def test_build_profile_rows_adds_price_tiers(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-minimax",
            name="MiniMax-M2.7",
            slug="minimax-m2-7",
            creator_name="MiniMax",
            coding_score=70.0,
            rank=1,
        )

        with patch.object(self.generator, "fetch_model_max_context", return_value=196608):
            rows = self.generator.build_profile_rows(
                candidate,
                "unsloth/MiniMax-M2.7-GGUF",
                metadata=GgufQuantMetadata(
                    quantizations=["UD-Q4_K_XL"],
                    vram_gb_by_quant={"UD-Q4_K_XL": 141.0},
                    architecture="llama",
                ),
                modal_gpu_catalog=[
                    ModalGpuSpec(value="A100-80GB", price_per_hour_usd=2.5),
                    ModalGpuSpec(value="RTX-PRO-6000", price_per_hour_usd=3.03),
                    ModalGpuSpec(value="B200", price_per_hour_usd=6.25),
                ],
            )

        self.assertEqual([row["resource_tier"] for row in rows], ["cheap", "rtx-pro", "b200"])
        self.assertEqual([row["resource_tier_label"] for row in rows], ["$", "$$", "$$$"])
        self.assertEqual(rows[0]["profile_label"], "Slow but cheap")
        self.assertEqual(rows[0]["gpu_type"], "A100-80GB")
        self.assertEqual(rows[0]["gpu_count"], 2)
        self.assertEqual(rows[0]["approx_cost_per_hour_usd"], 5.0)
        self.assertEqual(rows[1]["profile_label"], "RTX PRO")
        self.assertEqual(rows[1]["gpu_type"], "RTX-PRO-6000")
        self.assertEqual(rows[1]["gpu_count"], 2)
        self.assertEqual(rows[1]["approx_cost_per_hour_usd"], 6.06)
        self.assertEqual(rows[2]["profile_label"], "B200")
        self.assertEqual(rows[2]["gpu_type"], "B200")
        self.assertEqual(rows[2]["gpu_count"], 1)
        self.assertEqual(rows[2]["approx_cost_per_hour_usd"], 6.25)

    def test_build_profile_rows_adds_low_vram_quant_gpu_tiers_when_available(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-minimax",
            name="MiniMax-M2.7",
            slug="minimax-m2-7",
            creator_name="MiniMax",
            coding_score=70.0,
            rank=1,
        )

        with patch.object(self.generator, "fetch_model_max_context", return_value=196608):
            rows = self.generator.build_profile_rows(
                candidate,
                "unsloth/MiniMax-M2.7-GGUF",
                metadata=GgufQuantMetadata(
                    quantizations=["UD-Q4_K_XL", "UD-Q2_K_XL"],
                    vram_gb_by_quant={"UD-Q4_K_XL": 141.0, "UD-Q2_K_XL": 70.0},
                    architecture="llama",
                ),
                modal_gpu_catalog=[
                    ModalGpuSpec(value="A100-80GB", price_per_hour_usd=2.5),
                    ModalGpuSpec(value="RTX-PRO-6000", price_per_hour_usd=3.03),
                    ModalGpuSpec(value="B200", price_per_hour_usd=6.25),
                ],
            )

        self.assertEqual(
            [(row["quant"], row["resource_tier"]) for row in rows],
            [
                ("UD-Q2_K_XL", "cheap"),
                ("UD-Q2_K_XL", "rtx-pro"),
                ("UD-Q2_K_XL", "b200"),
                ("UD-Q4_K_XL", "cheap"),
                ("UD-Q4_K_XL", "rtx-pro"),
                ("UD-Q4_K_XL", "b200"),
            ],
        )
        self.assertEqual(rows[0]["gpu_type"], "A100-80GB")
        self.assertEqual(rows[0]["gpu_count"], 1)
        self.assertEqual(rows[0]["approx_cost_per_hour_usd"], 2.5)
        self.assertEqual(rows[0]["required_vram_gb"], 70.0)
        self.assertEqual(rows[0]["id"], "minimax-m2-7-q2xl-cheap-a100-80gb")
        self.assertEqual(rows[1]["gpu_type"], "RTX-PRO-6000")
        self.assertEqual(rows[1]["gpu_count"], 1)
        self.assertEqual(rows[2]["gpu_type"], "B200")
        self.assertEqual(rows[2]["gpu_count"], 1)
        self.assertEqual(rows[3]["gpu_type"], "A100-80GB")
        self.assertEqual(rows[3]["gpu_count"], 2)
        self.assertEqual(rows[3]["id"], "minimax-m2-7-q4xl-cheap-a100-80gb")

    def test_build_profile_rows_deduplicates_identical_quant_gpu_shapes(self) -> None:
        candidate = self.generator.AAModelCandidate(
            aa_model_id="aa-kimi",
            name="Kimi K2.6",
            slug="kimi-k2-6",
            creator_name="Moonshot",
            coding_score=70.0,
            rank=1,
        )

        with patch.object(self.generator, "fetch_model_max_context", return_value=262144):
            rows = self.generator.build_profile_rows(
                candidate,
                "unsloth/Kimi-K2.6-GGUF",
                metadata=GgufQuantMetadata(
                    quantizations=["UD-Q4_K_XL", "UD-Q2_K_XL"],
                    vram_gb_by_quant={"UD-Q4_K_XL": 584.0, "UD-Q2_K_XL": 340.0},
                    architecture="llama",
                ),
                modal_gpu_catalog=[
                    ModalGpuSpec(value="A100-80GB", price_per_hour_usd=2.5),
                    ModalGpuSpec(value="RTX-PRO-6000", price_per_hour_usd=3.03),
                    ModalGpuSpec(value="B200", price_per_hour_usd=6.25),
                ],
            )

        q2_rows = [row for row in rows if row["quant"] == "UD-Q2_K_XL"]
        self.assertEqual([(row["resource_tier"], row["gpu_type"], row["gpu_count"]) for row in q2_rows], [
            ("cheap", "RTX-PRO-6000", 4),
            ("b200", "B200", 2),
        ])
        self.assertEqual(q2_rows[0]["resource_tier_label"], "$/$$")
        self.assertEqual(q2_rows[0]["profile_label"], "Slow but cheap / RTX PRO")

    def test_cost_minimizing_gpu_shape_chooses_lowest_hourly_cost(self) -> None:
        shape = self.generator._cost_minimizing_gpu_shape(
            141.0,
            [
                ModalGpuSpec(value="B200", price_per_hour_usd=6.25),
                ModalGpuSpec(value="H200", price_per_hour_usd=4.54),
                ModalGpuSpec(value="A100-80GB", price_per_hour_usd=2.50),
                ModalGpuSpec(value="RTX-PRO-6000", price_per_hour_usd=3.03),
            ],
        )

        self.assertEqual(shape, ("A100-80GB", 2, 5.0))

    def test_cost_minimizing_gpu_shape_uses_newer_gpu_when_it_is_cheapest(self) -> None:
        shape = self.generator._cost_minimizing_gpu_shape(
            130.0,
            [
                ModalGpuSpec(value="B200", price_per_hour_usd=6.25),
                ModalGpuSpec(value="H200", price_per_hour_usd=4.54),
                ModalGpuSpec(value="A100-80GB", price_per_hour_usd=2.50),
            ],
        )

        self.assertEqual(shape, ("H200", 1, 4.54))

    def test_quant_selection_prefers_ud_q4_when_available(self) -> None:
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL", "UD-Q4_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0, "UD-Q4_K_XL": 80.0},
            architecture="llama",
        )

        selected = self.generator._select_quant_and_gpu(
            metadata,
            [
                ModalGpuSpec(value="L40S", price_per_hour_usd=1.0),
                ModalGpuSpec(value="A100-80GB", price_per_hour_usd=3.0),
            ],
        )

        assert selected is not None
        self.assertEqual(selected[0], "UD-Q4_K_XL")

    def test_main_uses_fallback_modal_catalog_when_live_fetch_fails(self) -> None:
        payload = {"data": []}
        catalog = {
            "schema_version": 1,
            "generated_at": "2026-06-16T00:00:00Z",
            "source": "Artificial Analysis coding rankings",
            "profiles": [{"id": "profile"}],
        }

        with (
            patch.dict("os.environ", {"ARTIFICIAL_ANALYSIS_API_KEY": "key"}),
            patch.object(self.generator, "fetch_aa_llm_models", return_value=payload) as fetch_aa,
            patch.object(self.generator, "fetch_modal_gpu_catalog", side_effect=RuntimeError("HTTP 403")),
            patch.object(self.generator, "build_catalog_payload", return_value=catalog) as build_payload,
            patch.object(self.generator, "write_catalog") as write_catalog,
        ):
            result = self.generator.main(
                ["--max-profiles", "4", "--output", "quick_deploy_catalog.json"]
            )

        self.assertEqual(result, 0)
        fetch_aa.assert_called_once_with("key")
        build_payload.assert_called_once()
        self.assertEqual(build_payload.call_args.kwargs["modal_gpu_catalog"], [])
        self.assertEqual(build_payload.call_args.kwargs["max_profiles"], 4)
        write_catalog.assert_called_once_with(catalog, path=Path("quick_deploy_catalog.json"))


if __name__ == "__main__":
    unittest.main()

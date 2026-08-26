from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core.hf_models import GgufQuantMetadata, ModelCandidate
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.quick_deploy_refresh import (
    AAModelCandidate,
    _AARankings,
    _load_aa_rankings,
    _write_aa_cache,
    build_live_quick_deploy_catalog,
    fetch_artificial_analysis_models,
    normalize_aa_model_candidates,
)


def _aa_candidate(
    name: str,
    parameter_count_b: float,
    score: float,
) -> AAModelCandidate:
    slug = name.casefold().replace(" ", "-")
    return AAModelCandidate(
        aa_model_id=f"aa-{slug}",
        name=name,
        slug=slug,
        creator_name="Test Org",
        coding_score=score,
        agentic_score=score - 1,
        intelligence_score=score - 2,
        capability_score=score - 0.75,
        rank=1,
        parameter_count_b=parameter_count_b,
        max_context_tokens=131_072,
    )


class QuickDeployRefreshTests(unittest.TestCase):
    def test_live_catalog_selects_one_aa_leader_per_model_size(self) -> None:
        candidates = (
            _aa_candidate("Compact Model 8B", 8, 80),
            _aa_candidate("Medium Model 70B", 70, 90),
            _aa_candidate("Large Model 300B", 300, 95),
        )
        rankings = _AARankings(candidates=candidates, freshness="live", tier="pro")
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL", "UD-Q4_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0, "UD-Q4_K_XL": 30.0},
        )
        gpu_catalog = [
            ModalGpuSpec("L4", 0.5),
            ModalGpuSpec("RTX-PRO-6000", 2.0),
            ModalGpuSpec("B200", 5.0),
        ]

        def matched_repo(candidate: AAModelCandidate, _api: object) -> str:
            return f"unsloth/{candidate.name.replace(' ', '-')}-GGUF"

        with patch(
            "llm_launchpad.core.quick_deploy_refresh._load_aa_rankings",
            return_value=rankings,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_modal_gpu_catalog",
            return_value=gpu_catalog,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match",
            side_effect=matched_repo,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_gguf_quant_metadata",
            return_value=metadata,
        ):
            info, profiles = build_live_quick_deploy_catalog()

        self.assertTrue(info.is_live)
        self.assertIn("Artificial Analysis size leaders", info.source_label)
        self.assertIn("live Modal pricing", info.source_label)
        self.assertEqual(
            {profile.model_size_label for profile in profiles},
            {"Compact ≤40B", "Medium 40–150B", "Large >150B"},
        )
        self.assertEqual({profile.source_label for profile in profiles}, {"Artificial Analysis"})
        self.assertEqual({profile.max_context_tokens for profile in profiles}, {131_072})

        compact_q4 = next(
            profile
            for profile in profiles
            if profile.model_size_label == "Compact ≤40B"
            and profile.quant == "UD-Q4_K_XL"
            and profile.resource_tier == "cheap"
        )
        self.assertEqual(compact_q4.gpu_type, "L4")
        self.assertEqual(compact_q4.gpu_count, 2)
        self.assertEqual(compact_q4.approx_cost_per_hour_usd, 1.0)
        self.assertEqual(compact_q4.aa_coding_score, 80)

    def test_fallback_catalog_uses_trending_models_without_aa_data(self) -> None:
        models = [ModelCandidate(repo_id="unsloth/Test-Model-GGUF")]
        metadata = GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": 10.0},
        )
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._load_aa_rankings",
            return_value=None,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.list_llamacpp_candidates",
            return_value=models,
        ) as list_models, patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_modal_gpu_catalog",
            return_value=[ModalGpuSpec("L4", 0.75)],
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_gguf_quant_metadata",
            return_value=metadata,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_model_max_context",
            return_value=None,
        ):
            info, profiles = build_live_quick_deploy_catalog(model_limit=1)

        list_models.assert_called_once_with(mode="trending", limit=6)
        self.assertIn("Hugging Face trending", info.source_label)
        self.assertEqual({profile.repo_id for profile in profiles}, {models[0].repo_id})
        self.assertEqual({profile.max_context_tokens for profile in profiles}, {65_536})

    def test_fallback_catalog_skips_unusable_trending_models(self) -> None:
        models = [
            ModelCandidate(repo_id="org/Broken-GGUF"),
            ModelCandidate(repo_id="org/Usable-GGUF"),
        ]

        def metadata_for(repo_id: str) -> GgufQuantMetadata:
            if repo_id == "org/Broken-GGUF":
                raise RuntimeError("metadata unavailable")
            return GgufQuantMetadata(
                quantizations=["Q4_K_M"],
                vram_gb_by_quant={"Q4_K_M": 10.0},
            )

        with patch(
            "llm_launchpad.core.quick_deploy_refresh._load_aa_rankings",
            return_value=None,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.list_llamacpp_candidates",
            return_value=models,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_modal_gpu_catalog",
            return_value=[ModalGpuSpec("L4", 0.75)],
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_gguf_quant_metadata",
            side_effect=metadata_for,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_model_max_context",
            return_value=None,
        ):
            _info, profiles = build_live_quick_deploy_catalog(model_limit=1)

        self.assertEqual({profile.repo_id for profile in profiles}, {"org/Usable-GGUF"})

    def test_catalog_fails_when_no_aa_or_trending_models_are_available(self) -> None:
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._load_aa_rankings",
            return_value=None,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.list_llamacpp_candidates",
            return_value=[],
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_modal_gpu_catalog",
            return_value=[],
        ):
            with self.assertRaisesRegex(RuntimeError, "Neither Artificial Analysis"):
                build_live_quick_deploy_catalog()

    def test_normalize_aa_candidates_deduplicates_variants_and_weights_scores(self) -> None:
        payload = {
            "data": [
                {
                    "id": "closed",
                    "name": "Closed 100B",
                    "slug": "closed",
                    "licensing": {"is_open_weights": False},
                    "evaluations": {"artificial_analysis_coding_index": 100},
                },
                {
                    "id": "low",
                    "name": "Example 8B (low)",
                    "slug": "example-8b",
                    "parameters": {"total": 8},
                    "evaluations": {
                        "artificial_analysis_coding_index": 40,
                        "artificial_analysis_agentic_index": 30,
                        "artificial_analysis_intelligence_index": 20,
                    },
                },
                {
                    "id": "high",
                    "name": "Example 8B (high)",
                    "slug": "example-8b",
                    "parameters": {"total": 8},
                    "context_window_tokens": 131072,
                    "evaluations": {
                        "artificial_analysis_coding_index": 80,
                        "artificial_analysis_agentic_index": 70,
                        "artificial_analysis_intelligence_index": 60,
                    },
                },
            ]
        }

        candidates = normalize_aa_model_candidates(payload)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].aa_model_id, "high")
        self.assertEqual(candidates[0].parameter_count_b, 8)
        self.assertEqual(candidates[0].max_context_tokens, 131_072)
        self.assertAlmostEqual(candidates[0].capability_score, 72.5)
        self.assertEqual(candidates[0].rank, 1)

    def test_fresh_aa_cache_avoids_api_request(self) -> None:
        now = datetime(2026, 8, 26, 12, tzinfo=timezone.utc)
        payload = {
            "tier": "free",
            "data": [
                {
                    "id": "cached",
                    "name": "Cached 8B",
                    "slug": "cached-8b",
                    "evaluations": {"artificial_analysis_coding_index": 50},
                }
            ],
        }
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            _write_aa_cache(
                cache_path,
                payload,
                fetched_at=now - timedelta(hours=1),
                api_key="key",
            )
            with patch(
                "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models"
            ) as fetch:
                rankings = _load_aa_rankings(
                    api_key="key",
                    cache_path=cache_path,
                    now=now,
                )

        fetch.assert_not_called()
        assert rankings is not None
        self.assertEqual(rankings.freshness, "cached")
        self.assertEqual(rankings.tier, "free")

    def test_stale_aa_cache_refreshes_once_and_is_rewritten(self) -> None:
        now = datetime(2026, 8, 26, 12, tzinfo=timezone.utc)
        stale_payload = {
            "data": [
                {
                    "id": "stale",
                    "name": "Stale 8B",
                    "slug": "stale-8b",
                    "evaluations": {"artificial_analysis_coding_index": 40},
                }
            ]
        }
        live_payload = {
            "tier": "pro",
            "data": [
                {
                    "id": "live",
                    "name": "Live 70B",
                    "slug": "live-70b",
                    "parameters": {"total": 70},
                    "evaluations": {"artificial_analysis_coding_index": 90},
                }
            ],
        }
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            _write_aa_cache(cache_path, stale_payload, fetched_at=now - timedelta(days=2))
            with patch(
                "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models",
                return_value=live_payload,
            ) as fetch:
                rankings = _load_aa_rankings(
                    api_key="key",
                    cache_path=cache_path,
                    now=now,
                )
            cache_text = cache_path.read_text(encoding="utf-8")

        fetch.assert_called_once_with("key")
        assert rankings is not None
        self.assertEqual(rankings.freshness, "live")
        self.assertEqual(rankings.candidates[0].aa_model_id, "live")
        self.assertIn('"live"', cache_text)

    def test_aa_fetch_falls_back_to_free_tier_and_collects_pages(self) -> None:
        responses = [
            SimpleNamespace(status_code=403, json=lambda: {}),
            SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "tier": "free",
                    "pagination": {"has_more": True},
                    "data": [{"id": "one"}],
                },
            ),
            SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "tier": "free",
                    "pagination": {"has_more": False},
                    "data": [{"id": "two"}],
                },
            ),
        ]
        with patch("requests.get", side_effect=responses) as get:
            payload = fetch_artificial_analysis_models("secret")

        self.assertEqual([row["id"] for row in payload["data"]], ["one", "two"])
        self.assertEqual(get.call_count, 3)
        self.assertEqual(
            get.call_args_list[0].args[0],
            "https://artificialanalysis.ai/api/v2/language/models",
        )
        self.assertEqual(
            get.call_args_list[1].args[0],
            "https://artificialanalysis.ai/api/v2/language/models/free",
        )
        self.assertEqual(get.call_args_list[2].kwargs["params"], {"page": 2})

    def test_stale_cache_remains_available_when_aa_refresh_fails(self) -> None:
        now = datetime(2026, 8, 26, 12, tzinfo=timezone.utc)
        payload = {
            "data": [
                {
                    "id": "stale",
                    "name": "Stale 8B",
                    "slug": "stale-8b",
                    "evaluations": {"artificial_analysis_coding_index": 40},
                }
            ]
        }
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            _write_aa_cache(cache_path, payload, fetched_at=now - timedelta(days=2))
            with patch(
                "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models",
                side_effect=RuntimeError("offline"),
            ):
                rankings = _load_aa_rankings(
                    api_key="key",
                    cache_path=cache_path,
                    now=now,
                )

        assert rankings is not None
        self.assertEqual(rankings.freshness, "cached")
        self.assertEqual(rankings.candidates[0].aa_model_id, "stale")


if __name__ == "__main__":
    unittest.main()

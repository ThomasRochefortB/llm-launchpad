from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Sequence
import unittest
from unittest.mock import patch

from llm_launchpad.core.gguf_metadata import GgufMtpCapability, GgufMtpStatus
from llm_launchpad.core.hf_models import GgufQuantMetadata, ModelCandidate
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.quick_deploy_refresh import (
    AAModelCandidate,
    QUICK_DEPLOY_CATALOG_CACHE_SCHEMA_VERSION,
    _AARankings,
    _available_gpu_types,
    _find_unsloth_gguf_match,
    _load_aa_rankings,
    _quick_deploy_profile_from_dict,
    _quick_deploy_profile_to_dict,
    _read_quick_deploy_catalog_cache,
    _write_aa_cache,
    attach_quick_deploy_mtp_recommendations,
    build_live_quick_deploy_catalog,
    fetch_artificial_analysis_models,
    is_fresh_cached_quick_deploy_catalog,
    load_cached_quick_deploy_catalog,
    normalize_aa_model_candidates,
    _mtp_recommendation,
)


def _aa_candidate(
    name: str,
    parameter_count_b: float,
    score: float,
    rank: int = 1,
) -> AAModelCandidate:
    slug = name.casefold().replace(" ", "-")
    return AAModelCandidate(
        aa_model_id=f"aa-{slug}",
        name=name,
        slug=slug,
        creator_name="Test Org",
        coding_score=score,
        intelligence_score=score,
        rank=rank,
        parameter_count_b=parameter_count_b,
        max_context_tokens=131_072,
    )


def _ordered_unique_names(profiles: Sequence[object]) -> list[str]:
    names: list[str] = []
    for profile in profiles:
        if profile.display_name not in names:
            names.append(str(profile.display_name))
    return names


class QuickDeployRefreshTests(unittest.TestCase):
    def test_mtp_recommendation_requires_model_and_runtime_support(self) -> None:
        supported = GgufQuantMetadata(
            quantizations=[],
            vram_gb_by_quant={},
            architecture="qwen35",
            mtp=GgufMtpCapability(
                status=GgufMtpStatus.SUPPORTED,
                nextn_predict_layers=1,
            ),
        )
        absent = GgufQuantMetadata(
            quantizations=[],
            vram_gb_by_quant={},
            architecture="deepseek4",
            mtp=GgufMtpCapability(
                status=GgufMtpStatus.UNSUPPORTED,
                nextn_predict_layers=0,
            ),
        )

        recommendation = _mtp_recommendation(supported)

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.num_speculative_tokens, 3)  # type: ignore[union-attr]
        self.assertIsNone(_mtp_recommendation(absent))

    def test_live_catalog_selects_top_open_models_in_rank_order(self) -> None:
        candidates = (
            _aa_candidate("Closed Model 8B", 8, 99, rank=1),
            _aa_candidate("Open Model One 8B", 8, 95, rank=2),
            _aa_candidate("Open Model Two 70B", 70, 90, rank=3),
            _aa_candidate("Open Model Three 300B", 300, 85, rank=4),
        )
        rankings = _AARankings(candidates=candidates, freshness="live", tier="pro")
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL", "UD-Q4_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0, "UD-Q4_K_XL": 30.0},
            architecture="llama",
        )
        gpu_catalog = [
            ModalGpuSpec("L4", 0.5),
            ModalGpuSpec("RTX-PRO-6000", 2.0),
            ModalGpuSpec("B200", 5.0),
        ]

        def matched_repo(candidate: AAModelCandidate, _api: object) -> str | None:
            if candidate.name.startswith("Closed"):
                return None
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
        ) as match_mock, patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_gguf_quant_metadata",
            return_value=metadata,
        ):
            info, profiles = build_live_quick_deploy_catalog()

        self.assertTrue(info.is_live)
        self.assertIn("Artificial Analysis top open models", info.source_label)
        self.assertIn("live Modal pricing", info.source_label)
        self.assertEqual(
            _ordered_unique_names(profiles),
            [
                "Open Model One 8B",
                "Open Model Two 70B",
                "Open Model Three 300B",
            ],
        )
        self.assertEqual(
            {profile.model_size_label for profile in profiles},
            {"Compact ≤40B", "Medium 40–150B", "Large >150B"},
        )
        self.assertEqual({profile.source_label for profile in profiles}, {"Artificial Analysis"})
        self.assertEqual({profile.max_context_tokens for profile in profiles}, {131_072})
        self.assertEqual(match_mock.call_count, len(candidates))

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
        self.assertEqual(compact_q4.aa_coding_score, 95)
        self.assertEqual(compact_q4.aa_intelligence_score, 95)

    def test_live_catalog_caps_selection_at_model_limit(self) -> None:
        candidates = tuple(
            _aa_candidate(f"Open Model {index} 8B", 8, 90 - index, rank=index)
            for index in range(1, 5)
        )
        rankings = _AARankings(candidates=candidates, freshness="live")
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0},
            architecture="llama",
        )
        gpu_catalog = [ModalGpuSpec("L4", 0.5)]

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
            _info, profiles = build_live_quick_deploy_catalog(model_limit=2)

        self.assertEqual(
            _ordered_unique_names(profiles),
            ["Open Model 1 8B", "Open Model 2 8B"],
        )

    def test_live_catalog_selects_top_models_in_each_size_bucket(self) -> None:
        candidates: list[AAModelCandidate] = []
        rank = 1
        for parameter_count_b, prefix in (
            (8, "Compact"),
            (70, "Medium"),
            (300, "Large"),
        ):
            for index in range(1, 5):
                candidates.append(
                    _aa_candidate(
                        f"{prefix} Model {index} {parameter_count_b}B",
                        parameter_count_b,
                        100 - rank,
                        rank=rank,
                    )
                )
                rank += 1
        rankings = _AARankings(candidates=tuple(candidates), freshness="live")
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0},
            architecture="llama",
        )
        gpu_catalog = [ModalGpuSpec("L4", 0.5)]

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
            _info, profiles = build_live_quick_deploy_catalog()

        self.assertEqual(
            _ordered_unique_names(profiles),
            [
                "Compact Model 1 8B",
                "Compact Model 2 8B",
                "Compact Model 3 8B",
                "Medium Model 1 70B",
                "Medium Model 2 70B",
                "Medium Model 3 70B",
                "Large Model 1 300B",
                "Large Model 2 300B",
                "Large Model 3 300B",
            ],
        )

    def test_live_catalog_keeps_size_buckets_beyond_global_rank_window(self) -> None:
        candidates: list[AAModelCandidate] = [
            _aa_candidate(
                f"Compact Model {index} 8B",
                8,
                100 - index,
                rank=index,
            )
            for index in range(1, 81)
        ]
        candidates.extend(
            _aa_candidate(f"Medium Model {index} 70B", 70, 10 - index, rank=80 + index)
            for index in range(1, 4)
        )
        candidates.extend(
            _aa_candidate(f"Large Model {index} 300B", 300, 5 - index, rank=83 + index)
            for index in range(1, 4)
        )
        rankings = _AARankings(candidates=tuple(candidates), freshness="live")
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0},
            architecture="llama",
        )
        gpu_catalog = [ModalGpuSpec("L4", 0.5)]

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
            _info, profiles = build_live_quick_deploy_catalog()

        names = _ordered_unique_names(profiles)
        self.assertEqual(len(names), 9)
        self.assertEqual(names[:3], [f"Compact Model {index} 8B" for index in range(1, 4)])
        self.assertEqual(names[3:6], [f"Medium Model {index} 70B" for index in range(1, 4)])
        self.assertEqual(names[6:], [f"Large Model {index} 300B" for index in range(1, 4)])

    def test_live_catalog_deduplicates_variants_sharing_a_gguf_repo(self) -> None:
        candidates = (
            _aa_candidate("Open Model One 8B", 8, 95, rank=1),
            _aa_candidate("Open Model One (low) 8B", 8, 90, rank=2),
            _aa_candidate("Open Model Two 70B", 70, 85, rank=3),
        )
        rankings = _AARankings(candidates=candidates, freshness="live")
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0},
            architecture="llama",
        )
        gpu_catalog = [ModalGpuSpec("L4", 0.5)]

        def matched_repo(candidate: AAModelCandidate, _api: object) -> str:
            if "One" in candidate.name:
                return "unsloth/Open-Model-One-GGUF"
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
        ) as match_mock, patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_gguf_quant_metadata",
            return_value=metadata,
        ):
            _info, profiles = build_live_quick_deploy_catalog()

        self.assertEqual(
            {profile.repo_id for profile in profiles},
            {
                "unsloth/Open-Model-One-GGUF",
                "unsloth/Open-Model-Two-70B-GGUF",
            },
        )
        self.assertEqual(
            _ordered_unique_names(profiles),
            ["Open Model One 8B", "Open Model Two 70B"],
        )
        self.assertEqual(match_mock.call_count, 2)

    def test_find_unsloth_gguf_match_finds_untagged_fresh_repos(self) -> None:
        class _FakeHfApi:
            def __init__(self) -> None:
                self.last_search_kwargs: dict[str, object] = {}

            def model_info(self, repo_id: str) -> object:
                raise RuntimeError(f"404 {repo_id}")

            def list_models(self, **kwargs: Any) -> list[dict[str, str]]:
                self.last_search_kwargs = kwargs
                if "gguf" in [str(tag) for tag in kwargs.get("filter", ())]:
                    # Fresh unsloth repos lack the gguf tag, so a hard
                    # filter excludes them from HF search results.
                    return []
                return [
                    {"id": "unsloth/Fresh Model 8B"},
                    {"id": "unsloth/Fresh Model 8B-GGUF"},
                ]

        candidate = _aa_candidate("Fresh Model 8B", 8, 90, rank=1)
        api = _FakeHfApi()

        repo_id = _find_unsloth_gguf_match(candidate, api)

        self.assertEqual(repo_id, "unsloth/Fresh Model 8B-GGUF")
        self.assertNotIn("filter", api.last_search_kwargs)

    def test_fallback_catalog_uses_trending_models_without_aa_data(self) -> None:
        models = [ModelCandidate(repo_id="unsloth/Test-Model-GGUF")]
        metadata = GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": 10.0},
            architecture="llama",
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

        list_models.assert_called_once_with(mode="trending", limit=8)
        self.assertIn("Hugging Face trending", info.source_label)
        self.assertEqual({profile.repo_id for profile in profiles}, {models[0].repo_id})
        self.assertEqual({profile.max_context_tokens for profile in profiles}, {65_536})

    def test_fallback_catalog_skips_unusable_trending_models(self) -> None:
        models = [
            ModelCandidate(repo_id="org/Broken-GGUF"),
            ModelCandidate(repo_id="org/Usable-GGUF"),
        ]

        def metadata_for(repo_id: str, **_kwargs: object) -> GgufQuantMetadata:
            if repo_id == "org/Broken-GGUF":
                raise RuntimeError("metadata unavailable")
            return GgufQuantMetadata(
                quantizations=["Q4_K_M"],
                vram_gb_by_quant={"Q4_K_M": 10.0},
                architecture="llama",
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

    def test_fallback_catalog_upgrades_context_length_after_profiles_exist(self) -> None:
        models = [ModelCandidate(repo_id="unsloth/Test-Model-GGUF")]
        metadata = GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": 10.0},
            architecture="llama",
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
            return_value=metadata,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_model_max_context",
            return_value=131_072,
        ), patch(
            "llm_launchpad.core.quick_deploy_refresh._write_quick_deploy_catalog_cache",
        ):
            _info, profiles = build_live_quick_deploy_catalog(model_limit=1)

        self.assertTrue(profiles)
        self.assertEqual(
            {profile.max_context_tokens for profile in profiles}, {131_072}
        )
        for profile in profiles:
            self.assertIn(str(131_072), " ".join(profile.server_args))

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

    def test_normalize_aa_candidates_deduplicates_variants_by_intelligence_index(self) -> None:
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
        self.assertAlmostEqual(candidates[0].intelligence_score, 60.0)
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
                    "evaluations": {"artificial_analysis_intelligence_index": 50},
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
                    "evaluations": {"artificial_analysis_intelligence_index": 40},
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
                    "evaluations": {"artificial_analysis_intelligence_index": 90},
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
                    "evaluations": {"artificial_analysis_intelligence_index": 40},
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

    def test_available_gpu_types_skips_unpriced_and_unknown_shapes(self) -> None:
        catalog = [
            ModalGpuSpec("T4", 0.5),
            ModalGpuSpec("H100!", None),
            ModalGpuSpec("H200", None),
            ModalGpuSpec("B200", 6.25),
            ModalGpuSpec("B300", 7.10),
        ]
        # H100!/H200/B300 have no usable Modal price or VRAM entry, so the
        # catalog must not offer them as deploy shapes. Previously B300 (no
        # VRAM entry) and unpriced H100!/H200 leaked through as profiles.
        self.assertEqual(
            _available_gpu_types(catalog),
            ["T4", "B200"],
        )

    def test_available_gpu_types_falls_back_to_priced_shapes(self) -> None:
        self.assertEqual(
            _available_gpu_types([]),
            ["T4", "L4", "A100", "L40S", "RTX-PRO-6000", "H100", "H200", "B200"],
        )

    def test_catalog_build_skips_mtp_inspection_on_first_pass(self) -> None:
        candidates = (
            _aa_candidate("Open Model One 8B", 8, 95, rank=1),
            _aa_candidate("Open Model Two 8B", 8, 90, rank=2),
        )
        rankings = _AARankings(candidates=candidates, freshness="live")
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0},
            architecture="llama",
        )
        gpu_catalog = [ModalGpuSpec("L4", 0.5)]

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
        ) as fetch_metadata, patch(
            "llm_launchpad.core.quick_deploy_refresh._write_quick_deploy_catalog_cache",
        ):
            _info, profiles = build_live_quick_deploy_catalog(model_limit=1)

        self.assertTrue(profiles)
        self.assertTrue(
            all(profile.speculative_decoding is None for profile in profiles)
        )
        for _args, kwargs in fetch_metadata.call_args_list:
            self.assertNotEqual(kwargs.get("inspect_mtp"), True)

    def test_attach_mtp_recommendations_upgrades_profiles_lazily(self) -> None:
        from llm_launchpad.core.quick_deploy import QuickDeployProfile

        profile = QuickDeployProfile(
            id="model-cheap-l4",
            display_name="Open Model One 8B",
            repo_id="unsloth/Open-Model-One-8B-GGUF",
            quant="UD-Q2_K_XL",
            gpu_type="L4",
            gpu_count=2,
            profile_label="Slow but cheap",
            approx_cost_per_hour_usd=1.0,
            max_context_tokens=131_072,
            instance_slug_hint="open-model-one",
            summary="Test profile.",
            server_args=("--ctx-size", "131072"),
        )
        metadata = GgufQuantMetadata(
            quantizations=["UD-Q2_K_XL"],
            vram_gb_by_quant={"UD-Q2_K_XL": 20.0},
            architecture="qwen35",
            mtp=GgufMtpCapability(
                status=GgufMtpStatus.SUPPORTED,
                nextn_predict_layers=1,
            ),
        )
        with patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_gguf_quant_metadata",
            return_value=metadata,
        ) as fetch_metadata:
            upgraded = attach_quick_deploy_mtp_recommendations((profile,))

        fetch_metadata.assert_called_once_with(
            "unsloth/Open-Model-One-8B-GGUF",
            inspect_mtp=True,
        )
        self.assertIsNone(profile.speculative_decoding)
        self.assertIsNotNone(upgraded[0].speculative_decoding)

    def test_cached_catalog_round_trip_marks_stale_snapshots(self) -> None:
        from llm_launchpad.core.quick_deploy import QuickDeployProfile

        profile = QuickDeployProfile(
            id="model-cheap-l4",
            display_name="Open Model One 8B",
            repo_id="unsloth/Open-Model-One-8B-GGUF",
            quant="UD-Q2_K_XL",
            gpu_type="L4",
            gpu_count=2,
            profile_label="Slow but cheap",
            approx_cost_per_hour_usd=1.0,
            max_context_tokens=131_072,
            instance_slug_hint="open-model-one",
            summary="Test profile.",
            server_args=("--ctx-size", "131072"),
        )
        payload = _quick_deploy_profile_to_dict(profile)
        restored = _quick_deploy_profile_from_dict(payload)
        self.assertEqual(restored, profile)

        with TemporaryDirectory() as temporary_directory:
            from llm_launchpad.core.quick_deploy import QuickDeployCatalogInfo

            cache_path = Path(temporary_directory) / "catalog.json"
            fresh_info = QuickDeployCatalogInfo(
                source_label="Test catalog",
                generated_at="2026-09-03T00:00:00Z",
                is_live=True,
                ready=True,
            )
            envelope = {
                "schema_version": QUICK_DEPLOY_CATALOG_CACHE_SCHEMA_VERSION,
                "info": {
                    "source_label": fresh_info.source_label,
                    "generated_at": fresh_info.generated_at,
                    "is_live": True,
                    "ready": True,
                },
                "profiles": [payload],
            }
            cache_path.write_text(json.dumps(envelope), encoding="utf-8")
            cached = _read_quick_deploy_catalog_cache(cache_path)
            assert cached is not None
            cached_info, cached_profiles = cached
            self.assertEqual(cached_profiles, (profile,))
            self.assertTrue(
                is_fresh_cached_quick_deploy_catalog(
                    cached_info,
                    now=datetime(2026, 9, 3, 1, tzinfo=timezone.utc),
                )
            )
            self.assertFalse(
                is_fresh_cached_quick_deploy_catalog(
                    cached_info,
                    now=datetime(2026, 9, 4, tzinfo=timezone.utc),
                )
            )
            with patch(
                "llm_launchpad.core.quick_deploy_refresh._quick_deploy_catalog_cache_path",
                return_value=cache_path,
            ):
                self.assertEqual(load_cached_quick_deploy_catalog(), cached)


if __name__ == "__main__":
    unittest.main()

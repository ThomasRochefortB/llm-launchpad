from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core import hf_models
from llm_launchpad.core.hf_models import ModelCandidate


class HFModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        hf_models._CACHE.clear()

    def test_normalize_keeps_text_generation_pipeline(self) -> None:
        row = SimpleNamespace(
            id="Qwen/Qwen2.5-7B-Instruct",
            downloads=123,
            likes=45,
            pipeline_tag="text-generation",
            tags=["transformers"],
        )
        candidate = hf_models._normalize_candidate(row)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.repo_id, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(candidate.downloads, 123)
        self.assertEqual(candidate.likes, 45)

    def test_normalize_keeps_text_generation_tag(self) -> None:
        row = SimpleNamespace(
            modelId="meta-llama/Llama-3.1-8B-Instruct",
            downloads="99",
            likes="7",
            pipeline_tag=None,
            tags=["text-generation", "transformers"],
        )
        candidate = hf_models._normalize_candidate(row)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.repo_id, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(candidate.downloads, 99)
        self.assertEqual(candidate.likes, 7)

    def test_normalize_filters_non_text_generation(self) -> None:
        row = SimpleNamespace(
            id="some/image-model",
            downloads=10,
            likes=1,
            pipeline_tag="image-classification",
            tags=["image-classification"],
        )
        self.assertIsNone(hf_models._normalize_candidate(row))

    def test_list_vllm_candidates_uses_ttl_cache(self) -> None:
        expected = [ModelCandidate(repo_id="Qwen/Qwen2.5-7B-Instruct")]
        with patch("llm_launchpad.core.hf_models._fetch_candidates", return_value=expected) as fetch:
            first = hf_models.list_vllm_candidates(mode="downloads", limit=10)
            second = hf_models.list_vllm_candidates(mode="downloads", limit=10)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(fetch.call_count, 1)

    def test_fetch_candidates_applies_sort_and_limit(self) -> None:
        calls: list[tuple[str, int]] = []
        rows = [
            SimpleNamespace(
                id="internlm/Intern-S1-Pro",
                downloads=10,
                likes=5,
                pipeline_tag="image-text-to-text",
                tags=["image-text-to-text"],
            ),
            SimpleNamespace(
                id="Qwen/Qwen3-Coder-Next",
                downloads=20,
                likes=9,
                pipeline_tag="text-generation",
                tags=["text-generation"],
            ),
            SimpleNamespace(
                id="zai-org/GLM-5",
                downloads=30,
                likes=12,
                pipeline_tag="text-generation",
                tags=["text-generation"],
            ),
        ]

        class FakeApi:
            def list_models(self, *, filter: str, sort: str, limit: int, full: bool):
                self._ = (filter, full)
                calls.append((sort, limit))
                return rows

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            result = hf_models._fetch_candidates(mode="trending", limit=2)
        self.assertEqual(calls, [("trending_score", 6)])
        self.assertEqual([m.repo_id for m in result], ["Qwen/Qwen3-Coder-Next", "zai-org/GLM-5"])


if __name__ == "__main__":
    unittest.main()


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
        hf_models._GGUF_QUANTS_CACHE.clear()

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

    def test_normalize_filters_llamacpp_without_gguf_tag(self) -> None:
        row = SimpleNamespace(
            id="Qwen/Qwen3-4B-Instruct",
            downloads=10,
            likes=1,
            pipeline_tag="text-generation",
            tags=["text-generation"],
        )
        self.assertIsNone(hf_models._normalize_candidate(row, target="llamacpp"))

    def test_list_vllm_candidates_uses_ttl_cache(self) -> None:
        expected = [ModelCandidate(repo_id="Qwen/Qwen2.5-7B-Instruct")]
        with patch("llm_launchpad.core.hf_models._fetch_candidates", return_value=expected) as fetch:
            first = hf_models.list_vllm_candidates(mode="downloads", limit=10)
            second = hf_models.list_vllm_candidates(mode="downloads", limit=10)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(fetch.call_count, 1)

    def test_list_llamacpp_candidates_uses_ttl_cache(self) -> None:
        expected = [ModelCandidate(repo_id="Qwen/Qwen3-Coder-Next-GGUF")]
        with patch("llm_launchpad.core.hf_models._fetch_candidates", return_value=expected) as fetch:
            first = hf_models.list_llamacpp_candidates(mode="trending", limit=10)
            second = hf_models.list_llamacpp_candidates(mode="trending", limit=10)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(fetch.call_count, 1)

    def test_fetch_candidates_applies_sort_and_limit(self) -> None:
        calls: list[tuple[object, str, int]] = []
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
            def list_models(self, *, filter: object, sort: str, limit: int, full: bool):
                self._ = (filter, full)
                calls.append((filter, sort, limit))
                return rows

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            result = hf_models._fetch_candidates(mode="trending", limit=2, target="vllm")
        self.assertEqual(calls, [("text-generation", "trending_score", 6)])
        self.assertEqual([m.repo_id for m in result], ["Qwen/Qwen3-Coder-Next", "zai-org/GLM-5"])

    def test_fetch_llamacpp_candidates_uses_gguf_filter(self) -> None:
        calls: list[tuple[object, str, int]] = []
        rows = [
            SimpleNamespace(
                id="Qwen/Qwen3-Coder-Next-GGUF",
                downloads=20,
                likes=9,
                pipeline_tag="text-generation",
                tags=["text-generation", "gguf"],
                siblings=[
                    SimpleNamespace(rfilename="Qwen3-Coder-Next-Q4_K_M.gguf"),
                    SimpleNamespace(rfilename="Qwen3-Coder-Next-Q8_0.gguf"),
                ],
            ),
            SimpleNamespace(
                id="Qwen/Qwen3-4B",
                downloads=30,
                likes=12,
                pipeline_tag="text-generation",
                tags=["text-generation"],
            ),
        ]

        class FakeApi:
            def list_models(self, *, filter: object, sort: str, limit: int, full: bool):
                self._ = (filter, full)
                calls.append((filter, sort, limit))
                return rows

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            result = hf_models._fetch_candidates(mode="downloads", limit=2, target="llamacpp")
        self.assertEqual(calls, [(["text-generation", "gguf"], "downloads", 6)])
        self.assertEqual([m.repo_id for m in result], ["Qwen/Qwen3-Coder-Next-GGUF"])
        self.assertEqual(result[0].quantizations, ("Q4_K_M", "Q8_0"))

    def test_extract_gguf_quantizations_parses_and_sorts(self) -> None:
        siblings = [
            SimpleNamespace(rfilename="BF16/model-BF16-00001-of-00002.gguf"),
            SimpleNamespace(rfilename="Q5_K_M/model-Q5_K_M.gguf"),
            SimpleNamespace(rfilename="Q4_K_M/model-Q4_K_M.gguf"),
            SimpleNamespace(rfilename="Q8_0/model-Q8_0.gguf"),
            SimpleNamespace(rfilename="README.md"),
        ]
        self.assertEqual(hf_models._extract_gguf_quantizations(siblings), ["Q4_K_M", "Q5_K_M", "Q8_0"])

    def test_fetch_gguf_quantizations_uses_cache(self) -> None:
        calls: list[tuple[str, str | None]] = []

        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, files_metadata: bool):
                self._ = files_metadata
                calls.append((repo_id, revision))
                return SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename="Q4_K_M/model-Q4_K_M.gguf"),
                        SimpleNamespace(rfilename="Q6_K/model-Q6_K.gguf"),
                    ]
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            first = hf_models.fetch_gguf_quantizations("Qwen/Qwen3-Coder-Next-GGUF")
            second = hf_models.fetch_gguf_quantizations("Qwen/Qwen3-Coder-Next-GGUF")
        self.assertEqual(first, ["Q4_K_M", "Q6_K"])
        self.assertEqual(second, ["Q4_K_M", "Q6_K"])
        self.assertEqual(calls, [("Qwen/Qwen3-Coder-Next-GGUF", None)])


if __name__ == "__main__":
    unittest.main()


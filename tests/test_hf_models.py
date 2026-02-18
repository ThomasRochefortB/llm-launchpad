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
        hf_models._GGUF_QUANT_METADATA_CACHE.clear()
        hf_models._VLLM_MEMORY_CACHE.clear()

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
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = expand
                calls.append((repo_id, revision))
                return SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename="Q4_K_M/model-Q4_K_M.gguf"),
                        SimpleNamespace(rfilename="Q6_K/model-Q6_K.gguf"),
                    ],
                    gguf={},
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch("llm_launchpad.core.hf_models._fetch_gguf_quantization_data_from_model_page", return_value=None),
        ):
            first = hf_models.fetch_gguf_quantizations("Qwen/Qwen3-Coder-Next-GGUF")
            second = hf_models.fetch_gguf_quantizations("Qwen/Qwen3-Coder-Next-GGUF")
        self.assertEqual(first, ["Q4_K_M", "Q6_K"])
        self.assertEqual(second, ["Q4_K_M", "Q6_K"])
        self.assertEqual(calls, [("Qwen/Qwen3-Coder-Next-GGUF", None)])

    def test_fetch_gguf_quant_metadata_uses_cache(self) -> None:
        calls: list[tuple[str, str | None]] = []

        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = expand
                calls.append((repo_id, revision))
                return SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename="Q4_K_M/model-Q4_K_M.gguf"),
                        SimpleNamespace(rfilename="Q6_K/model-Q6_K.gguf"),
                    ],
                    gguf={
                        "compatibility": [
                            {"quantization": "Q4_K_M", "memory": "4.66 GB"},
                            {"quantization": "Q6_K", "memory": "7.2 GB"},
                            {"quantization": "Q5_K_M", "memory": "6.0 GB"},
                        ]
                    },
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch("llm_launchpad.core.hf_models._fetch_gguf_quantization_data_from_model_page", return_value=None),
        ):
            first = hf_models.fetch_gguf_quant_metadata("Qwen/Qwen3-Coder-Next-GGUF")
            second = hf_models.fetch_gguf_quant_metadata("Qwen/Qwen3-Coder-Next-GGUF")

        self.assertEqual(first.quantizations, ["Q4_K_M", "Q6_K"])
        self.assertEqual(second.quantizations, ["Q4_K_M", "Q6_K"])
        self.assertAlmostEqual(first.vram_gb_by_quant["Q4_K_M"], 4.66, places=2)
        self.assertAlmostEqual(first.vram_gb_by_quant["Q6_K"], 7.2, places=2)
        self.assertNotIn("Q5_K_M", first.vram_gb_by_quant)
        self.assertEqual(calls, [("Qwen/Qwen3-Coder-Next-GGUF", None)])

    def test_parse_memory_to_gb_supports_gb_tb_and_bytes(self) -> None:
        self.assertAlmostEqual(hf_models._parse_memory_to_gb("4.66 GB") or 0, 4.66, places=2)
        self.assertAlmostEqual(hf_models._parse_memory_to_gb("1.5 TB") or 0, 1500.0, places=1)
        self.assertAlmostEqual(hf_models._parse_memory_to_gb(5 * 1024**3) or 0, 5.369, places=3)

    def test_extract_gguf_vram_by_quant_keeps_max_duplicate(self) -> None:
        payload = {
            "hardware_compatibility": [
                {"quant": "Q4_K_M", "vram": "4.6 GB"},
                {"quant": "Q4_K_M", "vram": "4.9 GB"},
                {"quant": "Q6_K", "vram": "7.3 GB"},
            ]
        }
        parsed = hf_models._extract_gguf_vram_by_quant(payload)
        self.assertAlmostEqual(parsed["Q4_K_M"], 4.9, places=1)
        self.assertAlmostEqual(parsed["Q6_K"], 7.3, places=1)

    def test_extract_quantization_data_from_model_page_html(self) -> None:
        html = (
            '<div data-target="ModelTensorsParams" '
            'data-props="{&quot;ggufQuantizationData&quot;:{&quot;variantsByQuantizationLevels&quot;:'
            '{&quot;4&quot;:[{&quot;label&quot;:&quot;Q4_1&quot;,&quot;size&quot;:472704107744}]}}}"></div>'
        )
        parsed = hf_models._extract_gguf_quantization_data_from_model_page_html(html)
        assert isinstance(parsed, dict)
        levels = parsed.get("variantsByQuantizationLevels")
        assert isinstance(levels, dict)
        row = levels.get("4", [])[0]
        self.assertEqual(row["label"], "Q4_1")
        self.assertEqual(row["size"], 472704107744)

    def test_extract_quantizations_and_vram_from_quantization_data_keeps_hf_labels(self) -> None:
        quant_data = {
            "variantsByQuantizationLevels": {
                "1": [{"label": "UD-IQ1_S", "size": 203699011168}],
                "4": [
                    {"label": "MXFP4_MOE", "size": 410557924192},
                    {"label": "Q4_1", "size": 472704107744},
                ],
                "16": [{"label": "BF16", "size": 1510000000000}],
            }
        }
        quantizations, vram = hf_models._extract_quantizations_and_vram_from_quantization_data(quant_data)
        self.assertEqual(quantizations, ["UD-IQ1_S", "MXFP4_MOE", "Q4_1", "BF16"])
        self.assertAlmostEqual(vram["UD-IQ1_S"], 203.699, places=2)
        self.assertAlmostEqual(vram["MXFP4_MOE"], 410.558, places=2)
        self.assertAlmostEqual(vram["Q4_1"], 472.704, places=2)
        self.assertAlmostEqual(vram["BF16"], 1510.0, places=1)

    def test_fetch_quant_metadata_falls_back_to_model_page_quantization_data(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename="Q4_1/model-Q4_1.gguf"),
                        SimpleNamespace(rfilename="Q8_0/model-Q8_0.gguf"),
                    ],
                    gguf={},
                )

        class FakeResponse:
            status_code = 200
            text = (
                '<div data-target="ModelTensorsParams" '
                'data-props="{&quot;ggufQuantizationData&quot;:{&quot;variantsByQuantizationLevels&quot;:'
                "{&quot;4&quot;:[{&quot;label&quot;:&quot;Q4_1&quot;,&quot;size&quot;:472704107744}],"
                '&quot;8&quot;:[{&quot;label&quot;:&quot;Q8_0&quot;,&quot;size&quot;:801344964608}]'
                "}}}\"></div>"
            )

        def fake_get(url: str, timeout: float, headers: dict[str, str]):
            self.assertIn("huggingface.co", url)
            self.assertIn("User-Agent", headers)
            self.assertEqual(timeout, 10.0)
            return FakeResponse()

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        fake_requests = types.SimpleNamespace(get=fake_get)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module, "requests": fake_requests}):
            metadata = hf_models.fetch_gguf_quant_metadata("unsloth/GLM-5-GGUF")
        self.assertCountEqual(metadata.quantizations, ["Q4_1", "Q8_0"])
        self.assertAlmostEqual(metadata.vram_gb_by_quant["Q4_1"], 472.704, places=2)
        self.assertAlmostEqual(metadata.vram_gb_by_quant["Q8_0"], 801.345, places=2)

    def test_fetch_vllm_memory_breakdown_uses_hf_config(self) -> None:
        calls: list[tuple[str, str | None]] = []

        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = expand
                calls.append((repo_id, revision))
                return SimpleNamespace(
                    config={
                        "num_hidden_layers": 32,
                        "hidden_size": 4096,
                        "max_position_embeddings": 32768,
                    },
                    safetensors={"total": 16_000_000_000},
                    tags=["text-generation", "bf16"],
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            estimate = hf_models.fetch_vllm_memory_breakdown("Qwen/Qwen3-8B-Instruct")
            estimate_cached = hf_models.fetch_vllm_memory_breakdown("Qwen/Qwen3-8B-Instruct")

        assert estimate is not None
        assert estimate_cached is not None
        self.assertEqual(calls, [("Qwen/Qwen3-8B-Instruct", None)])
        self.assertGreater(estimate.total_gb, estimate.weights_gb)
        self.assertEqual(estimate.context_tokens, 8192)

    def test_fetch_vllm_memory_breakdown_falls_back_to_repo_name(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(config={}, safetensors={}, tags=["text-generation", "fp8"])

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            estimate = hf_models.fetch_vllm_memory_breakdown("Qwen/Qwen3-8B-FP8")

        assert estimate is not None
        self.assertGreater(estimate.weights_gb, 0.0)
        self.assertGreater(estimate.total_gb, estimate.weights_gb)


if __name__ == "__main__":
    unittest.main()

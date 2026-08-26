from __future__ import annotations

import types
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core import hf_models
from llm_launchpad.core.hf_models import ModelCandidate


class HFModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        hf_models._CACHE.clear()
        hf_models._GGUF_QUANT_METADATA_CACHE.clear()
        hf_models._VLLM_MEMORY_CACHE.clear()
        hf_models._HF_JSON_FILE_CACHE.clear()

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

    def test_extract_hardware_compatibility_vram_from_model_page_html(self) -> None:
        html = """
        <section>
          <h3>Hardware compatibility</h3>
          <p>Log In to add your hardware</p>
          <p>4-bit</p>
          <p>UD-Q4_K_M</p>
          <p>140 GB UD_Q4_K_XL</p>
          <p>141 GB</p>
          <p>8-bit</p>
          <p>Q8_0</p>
          <p>243 GB</p>
          <h2>Inference Providers</h2>
        </section>
        """

        parsed = hf_models._extract_hardware_compatibility_quantization_data_from_model_page_html(html)

        assert isinstance(parsed, dict)
        quantizations, vram = hf_models._extract_quantizations_and_vram_from_quantization_data(parsed)
        self.assertEqual(quantizations, ["UD-Q4_K_M", "UD-Q4_K_XL", "Q8_0"])
        self.assertAlmostEqual(vram["UD-Q4_K_M"], 140.0, places=1)
        self.assertAlmostEqual(vram["UD-Q4_K_XL"], 141.0, places=1)
        self.assertAlmostEqual(vram["Q8_0"], 243.0, places=1)

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

    def test_quantization_page_treats_sub_gigabyte_numeric_size_as_bytes(self) -> None:
        quantizations, vram = (
            hf_models._extract_quantizations_and_vram_from_quantization_data(
                {
                    "variantsByQuantizationLevels": {
                        "4": [{"label": "Q4_K_M", "size": 398_000_000}],
                    }
                }
            )
        )

        self.assertEqual(quantizations, ["Q4_K_M"])
        self.assertAlmostEqual(vram["Q4_K_M"], 0.398, places=3)

    def test_fetch_quant_metadata_prefers_hardware_compatibility_rows(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(
                    siblings=[SimpleNamespace(rfilename="UD-Q4_K_XL/model-UD-Q4_K_XL.gguf")],
                    gguf={},
                )

        class FakeResponse:
            status_code = 200
            text = (
                '<div data-target="ModelTensorsParams" '
                'data-props="{&quot;ggufQuantizationData&quot;:{&quot;variantsByQuantizationLevels&quot;:'
                "{&quot;4&quot;:[{&quot;label&quot;:&quot;UD-Q4_K_XL&quot;,&quot;size&quot;:90000000000}]"
                "}}}\"></div>"
                "<h3>Hardware compatibility</h3>"
                "<p>4-bit</p><p>UD-Q4_K_XL</p><p>141 GB</p>"
                "<h2>Inference Providers</h2>"
            )

        def fake_get(url: str, timeout: float, headers: dict[str, str]):
            self.assertIn("huggingface.co", url)
            self.assertIn("User-Agent", headers)
            self.assertEqual(timeout, 10.0)
            return FakeResponse()

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        fake_requests = types.SimpleNamespace(get=fake_get)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module, "requests": fake_requests}):
            metadata = hf_models.fetch_gguf_quant_metadata("unsloth/MiniMax-M2.7-GGUF")

        self.assertEqual(metadata.quantizations, ["UD-Q4_K_XL"])
        self.assertAlmostEqual(metadata.vram_gb_by_quant["UD-Q4_K_XL"], 141.0, places=1)

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
        self.assertEqual(estimate.context_tokens, 32768)

    def test_fetch_vllm_memory_breakdown_uses_timeout_and_card_data_expand(self) -> None:
        calls: list[tuple[float, tuple[str, ...]]] = []

        class FakeApi:
            def model_info(
                self,
                *,
                repo_id: str,
                revision: str | None,
                timeout: float,
                expand: list[str] | None = None,
            ):
                self._ = (repo_id, revision)
                calls.append((timeout, tuple(expand or [])))
                return SimpleNamespace(
                    config={
                        "num_hidden_layers": 32,
                        "hidden_size": 4096,
                        "max_position_embeddings": 32768,
                    },
                    safetensors={"total": 16_000_000_000},
                    tags=["text-generation", "bf16"],
                    cardData={"context_length": 32768},
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            estimate = hf_models.fetch_vllm_memory_breakdown("Qwen/Qwen3-8B-Instruct")
        assert estimate is not None
        self.assertEqual(calls[0][0], hf_models._HF_REQUEST_TIMEOUT_SECONDS)
        self.assertIn("cardData", calls[0][1])

    def test_fetch_vllm_memory_breakdown_skips_repo_json_fetch_when_model_info_is_complete(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(
                    config={
                        "num_hidden_layers": 32,
                        "hidden_size": 4096,
                        "max_position_embeddings": 32768,
                    },
                    safetensors={"total": 16_000_000_000},
                    tags=["text-generation", "bf16"],
                    cardData={"context_length": 32768},
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch(
                "llm_launchpad.core.hf_models._load_repo_json_file",
                side_effect=AssertionError("should not fetch repo JSON"),
            ),
        ):
            estimate = hf_models.fetch_vllm_memory_breakdown("Qwen/Qwen3-8B-Instruct")
        assert estimate is not None
        self.assertEqual(estimate.context_tokens, 32768)

    def test_fetch_vllm_memory_breakdown_uses_default_context_without_metadata(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand: list[str]):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(config={}, safetensors={}, tags=["text-generation", "fp8"])

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch("llm_launchpad.core.hf_models._load_repo_json_file", return_value=None),
        ):
            estimate = hf_models.fetch_vllm_memory_breakdown("Qwen/Qwen3-8B-FP8")
        assert estimate is not None
        self.assertEqual(estimate.context_tokens, hf_models._DEFAULT_CONTEXT_TOKENS)

    def test_fetch_vllm_memory_breakdown_retries_without_expand(self) -> None:
        class FakeApi:
            def __init__(self) -> None:
                self.calls = 0

            def model_info(self, *, repo_id: str, revision: str | None, expand=None):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("expand not supported")
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(
                    config={"max_position_embeddings": 131072, "num_hidden_layers": 32, "hidden_size": 4096},
                    safetensors={"total": 16_000_000_000},
                    tags=["bf16"],
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            estimate = hf_models.fetch_vllm_memory_breakdown("Qwen/Qwen3-8B-Instruct")
        assert estimate is not None
        self.assertEqual(estimate.context_tokens, 131072)

    def test_extract_model_max_context_reads_nested_config(self) -> None:
        config = {
            "text_config": {"max_position_embeddings": 32768},
            "rope_scaling": {"original_max_position_embeddings": 65536},
            "tokenizer_config": {"model_max_length": 100000000000000000000},
        }
        value = hf_models._extract_model_max_context(config)
        self.assertEqual(value, 32768)

    def test_extract_model_max_context_handles_rope_scaling_factor(self) -> None:
        config = {
            "max_position_embeddings": 32768,
            "rope_scaling": {"factor": 8.0, "original_max_position_embeddings": 32768},
        }
        value = hf_models._extract_model_max_context(config)
        self.assertEqual(value, 262144)

    def test_extract_model_max_context_prefers_original_rope_base(self) -> None:
        config = {
            "max_position_embeddings": 131072,
            "rope_scaling": {"factor": 32.0, "original_max_position_embeddings": 4096},
        }
        value = hf_models._extract_model_max_context(config)
        self.assertEqual(value, 131072)

    def test_extract_model_max_context_parses_k_suffix(self) -> None:
        config = {"context_length": "200K"}
        value = hf_models._extract_model_max_context(config)
        self.assertEqual(value, 200000)

    def test_load_repo_json_file_falls_back_to_http_when_hf_hub_download_fails(self) -> None:
        def fake_hf_hub_download(*, repo_id: str, filename: str, revision: str | None, etag_timeout: float):
            self._ = (repo_id, filename, revision, etag_timeout)
            raise RuntimeError("network error")

        fake_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch(
                "llm_launchpad.core.hf_models._fetch_repo_json_file_via_http",
                return_value={"max_position_embeddings": 262144},
            ) as fetch_http,
        ):
            parsed = hf_models._load_repo_json_file("Qwen/Qwen3-Coder-Next", None, "config.json")
        assert parsed is not None
        self.assertEqual(parsed.get("max_position_embeddings"), 262144)
        self.assertEqual(fetch_http.call_count, 1)

    def test_load_repo_json_file_falls_back_to_http_when_downloaded_json_is_invalid(self) -> None:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write("{not-json")
            temp_path = Path(handle.name)

        def fake_hf_hub_download(*, repo_id: str, filename: str, revision: str | None, etag_timeout: float):
            self._ = (repo_id, filename, revision, etag_timeout)
            return str(temp_path)

        fake_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch(
                "llm_launchpad.core.hf_models._fetch_repo_json_file_via_http",
                return_value={"max_position_embeddings": 262144},
            ) as fetch_http,
        ):
            parsed = hf_models._load_repo_json_file("Qwen/Qwen3-Coder-Next", None, "config.json")
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        assert parsed is not None
        self.assertEqual(parsed.get("max_position_embeddings"), 262144)
        self.assertEqual(fetch_http.call_count, 1)

    def test_fetch_vllm_memory_breakdown_uses_repo_json_context_when_model_info_missing(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand=None):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(config={}, safetensors={"total": 16_000_000_000}, tags=["bf16"])

        def fake_load(repo_id: str, revision: str | None, filename: str):
            self.assertEqual(repo_id, "zai-org/GLM-5")
            if filename == "config.json":
                return {
                    "num_hidden_layers": 64,
                    "hidden_size": 7168,
                    "max_position_embeddings": 32768,
                    "rope_scaling": {"factor": 8.0, "original_max_position_embeddings": 32768},
                }
            if filename == "tokenizer_config.json":
                return {"model_max_length": 262144}
            return None

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch("llm_launchpad.core.hf_models._load_repo_json_file", side_effect=fake_load),
        ):
            estimate = hf_models.fetch_vllm_memory_breakdown("zai-org/GLM-5")

        assert estimate is not None
        self.assertEqual(estimate.context_tokens, 262144)

    def test_fetch_vllm_memory_breakdown_uses_card_data_context(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand=None):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(
                    config={},
                    cardData={"context_length": "256K"},
                    safetensors={"total": 16_000_000_000},
                    tags=["bf16"],
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch("llm_launchpad.core.hf_models._load_repo_json_file", return_value=None),
        ):
            estimate = hf_models.fetch_vllm_memory_breakdown("zai-org/GLM-5")
        assert estimate is not None
        self.assertEqual(estimate.context_tokens, 256000)

    def test_fetch_model_max_context_uses_model_info_metadata(self) -> None:
        calls: list[tuple[str, str | None, tuple[str, ...]]] = []

        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, timeout: float, expand=None):
                self._ = timeout
                calls.append((repo_id, revision, tuple(expand or [])))
                return SimpleNamespace(
                    config={"max_position_embeddings": 32768},
                    cardData={"context_window": "256K"},
                )

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            context = hf_models.fetch_model_max_context("unsloth/Test-GGUF")

        self.assertEqual(context, 256000)
        self.assertEqual(calls[0][0], "unsloth/Test-GGUF")
        self.assertIn("cardData", calls[0][2])

    def test_fetch_model_max_context_falls_back_to_repo_json(self) -> None:
        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand=None):
                self._ = (repo_id, revision, expand)
                return SimpleNamespace(config={}, cardData={})

        def fake_load(repo_id: str, revision: str | None, filename: str):
            self.assertEqual(repo_id, "unsloth/Test-GGUF")
            if filename == "config.json":
                return {"max_context_tokens": "64K"}
            if filename == "tokenizer_config.json":
                return {"model_max_length": 32768}
            return None

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch("llm_launchpad.core.hf_models._load_repo_json_file", side_effect=fake_load),
        ):
            context = hf_models.fetch_model_max_context("unsloth/Test-GGUF")

        self.assertEqual(context, 64000)

    def test_fetch_model_max_context_follows_base_model_tag(self) -> None:
        calls: list[str] = []

        class FakeApi:
            def model_info(self, *, repo_id: str, revision: str | None, expand=None):
                self._ = (revision, expand)
                calls.append(repo_id)
                if repo_id == "unsloth/Test-GGUF":
                    return SimpleNamespace(
                        config={},
                        cardData={},
                        tags=["base_model:Owner/Base-Model"],
                    )
                if repo_id == "Owner/Base-Model":
                    return SimpleNamespace(
                        config={"max_position_embeddings": 196608},
                        cardData={},
                        tags=[],
                    )
                return SimpleNamespace(config={}, cardData={}, tags=[])

        fake_module = types.SimpleNamespace(HfApi=FakeApi)
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_module}),
            patch("llm_launchpad.core.hf_models._load_repo_json_file", return_value=None),
        ):
            context = hf_models.fetch_model_max_context("unsloth/Test-GGUF")

        self.assertEqual(context, 196608)
        self.assertEqual(calls, ["unsloth/Test-GGUF", "Owner/Base-Model"])


if __name__ == "__main__":
    unittest.main()

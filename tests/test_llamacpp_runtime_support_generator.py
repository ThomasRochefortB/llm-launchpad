from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


def _load_generator_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "refresh_llamacpp_runtime_support.py"
    )
    spec = importlib.util.spec_from_file_location(
        "refresh_llamacpp_runtime_support",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LlamaCppRuntimeSupportGeneratorTests(unittest.TestCase):
    def test_manifest_records_generated_mtp_architectures(self) -> None:
        generator = _load_generator_module()
        architecture_source = """
        { LLM_ARCH_QWEN35, "qwen35" },
        { LLM_ARCH_LLAMA, "llama" },
        """
        model_source = """
        llm_build_qwen35::llm_build_qwen35() {}
        LLM_KV_NEXTN_PREDICT_LAYERS
        LLM_GRAPH_TYPE_DECODER_MTP
        """

        payload = generator.build_manifest(
            architecture_source,
            model_sources=(model_source,),
            source_revision="abc",
            image_ref="example/runtime:server-cuda-b1",
            image_digest="sha256:test",
            runtime_build="b1",
            generated_at="2026-09-02T00:00:00Z",
        )

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["mtp_architectures"], ["qwen35"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from llm_launchpad.core.naming import (
    auto_instance_name_for_backend,
    build_app_name,
    default_served_model_name,
    infer_backend_from_app_name,
    infer_instance_from_app_name,
    legacy_app_name,
    modal_function_name,
    slugify_instance_name,
)
from llm_launchpad.protocol.enums import BackendType


class NamingTests(unittest.TestCase):
    def test_slugify_instance_name_normalizes_model_ids(self) -> None:
        slug = slugify_instance_name("Qwen/Qwen2.5-Coder_7B-Instruct")
        self.assertEqual(slug, "qwen-qwen2-5-coder-7b-instruct")

    def test_auto_instance_name_uses_model_hint(self) -> None:
        name = auto_instance_name_for_backend(BackendType.VLLM, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(name, "meta-llama-llama-3-1-8b-instruct")

    def test_default_served_model_name_uses_model_id_suffix(self) -> None:
        alias = default_served_model_name("Qwen/Qwen3-0.6B")
        self.assertEqual(alias, "Qwen3-0.6B")

    def test_default_served_model_name_falls_back_to_default(self) -> None:
        self.assertEqual(default_served_model_name(""), "llm")
        self.assertEqual(default_served_model_name(None), "llm")

    def test_build_app_name(self) -> None:
        self.assertEqual(build_app_name(BackendType.VLLM, "qwen3"), "vllm-qwen3")
        self.assertEqual(build_app_name(BackendType.LLAMACPP, "qwen25"), "llamacpp-qwen25")

    def test_infer_backend_and_instance_from_app_name(self) -> None:
        app_name = "vllm-qwen3-coder"
        backend = infer_backend_from_app_name(app_name)
        self.assertEqual(backend, BackendType.VLLM)
        self.assertEqual(infer_instance_from_app_name(app_name, backend), "qwen3-coder")

    def test_legacy_app_name_roundtrip(self) -> None:
        app_name = legacy_app_name(BackendType.LLAMACPP)
        backend = infer_backend_from_app_name(app_name)
        self.assertEqual(backend, BackendType.LLAMACPP)
        self.assertEqual(infer_instance_from_app_name(app_name, backend), "default")

    def test_modal_function_name_appends_slug(self) -> None:
        self.assertEqual(modal_function_name("serve", "alpha-bravo"), "serve-alpha-bravo")
        self.assertEqual(modal_function_name("serve", None), "serve")


if __name__ == "__main__":
    unittest.main()

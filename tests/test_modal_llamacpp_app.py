from __future__ import annotations

import unittest
import os
from unittest.mock import patch


class ModalLlamaCppAppTests(unittest.TestCase):
    def test_glm_runtime_builds_bundled_recipe_instead_of_pulling_local_tag(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app as backend

        with (
            patch.dict(os.environ, {"LLAMA_CPP_BUILD_RECIPE": "llamacpp_glm5next.dockerfile", "LLAMA_CPP_CUDA_ARCHITECTURES": "100"}),
            patch.object(backend.modal.Image, "from_dockerfile") as build,
            patch.object(backend.modal.Image, "from_registry") as pull,
        ):
            backend._serving_image()
        pull.assert_not_called()
        path = build.call_args.args[0]
        self.assertTrue(path.is_file())
        self.assertIn("NVIDIA_TF32_OVERRIDE=0", path.read_text())
        self.assertEqual(build.call_args.kwargs["add_python"], "3.12")
        self.assertEqual(build.call_args.kwargs["build_args"], {"CUDA_ARCHITECTURES": "100"})
        build.return_value.entrypoint.assert_called_once_with([])

    def _serve_command(self, n_gpu_layers: int | None) -> list[str]:
        from llm_launchpad.backends import modal_llamacpp_app

        config = {
            "repo_id": "unsloth/Test-GGUF",
            "quant": "Q4_K_M",
            "revision": None,
            "served_model_name": "test-model",
            "server_args": ["--ctx-size", "131072"],
            "host": "0.0.0.0",
            "port": 8080,
            "n_gpu_layers": n_gpu_layers,
        }
        with (
            patch.object(modal_llamacpp_app, "_load_config", return_value=config),
            patch.object(
                modal_llamacpp_app,
                "_resolve_or_download_model_entrypoint",
                return_value="/models/test.gguf",
            ),
            patch.object(
                modal_llamacpp_app,
                "_resolve_llama_server_binary",
                return_value="llama-server",
            ),
            patch.object(
                modal_llamacpp_app,
                "_llama_server_runtime_env",
                return_value=({}, None),
            ),
            patch.object(modal_llamacpp_app.subprocess, "Popen") as popen,
        ):
            modal_llamacpp_app.serve.local()

        return list(popen.call_args.args[0])

    def test_serve_leaves_gpu_layers_unset_for_llamacpp_auto_fit(self) -> None:
        command = self._serve_command(n_gpu_layers=None)

        self.assertNotIn("--n-gpu-layers", command)
        self.assertEqual(command[-2:], ["--ctx-size", "131072"])

    def test_serve_preserves_explicit_gpu_layer_override(self) -> None:
        command = self._serve_command(n_gpu_layers=42)

        index = command.index("--n-gpu-layers")
        self.assertEqual(command[index + 1], "42")


if __name__ == "__main__":
    unittest.main()

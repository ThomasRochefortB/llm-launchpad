from __future__ import annotations

from collections.abc import Sequence
import io
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


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
            patch.object(modal_llamacpp_app, "_warm_volume_paths"),
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

    def test_gguf_weight_paths_returns_entrypoint_for_unsplit_file(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model-Q4_K_M.gguf"
            path.write_bytes(b"gguf")
            self.assertEqual(modal_llamacpp_app._gguf_weight_paths(path), [path])

    def test_gguf_weight_paths_collects_split_shards_in_index_order(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            shards = []
            for idx in (2, 1, 3):
                shard = parent / f"GLM-UD-Q2_K_XL-{idx:05d}-of-00003.gguf"
                shard.write_bytes(bytes([idx]) * 4)
                shards.append(shard)
            other = parent / "GLM-UD-Q3_K_XL-00001-of-00003.gguf"
            other.write_bytes(b"skip")
            found = modal_llamacpp_app._gguf_weight_paths(shards[0])
            self.assertEqual(
                [path.name for path in found],
                [
                    "GLM-UD-Q2_K_XL-00001-of-00003.gguf",
                    "GLM-UD-Q2_K_XL-00002-of-00003.gguf",
                    "GLM-UD-Q2_K_XL-00003-of-00003.gguf",
                ],
            )

    def test_gguf_weight_paths_keeps_entrypoint_when_parent_is_missing(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        path = Path("/models/missing-00001-of-00002.gguf")
        self.assertEqual(modal_llamacpp_app._gguf_weight_paths(path), [path])

    def test_warm_volume_paths_reads_each_file_and_skips_missing(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.gguf"
            second = Path(tmp) / "b.gguf"
            first.write_bytes(b"alpha-data")
            second.write_bytes(b"beta-payload")
            missing = Path(tmp) / "gone.gguf"
            logs = io.StringIO()
            with patch("sys.stdout", logs):
                modal_llamacpp_app._warm_volume_paths(
                    [first, second, missing, first],
                    chunk_bytes=4,
                )
            output = logs.getvalue()
            self.assertIn("warming 2 volume file(s)", output)
            self.assertIn("volume warm complete", output)
            self.assertIn("skip volume warm; missing", output)

    def test_serve_warms_weight_files_before_starting_llama_server(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        config = {
            "repo_id": "unsloth/Test-GGUF",
            "quant": "Q4_K_M",
            "revision": None,
            "served_model_name": "test-model",
            "server_args": ["--ctx-size", "131072"],
            "host": "0.0.0.0",
            "port": 8080,
            "n_gpu_layers": None,
        }
        order: list[str] = []
        model_path = Path("/models/test-00001-of-00002.gguf")
        shards = [
            Path("/models/test-00001-of-00002.gguf"),
            Path("/models/test-00002-of-00002.gguf"),
        ]

        def _record_warm(paths: Sequence[Path | str], **_kwargs: object) -> None:
            order.append("warm")
            self.assertEqual(list(paths), shards)

        def _record_popen(*_args: object, **_kwargs: object) -> object:
            order.append("popen")
            return MagicMock()

        with (
            patch.object(modal_llamacpp_app, "_load_config", return_value=config),
            patch.object(
                modal_llamacpp_app,
                "_resolve_or_download_model_entrypoint",
                return_value=model_path,
            ),
            patch.object(modal_llamacpp_app, "_gguf_weight_paths", return_value=shards),
            patch.object(modal_llamacpp_app, "WARM_VOLUME", True),
            patch.object(modal_llamacpp_app, "_warm_volume_paths", side_effect=_record_warm),
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
            patch.object(modal_llamacpp_app.subprocess, "Popen", side_effect=_record_popen),
        ):
            modal_llamacpp_app.serve.local()

        self.assertEqual(order, ["warm", "popen"])

    def test_serve_warms_projector_with_model_weights(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        config = {
            "repo_id": "unsloth/Test-GGUF",
            "quant": "Q4_K_M",
            "revision": None,
            "served_model_name": "test-model",
            "server_args": ["--ctx-size", "131072"],
            "host": "0.0.0.0",
            "port": 8080,
            "n_gpu_layers": None,
            "vision": {"enabled": True, "projector": {"repo_id": "x", "revision": "r", "filename": "mmproj.gguf"}},
        }
        warmed: list[list[object]] = []

        with (
            patch.object(modal_llamacpp_app, "_load_config", return_value=config),
            patch.object(
                modal_llamacpp_app,
                "_resolve_or_download_model_entrypoint",
                return_value=Path("/models/test.gguf"),
            ),
            patch.object(
                modal_llamacpp_app,
                "_gguf_weight_paths",
                return_value=[Path("/models/test.gguf")],
            ),
            patch.object(modal_llamacpp_app, "WARM_VOLUME", True),
            patch.object(modal_llamacpp_app, "_warm_volume_paths", side_effect=lambda paths: warmed.append(list(paths))),
            patch.object(modal_llamacpp_app.download_projector, "remote", return_value="/models/mmproj.gguf"),
            patch.object(modal_llamacpp_app.model_cache, "reload"),
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

        self.assertEqual(
            warmed,
            [[Path("/models/test.gguf"), "/models/mmproj.gguf"]],
        )
        command = list(popen.call_args.args[0])
        self.assertIn("--mmproj", command)
        self.assertEqual(command[command.index("--mmproj") + 1], "/models/mmproj.gguf")

    def test_serve_skips_volume_warm_when_disabled(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        config = {
            "repo_id": "unsloth/Test-GGUF",
            "quant": "Q4_K_M",
            "revision": None,
            "served_model_name": "test-model",
            "server_args": ["--ctx-size", "131072"],
            "host": "0.0.0.0",
            "port": 8080,
            "n_gpu_layers": None,
        }
        with (
            patch.object(modal_llamacpp_app, "_load_config", return_value=config),
            patch.object(
                modal_llamacpp_app,
                "_resolve_or_download_model_entrypoint",
                return_value="/models/test.gguf",
            ),
            patch.object(modal_llamacpp_app, "WARM_VOLUME", False),
            patch.object(modal_llamacpp_app, "_warm_volume_paths") as warm,
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
            patch.object(modal_llamacpp_app.subprocess, "Popen"),
        ):
            modal_llamacpp_app.serve.local()

        warm.assert_not_called()

    def test_serve_startup_timeout_covers_large_cold_volume_read(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        self.assertGreaterEqual(modal_llamacpp_app.SERVE_STARTUP_TIMEOUT_MINUTES, 90)
        self.assertGreaterEqual(modal_llamacpp_app.SERVE_TIMEOUT_MINUTES, 60)


if __name__ == "__main__":
    unittest.main()

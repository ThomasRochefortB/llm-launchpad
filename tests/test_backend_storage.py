from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.protocol.events import OperationCompleteEvent


class BackendStorageTests(unittest.TestCase):
    def test_llamacpp_and_vllm_share_hf_cache_storage(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app, modal_vllm_app

        self.assertEqual(modal_llamacpp_app.HF_CACHE_VOLUME_NAME, "huggingface-cache")
        self.assertEqual(modal_llamacpp_app.HF_CACHE_DIR, modal_vllm_app.HF_CACHE_DIR)
        self.assertEqual(str(modal_llamacpp_app.HF_HUB_DIR), str(modal_vllm_app.HF_HUB_DIR))

    def test_build_modal_entrypoint_command(self) -> None:
        cmd = ModalBackend.build_modal_entrypoint_command(
            "llm_launchpad/backends/modal_vllm_app.py",
            "predownload_model",
            args=["--repo-id", "Qwen/Qwen3-4B"],
        )
        self.assertEqual(
            cmd,
            [
                "modal",
                "run",
                "llm_launchpad/backends/modal_vllm_app.py::predownload_model",
                "--repo-id",
                "Qwen/Qwen3-4B",
            ],
        )

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_list_volume_parses_list_payload(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '[{"path":"/models","type":"directory"}]'
        rows = ModalBackend.list_volume("llamacpp-cache", "/")
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(rows[0]["path"], "/models")

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_list_volume_parses_wrapped_entries(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"entries":[{"path":"/hub/models--foo--bar","type":"directory"}]}'
        rows = ModalBackend.list_volume("huggingface-cache", "/hub")
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(rows[0]["path"], "/hub/models--foo--bar")

    @patch("llm_launchpad.core.backend.ModalBackend.run_streaming")
    def test_run_modal_script_entrypoint_delegates_streaming(self, mock_stream) -> None:  # type: ignore[no-untyped-def]
        mock_stream.return_value = iter(
            [OperationCompleteEvent(success=True, exit_code=0)]  # type: ignore[call-arg]
        )
        events = list(
            ModalBackend.run_modal_script_entrypoint(
                "llm_launchpad/backends/modal_llamacpp_app.py",
                "predownload_model",
                args=["--repo-id", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"],
            )
        )
        self.assertEqual(len(events), 1)
        called_cmd = mock_stream.call_args.args[0]
        self.assertIn(
            "llm_launchpad/backends/modal_llamacpp_app.py::predownload_model",
            called_cmd,
        )

    @patch("llm_launchpad.core.backend.ModalBackend.run_streaming")
    def test_run_volume_remove_builds_expected_command(self, mock_stream) -> None:  # type: ignore[no-untyped-def]
        mock_stream.return_value = iter([])
        list(ModalBackend.run_volume_remove("huggingface-cache", "/hub/models--Qwen--Qwen3-4B", recursive=True))
        called_cmd = mock_stream.call_args.args[0]
        self.assertEqual(
            called_cmd,
            [
                "modal",
                "volume",
                "rm",
                "huggingface-cache",
                "/hub/models--Qwen--Qwen3-4B",
                "--recursive",
            ],
        )


if __name__ == "__main__":
    unittest.main()

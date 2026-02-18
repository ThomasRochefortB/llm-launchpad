from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend, _extract_modal_app_rows
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import DeploymentConfig


class BackendParsingAndCommandTests(unittest.TestCase):
    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_list_apps_returns_empty_list_for_empty_json_payload(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "[]"
        self.assertEqual(ModalBackend.list_apps(), [])

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_list_apps_returns_none_on_timeout(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["modal", "app", "list"], timeout=8)
        self.assertIsNone(ModalBackend.list_apps())

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_list_volume_returns_none_on_timeout(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["modal", "volume", "ls"], timeout=8)
        self.assertIsNone(ModalBackend.list_volume("huggingface-cache", "/hub"))

    def test_extract_modal_app_rows_accepts_apps_and_data_wrappers(self) -> None:
        wrapped_apps = {
            "apps": [
                {"app_name": "vllm-qwen", "id": "ap-1", "status": "running"},
            ]
        }
        wrapped_data = {
            "data": [
                {"label": "llamacpp-phi", "appId": "ap-2", "phase": "deployed"},
            ]
        }

        rows_apps = _extract_modal_app_rows(wrapped_apps)
        rows_data = _extract_modal_app_rows(wrapped_data)

        self.assertEqual(rows_apps[0].name, "vllm-qwen")
        self.assertEqual(rows_apps[0].app_id, "ap-1")
        self.assertEqual(rows_apps[0].state, "running")
        self.assertEqual(rows_apps[0].backend, BackendType.VLLM)
        self.assertEqual(rows_apps[0].instance_name, "qwen")

        self.assertEqual(rows_data[0].name, "llamacpp-phi")
        self.assertEqual(rows_data[0].app_id, "ap-2")
        self.assertEqual(rows_data[0].state, "deployed")
        self.assertEqual(rows_data[0].backend, BackendType.LLAMACPP)
        self.assertEqual(rows_data[0].instance_name, "phi")

    def test_extract_modal_app_rows_skips_non_dict_and_non_list_payload(self) -> None:
        self.assertEqual(_extract_modal_app_rows("not-a-list"), [])
        rows = _extract_modal_app_rows([None, 1, {"name": "vllm-qwen", "state": "running"}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "vllm-qwen")

    def test_build_run_command_llamacpp_includes_full_option_set(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            preset="qwen3-coder-30b",
            repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF",
            quant="Q4_K_M",
            revision="main",
            preload=False,
            do_deploy=True,
            server_args="--ctx-size 65536",
            host="0.0.0.0",
            port=8080,
            n_gpu_layers=100,
        )
        cmd = ModalBackend.build_run_command(config)
        self.assertIn("--preset", cmd)
        self.assertIn("--repo-id", cmd)
        self.assertIn("--quant", cmd)
        self.assertIn("--revision", cmd)
        self.assertIn("--no-preload", cmd)
        self.assertIn("--deploy", cmd)
        self.assertIn("--server_args", cmd)
        self.assertIn("--host", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("--n_gpu_layers", cmd)

    def test_env_for_backend_non_vllm_only_sets_app_name(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-prod",
            model_name="should-not-be-used",
            served_model_name="ignored",
        )
        env = ModalBackend.env_for_backend(config)
        self.assertEqual(env, {"MODAL_APP_NAME": "llamacpp-prod"})

    def test_test_curl_command_vllm_prefers_provided_served_model_name(self) -> None:
        cmd = ModalBackend.test_curl_command(
            BackendType.VLLM,
            "https://example.modal.run",
            served_model_name="Qwen3-0.6B",
        )
        self.assertIn('"model":"Qwen3-0.6B"', cmd)

    def test_test_curl_command_llamacpp_is_copy_paste_ready(self) -> None:
        cmd = ModalBackend.test_curl_command(
            BackendType.LLAMACPP,
            "https://example.modal.run/",
        )
        self.assertTrue(
            cmd.startswith("curl -s -X POST https://example.modal.run/v1/completions ")
        )
        self.assertIn('"prompt":"Say hello in one short sentence."', cmd)


if __name__ == "__main__":
    unittest.main()

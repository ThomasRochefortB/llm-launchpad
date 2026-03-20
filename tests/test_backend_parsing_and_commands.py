from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend, _extract_modal_app_rows
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import DeploymentConfig


class BackendParsingAndCommandTests(unittest.TestCase):
    def test_modal_cli_path_prefers_active_env_scripts_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.makedirs(bin_dir, exist_ok=True)
            modal_path = os.path.join(bin_dir, "modal")
            with open(modal_path, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n")
            with (
                patch("llm_launchpad.core.backend.sys.prefix", tmp),
                patch("llm_launchpad.core.backend.shutil.which", return_value=None),
            ):
                self.assertEqual(ModalBackend.modal_cli_path(), modal_path)

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

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_billing_report_json_returns_payload(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"summary":{"total_usd":3.25}}'
        payload, error = ModalBackend.billing_report_json()
        self.assertIsInstance(payload, dict)
        self.assertIsNone(error)
        assert isinstance(payload, dict)
        self.assertEqual(payload["summary"]["total_usd"], 3.25)
        called_command = mock_run.call_args.args[0]
        self.assertEqual(
            called_command[-5:],
            ["billing", "report", "--for", "this month", "--json"],
        )

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_billing_report_json_returns_none_on_timeout(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["modal", "billing", "report", "--for", "this month", "--json"],
            timeout=8,
        )
        payload, error = ModalBackend.billing_report_json()
        self.assertIsNone(payload)
        self.assertIn("Timed out", error or "")

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_billing_report_json_falls_back_to_workspace_command(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        first = subprocess.CompletedProcess(
            args=["modal", "billing", "report", "--for", "this month", "--json"],
            returncode=2,
            stdout="",
            stderr="No such command 'billing'.",
        )
        second = subprocess.CompletedProcess(
            args=["modal", "workspace", "billing", "report", "--for", "this month", "--json"],
            returncode=0,
            stdout='{"summary":{"total_usd":7.0}}',
            stderr="",
        )
        mock_run.side_effect = [first, second]

        payload, error = ModalBackend.billing_report_json()
        self.assertIsNone(error)
        assert isinstance(payload, dict)
        self.assertEqual(payload["summary"]["total_usd"], 7.0)

    @patch("llm_launchpad.core.backend.subprocess.run")
    def test_billing_report_json_returns_upgrade_hint_when_billing_command_missing(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        first = subprocess.CompletedProcess(
            args=["modal", "billing", "report", "--for", "this month", "--json"],
            returncode=2,
            stdout="",
            stderr="No such command 'billing'.",
        )
        second = subprocess.CompletedProcess(
            args=["modal", "workspace", "billing", "report", "--for", "this month", "--json"],
            returncode=2,
            stdout="Usage: modal [OPTIONS] COMMAND [ARGS]...",
            stderr="",
        )
        mock_run.side_effect = [first, second]

        payload, error = ModalBackend.billing_report_json()
        self.assertIsNone(payload)
        self.assertIn("Upgrade Modal CLI", error or "")

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

    def test_extract_modal_app_rows_extracts_web_url_and_model_metadata(self) -> None:
        payload = [
            {
                "name": "vllm-qwen",
                "state": "running",
                "details": {
                    "webUrl": "https://alice--vllm-qwen-serve.modal.run",
                    "env": {
                        "MODEL_NAME": "Qwen/Qwen3-4B-Thinking-2507-FP8",
                        "SERVED_MODEL_NAME": "Qwen3-4B",
                    },
                },
            },
            {
                "name": "llamacpp-phi",
                "state": "deployed",
                "metadata": {
                    "config": {
                        "repo_id": "unsloth/Phi-3-GGUF",
                        "quant": "Q4_K_M",
                    }
                },
                "url": "https://alice--llamacpp-phi-serve-sly-otter.modal.run",
            },
        ]

        rows = _extract_modal_app_rows(payload)
        self.assertEqual(rows[0].web_url, "https://alice--vllm-qwen-serve.modal.run")
        self.assertEqual(rows[0].model_name, "Qwen/Qwen3-4B-Thinking-2507-FP8")
        self.assertEqual(rows[0].served_model_name, "Qwen3-4B")
        self.assertEqual(rows[1].web_url, "https://alice--llamacpp-phi-serve-sly-otter.modal.run")
        self.assertEqual(rows[1].repo_id, "unsloth/Phi-3-GGUF")
        self.assertEqual(rows[1].quant, "Q4_K_M")

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
        self.assertIn("--server-args=--ctx-size 65536", cmd)
        self.assertIn("--host", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("--n-gpu-layers", cmd)

    def test_env_for_backend_non_vllm_only_sets_app_name(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-prod",
            model_name="should-not-be-used",
            served_model_name="ignored",
        )
        env = ModalBackend.env_for_backend(config)
        self.assertEqual(env, {"MODAL_APP_NAME": "llamacpp-prod"})

    def test_env_for_backend_llamacpp_forwards_image_no_cache_toggle(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-prod",
            llamacpp_image_no_cache=True,
        )
        env = ModalBackend.env_for_backend(config)
        self.assertEqual(
            env,
            {
                "MODAL_APP_NAME": "llamacpp-prod",
                "LLAMA_CPP_IMAGE_NO_CACHE": "true",
            },
        )

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

    def test_test_curl_command_llamacpp_uses_served_model_name_when_provided(self) -> None:
        cmd = ModalBackend.test_curl_command(
            BackendType.LLAMACPP,
            "https://example.modal.run/",
            served_model_name="Nanbeige4.1-3B-Q4_K_M-GGUF",
        )
        self.assertIn('"model":"Nanbeige4.1-3B-Q4_K_M-GGUF"', cmd)


if __name__ == "__main__":
    unittest.main()

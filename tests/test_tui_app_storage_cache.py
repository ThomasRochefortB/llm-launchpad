from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.app import WizardApp, _deploy_connection_summary_lines


class TuiAppStorageCacheTests(unittest.TestCase):
    def test_deploy_connection_summary_lines_for_llamacpp_are_simple_and_boxed(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
            quant="Q4_K_M",
            app_name="llamacpp-edge-quant-nanbeige4-1-3b-q4-k-m-gguf",
            instance_name="edge-quant-nanbeige4-1-3b-q4-k-m-gguf",
        )
        lines = _deploy_connection_summary_lines(
            config,
            "https://example.modal.run",
        )
        joined = "\n".join(lines)
        self.assertIn("=== OpenAI-compatible ===", joined)
        self.assertIn("Base URL: https://example.modal.run/v1", joined)
        self.assertIn("Model ID: Nanbeige4.1-3B-Q4_K_M-GGUF", joined)
        self.assertIn("Display name: Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF (Q4_K_M)", joined)
        self.assertIn("API key: (leave blank; no auth by default)", joined)
        self.assertIn("=========================", joined)
        self.assertNotIn("OpenCode custom provider:", joined)

    def test_deploy_connection_summary_lines_for_vllm_use_served_model_alias(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            model_name="Nanbeige/Nanbeige4.1-3B",
            served_model_name="Nanbeige4.1-3B",
            app_name="vllm-nanbeige4-1-3b",
            instance_name="nanbeige4-1-3b",
        )
        lines = _deploy_connection_summary_lines(
            config,
            "https://example.modal.run",
        )
        joined = "\n".join(lines)
        self.assertIn("Base URL: https://example.modal.run/v1", joined)
        self.assertIn("Model ID: Nanbeige4.1-3B", joined)
        self.assertIn("Display name: Nanbeige4.1-3B", joined)

    def test_run_deploy_warmup_uses_function_slug_for_default_url(self) -> None:
        warmup_urls: list[str] = []
        app = WizardApp()
        app._username = "alice"
        app._orchestrator = type(
            "FakeOrchestrator",
            (),
            {
                "deploy": staticmethod(
                    lambda _config: [
                        OperationCompleteEvent(success=True, operation=OperationType.DEPLOY)
                    ]
                ),
                "warmup": staticmethod(
                    lambda backend, server_url, *_args, **_kwargs: warmup_urls.append(server_url) or []
                ),
            },
        )()

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            do_deploy=True,
            do_warmup=True,
            app_name="llamacpp-test",
            function_slug="alpha-bravo",
        )
        with patch("llm_launchpad.tui.app._dispatch_event", return_value=None):
            app._run_deploy(config, monitor=object())

        self.assertEqual(
            warmup_urls,
            ["https://alice--llamacpp-test-serve-alpha-bravo.modal.run"],
        )

    def test_run_deploy_warmup_prefers_created_web_function_url_over_printed_endpoint(self) -> None:
        warmup_urls: list[str] = []
        app = WizardApp()
        app._username = "alice"
        created_dev_url = "https://alice--llamacpp-test-serve-alpha-bravo-dev.modal.run"
        created_prod_url = "https://alice--llamacpp-test-serve-abcd1234.modal.run"
        guessed_url = "https://alice--llamacpp-test-serve-alpha-bravo.modal.run"
        app._orchestrator = type(
            "FakeOrchestrator",
            (),
            {
                "deploy": staticmethod(
                    lambda _config: [
                        LogEvent(line=f"└── 🔨 Created web function serve => {created_dev_url}"),
                        LogEvent(
                            line=(
                                "└── 🔨 Created web function serve => "
                                f"{created_prod_url} (label truncated)"
                            )
                        ),
                        LogEvent(line=f"   Endpoint base URL: {guessed_url}"),
                        OperationCompleteEvent(success=True, operation=OperationType.DEPLOY),
                    ]
                ),
                "warmup": staticmethod(
                    lambda backend, server_url, *_args, **_kwargs: warmup_urls.append(server_url) or []
                ),
            },
        )()

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            do_deploy=True,
            do_warmup=True,
            app_name="llamacpp-test",
            function_slug="alpha-bravo",
        )
        with patch("llm_launchpad.tui.app._dispatch_event", return_value=None):
            app._run_deploy(config, monitor=object())

        self.assertEqual(warmup_urls, [created_prod_url])

    def test_run_deploy_suppresses_intermediate_deploy_completion_when_warmup_follows(self) -> None:
        app = WizardApp()
        app._username = "alice"
        app._orchestrator = type(
            "FakeOrchestrator",
            (),
            {
                "deploy": staticmethod(
                    lambda _config: [
                        OperationCompleteEvent(success=True, operation=OperationType.DEPLOY)
                    ]
                ),
                "warmup": staticmethod(
                    lambda *_args, **_kwargs: [
                        LogEvent(line="Probing readiness at: https://example.com/v1/completions"),
                        OperationCompleteEvent(success=True, operation=OperationType.WARMUP),
                    ]
                ),
            },
        )()

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            do_deploy=True,
            do_warmup=True,
            app_name="llamacpp-test",
            function_slug="alpha-bravo",
        )
        with patch("llm_launchpad.tui.app._dispatch_event", return_value=None) as dispatch:
            app._run_deploy(config, monitor=object())

        dispatched_events = [call.args[1] for call in dispatch.call_args_list]
        self.assertFalse(
            any(
                isinstance(event, OperationCompleteEvent)
                and event.operation == OperationType.DEPLOY
                and event.success
                for event in dispatched_events
            )
        )
        self.assertTrue(
            any(
                isinstance(event, OperationCompleteEvent)
                and event.operation == OperationType.WARMUP
                and event.success
                for event in dispatched_events
            )
        )

    def test_run_deploy_emits_connection_summary_on_warmup_success(self) -> None:
        app = WizardApp()
        app._username = "alice"
        app._orchestrator = type(
            "FakeOrchestrator",
            (),
            {
                "deploy": staticmethod(
                    lambda _config: [
                        LogEvent(line="└── 🔨 Created web function serve => https://alice--llamacpp-test-serve-abcd.modal.run"),
                        OperationCompleteEvent(success=True, operation=OperationType.DEPLOY),
                    ]
                ),
                "warmup": staticmethod(
                    lambda *_args, **_kwargs: [
                        OperationCompleteEvent(
                            success=True,
                            operation=OperationType.WARMUP,
                            data={"url": "https://alice--llamacpp-test-serve-abcd.modal.run"},
                        )
                    ]
                ),
            },
        )()

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
            quant="Q4_K_M",
            do_deploy=True,
            do_warmup=True,
            app_name="llamacpp-test",
            instance_name="llamacpp-test",
            function_slug="alpha-bravo",
        )
        with patch("llm_launchpad.tui.app._dispatch_event", return_value=None) as dispatch:
            app._run_deploy(config, monitor=object())

        dispatched_events = [call.args[1] for call in dispatch.call_args_list]
        summary_lines = [
            event.line
            for event in dispatched_events
            if isinstance(event, LogEvent)
            and (
                "=== OpenAI-compatible ===" in event.line
                or "Base URL:" in event.line
                or "Model ID:" in event.line
                or "Display name:" in event.line
            )
        ]
        self.assertTrue(any("=== OpenAI-compatible ===" in line for line in summary_lines))
        self.assertTrue(any("Model ID: Nanbeige4.1-3B-Q4_K_M-GGUF" in line for line in summary_lines))
        self.assertTrue(
            any("Display name: Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF (Q4_K_M)" in line for line in summary_lines)
        )

    def test_list_instances_merges_cached_deploy_connection_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = WizardApp()
            app._deploy_connection_cache_path = Path(tmp) / "deployment_connection_summaries.json"
            app._deploy_connection_cache = {}
            config = DeploymentConfig(
                backend=BackendType.LLAMACPP,
                repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
                quant="Q4_K_M",
                app_name="llamacpp-edge-quant",
                instance_name="edge-quant",
                served_model_name="Nanbeige4.1-3B-Q4_K_M-GGUF",
            )
            app._cache_deploy_connection_summary(config, "https://alice--llamacpp-edge-quant-serve.modal.run")

            rows = [
                EndpointInfo(
                    name="llamacpp-edge-quant",
                    app_id="ap-123",
                    state="running",
                    backend=BackendType.LLAMACPP,
                    instance_name="edge-quant",
                )
            ]
            with patch("llm_launchpad.tui.app.ModalBackend.list_apps", return_value=rows):
                merged = app.list_instances()

            self.assertEqual(len(merged), 1)
            self.assertEqual(
                merged[0].web_url,
                "https://alice--llamacpp-edge-quant-serve.modal.run/v1",
            )
            self.assertEqual(merged[0].served_model_name, "Nanbeige4.1-3B-Q4_K_M-GGUF")
            self.assertEqual(
                merged[0].display_name,
                "Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF (Q4_K_M)",
            )

    def test_snapshot_persist_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "storage_snapshot.json"
            app = WizardApp()
            app._storage_cache_path = cache_path
            app._storage_snapshot_cache = None
            app._storage_snapshot_cached_at_epoch = 0.0

            snapshot = StorageSnapshot(
                llamacpp_models=[
                    StoredModelInfo(
                        backend=BackendType.LLAMACPP,
                        model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                        revision="main",
                        quant="Q4_K_M",
                        size_bytes=1024,
                        file_count=1,
                        source_volume="huggingface-cache",
                        paths=["models/Qwen__Qwen2.5-Coder-7B-Instruct-GGUF/main/model.Q4_K_M.gguf"],
                        incomplete=True,
                    )
                ],
                vllm_models=[
                    StoredModelInfo(
                        backend=BackendType.VLLM,
                        model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
                        revision=None,
                        quant=None,
                        size_bytes=2048,
                        file_count=2,
                        source_volume="huggingface-cache",
                        paths=["hub/models--Qwen--Qwen3-4B-Thinking-2507-FP8"],
                    )
                ],
            )

            app._cache_storage_snapshot(snapshot)
            self.assertTrue(cache_path.exists())

            reloaded = WizardApp()
            reloaded._storage_cache_path = cache_path
            reloaded._storage_snapshot_cache = None
            reloaded._storage_snapshot_cached_at_epoch = 0.0
            reloaded._load_persisted_storage_cache()

            self.assertIsNotNone(reloaded._storage_snapshot_cache)
            loaded = reloaded._storage_snapshot_cache
            assert loaded is not None
            self.assertEqual(len(loaded.llamacpp_models), 1)
            self.assertEqual(len(loaded.vllm_models), 1)
            self.assertEqual(loaded.llamacpp_models[0].model_id, snapshot.llamacpp_models[0].model_id)
            self.assertEqual(loaded.vllm_models[0].model_id, snapshot.vllm_models[0].model_id)
            self.assertTrue(loaded.llamacpp_models[0].incomplete)

    def test_invalidate_storage_cache_removes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "storage_snapshot.json"
            app = WizardApp()
            app._storage_cache_path = cache_path
            snapshot = StorageSnapshot(llamacpp_models=[], vllm_models=[])
            app._cache_storage_snapshot(snapshot)
            self.assertTrue(cache_path.exists())

            app._invalidate_storage_cache()
            self.assertFalse(cache_path.exists())
            self.assertIsNone(app._storage_snapshot_cache)

    def test_load_persisted_storage_cache_handles_string_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "storage_snapshot.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "cached_at_epoch": 123.0,
                        "snapshot": {
                            "llamacpp_models": [
                                {
                                    "backend": "llamacpp",
                                    "model_id": "legacy:model-Q4_K_M",
                                    "revision": None,
                                    "quant": "Q4_K_M",
                                    "size_bytes": 1024,
                                    "file_count": 1,
                                    "source_volume": "huggingface-cache",
                                    "paths": "/legacy/model-Q4_K_M.gguf",
                                    "incomplete": False,
                                }
                            ],
                            "vllm_models": [],
                        },
                    }
                )
            )

            app = WizardApp()
            app._storage_cache_path = cache_path
            app._storage_snapshot_cache = None
            app._storage_snapshot_cached_at_epoch = 0.0
            app._load_persisted_storage_cache()

            self.assertIsNotNone(app._storage_snapshot_cache)
            snapshot = app._storage_snapshot_cache
            assert snapshot is not None
            self.assertEqual(
                snapshot.llamacpp_models[0].paths,
                ["/legacy/model-Q4_K_M.gguf"],
            )


if __name__ == "__main__":
    unittest.main()

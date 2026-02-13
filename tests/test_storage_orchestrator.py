from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent, StateChangeEvent
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo


class StorageOrchestratorTests(unittest.TestCase):
    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_builds_snapshot(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("llamacpp-cache", "/models"): [{"path": "/models/Qwen__Coder", "type": "directory"}],
                ("llamacpp-cache", "/models/Qwen__Coder"): [
                    {"path": "/models/Qwen__Coder/main", "type": "directory"}
                ],
                ("llamacpp-cache", "/models/Qwen__Coder/main"): [
                    {
                        "path": "/models/Qwen__Coder/main/model.Q4_K_M.gguf",
                        "type": "file",
                        "size": 2000,
                    }
                ],
                ("llamacpp-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Qwen3-4B-Thinking-2507-FP8", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Qwen3-4B-Thinking-2507-FP8"): [
                    {
                        "path": "/hub/models--Qwen--Qwen3-4B-Thinking-2507-FP8/snapshots/abc/config.json",
                        "type": "file",
                        "size": 500,
                    }
                ],
            }
            return mapping.get((volume_name, path), [])

        mock_list_volume.side_effect = side_effect

        orch = Orchestrator()
        events = list(orch.list_storage())
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        self.assertEqual(done.operation, OperationType.STORAGE_LIST)
        self.assertIsInstance(done.data, StorageSnapshot)
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        self.assertEqual(len(snapshot.vllm_models), 1)
        self.assertEqual(snapshot.total_models, 2)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.run_modal_script_entrypoint")
    def test_predownload_model_emits_storage_operation(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        mock_run.return_value = iter(
            [
                LogEvent(line="download started"),
                OperationCompleteEvent(success=True, exit_code=0),
            ]
        )
        orch = Orchestrator()
        events = list(
            orch.predownload_model(
                backend=BackendType.VLLM,
                model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
            )
        )
        state = next(e for e in events if isinstance(e, StateChangeEvent))
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertEqual(state.operation, OperationType.STORAGE_PREDOWNLOAD)
        self.assertEqual(done.operation, OperationType.STORAGE_PREDOWNLOAD)
        self.assertTrue(done.success)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_handles_uppercase_volume_keys(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("llamacpp-cache", "/models"): [{"Name": "Qwen__Coder", "Type": "directory"}],
                ("llamacpp-cache", "/models/Qwen__Coder"): [{"Name": "main", "Type": "directory"}],
                ("llamacpp-cache", "/models/Qwen__Coder/main"): [
                    {"Name": "weights.Q4_K_M.gguf", "Type": "file", "Size": 2048}
                ],
                ("llamacpp-cache", "/"): [],
                ("huggingface-cache", "/hub"): [],
            }
            return mapping.get((volume_name, path), [])

        mock_list_volume.side_effect = side_effect
        orch = Orchestrator()
        events = list(orch.list_storage())
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.llamacpp_models), 1)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_handles_hub_prefixed_filename_paths(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("llamacpp-cache", "/models"): [],
                ("llamacpp-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"Filename": "hub/models--Qwen--Qwen3-4B", "Type": "dir"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Qwen3-4B"): [
                    {
                        "Filename": "hub/models--Qwen--Qwen3-4B/snapshots/abc/config.json",
                        "Type": "file",
                        "Size": "123 B",
                    }
                ],
            }
            return mapping.get((volume_name, path), [])

        mock_list_volume.side_effect = side_effect
        orch = Orchestrator()
        events = list(orch.list_storage())
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.vllm_models), 1)
        self.assertEqual(snapshot.vllm_models[0].model_id, "Qwen/Qwen3-4B")
        self.assertEqual(snapshot.vllm_models[0].size_bytes, 123)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_legacy_llamacpp_shards_are_grouped(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("llamacpp-cache", "/models"): [],
                ("llamacpp-cache", "/"): [
                    {
                        "Filename": "GLM-5-Q4_K_M-00001-of-00003.gguf",
                        "Type": "file",
                        "Size": "1 MB",
                    },
                    {
                        "Filename": "GLM-5-Q4_K_M-00002-of-00003.gguf",
                        "Type": "file",
                        "Size": "2 MB",
                    },
                    {
                        "Filename": "GLM-5-Q4_K_M-00003-of-00003.gguf",
                        "Type": "file",
                        "Size": "3 MB",
                    },
                ],
                ("huggingface-cache", "/hub"): [],
            }
            return mapping.get((volume_name, path), [])

        mock_list_volume.side_effect = side_effect
        orch = Orchestrator()
        events = list(orch.list_storage())
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        model = snapshot.llamacpp_models[0]
        self.assertEqual(model.model_id, "legacy:GLM-5-Q4_K_M")
        self.assertEqual(model.file_count, 3)
        self.assertEqual(model.size_bytes, 6 * 1024 * 1024)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.run_volume_remove")
    def test_delete_stored_model_vllm_uses_model_directory(self, mock_remove) -> None:  # type: ignore[no-untyped-def]
        mock_remove.return_value = iter([OperationCompleteEvent(success=True, exit_code=0)])  # type: ignore[call-arg]
        orch = Orchestrator()
        model = StoredModelInfo(
            backend=BackendType.VLLM,
            model_id="Qwen/Qwen3-4B",
            source_volume="huggingface-cache",
            paths=[],
        )
        events = list(orch.delete_stored_model(model))
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        mock_remove.assert_called_with(
            "huggingface-cache",
            "/hub/models--Qwen--Qwen3-4B",
            recursive=True,
        )


if __name__ == "__main__":
    unittest.main()

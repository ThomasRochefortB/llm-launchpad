from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent, StateChangeEvent
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo


class StorageOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._capture_patcher = patch(
            "llm_launchpad.core.orchestrator.ModalBackend.run_modal_script_entrypoint_capture",
            return_value=None,
        )
        self._capture_patcher.start()
        self.addCleanup(self._capture_patcher.stop)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_builds_snapshot(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Coder", "type": "directory"},
                    {"path": "/hub/models--Qwen--Qwen3-4B-Thinking-2507-FP8", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Coder"): [
                    {
                        "path": "/hub/models--Qwen--Coder/snapshots/abc/model.Q4_K_M.gguf",
                        "type": "file",
                        "size": 2000,
                    }
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
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [{"Name": "models--Qwen--Coder", "Type": "directory"}],
                ("huggingface-cache", "/hub/models--Qwen--Coder"): [
                    {"Name": "snapshots", "Type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Coder/snapshots"): [
                    {"Name": "abc", "Type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Coder/snapshots/abc"): [
                    {"Name": "weights.Q4_K_M.gguf", "Type": "file", "Size": 2048}
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
        self.assertEqual(len(snapshot.llamacpp_models), 1)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_handles_hub_prefixed_filename_paths(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
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
    def test_list_storage_detects_llamacpp_hub_snapshot_gguf(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--unsloth--GLM-5-GGUF", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--unsloth--GLM-5-GGUF"): [
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/snapshots/abc/GLM-5-Q4_K_M.gguf",
                        "type": "file",
                        "size": 1024,
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
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        self.assertEqual(snapshot.llamacpp_models[0].model_id, "unsloth/GLM-5-GGUF")
        self.assertEqual(snapshot.llamacpp_models[0].quant, "Q4_K_M")

    @patch("llm_launchpad.core.orchestrator.ModalBackend.run_modal_script_entrypoint_capture")
    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_uses_llamacpp_backend_inventory_for_real_sizes(
        self, mock_list_volume, mock_capture
    ) -> None:  # type: ignore[no-untyped-def]
        mock_list_volume.return_value = []
        mock_capture.return_value = (
            0,
            "\n".join(
                [
                    "modal noise",
                    "LLM_LAUNCHPAD_STORAGE_JSON_BEGIN",
                    (
                        '[{"backend":"llamacpp","model_id":"unsloth/GLM-5-GGUF","revision":"abc",'
                        '"quant":"Q4_K_M","size_bytes":123456789,"file_count":11,'
                        '"source_volume":"huggingface-cache","paths":["hub/models--unsloth--GLM-5-GGUF/'
                        'snapshots/abc/Q4_K_M/GLM-5-Q4_K_M-00001-of-00011.gguf"]}]'
                    ),
                    "LLM_LAUNCHPAD_STORAGE_JSON_END",
                ]
            ),
            "",
        )

        orch = Orchestrator()
        events = list(orch.list_storage())
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        row = snapshot.llamacpp_models[0]
        self.assertEqual(row.model_id, "unsloth/GLM-5-GGUF")
        self.assertEqual(row.revision, "abc")
        self.assertEqual(row.quant, "Q4_K_M")
        self.assertEqual(row.size_bytes, 123456789)
        self.assertEqual(row.file_count, 11)
        self.assertFalse(row.incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_marks_llamacpp_model_incomplete_from_blob_sidecar(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--unsloth--GLM-5-GGUF", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--unsloth--GLM-5-GGUF"): [
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/snapshots/abc/GLM-5-Q4_K_M.gguf",
                        "type": "file",
                        "size": 79,
                    },
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/blobs/weights.gguf.incomplete",
                        "type": "file",
                        "size": 2048,
                    },
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
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        self.assertTrue(snapshot.llamacpp_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_ignores_llamacpp_incomplete_sidecar_when_blob_exists(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--unsloth--GLM-5-GGUF", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--unsloth--GLM-5-GGUF"): [
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/snapshots/abc/GLM-5-Q4_K_M.gguf",
                        "type": "file",
                        "size": 79,
                    },
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/blobs/weights.gguf",
                        "type": "file",
                        "size": 1024,
                    },
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/blobs/weights.gguf.incomplete",
                        "type": "file",
                        "size": 128,
                    },
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
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        self.assertFalse(snapshot.llamacpp_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_marks_llamacpp_tiny_snapshot_without_blob_incomplete(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--unsloth--GLM-5-GGUF", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--unsloth--GLM-5-GGUF"): [
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/snapshots/abc/GLM-5-Q4_K_M.gguf",
                        "type": "file",
                        "size": 79,
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
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        self.assertTrue(snapshot.llamacpp_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_keeps_llamacpp_complete_when_tiny_snapshot_has_blob(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--unsloth--GLM-5-GGUF", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--unsloth--GLM-5-GGUF"): [
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/snapshots/abc/GLM-5-Q4_K_M.gguf",
                        "type": "file",
                        "size": 79,
                    },
                    {
                        "path": "/hub/models--unsloth--GLM-5-GGUF/blobs/weights.gguf",
                        "type": "file",
                        "size": 1024,
                    },
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
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        self.assertFalse(snapshot.llamacpp_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_marks_vllm_model_incomplete(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Qwen3-4B", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Qwen3-4B"): [
                    {
                        "path": "/hub/models--Qwen--Qwen3-4B/blobs/weights.bin.incomplete",
                        "type": "file",
                        "size": 1024,
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
        self.assertTrue(snapshot.vllm_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_ignores_vllm_incomplete_sidecar_when_blob_exists(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Qwen3-4B", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Qwen3-4B"): [
                    {
                        "path": "/hub/models--Qwen--Qwen3-4B/blobs/weights.bin",
                        "type": "file",
                        "size": 1024,
                    },
                    {
                        "path": "/hub/models--Qwen--Qwen3-4B/blobs/weights.bin.incomplete",
                        "type": "file",
                        "size": 256,
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
        self.assertFalse(snapshot.vllm_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_marks_vllm_incomplete_for_refs_without_snapshots(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Qwen3-4B", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Qwen3-4B"): [
                    {"path": "/hub/models--Qwen--Qwen3-4B/refs/main", "type": "file", "size": 40}
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
        self.assertTrue(snapshot.vllm_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_ignores_orphan_vllm_incomplete_blobs_when_snapshot_exists(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Qwen3-4B", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Qwen3-4B"): [
                    {
                        "path": "/hub/models--Qwen--Qwen3-4B/snapshots/abc/config.json",
                        "type": "file",
                        "size": 500,
                    },
                    {
                        "path": "/hub/models--Qwen--Qwen3-4B/blobs/weights.bin.incomplete",
                        "type": "file",
                        "size": 1024,
                    },
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
        self.assertFalse(snapshot.vllm_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_accepts_vllm_snapshot_only_layout(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Qwen3-4B", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Qwen3-4B"): [
                    {
                        "path": "/hub/models--Qwen--Qwen3-4B/snapshots/abc/config.json",
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
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.vllm_models), 1)
        self.assertFalse(snapshot.vllm_models[0].incomplete)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_ignores_llamacpp_incomplete_without_gguf(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Coder", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Coder"): [
                    {
                        "path": "/hub/models--Qwen--Coder/blobs/model.gguf.incomplete",
                        "type": "file",
                        "size": 2048,
                    }
                ],
                ("huggingface-cache", "/"): [],
            }
            return mapping.get((volume_name, path), [])

        mock_list_volume.side_effect = side_effect
        orch = Orchestrator()
        events = list(orch.list_storage())
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.llamacpp_models), 0)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_list_storage_ignores_llamacpp_metadata_without_gguf_payload(
        self, mock_list_volume
    ) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Coder", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Coder"): [
                    {
                        "path": "/hub/models--Qwen--Coder/refs/main",
                        "type": "file",
                        "size": 40,
                    }
                ],
                ("huggingface-cache", "/"): [],
            }
            return mapping.get((volume_name, path), [])

        mock_list_volume.side_effect = side_effect
        orch = Orchestrator()
        events = list(orch.list_storage())
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        snapshot = done.data
        assert isinstance(snapshot, StorageSnapshot)
        self.assertEqual(len(snapshot.llamacpp_models), 0)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_legacy_llamacpp_shards_are_ignored(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [
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
        self.assertEqual(len(snapshot.llamacpp_models), 0)

    @patch("llm_launchpad.core.orchestrator.ModalBackend.list_volume")
    def test_llamacpp_storage_scan_uses_hub_only(self, mock_list_volume) -> None:  # type: ignore[no-untyped-def]
        def side_effect(volume_name, path):  # type: ignore[no-untyped-def]
            mapping = {
                ("huggingface-cache", "/models"): [],
                ("huggingface-cache", "/"): [
                    {"path": "/models", "type": "directory"},
                    {"path": "/legacy.Q4_K_M.gguf", "type": "file", "size": 1024},
                ],
                ("huggingface-cache", "/hub"): [
                    {"path": "/hub/models--Qwen--Coder", "type": "directory"}
                ],
                ("huggingface-cache", "/hub/models--Qwen--Coder"): [
                    {
                        "path": "/hub/models--Qwen--Coder/snapshots/abc/model.Q4_K_M.gguf",
                        "type": "file",
                        "size": 2048,
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
        self.assertEqual(len(snapshot.llamacpp_models), 1)
        self.assertEqual(snapshot.llamacpp_models[0].model_id, "Qwen/Coder")

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

    @patch("llm_launchpad.core.orchestrator.ModalBackend.run_volume_remove")
    def test_delete_stored_model_llamacpp_hub_layout_uses_model_directory(
        self, mock_remove
    ) -> None:  # type: ignore[no-untyped-def]
        mock_remove.return_value = iter([OperationCompleteEvent(success=True, exit_code=0)])  # type: ignore[call-arg]
        orch = Orchestrator()
        model = StoredModelInfo(
            backend=BackendType.LLAMACPP,
            model_id="unsloth/GLM-5-GGUF",
            source_volume="huggingface-cache",
            paths=["/hub/models--unsloth--GLM-5-GGUF/snapshots/abc/GLM-5-Q4_K_M.gguf"],
        )
        events = list(orch.delete_stored_model(model))
        done = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(done.success)
        mock_remove.assert_called_with(
            "huggingface-cache",
            "/hub/models--unsloth--GLM-5-GGUF",
            recursive=True,
        )


if __name__ == "__main__":
    unittest.main()

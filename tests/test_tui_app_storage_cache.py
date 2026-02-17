from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.app import WizardApp


class TuiAppStorageCacheTests(unittest.TestCase):
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

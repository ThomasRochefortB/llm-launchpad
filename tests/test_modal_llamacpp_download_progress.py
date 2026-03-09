from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from llm_launchpad.backends import modal_llamacpp_app


class LlamacppDownloadProgressTests(unittest.TestCase):
    def test_estimate_matched_snapshot_size_uses_allow_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"
            model_dir = hub_dir / "models--unsloth--GLM-5-GGUF"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            refs_dir = model_dir / "refs"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123\n", encoding="utf-8")
            (snapshot_dir / "GLM-5-Q4_K_M.gguf").write_bytes(b"a" * 10)
            (snapshot_dir / "GLM-5-Q8_0.gguf").write_bytes(b"b" * 20)
            (snapshot_dir / "README.md").write_bytes(b"c" * 5)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                size_bytes, file_count = modal_llamacpp_app._estimate_matched_snapshot_size(
                    repo_id="unsloth/GLM-5-GGUF",
                    revision=None,
                    allow_patterns=["*Q4_K_M*.gguf"],
                )

            self.assertEqual(size_bytes, 10)
            self.assertEqual(file_count, 1)

    def test_estimate_incomplete_blob_size_counts_incomplete_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"
            blobs_dir = hub_dir / "models--unsloth--GLM-5-GGUF" / "blobs"
            blobs_dir.mkdir(parents=True, exist_ok=True)
            (blobs_dir / "blob-a.incomplete").write_bytes(b"a" * 11)
            (blobs_dir / "blob-b.incomplete").write_bytes(b"b" * 13)
            (blobs_dir / "blob-c").write_bytes(b"c" * 17)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                size_bytes, file_count = modal_llamacpp_app._estimate_incomplete_blob_size(
                    repo_id="unsloth/GLM-5-GGUF"
                )

            self.assertEqual(size_bytes, 24)
            self.assertEqual(file_count, 2)

    def test_resolve_model_entrypoint_prefers_first_split_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"
            model_dir = hub_dir / "models--unsloth--GLM-5-GGUF"
            snapshot_dir = model_dir / "snapshots" / "abc123" / "Q4_K_M"
            refs_dir = model_dir / "refs"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123\n", encoding="utf-8")

            # Make shard 2 larger to reproduce the previous size-based misselection.
            shard1 = snapshot_dir / "GLM-5-Q4_K_M-00001-of-00011.gguf"
            shard2 = snapshot_dir / "GLM-5-Q4_K_M-00002-of-00011.gguf"
            shard1.write_bytes(b"a" * 10)
            shard2.write_bytes(b"b" * 20)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                selected = modal_llamacpp_app._resolve_model_entrypoint(
                    repo_id="unsloth/GLM-5-GGUF",
                    revision=None,
                    quant="Q4_K_M",
                )

            self.assertEqual(selected.name, "GLM-5-Q4_K_M-00001-of-00011.gguf")

    def test_resolve_model_entrypoint_does_not_fallback_to_other_repo_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"

            # Requested repo snapshot exists but does not contain the requested quant.
            requested_model_dir = hub_dir / "models--Edge-Quant--Nanbeige4.1-3B-Q4_K_M-GGUF"
            requested_snapshot = requested_model_dir / "snapshots" / "abc123"
            requested_refs = requested_model_dir / "refs"
            requested_snapshot.mkdir(parents=True, exist_ok=True)
            requested_refs.mkdir(parents=True, exist_ok=True)
            (requested_refs / "main").write_text("abc123\n", encoding="utf-8")
            (requested_snapshot / "README.md").write_text("not a model", encoding="utf-8")

            # Unrelated cached repo has a matching GGUF; old behavior would incorrectly pick this.
            other_model_dir = hub_dir / "models--unsloth--GLM-5-GGUF"
            other_snapshot = other_model_dir / "snapshots" / "def456"
            other_refs = other_model_dir / "refs"
            other_snapshot.mkdir(parents=True, exist_ok=True)
            other_refs.mkdir(parents=True, exist_ok=True)
            (other_refs / "main").write_text("def456\n", encoding="utf-8")
            (other_snapshot / "GLM-5-Q4_K_M.gguf").write_bytes(b"x" * 10)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                with self.assertRaises(RuntimeError) as ctx:
                    modal_llamacpp_app._resolve_model_entrypoint(
                        repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
                        revision=None,
                        quant="Q4_K_M",
                    )

            self.assertIn("Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF", str(ctx.exception))

    def test_resolve_model_entrypoint_matches_lowercase_quant_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"
            model_dir = hub_dir / "models--Edge-Quant--Nanbeige4.1-3B-Q4_K_M-GGUF"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            refs_dir = model_dir / "refs"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123\n", encoding="utf-8")
            gguf = snapshot_dir / "nanbeige4.1-3b-q4_k_m.gguf"
            gguf.write_bytes(b"x" * 10)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                with patch.object(modal_llamacpp_app, "cache_dir", str(Path(tmp))):
                    selected = modal_llamacpp_app._resolve_model_entrypoint(
                        repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
                        revision=None,
                        quant="Q4_K_M",
                    )
                    matches = modal_llamacpp_app._collect_hub_gguf_matches(
                        repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
                        revision=None,
                        allow_patterns=["*Q4_K_M*.gguf"],
                    )

            self.assertEqual(selected.name, gguf.name)
            self.assertEqual(len(matches), 1)
            self.assertTrue(matches[0].endswith("nanbeige4.1-3b-q4_k_m.gguf"))

    def test_resolve_or_download_model_entrypoint_skips_download_on_cache_hit(self) -> None:
        cached_path = Path("/tmp/cached-model.gguf")
        fake_download = SimpleNamespace()
        fake_download.remote_called = False

        def _remote(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            fake_download.remote_called = True
            return []

        fake_download.remote = _remote

        with patch.object(modal_llamacpp_app, "download_model", fake_download):
            with patch.object(modal_llamacpp_app, "_resolve_model_entrypoint", return_value=cached_path):
                selected = modal_llamacpp_app._resolve_or_download_model_entrypoint(
                    repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
                    revision=None,
                    quant="Q4_K_M",
                )

        self.assertEqual(selected, cached_path)
        self.assertFalse(fake_download.remote_called)

    def test_resolve_or_download_model_entrypoint_downloads_on_cache_miss(self) -> None:
        downloaded_path = Path("/tmp/downloaded-model.gguf")
        calls: list[tuple[object, ...]] = []
        fake_download = SimpleNamespace()

        def _remote(repo_id, allow_patterns, revision):  # type: ignore[no-untyped-def]
            calls.append((repo_id, tuple(allow_patterns), revision))
            return ["hub/models--Edge-Quant--Nanbeige.../nanbeige4.1-3b-q4_k_m.gguf"]

        fake_download.remote = _remote

        with patch.object(modal_llamacpp_app, "download_model", fake_download):
            with patch.object(
                modal_llamacpp_app,
                "_resolve_model_entrypoint",
                side_effect=[RuntimeError("cache miss"), downloaded_path],
            ):
                selected = modal_llamacpp_app._resolve_or_download_model_entrypoint(
                    repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
                    revision=None,
                    quant="Q4_K_M",
                )

        self.assertEqual(selected, downloaded_path)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF")
        self.assertIn("*Q4_K_M*.gguf", calls[0][1])


if __name__ == "__main__":
    unittest.main()

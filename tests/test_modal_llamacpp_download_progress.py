from __future__ import annotations

from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from llm_launchpad.backends import modal_llamacpp_app


class FakeModalDict:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def get(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)

    def put(self, key: str, value: object, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = value
        return True

    def pop(self, key: str, default: object = None) -> object:
        return self.data.pop(key, default)


class LlamacppDownloadProgressTests(unittest.TestCase):
    def test_download_image_env_disables_xet_by_default(self) -> None:
        with patch.object(modal_llamacpp_app, "HF_HUB_DISABLE_XET_DEFAULT", True):
            with patch.object(modal_llamacpp_app, "HF_XET_HIGH_PERFORMANCE_DEFAULT", True):
                env = modal_llamacpp_app._download_image_env()

        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", env)

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
            for idx in range(1, 12):
                shard = snapshot_dir / f"GLM-5-Q4_K_M-{idx:05d}-of-00011.gguf"
                size = 20 if idx == 2 else 10
                shard.write_bytes(bytes([96 + idx]) * size)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                selected = modal_llamacpp_app._resolve_model_entrypoint(
                    repo_id="unsloth/GLM-5-GGUF",
                    revision=None,
                    quant="Q4_K_M",
                )

            self.assertEqual(selected.name, "GLM-5-Q4_K_M-00001-of-00011.gguf")

    def test_collect_hub_gguf_matches_ignores_incomplete_split_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"
            model_dir = hub_dir / "models--unsloth--GLM-5-GGUF"
            snapshot_dir = model_dir / "snapshots" / "abc123" / "UD-Q2_K_XL"
            refs_dir = model_dir / "refs"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123\n", encoding="utf-8")
            (snapshot_dir / "GLM-5-UD-Q2_K_XL-00001-of-00003.gguf").write_bytes(b"a" * 10)
            (snapshot_dir / "GLM-5-UD-Q2_K_XL-00003-of-00003.gguf").write_bytes(b"b" * 10)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                with patch.object(modal_llamacpp_app, "cache_dir", str(Path(tmp))):
                    matches = modal_llamacpp_app._collect_hub_gguf_matches(
                        repo_id="unsloth/GLM-5-GGUF",
                        revision=None,
                        allow_patterns=["*UD-Q2_K_XL*.gguf"],
                    )

            self.assertEqual(matches, [])

    def test_validate_cached_gguf_matches_rejects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"
            model_dir = hub_dir / "models--unsloth--GLM-5-GGUF"
            snapshot_dir = model_dir / "snapshots" / "abc123" / "UD-Q2_K_XL"
            refs_dir = model_dir / "refs"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123\n", encoding="utf-8")
            shard = snapshot_dir / "GLM-5-UD-Q2_K_XL-00001-of-00001.gguf"
            shard.write_bytes(b"a" * 10)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                with patch.object(
                    modal_llamacpp_app,
                    "_fetch_expected_gguf_sizes",
                    return_value={"UD-Q2_K_XL/GLM-5-UD-Q2_K_XL-00001-of-00001.gguf": 20},
                ):
                    with patch.object(modal_llamacpp_app, "cache_dir", str(Path(tmp))):
                        matches, problems = modal_llamacpp_app._validate_cached_gguf_matches(
                            repo_id="unsloth/GLM-5-GGUF",
                            revision=None,
                            allow_patterns=["*UD-Q2_K_XL*.gguf"],
                        )

            self.assertEqual(matches, [])
            self.assertEqual(len(problems), 1)
            self.assertIn("size=10 expected=20", problems[0])

    def test_resolve_model_entrypoint_raises_on_incomplete_split_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_dir = Path(tmp) / "hub"
            model_dir = hub_dir / "models--unsloth--GLM-5-GGUF"
            snapshot_dir = model_dir / "snapshots" / "abc123" / "UD-Q2_K_XL"
            refs_dir = model_dir / "refs"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123\n", encoding="utf-8")
            (snapshot_dir / "GLM-5-UD-Q2_K_XL-00001-of-00003.gguf").write_bytes(b"a" * 10)

            with patch.object(modal_llamacpp_app, "HF_HUB_DIR", hub_dir):
                with self.assertRaises(RuntimeError) as ctx:
                    modal_llamacpp_app._resolve_model_entrypoint(
                        repo_id="unsloth/GLM-5-GGUF",
                        revision=None,
                        quant="UD-Q2_K_XL",
                    )

            self.assertIn("Incomplete GGUF shard set", str(ctx.exception))

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

    def test_acquire_download_lease_raises_when_another_owner_is_active(self) -> None:
        fake_dict = FakeModalDict()
        lease_key = modal_llamacpp_app._download_lease_key("unsloth/GLM-5-GGUF", None)
        fake_dict.put(
            lease_key,
            {
                "owner_id": "other-owner",
                "repo_id": "unsloth/GLM-5-GGUF",
                "revision": None,
                "allow_patterns": ["*UD-Q2_K_XL*.gguf"],
                "acquired_at": 100.0,
                "heartbeat_at": 100.0,
            },
        )

        with patch.object(modal_llamacpp_app, "download_leases", fake_dict):
            with patch.object(modal_llamacpp_app, "HF_DOWNLOAD_LOCK_WAIT_TIMEOUT_SECONDS", 0):
                with patch.object(modal_llamacpp_app.time, "time", return_value=100.0):
                    with self.assertRaises(RuntimeError) as ctx:
                        with modal_llamacpp_app._acquire_download_lease(
                            "unsloth/GLM-5-GGUF",
                            None,
                            ["*UD-Q2_K_XL*.gguf"],
                        ):
                            pass

        self.assertIn("already in progress on Modal", str(ctx.exception))

    def test_download_model_files_skips_snapshot_after_cache_recheck(self) -> None:
        @contextmanager
        def fake_lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            yield lambda: None

        fake_cache = SimpleNamespace(commit=Mock(), reload=Mock())

        with patch.object(modal_llamacpp_app, "_acquire_download_lease", fake_lease):
            with patch.object(
                modal_llamacpp_app,
                "_validate_cached_gguf_matches",
                side_effect=[
                    ([], []),
                    (["hub/models--unsloth--GLM-5-GGUF/snapshots/abc123/UD-Q2_K_XL/model.gguf"], []),
                ],
            ):
                with patch.object(modal_llamacpp_app, "_snapshot_download_with_keepalive") as mock_snapshot:
                    with patch.object(modal_llamacpp_app, "model_cache", fake_cache):
                        matches = modal_llamacpp_app._download_model_files(
                            repo_id="unsloth/GLM-5-GGUF",
                            allow_patterns=["*UD-Q2_K_XL*.gguf"],
                            revision=None,
                        )

        self.assertEqual(
            matches,
            ["hub/models--unsloth--GLM-5-GGUF/snapshots/abc123/UD-Q2_K_XL/model.gguf"],
        )
        fake_cache.reload.assert_called_once()
        fake_cache.commit.assert_not_called()
        mock_snapshot.assert_not_called()

    def test_download_model_files_forces_redownload_when_cached_sizes_are_invalid(self) -> None:
        @contextmanager
        def fake_lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            yield lambda: None

        fake_cache = SimpleNamespace(commit=Mock(), reload=Mock())

        with patch.object(modal_llamacpp_app, "_acquire_download_lease", fake_lease):
            with patch.object(
                modal_llamacpp_app,
                "_validate_cached_gguf_matches",
                side_effect=[
                    ([], ["bad size"]),
                    ([], ["bad size"]),
                    (["hub/models--unsloth--GLM-5-GGUF/snapshots/abc123/UD-Q2_K_XL/model.gguf"], []),
                ],
            ):
                with patch.object(modal_llamacpp_app, "model_cache", fake_cache):
                    with patch.object(modal_llamacpp_app, "_snapshot_download_with_keepalive") as mock_snapshot:
                        matches = modal_llamacpp_app._download_model_files(
                            repo_id="unsloth/GLM-5-GGUF",
                            allow_patterns=["*UD-Q2_K_XL*.gguf"],
                            revision=None,
                        )

        self.assertEqual(
            matches,
            ["hub/models--unsloth--GLM-5-GGUF/snapshots/abc123/UD-Q2_K_XL/model.gguf"],
        )
        self.assertTrue(mock_snapshot.called)
        self.assertTrue(mock_snapshot.call_args.kwargs["force_download"])
        fake_cache.commit.assert_called_once()

    def test_snapshot_download_retries_once_with_xet_disabled(self) -> None:
        popen_envs: list[dict[str, str]] = []

        class FakeProcess:
            def __init__(self, returncode: int) -> None:
                self.returncode = returncode

            def poll(self) -> int:
                return self.returncode

        with patch.dict(modal_llamacpp_app.os.environ, {"HF_XET_HIGH_PERFORMANCE": "1"}, clear=True):
            with patch.object(
                modal_llamacpp_app.subprocess,
                "Popen",
                side_effect=[
                    FakeProcess(23),
                    FakeProcess(0),
                ],
            ) as popen_mock:
                modal_llamacpp_app._snapshot_download_with_keepalive(
                    repo_id="unsloth/GLM-5-GGUF",
                    revision=None,
                    cache_dir="/tmp/hub",
                    allow_patterns=["*UD-Q2_K_XL*.gguf"],
                    max_workers=8,
                )

        self.assertEqual(popen_mock.call_count, 2)
        for call in popen_mock.call_args_list:
            popen_envs.append(call.kwargs["env"])
        self.assertEqual(popen_envs[0]["HF_XET_HIGH_PERFORMANCE"], "1")
        self.assertNotIn("HF_HUB_DISABLE_XET", popen_envs[0])
        self.assertEqual(popen_envs[1]["HF_HUB_DISABLE_XET"], "1")
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", popen_envs[1])

    def test_snapshot_download_prints_percent_when_expected_size_is_known(self) -> None:
        printed: list[str] = []

        class FakeProcess:
            def __init__(self) -> None:
                self._poll_results = [None, 0]
                self.returncode = 0

            def poll(self) -> int | None:
                value = self._poll_results.pop(0)
                if value is not None:
                    self.returncode = value
                return value

        with patch.object(
            modal_llamacpp_app,
            "_fetch_expected_gguf_sizes",
            return_value={"UD-Q2_K_XL/shard-1.gguf": 100},
        ):
            with patch.object(
                modal_llamacpp_app,
                "_estimate_completed_expected_gguf_size",
                return_value=(20, 0),
            ):
                with patch.object(
                    modal_llamacpp_app,
                    "_estimate_incomplete_blob_size",
                    return_value=(10, 1),
                ):
                    with patch.object(modal_llamacpp_app.subprocess, "Popen", return_value=FakeProcess()):
                        with patch.object(modal_llamacpp_app.time, "sleep", return_value=None):
                            with patch.object(modal_llamacpp_app.time, "time", side_effect=[0.0, 20.0]):
                                with patch("builtins.print", side_effect=printed.append):
                                    modal_llamacpp_app._snapshot_download_with_keepalive(
                                        repo_id="unsloth/GLM-5-GGUF",
                                        revision=None,
                                        cache_dir="/tmp/hub",
                                        allow_patterns=["*UD-Q2_K_XL*.gguf"],
                                        max_workers=8,
                                    )

        self.assertTrue(any("pct=30%" in line for line in printed))

    def test_snapshot_download_raises_after_xet_disabled_retry_fails(self) -> None:
        class FakeProcess:
            def __init__(self, returncode: int) -> None:
                self.returncode = returncode

            def poll(self) -> int:
                return self.returncode

        with patch.dict(modal_llamacpp_app.os.environ, {"HF_XET_HIGH_PERFORMANCE": "1"}, clear=True):
            with patch.object(
                modal_llamacpp_app.subprocess,
                "Popen",
                side_effect=[
                    FakeProcess(23),
                    FakeProcess(17),
                ],
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    modal_llamacpp_app._snapshot_download_with_keepalive(
                        repo_id="unsloth/GLM-5-GGUF",
                        revision=None,
                        cache_dir="/tmp/hub",
                        allow_patterns=["*UD-Q2_K_XL*.gguf"],
                        max_workers=8,
                    )

        self.assertIn("after retrying with HF_HUB_DISABLE_XET=1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

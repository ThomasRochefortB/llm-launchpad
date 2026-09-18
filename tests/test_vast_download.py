"""Download accounting uses real cache files and a complete HF size manifest."""

from types import SimpleNamespace
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core.vast_download import DownloadFile, fetch_download_files, measure_download
from llm_launchpad.core.vast_runtime import VAST_RUNTIME_DIR, VAST_STARTUP_PROBE_COMMAND
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import DeploymentConfig


def sibling(name: str, size: int | None, oid: str) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=size, lfs=SimpleNamespace(sha256=oid), blob_id=None)


class DownloadMetadataTests(unittest.TestCase):
    def test_reads_exact_shard_sizes_and_blob_ids(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/model")
        info = SimpleNamespace(siblings=[
            sibling("Q4/model-Q4-00001-of-00002.gguf", 1000, "a" * 64),
            sibling("Q4/model-Q4-00002-of-00002.gguf", 2000, "b" * 64),
            sibling("config.json", 100, "c" * 64),
        ])
        with patch("llm_launchpad.core.vast_download.HfApi") as api:
            api.return_value.model_info.return_value = info
            files = fetch_download_files(config)
        self.assertEqual([file.size for file in files], [1000, 2000])
        self.assertEqual({file.group for file in files}, {"Q4/model-Q4"})
        self.assertEqual(files[0].path, "models/models--acme--model/blobs/" + "a" * 64)
        api.return_value.model_info.assert_called_once_with(
            "acme/model", revision=None, files_metadata=True, timeout=10,
        )

    def test_missing_shard_or_size_never_produces_a_partial_total(self) -> None:
        config = DeploymentConfig(backend=BackendType.LLAMACPP, repo_id="acme/model")
        for rows in (
            [sibling("m-00001-of-00002.gguf", 1000, "a" * 64)],
            [sibling("m-00001-of-00002.gguf", 1000, "a" * 64),
             sibling("m-00002-of-00002.gguf", None, "b" * 64)],
        ):
            with self.subTest(rows=rows), patch("llm_launchpad.core.vast_download.HfApi") as api:
                api.return_value.model_info.return_value = SimpleNamespace(siblings=rows)
                self.assertEqual(fetch_download_files(config), ())

    def test_vllm_uses_the_selected_revision_and_hub_cache(self) -> None:
        config = DeploymentConfig(backend=BackendType.VLLM, model_name="acme/model", revision="commit")
        with patch("llm_launchpad.core.vast_download.HfApi") as api:
            api.return_value.model_info.return_value = SimpleNamespace(siblings=[
                sibling("model.safetensors", 1000, "a" * 64),
            ])
            files = fetch_download_files(config)
        self.assertTrue(files[0].path.startswith("hf/hub/models--acme--model/blobs/"))
        self.assertEqual(api.return_value.model_info.call_args.kwargs["revision"], "commit")

    def test_metadata_failure_is_optional(self) -> None:
        with patch("llm_launchpad.core.vast_download.HfApi", side_effect=RuntimeError("offline")):
            self.assertEqual(fetch_download_files(DeploymentConfig(
                backend=BackendType.LLAMACPP, repo_id="acme/model",
            )), ())


class DownloadAccountingTests(unittest.TestCase):
    files = (DownloadFile("models/blobs/a", 1000, "q4"), DownloadFile("models/blobs/b", 2000, "q4"),
             DownloadFile("models/blobs/c", 6000, "q8"))

    def test_completed_shards_remain_in_the_download_count(self) -> None:
        progress = measure_download("FILE 1000 8 models/blobs/a\nFILE 500 8 models/blobs/b.downloadInProgress", self.files)
        assert progress is not None
        self.assertEqual((progress.downloaded, progress.total, progress.active), (1500, 3000, True))

    def test_not_yet_started_shards_are_in_the_total(self) -> None:
        progress = measure_download("FILE 500 8 models/blobs/a.downloadInProgress", self.files)
        assert progress is not None
        self.assertEqual((progress.downloaded, progress.total), (500, 3000))

    def test_final_rename_completes_the_download_without_double_counting(self) -> None:
        progress = measure_download(
            "FILE 1000 8 models/blobs/a\nFILE 2000 8 models/blobs/b.incomplete\nFILE 2000 8 models/blobs/b", self.files,
        )
        assert progress is not None
        self.assertEqual((progress.downloaded, progress.total, progress.active), (3000, 3000, False))

    def test_sparse_partial_file_uses_allocated_bytes(self) -> None:
        progress = measure_download("FILE 2000 1 models/blobs/b.incomplete", self.files)
        assert progress is not None
        self.assertEqual(progress.downloaded, 512)

    def test_unknown_partial_file_disables_the_percentage(self) -> None:
        progress = measure_download("FILE 1000 8 models/blobs/a\nFILE 500 8 models/blobs/unknown.incomplete", self.files)
        assert progress is not None
        self.assertEqual((progress.downloaded, progress.total, progress.active), (1500, None, True))

    def test_bad_probe_output_has_no_measurement(self) -> None:
        for output in ("", "FILE invalid", "FILE -1 0 models/blobs/a", "LOG loading"):
            self.assertIsNone(measure_download(output, self.files))

    def test_shell_probe_counts_blobs_but_not_snapshot_symlinks(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "models/blobs"
            blobs.mkdir(parents=True)
            (blobs / "a").write_bytes(b"x" * 1000)
            (blobs / "b.downloadInProgress").write_bytes(b"x" * 500)
            snapshot = root / "models/snapshots"
            snapshot.mkdir()
            (snapshot / "model.gguf").symlink_to(blobs / "a")
            (root / "server.log").write_text("loading\n")
            result = subprocess.run(
                ["sh", "-c", VAST_STARTUP_PROBE_COMMAND.replace(VAST_RUNTIME_DIR, shlex.quote(directory))],
                capture_output=True, text=True, check=True,
            )
        progress = measure_download(result.stdout, self.files)
        assert progress is not None
        self.assertEqual((progress.downloaded, progress.total, progress.active), (1500, 3000, True))

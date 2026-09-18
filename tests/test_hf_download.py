"""Shared Hugging Face download policy: defaults, overrides, and failure classes."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_launchpad.core import hf_download


class ResolveXetTransportTests(unittest.TestCase):
    def test_defaults_to_xet_high_performance(self) -> None:
        self.assertEqual(
            hf_download.resolve_xet_transport(environ={}),
            {"HF_XET_HIGH_PERFORMANCE": "1"},
        )

    def test_explicit_disable_wins_over_perf_override(self) -> None:
        env = {"HF_HUB_DISABLE_XET": "1", "HF_XET_HIGH_PERFORMANCE": "1"}
        self.assertEqual(
            hf_download.resolve_xet_transport(environ=env),
            {"HF_HUB_DISABLE_XET": "1"},
        )

    def test_explicit_perf_off_emits_no_perf_marker(self) -> None:
        self.assertEqual(
            hf_download.resolve_xet_transport(
                environ={"HF_XET_HIGH_PERFORMANCE": "0"}
            ),
            {},
        )

    def test_argument_override_beats_environment(self) -> None:
        self.assertEqual(
            hf_download.resolve_xet_transport(
                disable_xet=True,
                high_performance=True,
                environ={"HF_HUB_DISABLE_XET": ""},
            ),
            {"HF_HUB_DISABLE_XET": "1"},
        )


class HfDownloadEnvTests(unittest.TestCase):
    def test_sets_timeouts_and_perf_by_default(self) -> None:
        env = hf_download.hf_download_env(environ={})
        self.assertEqual(env["HF_HUB_ETAG_TIMEOUT"], "30")
        self.assertEqual(env["HF_HUB_DOWNLOAD_TIMEOUT"], "120")
        self.assertEqual(env["HF_XET_HIGH_PERFORMANCE"], "1")
        self.assertNotIn("HF_HUB_DISABLE_XET", env)

    def test_disable_xet_removes_perf_marker(self) -> None:
        env = hf_download.hf_download_env(
            base={"HF_XET_HIGH_PERFORMANCE": "1"},
            environ={"HF_HUB_DISABLE_XET": "true"},
        )
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", env)

    def test_explicit_timeouts_win_over_environment(self) -> None:
        env = hf_download.hf_download_env(
            etag_timeout="5",
            download_timeout="60",
            environ={"HF_HUB_ETAG_TIMEOUT": "9", "HF_HUB_DOWNLOAD_TIMEOUT": "10"},
        )
        self.assertEqual(env["HF_HUB_ETAG_TIMEOUT"], "5")
        self.assertEqual(env["HF_HUB_DOWNLOAD_TIMEOUT"], "60")


class DescribeTransportTests(unittest.TestCase):
    def test_labels_each_mode(self) -> None:
        self.assertEqual(
            hf_download.describe_transport({"HF_HUB_DISABLE_XET": "1"}),
            "http",
        )
        self.assertEqual(
            hf_download.describe_transport({"HF_XET_HIGH_PERFORMANCE": "1"}),
            "xet-high-performance",
        )
        self.assertEqual(hf_download.describe_transport({}), "xet")


class ClampMaxWorkersTests(unittest.TestCase):
    def test_coerces_and_clamps(self) -> None:
        self.assertEqual(hf_download.clamp_max_workers("32", 8), 32)
        self.assertEqual(hf_download.clamp_max_workers(0, 8), 1)
        self.assertEqual(hf_download.clamp_max_workers(10_000, 8), 64)
        self.assertEqual(hf_download.clamp_max_workers("nope", 8), 8)
        self.assertEqual(hf_download.clamp_max_workers(True, 8), 8)


class AllocatedFileSizeTests(unittest.TestCase):
    def test_missing_file_reads_as_zero(self) -> None:
        self.assertEqual(
            hf_download.allocated_file_size("/nonexistent/file.incomplete"), 0
        )

    def test_sparse_or_dense_files_never_exceed_logical_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "blob.incomplete"
            target.write_bytes(b"x" * 4096)
            size = hf_download.allocated_file_size(target)
            self.assertGreaterEqual(size, 0)
            self.assertLessEqual(size, 4096)


class ModalParityTests(unittest.TestCase):
    def test_modal_download_env_matches_shared_policy(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app, modal_vllm_app

        with mock.patch.dict(os.environ, {}, clear=True):
            shared = hf_download.resolve_xet_transport(environ={})
            llamacpp_env = modal_llamacpp_app._download_image_env()
            vllm_env = modal_vllm_app._download_image_env()
        self.assertEqual(shared, {"HF_XET_HIGH_PERFORMANCE": "1"})
        self.assertEqual(llamacpp_env.get("HF_XET_HIGH_PERFORMANCE"), "1")
        self.assertNotIn("HF_HUB_DISABLE_XET", llamacpp_env)
        self.assertEqual(vllm_env.get("HF_XET_HIGH_PERFORMANCE"), "1")
        self.assertNotIn("HF_HUB_DISABLE_XET", vllm_env)


class ClassifyDownloadFailureTests(unittest.TestCase):
    def test_auth_missing_and_space_failures_are_terminal(self) -> None:
        self.assertEqual(
            hf_download.classify_download_failure("401 Unauthorized for gated repo"),
            "auth",
        )
        self.assertEqual(
            hf_download.classify_download_failure("Revision Not Found for url"),
            "not_found",
        )
        self.assertEqual(
            hf_download.classify_download_failure("OSError: No space left on device"),
            "no_space",
        )

    def test_transport_failures_stay_retryable(self) -> None:
        self.assertIsNone(hf_download.classify_download_failure("connection reset by peer"))
        self.assertIsNone(hf_download.classify_download_failure(""))


if __name__ == "__main__":
    unittest.main()

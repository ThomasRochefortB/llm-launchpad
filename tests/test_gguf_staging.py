"""Accelerated GGUF staging shared by Prime and Vast llama.cpp runtimes."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.serving_runtime import (
    _GGUF_STAGE_PROGRAM,
    gguf_stage_args,
    gguf_stage_command,
)


class GgufStageCommandTests(unittest.TestCase):
    def test_staging_uses_xet_high_performance_and_pins_revision(self) -> None:
        setup, flag = gguf_stage_command(
            repo_id="acme/model-GGUF",
            revision="abc123",
            quant="Q4_K_M",
            dest_dir="/data/staged",
        )

        self.assertEqual(flag, ("--hf-file", "/data/staged/weights.args"))
        self.assertIn("HF_XET_HIGH_PERFORMANCE=1", setup)
        self.assertIn("snapshot_download", setup)
        self.assertIn("LLM_LAUNCHPAD_GGUF_REVISION=abc123", setup)
        self.assertIn("weights.args", setup)

    def test_staging_rejects_missing_repo_or_quant(self) -> None:
        with self.assertRaises(ValueError):
            gguf_stage_command(
                repo_id="not-a-repo", revision=None, quant="Q4_K_M", dest_dir="/tmp/x"
            )
        with self.assertRaises(ValueError):
            gguf_stage_command(
                repo_id="acme/model", revision=None, quant=" ", dest_dir="/tmp/x"
            )

    def test_stage_program_resolves_split_shards_and_emits_first_shard(self) -> None:
        self.assertIn("snapshot_download", _GGUF_STAGE_PROGRAM)
        self.assertIn("weights.args", _GGUF_STAGE_PROGRAM)
        self.assertIn("weights.list", _GGUF_STAGE_PROGRAM)
        # Every shard is staged; the server resolves siblings from the first.
        self.assertIn("allow_patterns", _GGUF_STAGE_PROGRAM)

    def test_stage_args_keep_repo_listing_with_pinned_file(self) -> None:
        args, setup = gguf_stage_args(
            repo_id="acme/model-GGUF",
            revision=None,
            quant="Q4_K_M",
            dest_dir="/data/staged",
        )

        self.assertEqual(args[:4], ["/app/llama-server", "--hf-repo", "acme/model-GGUF", "--hf-file"])
        self.assertIn("/data/staged/weights.args", args)
        self.assertIn("snapshot_download", setup)

    def test_explicit_xet_disable_is_respected(self) -> None:
        with patch.dict("os.environ", {"HF_HUB_DISABLE_XET": "1"}, clear=False):
            setup, _ = gguf_stage_command(
                repo_id="acme/model-GGUF",
                revision=None,
                quant="Q4_K_M",
                dest_dir="/data/staged",
            )
        self.assertIn("HF_HUB_DISABLE_XET=1", setup)
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", setup)


if __name__ == "__main__":
    unittest.main()

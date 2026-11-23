from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.deployment_preflight import preflight_config, resolve_dspark_draft
from llm_launchpad.core.quick_deploy import config_from_request, request_from_config
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig


class DSparkPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            repo_id="org/model-GGUF",
            quant="Q4_K_M",
            gpu_type="T4",
            gpu_count=1,
            dspark=True,
        )
        self.target = SimpleNamespace(sha="target-sha", siblings=[
            SimpleNamespace(rfilename="model-Q4_K_M.gguf", size=2_500_000_000),
            SimpleNamespace(rfilename="dspark-model-BF16.gguf", size=800_000_000),
            SimpleNamespace(rfilename="dspark-model-Q8_0.gguf", size=600_000_000),
        ])

    def test_selects_q8_draft_in_same_repository_and_pins_both(self) -> None:
        with patch("huggingface_hub.HfApi.model_info", return_value=self.target) as info:
            self.config.dspark_draft = resolve_dspark_draft(self.config)

        info.assert_called_once()
        command = ModalBackend.build_run_command(self.config)
        self.assertEqual(command[command.index("--revision") + 1], "target-sha")
        self.assertEqual(json.loads(command[command.index("--dspark-json") + 1]), {
            "repo_id": "org/model-GGUF",
            "revision": "target-sha",
            "filename": "dspark-model-Q8_0.gguf",
            "size_bytes": 600_000_000,
        })

    def test_preserves_explicit_external_draft_through_request_and_pins_it(self) -> None:
        self.config.dspark_repo = "org/draft-GGUF"
        self.config.dspark_file = "dspark-Q4_K_M.gguf"
        config = config_from_request(request_from_config(self.config))
        draft_info = SimpleNamespace(sha="draft-sha", siblings=[
            SimpleNamespace(rfilename=self.config.dspark_file, size=500_000_000),
        ])
        with patch("huggingface_hub.HfApi.model_info", side_effect=(self.target, draft_info)):
            artifact = resolve_dspark_draft(config)

        self.assertTrue(config.dspark)
        self.assertEqual(artifact.repo_id, "org/draft-GGUF")
        self.assertEqual(artifact.revision, "draft-sha")
        self.assertEqual(config.revision, "target-sha")

    def test_rejects_gpu_that_fits_main_model_but_not_draft(self) -> None:
        self.config.required_vram_gb = 15.7
        with (
            patch("huggingface_hub.HfApi.model_info", return_value=self.target),
            self.assertRaisesRegex(ValueError, "at least 16.3 GB"),
        ):
            resolve_dspark_draft(self.config)

        self.assertIsNone(self.config.revision)

    def test_rejects_unsupported_provider_before_model_lookup(self) -> None:
        self.config.provider = ComputeProvider.PRIME
        self.assertIn("dspark-provider", [finding.code for finding in preflight_config(self.config).findings])

"""Security boundaries that must hold independently of installed optional tools."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import jwt
from typer.testing import CliRunner

from llm_launchpad.cli.main import app
from llm_launchpad.core.benchmark import BENCHMARK_UNAVAILABLE_REASON
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.app import TuiApp


class BenchmarkSecurityGuards(unittest.TestCase):
    def test_cli_refuses_benchmarking_before_provider_access(self) -> None:
        with patch("llm_launchpad.cli.main._preflight") as preflight:
            result = CliRunner().invoke(app, ["benchmark"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn(BENCHMARK_UNAVAILABLE_REASON, result.output)
        preflight.assert_not_called()

    def test_tui_refuses_benchmarking_without_starting_a_worker(self) -> None:
        fake_app = SimpleNamespace(notify=Mock(), run_worker=Mock(), push_screen=Mock())
        TuiApp.begin_benchmark(fake_app, EndpointInfo())
        fake_app.notify.assert_called_once_with(
            BENCHMARK_UNAVAILABLE_REASON, severity="warning", timeout=8
        )
        fake_app.run_worker.assert_not_called()
        fake_app.push_screen.assert_not_called()


class DependencySecurityGuards(unittest.TestCase):
    def test_jwt_unverified_read_does_not_disable_later_claim_checks(self) -> None:
        key = "test-signing-key-with-at-least-32-bytes"
        token = jwt.encode({"exp": 1, "aud": "launchpad"}, key, algorithm="HS256")
        options = {"verify_signature": False}
        jwt.decode(token, algorithms=["HS256"], options=options)
        self.assertEqual(options, {"verify_signature": False})
        options["verify_signature"] = True
        with self.assertRaises(jwt.ExpiredSignatureError):
            jwt.decode(token, key, algorithms=["HS256"], audience="launchpad", options=options)

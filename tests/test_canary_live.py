"""The live canary always stops what it starts, and never uploads a secret."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "canary_live.py"
_spec = importlib.util.spec_from_file_location("canary_live", _SCRIPT)
assert _spec is not None and _spec.loader is not None
canary = importlib.util.module_from_spec(_spec)
sys.modules["canary_live"] = canary
_spec.loader.exec_module(canary)


class CanaryTests(unittest.TestCase):
    def test_a_provider_without_credentials_is_skipped_not_failed(self) -> None:
        with patch.object(canary, "_credentials_present", return_value="no Prime API key"), \
                patch.object(canary, "_run") as run:
            result = canary.run_provider("prime", run_id="r1", max_hourly_cost=0.25, max_minutes=5)
        self.assertEqual(result.status, "skipped")
        run.assert_not_called()

    def test_a_failed_deploy_is_still_stopped(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str], timeout: int) -> tuple[int, list[str]]:
            calls.append(command)
            return (1, ["boom"]) if "deploy" in command else (0, [])

        with patch.object(canary, "_credentials_present", return_value=None), \
                patch.object(canary, "_run", side_effect=fake_run):
            result = canary.run_provider("vast", run_id="r1", max_hourly_cost=0.25, max_minutes=5)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.stopped)
        stop = calls[-1]
        self.assertIn("stop", stop)
        self.assertIn(result.app_name, stop)
        self.assertIn("--yes", stop)

    def test_a_failed_stop_fails_an_otherwise_passing_run(self) -> None:
        def fake_run(command: list[str], timeout: int) -> tuple[int, list[str]]:
            return (0, []) if "deploy" in command else (1, ["still billing"])

        with patch.object(canary, "_credentials_present", return_value=None), \
                patch.object(canary, "_run", side_effect=fake_run):
            result = canary.run_provider("prime", run_id="r1", max_hourly_cost=0.25, max_minutes=5)
        self.assertEqual(result.status, "failed")
        self.assertIn("leftover rental", result.detail)

    def test_rentals_carry_an_idle_backstop_and_vast_a_price_cap(self) -> None:
        vast = canary._deploy_args("vast", "llp-canary-vast-x", 0.25)
        prime = canary._deploy_args("prime", "llp-canary-prime-x", 0.25)
        modal = canary._deploy_args("modal", "llp-canary-modal-x", 0.25)
        for args in (vast, prime):
            self.assertIn("--idle-shutdown", args)
        self.assertIn("--max-hourly-cost", vast)
        self.assertIn("--do-warmup", modal)
        self.assertNotIn("--idle-shutdown", modal)  # Modal scales to zero

    def test_report_lines_are_redacted(self) -> None:
        with patch.dict(os.environ, {"PRIME_API_KEY": "pk-live-secret-123"}):
            lines = canary._redact([
                "API key: sk-endpoint-abc",
                "Authorization: Bearer tok_xyz",
                "prime key pk-live-secret-123 in use",
            ])
        joined = "\n".join(lines)
        for secret in ("sk-endpoint-abc", "tok_xyz", "pk-live-secret-123"):
            self.assertNotIn(secret, joined)


if __name__ == "__main__":
    unittest.main()

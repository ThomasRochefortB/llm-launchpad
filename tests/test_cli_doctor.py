from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from llm_launchpad.cli.main import app
from llm_launchpad.core import doctor as doctor_module
from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.doctor import (
    DoctorCheck,
    doctor_exit_code,
    run_doctor_checks,
)
from llm_launchpad.core.hf_auth import HuggingFaceAuthStatus
from llm_launchpad.core.modal_auth import ModalAuthStatus
from llm_launchpad.core.prime_auth import PrimeAuthStatus
from llm_launchpad.core.artificial_analysis import ArtificialAnalysisAuthStatus


def _ok_checks() -> tuple[DoctorCheck, ...]:
    return (
        DoctorCheck(name="Local state directory", ok=True, detail="/tmp/state"),
        DoctorCheck(name="Modal auth", ok=True, detail="authenticated"),
        DoctorCheck(name="Prime Intellect auth", ok=True, detail="API key found"),
        DoctorCheck(name="Hugging Face auth", ok=True, detail="logged in"),
        DoctorCheck(
            name="Artificial Analysis key", ok=True, required=False, detail="validated"
        ),
        DoctorCheck(name="Debug log", ok=True, detail="/tmp/log"),
    )


class DoctorCheckLogicTests(unittest.TestCase):
    def test_exit_code_zero_when_all_required_pass(self) -> None:
        self.assertEqual(doctor_exit_code(_ok_checks()), 0)

    def test_exit_code_one_when_required_check_fails(self) -> None:
        checks = (
            DoctorCheck(name="Modal auth", ok=False, hint="run: modal setup"),
            DoctorCheck(name="Hugging Face auth", ok=True),
        )
        self.assertEqual(doctor_exit_code(checks), 1)

    def test_optional_failure_does_not_change_exit_code(self) -> None:
        checks = (
            DoctorCheck(name="Hugging Face auth", ok=True),
            DoctorCheck(
                name="Artificial Analysis key",
                ok=False,
                required=False,
                hint="optional",
            ),
        )
        self.assertEqual(doctor_exit_code(checks), 0)


class RunDoctorChecksTests(unittest.TestCase):
    def setUp(self) -> None:
        patchers = [
            mock.patch.object(ModalBackend, "is_cli_available", return_value=True),
            mock.patch.object(
                doctor_module,
                "get_modal_auth_status",
                return_value=ModalAuthStatus(authenticated=True, profile="default"),
            ),
            mock.patch.object(
                doctor_module,
                "get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=True),
            ),
            mock.patch.object(
                doctor_module,
                "get_huggingface_auth_status",
                return_value=HuggingFaceAuthStatus(authenticated=True, username="hf-user"),
            ),
            mock.patch.object(
                doctor_module,
                "get_artificial_analysis_auth_status",
                return_value=ArtificialAnalysisAuthStatus(authenticated=True),
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_all_ok_when_every_service_is_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checks = run_doctor_checks(settings_dir=Path(tmp))

        self.assertTrue(all(check.ok for check in checks))
        by_name = {check.name for check in checks}
        self.assertIn("Modal auth", by_name)
        self.assertIn("Prime Intellect auth", by_name)
        self.assertIn("Hugging Face auth", by_name)
        self.assertIn("Debug log", by_name)
        self.assertEqual(doctor_exit_code(checks), 0)

    def test_missing_modal_cli_reports_hint(self) -> None:
        with mock.patch.object(ModalBackend, "is_cli_available", return_value=False):
            checks = run_doctor_checks(settings_dir=Path(tempfile.gettempdir()))

        modal_cli = next(check for check in checks if check.name == "Modal CLI")
        self.assertFalse(modal_cli.ok)
        self.assertIn("modal setup", modal_cli.hint or "")
        self.assertEqual(doctor_exit_code(checks), 1)

    def test_unauthenticated_modal_reports_hint(self) -> None:
        with mock.patch.object(
            doctor_module,
            "get_modal_auth_status",
            return_value=ModalAuthStatus(authenticated=False),
        ):
            checks = run_doctor_checks(settings_dir=Path(tempfile.gettempdir()))

        modal_auth = next(check for check in checks if check.name == "Modal auth")
        self.assertFalse(modal_auth.ok)
        self.assertIn("modal setup", modal_auth.hint or "")
        self.assertEqual(doctor_exit_code(checks), 1)

    def test_unwritable_state_dir_fails(self) -> None:
        blocker = Path(tempfile.gettempdir()) / "llm_launchpad_doctor_blocker"
        blocker.write_text("not a directory")
        self.addCleanup(blocker.unlink, missing_ok=True)

        checks = run_doctor_checks(settings_dir=blocker / "state")

        state_check = next(check for check in checks if check.name == "Local state directory")
        self.assertFalse(state_check.ok)
        self.assertEqual(doctor_exit_code(checks), 1)

    def test_optional_aai_failure_warns_without_failing(self) -> None:
        with mock.patch.object(
            doctor_module,
            "get_artificial_analysis_auth_status",
            return_value=ArtificialAnalysisAuthStatus(
                authenticated=False, error="no API key configured"
            ),
        ):
            checks = run_doctor_checks(settings_dir=Path(tempfile.gettempdir()))

        aai = next(check for check in checks if check.name == "Artificial Analysis key")
        self.assertFalse(aai.ok)
        self.assertFalse(aai.required)
        self.assertEqual(doctor_exit_code(checks), 0)

    def test_aai_check_can_be_skipped(self) -> None:
        with mock.patch.object(
            doctor_module,
            "get_artificial_analysis_auth_status",
            side_effect=AssertionError("must not be called"),
        ):
            checks = run_doctor_checks(
                settings_dir=Path(tempfile.gettempdir()),
                check_artificial_analysis=False,
            )

        self.assertFalse(any(check.name == "Artificial Analysis key" for check in checks))


class DoctorCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_cli_exits_zero_and_lists_checks_when_healthy(self) -> None:
        with mock.patch.object(
            doctor_module, "run_doctor_checks", return_value=_ok_checks()
        ):
            result = self.runner.invoke(app, ["doctor"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("[ok] Modal auth", result.output)
        self.assertIn("[ok] Hugging Face auth", result.output)
        self.assertIn("Debug log:", result.output)

    def test_cli_exits_nonzero_and_prints_hint_when_required_check_fails(self) -> None:
        checks = tuple(
            replace(
                check,
                ok=check.name != "Modal auth",
                detail="not authenticated"
                if check.name == "Modal auth"
                else check.detail,
                hint="run: modal setup" if check.name == "Modal auth" else check.hint,
            )
            for check in _ok_checks()
        )
        with mock.patch.object(
            doctor_module, "run_doctor_checks", return_value=checks
        ):
            result = self.runner.invoke(app, ["doctor"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("[FAIL] Modal auth", result.output)
        self.assertIn("fix: run: modal setup", result.output)

    def test_cli_marks_optional_failure_as_warn(self) -> None:
        checks = tuple(
            replace(
                check,
                ok=check.name != "Artificial Analysis key",
                hint="optional: run: llm-launchpad aai-auth login"
                if check.name == "Artificial Analysis key"
                else check.hint,
            )
            for check in _ok_checks()
        )
        with mock.patch.object(
            doctor_module, "run_doctor_checks", return_value=checks
        ):
            result = self.runner.invoke(app, ["doctor"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("[warn] Artificial Analysis key", result.output)


if __name__ == "__main__":
    unittest.main()

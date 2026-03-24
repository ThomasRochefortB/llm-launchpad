from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_launchpad.core.modal_auth import (
    ModalAuthStatus,
    _modal_cli_path,
    get_modal_auth_status,
    get_modal_profile,
)


class ModalAuthTests(unittest.TestCase):
    def test_modal_cli_path_prefers_active_env_scripts_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.makedirs(bin_dir, exist_ok=True)
            modal_path = os.path.join(bin_dir, "modal")
            with open(modal_path, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n")
            with (
                patch("llm_launchpad.core.modal_auth.sys.prefix", tmp),
                patch("llm_launchpad.core.modal_auth.shutil.which", return_value=None),
            ):
                self.assertEqual(_modal_cli_path(), modal_path)

    @patch("llm_launchpad.core.modal_auth._modal_cli_path", return_value=None)
    def test_get_modal_auth_status_reports_missing_cli(self, _mock_modal_path) -> None:
        status = get_modal_auth_status()
        self.assertEqual(
            status,
            ModalAuthStatus(
                authenticated=False,
                error="Modal CLI not found (install with: pip install modal)",
            ),
        )

    @patch("llm_launchpad.core.modal_auth.shutil.which", return_value="/usr/bin/modal")
    @patch("llm_launchpad.core.modal_auth.subprocess.run")
    def test_get_modal_profile_returns_current_profile(self, mock_run, _mock_which) -> None:  # type: ignore[no-untyped-def]
        mock_run.return_value = subprocess.CompletedProcess(
            args=["modal", "profile", "current"],
            returncode=0,
            stdout="default\n",
            stderr="",
        )
        self.assertEqual(get_modal_profile(), "default")

    @patch("llm_launchpad.core.modal_auth.shutil.which", return_value="/usr/bin/modal")
    @patch("llm_launchpad.core.modal_auth.subprocess.run")
    def test_get_modal_auth_status_returns_authenticated_status(self, mock_run, _mock_which) -> None:  # type: ignore[no-untyped-def]
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["modal", "profile", "current"],
                returncode=0,
                stdout="default\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["modal", "token", "info"],
                returncode=0,
                stdout="Authenticated to workspace acme\nToken ID: tk-123\n",
                stderr="",
            ),
        ]

        status = get_modal_auth_status()
        self.assertTrue(status.authenticated)
        self.assertEqual(status.profile, "default")
        self.assertEqual(status.detail, "Authenticated to workspace acme")

    @patch("llm_launchpad.core.modal_auth.shutil.which", return_value="/usr/bin/modal")
    @patch("llm_launchpad.core.modal_auth.subprocess.run")
    def test_get_modal_auth_status_treats_setup_hint_as_unauthenticated(self, mock_run, _mock_which) -> None:  # type: ignore[no-untyped-def]
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["modal", "profile", "current"],
                returncode=0,
                stdout="default\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["modal", "token", "info"],
                returncode=1,
                stdout="",
                stderr="Run `modal setup` to authenticate.",
            ),
        ]

        status = get_modal_auth_status()
        self.assertFalse(status.authenticated)
        self.assertEqual(status.profile, "default")
        self.assertIsNone(status.error)

    @patch("llm_launchpad.core.modal_auth.shutil.which", return_value="/usr/bin/modal")
    @patch("llm_launchpad.core.modal_auth.subprocess.run")
    def test_get_modal_auth_status_surfaces_auth_service_errors(self, mock_run, _mock_which) -> None:  # type: ignore[no-untyped-def]
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["modal", "profile", "current"],
                returncode=0,
                stdout="default\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["modal", "token", "info"],
                returncode=2,
                stdout="",
                stderr="Authentication service unavailable.",
            ),
        ]

        status = get_modal_auth_status()
        self.assertFalse(status.authenticated)
        self.assertEqual(status.profile, "default")
        self.assertEqual(status.error, "Authentication service unavailable.")

    @patch("llm_launchpad.core.modal_auth.shutil.which", return_value="/usr/bin/modal")
    @patch("llm_launchpad.core.modal_auth.subprocess.run")
    def test_get_modal_auth_status_surfaces_unexpected_errors(self, mock_run, _mock_which) -> None:  # type: ignore[no-untyped-def]
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["modal", "profile", "current"],
                returncode=0,
                stdout="default\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["modal", "token", "info"],
                returncode=2,
                stdout="",
                stderr="No such command 'token'.",
            ),
        ]

        status = get_modal_auth_status()
        self.assertFalse(status.authenticated)
        self.assertEqual(status.profile, "default")
        self.assertEqual(status.error, "No such command 'token'.")

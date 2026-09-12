from __future__ import annotations

import unittest
from unittest import mock

from textual.app import App
from typer.testing import CliRunner

from llm_launchpad.cli.main import app as cli_app
from llm_launchpad.tui.app import TuiApp


class TuiCrashLoggingTests(unittest.TestCase):
    """A crash that kills the TUI has to leave something to read.

    Textual renders the traceback to a terminal it is about to restore and lets
    run() return normally, so neither a try/except around run() nor
    sys.excepthook observes it.
    """

    def test_an_unhandled_exception_is_logged_with_its_type(self) -> None:
        error = KeyError("No quick deploy profile supplies recipe 'synthetic'.")
        with mock.patch("llm_launchpad.tui.app.log_exception") as logged, mock.patch.object(
            App, "_handle_exception"
        ) as parent:
            TuiApp(mouse_enabled=False)._handle_exception(error)
        logged.assert_called_once()
        self.assertIn("KeyError", logged.call_args.args[0])
        # Textual still owns rendering and teardown.
        parent.assert_called_once_with(error)


class TuiExitCodeTests(unittest.TestCase):
    """Reporting success after a crash hides it from anything scripting the CLI."""

    def setUp(self) -> None:
        self.runner = CliRunner()

    def _run_with(self, return_code: int | None) -> int:
        instance = mock.MagicMock()
        instance.mouse_enabled = False
        instance.return_code = return_code
        with mock.patch("llm_launchpad.tui.app.TuiApp", return_value=instance), mock.patch(
            "llm_launchpad.cli.main._ensure_tui_runtime"
        ), mock.patch("llm_launchpad.core.backend.ModalBackend.terminate_all"):
            return self.runner.invoke(cli_app, ["tui"]).exit_code

    def test_a_crashed_session_exits_non_zero(self) -> None:
        self.assertEqual(self._run_with(1), 1)

    def test_a_clean_session_still_exits_zero(self) -> None:
        for code in (0, None):
            with self.subTest(return_code=code):
                self.assertEqual(self._run_with(code), 0)


if __name__ == "__main__":
    unittest.main()

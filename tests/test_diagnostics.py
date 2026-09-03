from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_launchpad.core import diagnostics


class DiagnosticsSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate each test from logging configured by other suites.
        self._restore_handlers = list(diagnostics.logging.getLogger(diagnostics._LOGGER_NAME).handlers)
        self._restore_level = diagnostics.logging.getLogger(diagnostics._LOGGER_NAME).level
        self._restore_flag = diagnostics._configured
        self._reset_logger()

    def tearDown(self) -> None:
        logger = diagnostics.logging.getLogger(diagnostics._LOGGER_NAME)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        for handler in self._restore_handlers:
            logger.addHandler(handler)
        logger.setLevel(self._restore_level)
        diagnostics._configured = self._restore_flag

    @staticmethod
    def _reset_logger() -> None:
        logger = diagnostics.logging.getLogger(diagnostics._LOGGER_NAME)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        diagnostics._configured = False

    def test_setup_creates_handler_lazily_in_target_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            result = diagnostics.setup_logging(log_dir=log_dir)

            self.assertEqual(result, log_dir / "llm_launchpad.log")
            self.assertFalse(result.exists(), "log file must only appear on first record")
            diagnostics.log_debug("first record")

            self.assertTrue(result.exists())
            content = result.read_text(encoding="utf-8")
            self.assertIn("first record", content)
            self.assertIn("DEBUG", content)

    def test_setup_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            first = diagnostics.setup_logging(log_dir=log_dir)
            second = diagnostics.setup_logging(log_dir=Path(tmp) / "elsewhere")

            logger = logging.getLogger(diagnostics._LOGGER_NAME)
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(first, second)

    def test_setup_survives_unwritable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory")
            log_dir = blocker / "logs"

            result = diagnostics.setup_logging(log_dir=log_dir)

            self.assertIsNone(result)
            # Logging calls must not raise after fallback.
            diagnostics.log_debug("ignored record")

    def test_log_file_path_reports_active_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(diagnostics.log_file_path())
            diagnostics.setup_logging(log_dir=Path(tmp))
            self.assertEqual(diagnostics.log_file_path(), Path(tmp) / "llm_launchpad.log")

    def test_log_exception_includes_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            diagnostics.setup_logging(log_dir=Path(tmp))
            try:
                raise ValueError("boom")
            except ValueError:
                diagnostics.log_exception("context: operation failed")

            content = (Path(tmp) / "llm_launchpad.log").read_text(encoding="utf-8")
            self.assertIn("context: operation failed", content)
            self.assertIn("ValueError: boom", content)
            self.assertIn("Traceback", content)

    def test_default_log_dir_sits_under_the_settings_dir(self) -> None:
        # Asserted relative to SETTINGS_DIR rather than to an absolute home
        # path: the suite redirects the settings root so no test can write the
        # real one, and where that root lives is config's invariant, not this
        # module's. See tests/test_settings_isolation.py.
        from llm_launchpad.core.config import SETTINGS_DIR

        self.assertEqual(diagnostics.LOG_DIR, SETTINGS_DIR / "logs")
        self.assertEqual(diagnostics.LOG_FILE, diagnostics.LOG_DIR / "llm_launchpad.log")

    def test_setup_handles_mkdir_failure(self) -> None:
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            result = diagnostics.setup_logging(log_dir=Path("/tmp/opencode/never"))

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

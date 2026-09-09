from __future__ import annotations

import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core.vast_auth import (
    VastCredentials, clear_vast_api_key, resolve_vast_credentials, save_vast_api_key,
)


class VastAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.saved = self.root / "launchpad.json"
        self.cli_key = self.root / "vastai" / "vast_api_key"
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def resolve(self) -> VastCredentials:
        return resolve_vast_credentials(path=self.saved, cli_path=self.cli_key)

    def test_missing_keys_are_not_configured(self) -> None:
        self.assertEqual(self.resolve(), VastCredentials())

    def test_environment_overrides_saved_and_cli_keys(self) -> None:
        save_vast_api_key("stored-secret", self.saved)
        with patch.dict(os.environ, {"VAST_API_KEY": "environment-secret"}):
            self.assertEqual(self.resolve(), VastCredentials("environment-secret", "environment"))

    def test_saved_key_overrides_cli_and_logout_restores_cli(self) -> None:
        self.cli_key.parent.mkdir()
        self.cli_key.write_text("cli-secret\n")
        save_vast_api_key("stored-secret", self.saved)
        self.assertEqual(self.resolve().source, "stored")
        self.assertTrue(clear_vast_api_key(self.saved))
        self.assertEqual(self.resolve(), VastCredentials("cli-secret", "Vast CLI"))
        self.assertEqual(self.cli_key.read_text(), "cli-secret\n")
        self.assertFalse(clear_vast_api_key(self.saved))

    def test_xdg_cli_key_path_is_supported_without_mutating_it(self) -> None:
        self.cli_key.parent.mkdir()
        self.cli_key.write_text("cli-secret")
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root)}):
            result = resolve_vast_credentials(path=self.saved)
        self.assertEqual(result.source, "Vast CLI")

    def test_saved_key_is_owner_only_and_repr_never_contains_it(self) -> None:
        save_vast_api_key("top-secret", self.saved)
        self.assertEqual(stat.S_IMODE(self.saved.stat().st_mode), 0o600)
        self.assertNotIn("top-secret", repr(self.resolve()))
        self.assertFalse(list(self.root.glob(".vast-auth-*")))

    def test_corrupt_saved_file_does_not_silently_fall_back(self) -> None:
        for content in ("not-json", "null", "[]", '{"api_key": 123}', '{"api_key": ""}'):
            with self.subTest(content=content):
                self.saved.write_text(content)
                with self.assertRaises(ValueError):
                    self.resolve()

    def test_invalid_key_does_not_replace_existing_key(self) -> None:
        save_vast_api_key("existing", self.saved)
        for key in ("", "key with spaces", "bad\r\nheader", "non-ascii-é"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                save_vast_api_key(key, self.saved)
        self.assertEqual(self.resolve().api_key, "existing")

    def test_failed_replace_cleans_up_temporary_key_and_keeps_old_key(self) -> None:
        save_vast_api_key("existing", self.saved)
        with patch.object(Path, "replace", side_effect=OSError("read-only")):
            with self.assertRaises(OSError):
                save_vast_api_key("replacement", self.saved)
        self.assertEqual(self.resolve().api_key, "existing")
        self.assertFalse(list(self.root.glob(".vast-auth-*")))

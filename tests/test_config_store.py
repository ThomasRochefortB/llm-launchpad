from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_launchpad.core.config import ConfigStore


class ConfigStoreTests(unittest.TestCase):
    def test_load_missing_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            settings = ConfigStore(path=path).load()
        self.assertEqual(settings.scaledown_window, 1800)

    def test_load_invalid_json_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{not-json")
            settings = ConfigStore(path=path).load()
        self.assertEqual(settings.scaledown_window, 1800)

    def test_save_then_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "settings.json"
            store = ConfigStore(path=path)
            settings = store.load()
            settings.scaledown_window = 900

            store.save(settings)
            loaded = store.load()

        self.assertEqual(loaded.scaledown_window, 900)


if __name__ == "__main__":
    unittest.main()

"""Guard: no test may touch the real ~/.llm_launchpad directory.

These caches hold live product state -- OpenCode registrations, deployment
connection summaries, Artificial Analysis credentials, Prime SSH key material,
and the Fast Deploy catalog. Reading one makes a test depend on the developer's
machine; writing one edits the installed product. Both have happened, so this
enumerates every SETTINGS_DIR-derived path and fails when one is not isolated.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
import unittest

import llm_launchpad

# Bound at import time, before the isolation fixture runs, so this is the real
# ~/.llm_launchpad. That is deliberate: it is the location leaks are measured
# against. Read config.SETTINGS_DIR at call time to see the redirected value.
from llm_launchpad.core.config import SETTINGS_DIR as REAL_SETTINGS_DIR


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _settings_paths() -> dict[str, Path]:
    """Return every module-level Path still pointing inside the real settings dir."""

    found: dict[str, Path] = {}
    for info in pkgutil.walk_packages(
        llm_launchpad.__path__, prefix=f"{llm_launchpad.__name__}."
    ):
        try:
            module = importlib.import_module(info.name)
        except Exception:
            # A module that cannot import here cannot leak a path either.
            continue
        for name, value in vars(module).items():
            if isinstance(value, Path) and _is_under(value, REAL_SETTINGS_DIR):
                found[f"{info.name}.{name}"] = value
    return found


class SettingsIsolationTests(unittest.TestCase):
    def test_no_module_path_resolves_into_the_real_settings_dir(self) -> None:
        leaked = _settings_paths()

        self.assertEqual(
            leaked,
            {},
            "These paths still resolve into the real ~/.llm_launchpad and would "
            "let the suite read or overwrite live user state. Add each one to "
            "_ISOLATED_SETTINGS_PATHS in tests/conftest.py:\n  "
            + "\n  ".join(sorted(leaked)),
        )

    def test_the_settings_root_itself_is_redirected(self) -> None:
        # config.SETTINGS_DIR is where the shipped product keeps user state;
        # under test it must never resolve to the developer's real directory.
        from llm_launchpad.core import config

        self.assertNotEqual(config.SETTINGS_DIR, REAL_SETTINGS_DIR)

    def test_the_guard_can_actually_see_an_unisolated_path(self) -> None:
        # A guard that cannot fail is not a guard: confirm the detector matches
        # a path shaped like the ones it is meant to catch.
        self.assertTrue(
            _is_under(REAL_SETTINGS_DIR / "leaked.json", REAL_SETTINGS_DIR)
        )
        self.assertFalse(_is_under(Path("/tmp/elsewhere.json"), REAL_SETTINGS_DIR))


if __name__ == "__main__":
    unittest.main()

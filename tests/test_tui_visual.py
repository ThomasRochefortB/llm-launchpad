from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.protocol.models import LaunchpadSettings
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.visual import (
    DEFAULT_TUI_DENSITY,
    DEFAULT_TUI_THEME,
    normalize_tui_density,
    normalize_tui_theme,
)


class VisualPreferenceTests(unittest.TestCase):
    def test_invalid_preferences_fall_back_to_accessible_defaults(self) -> None:
        self.assertEqual(normalize_tui_theme("unknown"), DEFAULT_TUI_THEME)
        self.assertEqual(normalize_tui_density("huge"), DEFAULT_TUI_DENSITY)

    def test_app_registers_and_switches_launchpad_themes(self) -> None:
        settings = LaunchpadSettings(
            tui_theme="launchpad-high-contrast",
            tui_density="compact",
        )
        with patch("llm_launchpad.tui.app.ConfigStore.load", return_value=settings):
            app = TuiApp()

        self.assertEqual(app.theme, "launchpad-high-contrast")
        self.assertEqual(app.tui_density, "compact")
        self.assertFalse(app._monochrome_filter.enabled)

        app.apply_visual_preferences("launchpad-monochrome", "comfortable")

        self.assertEqual(app.theme, "launchpad-monochrome")
        self.assertEqual(app.tui_density, "comfortable")
        self.assertTrue(app._monochrome_filter.enabled)


if __name__ == "__main__":
    unittest.main()

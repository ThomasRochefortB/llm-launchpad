from __future__ import annotations

import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import Input, Select, Static

from llm_launchpad.tui.screens.settings import (
    SettingsScreen,
    _build_gpu_type_options,
    _parse_gpu_config,
)


class _TestApp(App[None]):
    pass


class SettingsScreenTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_gpu_config(self) -> None:
        self.assertEqual(_parse_gpu_config("A100-80GB:4"), ("A100-80GB", 4))
        self.assertEqual(_parse_gpu_config("h100"), ("H100", 1))
        self.assertEqual(_parse_gpu_config("H100:invalid"), ("H100", 1))

    def test_build_gpu_type_options_normalizes_and_deduplicates(self) -> None:
        options = _build_gpu_type_options(["h100", "A100-80GB", "H100"])
        self.assertEqual(options, ["H100", "A100-80GB"])

    async def test_gpu_types_populate_dropdown_and_save_type_and_count(self) -> None:
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.settings.fetch_modal_gpu_types",
            return_value=["H100", "A100-80GB"],
        ):
            async with app.run_test() as pilot:
                app.push_screen(SettingsScreen())
                await pilot.pause()
                for _ in range(5):
                    await pilot.pause()

                screen = app.screen
                assert isinstance(screen, SettingsScreen)

                dropdown = screen.query_one("#gpu-type-dropdown", Select)
                self.assertEqual(dropdown.value, "A100-80GB")

                screen.on_select_changed(Select.Changed(dropdown, "H100"))
                screen.query_one("#gpu-count", Input).value = "3"

                saved: dict[str, object] = {}

                def fake_save(settings: object) -> None:
                    saved["settings"] = settings

                screen._store.save = fake_save  # type: ignore[method-assign]
                screen._save()
                saved_settings = saved.get("settings")
                self.assertIsNotNone(saved_settings)
                assert saved_settings is not None
                self.assertEqual(saved_settings.gpu_config, "H100:3")

                status = str(screen.query_one("#gpu-types-status", Static).content)
                self.assertIn("GPU type list ready", status)

    async def test_gpu_types_failure_keeps_existing_selection(self) -> None:
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.settings.fetch_modal_gpu_types",
            side_effect=RuntimeError("network unavailable"),
        ):
            async with app.run_test() as pilot:
                app.push_screen(SettingsScreen())
                await pilot.pause()
                for _ in range(5):
                    await pilot.pause()

                screen = app.screen
                assert isinstance(screen, SettingsScreen)

                dropdown = screen.query_one("#gpu-type-dropdown", Select)
                self.assertEqual(dropdown.value, "A100-80GB")

                status = str(screen.query_one("#gpu-types-status", Static).content)
                self.assertIn("Could not load GPU types", status)


if __name__ == "__main__":
    unittest.main()

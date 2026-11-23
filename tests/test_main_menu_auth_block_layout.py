from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Footer, Static

from llm_launchpad.core.provider_readiness import ProviderReadiness, ProviderReadinessStage
from llm_launchpad.protocol.enums import ComputeProvider
from llm_launchpad.tui import app as tui_app_module
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.main_menu import MainMenuScreen

_THEME_CSS = Path(tui_app_module.__file__).with_name("theme.tcss")


class _ThemedTuiApp(TuiApp):
    CSS_PATH = _THEME_CSS

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen(username="alice", version="1.0.0"))


class MainMenuAuthBlockLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_home_shows_only_a_single_actionable_provider_line(self) -> None:
        app = _ThemedTuiApp()
        with (
            patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_prime_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_hf_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_aai_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_panels", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_quick_deploy_catalog", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_storage_estimate", lambda self: None),
        ):
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                block = screen.query_one("#auth-status-block", Static)
                self.assertFalse(block.display)
                screen.on_provider_readiness((
                    ProviderReadiness(ComputeProvider.MODAL, ProviderReadinessStage.READY),
                    ProviderReadiness(ComputeProvider.PRIME, ProviderReadinessStage.AUTH_FAILED),
                    ProviderReadiness(ComputeProvider.VAST, ProviderReadinessStage.MISSING_CREDENTIALS),
                ))
                await pilot.pause()

                self.assertEqual(block.region.height, 1)
                self.assertLess(block.region.bottom, screen.query_one(Footer).region.y)
                text = str(block.content)
                self.assertIn("Check Prime Intellect", text)
                self.assertNotIn("Modal", text)
                self.assertNotIn("Vast.ai", text)


if __name__ == "__main__":
    unittest.main()

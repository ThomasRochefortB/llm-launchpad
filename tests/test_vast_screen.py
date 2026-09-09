from __future__ import annotations

import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import Button, DataTable, Input, Static

from llm_launchpad.core.vast_auth import VastCredentials
from llm_launchpad.core.vast_backend import parse_vast_offer
from llm_launchpad.protocol.models import VastAuthStatus, VastOfferQuery
from llm_launchpad.tui.screens.settings import SettingsScreen
from llm_launchpad.tui.screens.setup import SetupRequiredScreen
from llm_launchpad.tui.screens.vast import VastPreviewScreen
from tests.test_vast_backend import offer_payload


class VastScreenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch("llm_launchpad.tui.screens.vast.resolve_vast_credentials", return_value=VastCredentials())
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_preview_is_accessible_from_settings_and_setup(self) -> None:
        for origin, button in ((SettingsScreen, "vast-preview-btn"), (SetupRequiredScreen, "setup-vast-preview-btn")):
            app = App()
            async with app.run_test() as pilot:
                await app.push_screen(origin())
                app.screen.query_one(f"#{button}", Button).press()
                await pilot.pause()
                self.assertIsInstance(app.screen, VastPreviewScreen)
                await pilot.press("escape")
                self.assertIsInstance(app.screen, origin)

    async def test_login_is_masked_and_success_is_shown(self) -> None:
        with patch("llm_launchpad.tui.screens.vast.VastBackend.auth_status", return_value=VastAuthStatus(True, account_id="12")), patch(
            "llm_launchpad.tui.screens.vast.save_vast_api_key"
        ) as save:
            app = App()
            async with app.run_test() as pilot:
                screen = VastPreviewScreen()
                await app.push_screen(screen)
                key = screen.query_one("#vast-key", Input)
                self.assertTrue(key.password)
                key.value = "private-key"
                screen.query_one("#vast-connect", Button).press()
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(key.value, "")
                self.assertIn("12", str(screen.query_one("#vast-feedback", Static).content))
                save.assert_called_once_with("private-key")

    async def test_offer_table_preserves_unknown_and_fractional_rates(self) -> None:
        row = parse_vast_offer(offer_payload(inet_up_cost=None), VastOfferQuery())
        with patch("llm_launchpad.tui.screens.vast.VastBackend.list_offers", return_value=[row]):
            app = App()
            async with app.run_test() as pilot:
                screen = VastPreviewScreen()
                await app.push_screen(screen)
                screen.query_one("#vast-refresh", Button).press()
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                table = screen.query_one("#vast-offers", DataTable)
                self.assertEqual(table.row_count, 1)
                cells = table.get_row("1001")
                self.assertIn("unknown", cells)
                self.assertIn("$0.0020", cells)
                self.assertFalse(screen.query_one("#vast-refresh", Button).disabled)

    async def test_failed_login_does_not_save_and_ui_recovers(self) -> None:
        with patch("llm_launchpad.tui.screens.vast.VastBackend.auth_status", return_value=VastAuthStatus(False, error="Denied")), patch(
            "llm_launchpad.tui.screens.vast.save_vast_api_key"
        ) as save:
            app = App()
            async with app.run_test() as pilot:
                screen = VastPreviewScreen()
                await app.push_screen(screen)
                screen.query_one("#vast-key", Input).value = "bad-key"
                screen.query_one("#vast-connect", Button).press()
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                save.assert_not_called()
                self.assertIn("Denied", str(screen.query_one("#vast-feedback", Static).content))
                self.assertFalse(screen.query_one("#vast-connect", Button).disabled)

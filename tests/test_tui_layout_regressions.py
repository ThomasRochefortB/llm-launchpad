"""Regressions from the visual and programmatic TUI review.

Each test here pins a defect that the rest of the suite could not see, either
because it lived in a rendered cell rather than in a widget's attributes, or
because the responsive checks deliberately exempt scrolling content.
"""

from __future__ import annotations

import re
import unicodedata
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from textual.app import App
from textual.geometry import Region
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets._footer import FooterKey

from llm_launchpad.protocol.models import StorageSnapshot
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens import main_menu as main_menu_module
from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.screens.monitor import _connection_card_markup, _result_card_markup
from llm_launchpad.tui.screens.settings import SettingsScreen
from llm_launchpad.tui.screens.storage import StorageScreen
from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
from llm_launchpad.tui.widgets.fitted_footer import FittedFooter
from llm_launchpad.tui.widgets.help_overlay import HelpOverlayScreen, _iter_screen_bindings
from llm_launchpad.tui.widgets.status_header import _STATE_MARKERS, _state_icon
from llm_launchpad.tui.workers import StorageLoaded
from tests.test_storage_screen import _sample_snapshot


class _StyledApp(App[None]):
    CSS_PATH = TuiApp.CSS_PATH


class _ReviewApp(TuiApp):
    """The real app, minus the main menu it would otherwise push on mount.

    Footer contents and the help overlay both depend on app-level bindings, so
    a bare ``App`` would test a footer and a shortcut list this product never
    shows.
    """

    def on_mount(self) -> None:
        return None


def _rendered_lines(app: App[object]) -> list[str]:
    """The screen exactly as the terminal would receive it."""
    strips = list(app.screen._compositor.render_strips())
    return [strip.text.rstrip() for strip in strips[: app.screen.size.height]]


def _is_drawn(app: App[object], widget: object) -> bool:
    """Whether the widget's cells actually reach the terminal.

    Region containment is not enough: a widget parked in a scroll container's
    own bottom padding is inside the screen by that measure and still never
    rendered.
    """
    region = app.screen.find_widget(widget).region
    viewport = Region(0, 0, *app.screen.size)
    if region.area == 0 or not viewport.contains_region(region):
        return False
    lines = _rendered_lines(app)
    return any(line.strip() for line in lines[region.y : region.bottom])


class ScrollbarGutterTests(unittest.IsolatedAsyncioTestCase):
    """`scrollbar-gutter: stable` on `*` taxed every widget a row and a column."""

    async def test_option_list_does_not_reserve_a_row_it_cannot_scroll(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(80, 24)) as pilot:
            option_list = OptionList("one", "two", "three")
            await app.screen.mount(option_list)
            await pilot.pause()
            # Three options plus the border, and nothing else.
            self.assertEqual(option_list.scrollable_content_region.height, 3)

    async def test_widgets_are_not_short_a_column_for_an_unused_scrollbar(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(80, 24)) as pilot:
            option_list = OptionList("one")
            await app.screen.mount(option_list)
            await pilot.pause()
            self.assertEqual(option_list.region.width, 80)

    async def test_footer_keeps_its_own_gutter_exemption(self) -> None:
        """The footer's single content row must never become a scrollbar."""
        app = _StyledApp()
        async with app.run_test(size=(80, 24)) as pilot:
            footer = FittedFooter()
            await app.screen.mount(footer)
            await pilot.pause()
            self.assertFalse(footer.show_horizontal_scrollbar)
            self.assertGreater(footer.content_size.height, 0)


class StorageInventoryTests(unittest.IsolatedAsyncioTestCase):
    """The inventory rendered zero rows -- not even its header -- at 80x24."""

    def _push(self, app: App[object]) -> StorageScreen:
        screen = StorageScreen()
        app.push_screen(screen)
        return screen

    async def test_inventory_shows_its_rows_on_a_small_terminal(self) -> None:
        for size in ((140, 45), (100, 30), (80, 24)):
            with self.subTest(size=size):
                app = _StyledApp()
                with patch.object(StorageScreen, "_refresh_storage_snapshot", lambda self: None):
                    async with app.run_test(size=size) as pilot:
                        screen = self._push(app)
                        await pilot.pause()
                        screen.on_storage_loaded(StorageLoaded(_sample_snapshot()))
                        await pilot.pause()

                        table = screen.query_one("#storage-table", AdaptiveDataTable)
                        self.assertGreater(table.row_count, 0)
                        # Header plus at least one data row have to be drawn.
                        self.assertGreaterEqual(
                            table.scrollable_content_region.height, 2
                        )
                        lines = "\n".join(_rendered_lines(app))
                        self.assertIn("backend", lines)
                        self.assertIn("Qwen", lines)

    async def test_an_empty_inventory_says_why_it_is_empty(self) -> None:
        app = _StyledApp()
        with patch.object(StorageScreen, "_refresh_storage_snapshot", lambda self: None):
            async with app.run_test(size=(100, 30)) as pilot:
                screen = self._push(app)
                await pilot.pause()
                screen.on_storage_loaded(
                    StorageLoaded(StorageSnapshot(llamacpp_models=[], vllm_models=[]))
                )
                await pilot.pause()

                empty = screen.query_one("#storage-empty", Static)
                self.assertFalse(empty.has_class("hidden"))
                self.assertIn("No models cached yet", str(empty.render()))

    async def test_a_filter_that_matches_nothing_is_not_reported_as_empty(self) -> None:
        app = _StyledApp()
        with patch.object(StorageScreen, "_refresh_storage_snapshot", lambda self: None):
            async with app.run_test(size=(100, 30)) as pilot:
                screen = self._push(app)
                await pilot.pause()
                screen.on_storage_loaded(StorageLoaded(_sample_snapshot()))
                await pilot.pause()
                screen.query_one("#storage-filter", Input).value = "no-such-model"
                await pilot.pause()

                empty = screen.query_one("#storage-empty", Static)
                self.assertFalse(empty.has_class("hidden"))
                self.assertIn("matches the current filter", str(empty.render()))


class SettingsFeedbackTests(unittest.IsolatedAsyncioTestCase):
    """Save, and every message about saving, sat permanently below the fold.

    `_assert_screen_families_fit_supported_viewports` skips vertical checks for
    anything inside a VerticalScroll, so nothing caught this.
    """

    async def test_a_rejected_save_puts_its_reason_on_screen(self) -> None:
        for size in ((140, 45), (120, 40), (100, 30), (80, 24)):
            with self.subTest(size=size):
                app = _StyledApp()
                async with app.run_test(size=size) as pilot:
                    screen = SettingsScreen()
                    app.push_screen(screen)
                    await pilot.pause()
                    screen.query_one("#scaledown-window", Input).value = "not-a-number"
                    await pilot.press("ctrl+s")
                    await pilot.pause()

                    feedback = screen.query_one("#save-feedback", Static)
                    self.assertIn("must be an integer", str(feedback.render()))
                    self.assertIn(
                        "Scaledown must be an integer",
                        "\n".join(_rendered_lines(app)),
                        f"save feedback never reached the terminal at {size}",
                    )

    async def test_the_discard_prompt_reaches_the_reader(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen._accepting_edits = True
            screen.query_one("#scaledown-window", Input).value = "123"
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            self.assertIsInstance(app.screen, SettingsScreen)
            self.assertIn("Press esc again to discard", "\n".join(_rendered_lines(app)))

    async def test_the_save_button_is_reachable_without_scrolling_on_a_full_screen(
        self,
    ) -> None:
        app = _StyledApp()
        async with app.run_test(size=(140, 45)) as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()
            self.assertTrue(_is_drawn(app, screen.query_one("#save-btn", Button)))


class FooterFitTests(unittest.IsolatedAsyncioTestCase):
    """Hints were clipped mid-word: `i Details` became `i D`."""

    async def _menu(self, pilot, app) -> MainMenuScreen:
        screen = MainMenuScreen(username="review")
        with patch.object(MainMenuScreen, "on_mount", lambda self: None):
            app.push_screen(screen)
            await pilot.pause()
        return screen

    async def test_no_hint_is_rendered_half_way_through(self) -> None:
        for width in (80, 100, 120, 140, 200):
            with self.subTest(width=width):
                app = _ReviewApp()
                async with app.run_test(size=(width, 30)) as pilot:
                    screen = await self._menu(pilot, app)
                    footer = screen.query_one(FittedFooter)
                    for key in footer.query(FooterKey):
                        self.assertLessEqual(
                            key.region.right,
                            width,
                            f"{key.description!r} is clipped at {width} columns",
                        )

    async def test_dropped_hints_are_declared_rather_than_silently_cut(self) -> None:
        app = _ReviewApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await self._menu(pilot, app)
            rendered = "\n".join(_rendered_lines(app))
            self.assertIn("more (?)", rendered)
            # The shortcut that opens the full list must itself survive.
            self.assertIn("? Help", rendered)

    async def test_every_hint_is_shown_when_they_all_fit(self) -> None:
        app = _ReviewApp()
        async with app.run_test(size=(200, 30)) as pilot:
            await self._menu(pilot, app)
            rendered = "\n".join(_rendered_lines(app))
            self.assertIn("i Details", rendered)
            self.assertNotIn("more (?)", rendered)

    async def test_widening_the_terminal_restores_the_dropped_hints(self) -> None:
        app = _ReviewApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await self._menu(pilot, app)
            self.assertNotIn("i Details", "\n".join(_rendered_lines(app)))

            await pilot.resize_terminal(200, 30)
            await pilot.pause()
            await pilot.pause()
            self.assertIn("i Details", "\n".join(_rendered_lines(app)))


class ProviderMarkerTests(unittest.TestCase):
    """A width-2 emoji among width-1 markers knocked one row out of column."""

    MARKERS = (
        main_menu_module.MODAL_MARKER,
        main_menu_module.PRIME_MARKER,
        main_menu_module.VAST_MARKER,
        main_menu_module.HUGGINGFACE_MARKER,
        main_menu_module.ARTIFICIAL_ANALYSIS_MARKER,
    )

    def test_every_marker_is_one_unambiguous_cell(self) -> None:
        for marker in self.MARKERS:
            with self.subTest(marker=marker):
                self.assertEqual(cell_len(marker), 1)
                # "A" (ambiguous) renders as two cells in a CJK locale while
                # Rich measures one, which corrupts the rest of the line.
                self.assertEqual(
                    unicodedata.east_asian_width(marker),
                    "N",
                    f"{marker!r} is width-ambiguous across terminals",
                )

    def test_markers_are_distinguishable_from_each_other(self) -> None:
        self.assertEqual(len(set(self.MARKERS)), len(self.MARKERS))

    def test_auth_rows_all_start_their_text_in_the_same_column(self) -> None:
        from rich.markup import render as render_markup

        rows = [
            main_menu_module._render_modal_auth_status(),
            main_menu_module._render_prime_auth_status(),
            main_menu_module._render_hf_auth_status(),
            main_menu_module._render_artificial_analysis_auth_status(),
        ]
        offsets = {cell_len(render_markup(row).plain.split(" ", 1)[0]) for row in rows}
        self.assertEqual(offsets, {1})


# Keys never contain two consecutive spaces, so the run of two or more is
# always the gap between a help row's key column and its description.
_HELP_ROW = re.compile(r"^  (?P<key>\S.*?) {2,}(?P<description>\S.*)$")

_LABELLED_ROW = re.compile(r"^\[(?:bold|dim)\](?P<label>.*?)\[/(?:bold|dim)?\](?P<pad> *)")


def _value_columns(markup: str) -> set[int]:
    """Column at which each row's value begins, as the terminal would see it.

    Measured from the markup rather than the rendered line, because a label
    padded to exactly one space is indistinguishable from a label followed by a
    value that happens to contain spaces.
    """
    columns = set()
    for line in markup.splitlines():
        match = _LABELLED_ROW.match(line)
        if match is None:
            continue
        columns.add(cell_len(match.group("label")) + len(match.group("pad")))
    return columns


class LabelledRowAlignmentTests(unittest.TestCase):
    """Result cards did not align their values; connection cards did."""

    def test_result_values_share_one_column(self) -> None:
        markup = _result_card_markup(
            [("Status", "Healthy"), ("Test command", "curl https://example.test")]
        )
        self.assertEqual(len(_value_columns(markup)), 1, markup)

    def test_connection_values_share_one_column(self) -> None:
        markup = _connection_card_markup(
            {
                "base_url": "https://example.test/v1",
                "model_id": "m",
                "display_name": "d",
                "api_key": "k",
            }
        )
        self.assertEqual(len(_value_columns(markup)), 1, markup)

    def test_a_long_label_widens_the_column_instead_of_breaking_it(self) -> None:
        markup = _result_card_markup([("A", "1"), ("A much longer label", "2")])
        self.assertEqual(len(_value_columns(markup)), 1, markup)
        self.assertEqual(_value_columns(markup), {len("A much longer label") + 2})


class ProfileSummaryAlignmentTests(unittest.IsolatedAsyncioTestCase):
    """Availability, Placement, Default slug, If left up and Spec decode all
    overflowed the hand-counted padding and pushed their values out of line."""

    async def test_every_summary_row_lands_on_the_same_column(self) -> None:
        from llm_launchpad.core import quick_deploy as quick_deploy_module
        from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen
        from tests.catalog_fixtures import activate_static_like_catalog

        quick_deploy_module._reset_quick_deploy_catalog_cache()
        activate_static_like_catalog()
        self.addCleanup(quick_deploy_module._reset_quick_deploy_catalog_cache)

        app = _StyledApp()
        async with app.run_test(size=(140, 45)) as pilot:
            app.push_screen(QuickDeployScreen(profile_id="kimi25-rtxpro"))
            await pilot.pause()
            summary = str(
                app.screen.query_one("#quick-deploy-profile-body", Static).content
            )

        rows = [line for line in summary.splitlines() if line.startswith("[bold]")]
        self.assertGreater(len(rows), 5)
        self.assertEqual(len(_value_columns("\n".join(rows))), 1, summary)

    def test_the_longest_label_still_leaves_a_gap(self) -> None:
        from llm_launchpad.tui.screens.quick_deploy import (
            _SUMMARY_LABEL_WIDTH,
            _summary_row,
        )

        for label in (
            "Availability", "Placement", "Default slug", "If left up",
            "Spec decode", "GPU", "Provider",
        ):
            with self.subTest(label=label):
                self.assertLessEqual(len(label), _SUMMARY_LABEL_WIDTH - 1)
                self.assertTrue(_summary_row(label, "value").endswith(" value"))


class StatusHeaderTests(unittest.IsolatedAsyncioTestCase):
    """A four-row box for one line of text, and markers of varying width."""

    def test_every_state_marker_is_the_same_width(self) -> None:
        from rich.markup import render as render_markup

        widths = {
            cell_len(render_markup(_state_icon(state)).plain)
            for state in list(_STATE_MARKERS) + ["something-unknown"]
        }
        self.assertEqual(widths, {2})

    def test_no_marker_is_swallowed_as_rich_markup(self) -> None:
        """The markers are interpolated into markup, so "[..]" would vanish."""
        from rich.markup import render as render_markup

        for state, (marker, _style) in _STATE_MARKERS.items():
            with self.subTest(state=state):
                self.assertFalse(marker.startswith("["))
                self.assertEqual(
                    render_markup(_state_icon(state)).plain.strip(), marker
                )

    def test_markers_survive_colour_being_stripped(self) -> None:
        from rich.markup import render as render_markup

        plains = {
            render_markup(_state_icon(state)).plain.strip()
            for state in _STATE_MARKERS
        }
        self.assertEqual(len(plains), len(_STATE_MARKERS))

    async def test_the_header_does_not_reserve_blank_rows(self) -> None:
        from llm_launchpad.tui.widgets.status_header import StatusHeader

        app = _StyledApp()
        async with app.run_test(size=(100, 30)) as pilot:
            header = StatusHeader()
            await app.screen.mount(header)
            await pilot.pause()
            # One content row plus the bottom border.
            self.assertEqual(header.region.height, 2)


class HelpOverlayTests(unittest.IsolatedAsyncioTestCase):
    def test_keys_that_share_a_description_are_listed_together(self) -> None:
        from textual.binding import Binding

        class _Screen:
            BINDINGS = [
                Binding("y", "copy_text", "Copy"),
                Binding("ctrl+shift+c", "copy_text", "Copy"),
            ]

        self.assertEqual(
            _iter_screen_bindings(_Screen()), [("y, ctrl+shift+c", "Copy")]
        )

    def test_a_key_the_app_claims_at_priority_is_dropped_from_the_screen(self) -> None:
        from textual.binding import Binding

        class _Screen:
            BINDINGS = [Binding("ctrl+c", "copy_text", "Copy selected text")]

        self.assertEqual(
            _iter_screen_bindings(_Screen(), shadowed_keys=frozenset({"ctrl+c"})), []
        )

    async def _open_help(self, pilot, app) -> None:
        screen = MainMenuScreen(username="review")
        with patch.object(MainMenuScreen, "on_mount", lambda self: None):
            app.push_screen(screen)
            await pilot.pause()
        app.push_screen(HelpOverlayScreen.from_screen(screen))
        await pilot.pause()

    async def test_the_overlay_names_the_key_that_quits(self) -> None:
        app = _ReviewApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await self._open_help(pilot, app)
            rendered = "\n".join(_rendered_lines(app))
            self.assertIn("Quit", rendered)
            # Textual's own ctrl+c row claims to copy; this app quits with it.
            self.assertNotIn("Copy selected text", rendered)

    async def _help_rows(self, pilot, app) -> list[str]:
        await self._open_help(pilot, app)
        screen = app.screen
        assert isinstance(screen, HelpOverlayScreen)
        return [
            line
            for line in (str(static.render()) for static in screen.query(Static))
            if line.startswith("  ") and line.strip()
        ]

    async def test_no_key_column_collides_with_its_description(self) -> None:
        app = _ReviewApp()
        async with app.run_test(size=(120, 40)) as pilot:
            rows = await self._help_rows(pilot, app)
            self.assertTrue(rows)
            for row in rows:
                match = _HELP_ROW.match(row)
                self.assertIsNotNone(
                    match, f"key and description ran together: {row!r}"
                )

    async def test_the_widest_key_sets_the_column_for_all_of_them(self) -> None:
        app = _ReviewApp()
        async with app.run_test(size=(120, 40)) as pilot:
            rows = await self._help_rows(pilot, app)
            columns = set()
            for row in rows:
                match = _HELP_ROW.match(row)
                assert match is not None
                columns.add(match.start("description"))
            self.assertEqual(len(columns), 1, columns)

    async def test_the_longest_key_is_present_and_still_separated(self) -> None:
        app = _ReviewApp()
        async with app.run_test(size=(120, 40)) as pilot:
            rows = await self._help_rows(pilot, app)
            combined = [row for row in rows if "ctrl+shift+c" in row]
            self.assertTrue(combined, "the grouped copy row is missing")
            self.assertIn("  Copy", combined[0])


class ToggleFieldTests(unittest.IsolatedAsyncioTestCase):
    """A switch distinguished on from off by colour alone, which the
    monochrome theme -- and the terminals it exists for -- flatten."""

    async def test_the_state_is_readable_without_colour(self) -> None:
        from textual.widgets import Switch

        app = _StyledApp()
        async with app.run_test(size=(100, 40)) as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()
            field = screen.query_one("#tui-mouse", Switch).parent
            state = str(field.query_one(".toggle-state", Static).render()).strip()
            self.assertIn(state, {"On", "Off"})
            self.assertIn(state, "\n".join(_rendered_lines(app)))

    async def test_the_state_word_follows_the_switch(self) -> None:
        from textual.widgets import Switch

        app = _StyledApp()
        async with app.run_test(size=(100, 40)) as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()
            field = screen.query_one("#tui-mouse", Switch).parent
            before = str(field.query_one(".toggle-state", Static).render()).strip()
            field.query_one("#tui-mouse", Switch).toggle()
            await pilot.pause()
            after = str(field.query_one(".toggle-state", Static).render()).strip()
            self.assertNotEqual(before, after)
            self.assertEqual({before, after}, {"On", "Off"})


class DuplicatedChromeTests(unittest.IsolatedAsyncioTestCase):
    async def test_manage_does_not_restate_the_footer_below_the_table(self) -> None:
        from llm_launchpad.tui.screens.manage import ManageScreen
        from llm_launchpad.tui.workers import EndpointsLoaded
        from tests.test_manage_screen_routing import _TestApp as ManageApp, _endpoint

        class _StyledManageApp(ManageApp):
            CSS_PATH = TuiApp.CSS_PATH

        app = _StyledManageApp()
        async with app.run_test(size=(100, 30)) as pilot:
            screen = ManageScreen()
            with patch.object(ManageScreen, "_refresh_endpoints", lambda self: None):
                app.push_screen(screen)
                await pilot.pause()
                screen.on_endpoints_loaded(EndpointsLoaded([_endpoint("vllm-x", "ap-x")]))
                await pilot.pause()

            rendered = "\n".join(_rendered_lines(app))
            self.assertIn("vllm-x", rendered)
            self.assertNotIn("Enter actions", rendered)
            self.assertNotIn("r refresh · Esc back", rendered)
            # The footer is where those keys belong, and it still carries them.
            self.assertIn("esc Back", rendered)
            self.assertIn("r Refresh", rendered)


if __name__ == "__main__":
    unittest.main()

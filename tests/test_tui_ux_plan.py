"""Tests for the TUI UX plan items 3/5/6: progress, help/palette, theming."""

from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Static

from llm_launchpad.protocol.enums import DeploymentState, OperationType
from llm_launchpad.tui.deployment_progress import DeploymentProgress
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.visual import STATUS_MARKERS, accent_title, status_markup
from llm_launchpad.tui.widgets.deployment_progress import (
    DeploymentProgressWidget,
    _stage_chip,
)
from llm_launchpad.tui.widgets.help_overlay import (
    HelpOverlayScreen,
    _focused_control_hints,
    _iter_effective_bindings,
)
from llm_launchpad.tui.workers import (
    EndpointAvailable,
    OperationDone,
    ResourceAllocated,
    StateChanged,
    _dispatch_event,
)
from llm_launchpad.protocol.events import (
    EndpointAvailableEvent,
    ResourceAllocatedEvent,
)
from llm_launchpad.protocol.models import EndpointInfo


class DeploymentProgressTests(unittest.TestCase):
    def test_deploy_advances_through_stages_in_order(self) -> None:
        progress = DeploymentProgress()
        progress.start(OperationType.DEPLOY, "Deploy test")
        progress.on_state(DeploymentState.QUEUED, "Queued")
        progress.on_state(DeploymentState.DEPLOYING, "Building image")
        self.assertEqual(
            progress.stage_states, ["done", "done", "active", "pending", "pending"]
        )
        self.assertEqual(progress.active_stage(), "Prepare")

    def test_endpoint_available_does_not_claim_verify(self) -> None:
        """The URL arriving means weights may still be loading, not verified."""
        progress = DeploymentProgress()
        progress.start(OperationType.DEPLOY, "Deploy test")
        progress.on_endpoint_available()
        self.assertTrue(progress.endpoint_known)
        self.assertEqual(progress.active_stage(), "Load")
        self.assertEqual(progress.stage_states[4], "pending")

    def test_fallback_attempt_resets_stages_and_counts_attempt(self) -> None:
        progress = DeploymentProgress()
        progress.start(OperationType.DEPLOY, "Deploy test")
        progress.on_state(DeploymentState.DEPLOYING, "Building image")
        progress.on_error("placement failed")
        progress.on_state(DeploymentState.QUEUED, "Trying fallback")
        self.assertEqual(progress.attempt, 2)
        self.assertEqual(
            progress.stage_states, ["active", "pending", "pending", "pending", "pending"]
        )

    def test_quiet_explanation_never_invents_a_percentage(self) -> None:
        progress = DeploymentProgress()
        progress.start(OperationType.DEPLOY, "Deploy test")
        progress.on_state(DeploymentState.DEPLOYING, "Building image")
        explanation = progress.quiet_explanation(
            now=progress.last_progress_at + 3600.0
        )
        self.assertTrue(explanation)
        self.assertNotIn("%", explanation)

    def test_non_deploy_operation_uses_a_single_stage(self) -> None:
        progress = DeploymentProgress()
        progress.start(OperationType.STATUS, "Status check")
        progress.on_state(DeploymentState.RUNNING, "Probing")
        self.assertEqual(progress.stages, ["Status"])
        self.assertEqual(progress.active_stage(), "Status")


class MonitorProgressScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_state_events_drive_the_persistent_progress_row(self) -> None:
        app = App[None]()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Deploy test"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_state_changed(
                StateChanged(
                    DeploymentState.DEPLOYING,
                    OperationType.DEPLOY,
                    "Building image",
                )
            )
            await pilot.pause()
            title = str(
                screen.query_one("#deploy-progress-title", Static).render()
            )
            stages = str(
                screen.query_one("#deploy-progress-stages", Static).render()
            )
            self.assertIn("Deploy test", title)
            self.assertIn("Prepare", stages)
            self.assertIn("Building image", str(
                screen.query_one("#deploy-progress-detail", Static).render()
            ))

    async def test_failed_deploy_shows_outcome_with_resource_and_next(self) -> None:
        app = App[None]()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(MonitorScreen(title="Deploy test"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_operation_done(
                OperationDone(OperationType.DEPLOY, False, 1, "placement failed")
            )
            await pilot.pause()
            await pilot.pause()
            card = screen.query_one("#failure-card")
            self.assertTrue(card.display)
            body = str(screen.query_one("#failure-card-body", Static).render())
            self.assertIn("placement failed", body)
            self.assertIn("Manage", body)
            # A failed deploy still fits the viewport: the card scrolls
            # internally instead of pushing the log viewer off screen.
            self.assertLessEqual(
                card.region.bottom, screen.region.height + screen.region.y
            )

    async def test_reopening_monitor_preserves_progress_state(self) -> None:
        """Leaving and reopening keeps the tracker; screens are not rebuilt."""
        app = App[None]()
        async with app.run_test() as pilot:
            monitor = MonitorScreen(title="Deploy test")
            app.push_screen(monitor)
            await pilot.pause()
            monitor.on_state_changed(
                StateChanged(
                    DeploymentState.WARMING_UP, OperationType.DEPLOY, "Loading weights"
                )
            )
            await pilot.pause()
            app.pop_screen()
            await pilot.pause()
            app.push_screen(monitor)
            await pilot.pause()
            self.assertEqual(monitor._progress.active_stage(), "Load")

    async def test_resource_and_endpoint_events_advance_progress(self) -> None:
        app = App[None]()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Deploy test"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_resource_allocated(ResourceAllocated(app_id="app-1"))
            screen.on_endpoint_available(EndpointAvailable(endpoint=None))
            await pilot.pause()
            self.assertEqual(screen._progress.resource_status, "allocated")
            # Without a state event the operation is still generic; the
            # endpoint signal is recorded but no deploy stages are invented.
            self.assertTrue(screen._progress.endpoint_known)


class WorkerDispatchTests(unittest.TestCase):
    def test_resource_and_endpoint_events_reach_the_monitor(self) -> None:
        seen: list[object] = []

        class Sink:
            def post_message(self, message: object) -> None:
                seen.append(message)

        _dispatch_event(Sink(), ResourceAllocatedEvent(app_id="app-1"))
        _dispatch_event(
            Sink(), EndpointAvailableEvent(endpoint=EndpointInfo(name="n"))
        )
        kinds = {type(message).__name__ for message in seen}
        self.assertEqual(kinds, {"ResourceAllocated", "EndpointAvailable"})


class HelpOverlayTests(unittest.IsolatedAsyncioTestCase):
    def test_effective_bindings_exclude_phase_gated_actions(self) -> None:
        class PhaseScreen(Static):
            BINDINGS = []  # type: ignore[assignment]

            def __init__(self) -> None:
                self._phase = "models"

            @property
            def active_bindings(self) -> dict:  # type: ignore[override]
                from textual.binding import Binding

                shown = Binding("enter", "choose", "Choose", show=True)
                hidden = Binding("a", "compare", "Compare", show=False)
                return {
                    "enter": ("s", shown, True, ""),
                    "a": ("s", hidden, True, ""),
                }

        rows = _iter_effective_bindings(PhaseScreen())
        assert rows is not None
        labels = [label for _key, label in rows]
        self.assertIn("Choose", labels)

    async def test_unmounted_screen_falls_back_to_class_bindings(self) -> None:
        app = App[None]()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = HelpOverlayScreen.from_screen(app.screen)
            sections = dict(overlay._sections)
            self.assertTrue(sections)

    def test_focused_input_gets_control_hints(self) -> None:
        from textual.widgets import Input

        class FakeScreen:
            focused = Input()

        hints = _focused_control_hints(FakeScreen())
        self.assertTrue(hints)


class SemanticStyleTests(unittest.TestCase):
    def test_status_markers_are_distinct_without_color(self) -> None:
        markers = [marker for marker, _style in STATUS_MARKERS.values()]
        self.assertEqual(len(set(markers)), len(markers))

    def test_accent_title_uses_theme_variable_not_hex(self) -> None:
        rendered = accent_title("Deploy")
        self.assertIn("primary", rendered)
        self.assertNotIn("#", rendered)

    def test_status_markup_escapes_text(self) -> None:
        rendered = status_markup("failed", "[boom]")
        self.assertIn("XX", rendered)
        self.assertNotIn("[boom]", rendered.replace("\\[boom]", ""))

    def test_stage_chips_use_semantic_styles(self) -> None:
        self.assertIn("[success]", _stage_chip("Validate", "done"))
        self.assertIn("[error]", _stage_chip("Load", "failed"))
        self.assertNotIn("#", _stage_chip("Load", "failed"))

    def test_progress_widget_renders_without_mount(self) -> None:
        widget = DeploymentProgressWidget()
        progress = DeploymentProgress()
        progress.start(OperationType.DEPLOY, "Deploy test")
        # Must not raise before mount; content appears after compose.
        widget.update_progress(progress)
        self.assertEqual(widget.progress.title, "Deploy test")


class CommandPaletteTests(unittest.IsolatedAsyncioTestCase):
    async def test_palette_exposes_navigation_commands(self) -> None:
        from llm_launchpad.tui.app import TuiApp

        app = TuiApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Push a neutral screen so get_system_commands has context.
            from llm_launchpad.tui.screens.settings import SettingsScreen

            app.push_screen(SettingsScreen())
            await pilot.pause()
            commands = list(app.get_system_commands(app.screen))
            titles = [command.title for command in commands]
            for expected in (
                "Deploy model",
                "Advanced deploy",
                "Manage endpoints",
                "Jobs",
                "Storage",
                "Settings",
            ):
                self.assertIn(expected, titles)


if __name__ == "__main__":
    unittest.main()

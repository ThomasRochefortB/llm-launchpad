"""Tests for completion notifications and actionable deployment failure recovery."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import Switch

from llm_launchpad.core.config import ConfigStore
from llm_launchpad.core.notifications import (
    emit_terminal_notification,
    format_deployment_notification,
    osc_9_sequence,
    sanitize_notification_text,
    terminal_notification_sequence,
    wrap_terminal_passthrough,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig, LaunchpadSettings
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.settings import SettingsScreen


class NotificationHelpersTests(unittest.TestCase):
    def test_sanitize_strips_control_chars_and_osc_delimiters(self) -> None:
        dirty = "Model ready\x1b]52;c;evil\x07\nnext;line"
        cleaned = sanitize_notification_text(dirty)
        self.assertNotIn("\x1b", cleaned)
        self.assertNotIn("\x07", cleaned)
        self.assertNotIn(";", cleaned)
        self.assertNotIn("\n", cleaned)
        self.assertIn("Model ready", cleaned)

    def test_terminal_notification_sequence_wraps_tmux_and_screen(self) -> None:
        plain = terminal_notification_sequence("LLM Launchpad", "Qwen3 ready")
        self.assertEqual(plain, "\x07\x1b]9;LLM Launchpad: Qwen3 ready\x07")

        tmux = terminal_notification_sequence("LLM Launchpad", "Qwen3 ready", tmux=True)
        self.assertTrue(tmux.startswith("\x07\x1bPtmux;"))
        self.assertTrue(tmux.endswith("\x1b\\"))
        self.assertIn("\x1b\x1b]9;LLM Launchpad: Qwen3 ready\x07", tmux)

        screen = terminal_notification_sequence("LLM Launchpad", "Qwen3 ready", screen=True)
        self.assertEqual(
            screen,
            f"\x07{wrap_terminal_passthrough(osc_9_sequence('LLM Launchpad: Qwen3 ready'), screen=True)}",
        )

    def test_format_deployment_notification_for_success_and_failure(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            served_model_name="Qwen3-32B",
        )
        title, body = format_deployment_notification(config, success=True)
        self.assertEqual(title, "LLM Launchpad")
        self.assertEqual(body, "Qwen3-32B is ready on Modal")

        title, fail_body = format_deployment_notification(
            config, success=False, detail="Timed out waiting for GPU"
        )
        self.assertEqual(title, "LLM Launchpad")
        self.assertIn("Qwen3-32B failed on Modal", fail_body)
        self.assertIn("Timed out waiting for GPU", fail_body)

    def test_emit_terminal_notification_writes_to_custom_writer(self) -> None:
        buf = io.StringIO()
        ok = emit_terminal_notification(
            "LLM Launchpad",
            "Model ready",
            writer=buf,
            env={"TMUX": "/tmp/tmux-1000/default,123,0"},
        )
        self.assertTrue(ok)
        self.assertIn("\x1bPtmux;", buf.getvalue())
        self.assertIn("Model ready", buf.getvalue())

    def test_launchpad_settings_notify_on_complete_round_trip(self) -> None:
        default = LaunchpadSettings.from_dict({})
        self.assertTrue(default.notify_on_complete)

        disabled = LaunchpadSettings(notify_on_complete=False)
        reloaded = LaunchpadSettings.from_dict(disabled.to_dict())
        self.assertFalse(reloaded.notify_on_complete)


class NotificationSettingsScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_screen_persists_notify_on_complete(self) -> None:
        class _App(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self._confirm_quit = True
                self._notify_on_complete = True

        app = _App()
        async with app.run_test(size=(100, 40)) as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()

            toggle = screen.query_one("#notify-on-complete", Switch)
            self.assertTrue(toggle.value)
            toggle.value = False
            screen._save()

            self.assertFalse(app._notify_on_complete)
            self.assertFalse(ConfigStore().load().notify_on_complete)


class TuiAppRecoveryAndNotificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_deployment_resets_ephemeral_fields_and_redeploys(self) -> None:
        started: list[DeploymentConfig] = []

        with (
            patch.object(TuiApp, "begin_deploy", side_effect=started.append),
            patch.object(TuiApp, "_sync_opencode", return_value=None),
            patch.object(TuiApp, "run_worker", return_value=None),
        ):
            app = TuiApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                original = DeploymentConfig(
                    backend=BackendType.LLAMACPP,
                    provider=ComputeProvider.MODAL,
                    repo_id="unsloth/Qwen3-4B-GGUF",
                    endpoint_api_key="old-secret",
                    tool_calling="failed",
                    retry_allowed=False,
                )
                app.push_screen(MonitorScreen(title="Deploy", deployment_config=original))
                await pilot.pause()

                app.retry_deployment(original)
                await pilot.pause()

                self.assertEqual(len(started), 1)
                retried = started[0]
                self.assertIsNone(retried.endpoint_api_key)
                self.assertIsNone(retried.tool_calling)
                self.assertTrue(retried.retry_allowed)
                self.assertEqual(retried.repo_id, original.repo_id)

    async def test_choose_other_placement_opens_fast_deploy_for_catalog_model(self) -> None:
        with (
            patch.object(TuiApp, "_catalog_model_id_for_config", return_value="qwen3-4b"),
            patch.object(TuiApp, "_sync_opencode", return_value=None),
            patch.object(TuiApp, "run_worker", return_value=None),
        ):
            app = TuiApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                config = DeploymentConfig(
                    backend=BackendType.LLAMACPP,
                    provider=ComputeProvider.VAST,
                    repo_id="unsloth/Qwen3-4B-GGUF",
                    max_context_tokens=32768,
                )
                app.push_screen(MonitorScreen(title="Deploy", deployment_config=config))
                await pilot.pause()

                app.choose_other_placement(config)
                await pilot.pause()

                self.assertIsInstance(app.screen, FastDeployScreen)
                assert isinstance(app.screen, FastDeployScreen)
                self.assertEqual(app.screen._initial_model_id, "qwen3-4b")
                self.assertEqual(app.screen._context_tokens, 32768)

    async def test_notify_deployment_finished_respects_setting_and_emits_osc9(self) -> None:
        written: list[str] = []

        with (
            patch.object(TuiApp, "_sync_opencode", return_value=None),
            patch.object(TuiApp, "run_worker", return_value=None),
        ):
            app = TuiApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app._driver is not None
                with patch.object(app._driver, "write", side_effect=written.append):
                    config = DeploymentConfig(
                        backend=BackendType.VLLM,
                        provider=ComputeProvider.MODAL,
                        model_name="Qwen/Qwen3-4B",
                    )
                    app._notify_on_complete = True
                    app.notify_deployment_finished(config, success=True)
                    self.assertEqual(len(written), 1)
                    self.assertIn("\x1b]9;", written[0])
                    self.assertIn("Qwen/Qwen3-4B is ready on Modal", written[0])

                    written.clear()
                    app._notify_on_complete = False
                    app.notify_deployment_finished(config, success=True)
                    self.assertEqual(written, [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import OptionList

from llm_launchpad.core import quick_deploy
from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.quick_deploy import QuickDeployCatalogInfo, QuickDeployProfile
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.workers import EndpointsLoaded


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.deploy_calls = 0
        self.custom_deploy_calls = 0

    def action_push_deploy(self) -> None:
        self.deploy_calls += 1
        self.push_screen(FastDeployScreen())

    def action_push_custom_deploy(self) -> None:
        self.custom_deploy_calls += 1

    def action_push_manage(self) -> None:
        pass

    def action_push_storage(self) -> None:
        pass

    def action_push_settings(self) -> None:
        pass


def _quiet_main_menu():
    return (
        patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None),
        patch.object(MainMenuScreen, "_refresh_prime_auth_status", lambda self: None),
        patch.object(MainMenuScreen, "_refresh_hf_auth_status", lambda self: None),
        patch.object(MainMenuScreen, "_refresh_aai_auth_status", lambda self: None),
        patch.object(MainMenuScreen, "_refresh_panels", lambda self: None),
        patch.object(MainMenuScreen, "_refresh_quick_deploy_catalog", lambda self: None),
        patch.object(MainMenuScreen, "_refresh_storage_estimate", lambda self: None),
        patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
            return_value=aggregate_compute_availability(),
        ),
    )


class MainMenuDeployDoorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        quick_deploy._reset_quick_deploy_catalog_cache()

    def tearDown(self) -> None:
        quick_deploy._reset_quick_deploy_catalog_cache()

    async def test_home_menu_has_deploy_and_custom_deploy_only(self) -> None:
        app = _TestApp()
        with ExitStack() as stack:
            for item in _quiet_main_menu():
                stack.enter_context(item)
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                option_ids = [
                    action_list.get_option_at_index(index).id
                    for index in range(action_list.option_count)
                ]
                self.assertEqual(
                    option_ids,
                    ["deploy", "custom-deploy", "manage", "storage", "settings"],
                )
                self.assertEqual(len(screen.query("#quick-deploy-list")), 0)
                self.assertEqual(len(screen.query("#quick-deploy-panel")), 0)
                self.assertFalse(any(binding.key == "f" for binding in screen.BINDINGS))

    async def test_deploy_entry_opens_model_first_flow(self) -> None:
        app = _TestApp()
        with ExitStack() as stack:
            for item in _quiet_main_menu():
                stack.enter_context(item)
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                deploy_option = next(
                    action_list.get_option_at_index(index)
                    for index in range(action_list.option_count)
                    if action_list.get_option_at_index(index).id == "deploy"
                )
                screen.on_option_list_option_selected(
                    SimpleNamespace(option=deploy_option, option_list=action_list)
                )
                await pilot.pause()

                self.assertEqual(app.deploy_calls, 1)
                self.assertIsInstance(app.screen, FastDeployScreen)

    async def test_custom_deploy_entry_uses_expert_form_action(self) -> None:
        app = _TestApp()
        with ExitStack() as stack:
            for item in _quiet_main_menu():
                stack.enter_context(item)
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                custom_option = next(
                    action_list.get_option_at_index(index)
                    for index in range(action_list.option_count)
                    if action_list.get_option_at_index(index).id == "custom-deploy"
                )
                screen.on_option_list_option_selected(
                    SimpleNamespace(option=custom_option, option_list=action_list)
                )

        self.assertEqual(app.custom_deploy_calls, 1)
        self.assertEqual(app.deploy_calls, 0)

    async def test_stale_endpoint_snapshot_does_not_duplicate_runtime_probe(self) -> None:
        app = _TestApp()
        with ExitStack() as stack:
            for item in _quiet_main_menu():
                stack.enter_context(item)
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                rows = [
                    EndpointInfo(
                        name="vllm-qwen",
                        app_id="ap-1",
                        state="running",
                        backend=BackendType.VLLM,
                    )
                ]
                screen._status_refresh_inflight = True

                with patch.object(screen, "run_worker") as run_worker:
                    screen.on_endpoints_loaded(EndpointsLoaded(rows, is_stale=True))
                    run_worker.assert_not_called()
                    self.assertTrue(screen._status_refresh_inflight)

                    screen.on_endpoints_loaded(EndpointsLoaded(rows))
                    run_worker.assert_called_once()

    def test_warm_catalog_activates_instantly_and_refreshes_in_background(self) -> None:
        profile = QuickDeployProfile(
            id="warm-model-cheap",
            display_name="Warm Model",
            repo_id="unsloth/Warm-Model-GGUF",
            quant="UD-Q2_K_XL",
            gpu_type="L4",
            gpu_count=1,
            profile_label="Slow but cheap",
            approx_cost_per_hour_usd=0.8,
            max_context_tokens=131_072,
            instance_slug_hint="warm-model",
            summary="Warm snapshot.",
            server_args=("--ctx-size", "131072"),
        )
        info = QuickDeployCatalogInfo(
            source_label="Cached snapshot",
            generated_at="2026-09-03T00:00:00Z",
            is_live=True,
            ready=True,
        )
        screen = MainMenuScreen(username="alice")
        posted: list[object] = []
        screen.post_message = posted.append  # type: ignore[method-assign]
        with patch(
            "llm_launchpad.tui.screens.main_menu.load_cached_quick_deploy_catalog",
            return_value=(info, (profile,)),
        ), patch(
            "llm_launchpad.tui.screens.main_menu.is_fresh_cached_quick_deploy_catalog",
            return_value=True,
        ), patch.object(
            screen, "run_worker", return_value=None
        ) as run_worker, patch(
            "llm_launchpad.tui.screens.main_menu.activate_quick_deploy_catalog",
            wraps=quick_deploy.activate_quick_deploy_catalog,
        ) as activate:
            screen._refresh_quick_deploy_catalog()

        self.assertTrue(screen._quick_deploy_catalog_refresh_inflight)
        activate.assert_called_once_with(info, (profile,))
        run_worker.assert_called_once()
        self.assertEqual(
            [p.id for p in quick_deploy.list_quick_deploy_profiles()],
            ["warm-model-cheap"],
        )

    def test_stale_snapshot_still_activates_before_refresh(self) -> None:
        profile = QuickDeployProfile(
            id="stale-model-cheap",
            display_name="Stale Model",
            repo_id="unsloth/Stale-Model-GGUF",
            quant="UD-Q2_K_XL",
            gpu_type="L4",
            gpu_count=1,
            profile_label="Slow but cheap",
            approx_cost_per_hour_usd=0.8,
            max_context_tokens=131_072,
            instance_slug_hint="stale-model",
            summary="Stale snapshot.",
            server_args=("--ctx-size", "131072"),
        )
        info = QuickDeployCatalogInfo(
            source_label="Cached snapshot",
            generated_at="2026-09-01T00:00:00Z",
            is_live=True,
            ready=True,
        )
        screen = MainMenuScreen(username="alice")
        with patch(
            "llm_launchpad.tui.screens.main_menu.load_cached_quick_deploy_catalog",
            return_value=(info, (profile,)),
        ), patch(
            "llm_launchpad.tui.screens.main_menu.is_fresh_cached_quick_deploy_catalog",
            return_value=False,
        ), patch.object(screen, "run_worker", return_value=None) as run_worker:
            screen._refresh_quick_deploy_catalog()

        run_worker.assert_called_once()
        self.assertEqual(
            [p.id for p in quick_deploy.list_quick_deploy_profiles()],
            ["stale-model-cheap"],
        )

    async def test_hidden_home_screen_skips_periodic_endpoint_refresh(self) -> None:
        app = _TestApp()
        with (
            patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_prime_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_hf_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_aai_auth_status", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_secondary_panels", lambda self: None),
            patch.object(MainMenuScreen, "_refresh_deployment_status") as refresh_status,
            patch(
                "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
                return_value=aggregate_compute_availability(),
            ),
        ):
            async with app.run_test() as pilot:
                main_menu = MainMenuScreen(username="alice", version="1.0.0")
                app.push_screen(main_menu)
                await pilot.pause()
                refresh_status.reset_mock()

                app.push_screen(FastDeployScreen())
                await pilot.pause()
                self.assertIsInstance(app.screen, FastDeployScreen)
                refresh_status.reset_mock()
                main_menu._refresh_panels()

                refresh_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()

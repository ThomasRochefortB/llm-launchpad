from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import OptionList, Static

from llm_launchpad.core import quick_deploy
from llm_launchpad.core.quick_deploy import QuickDeployProfile
from llm_launchpad.tui.screens.main_menu import MainMenuScreen, _quick_deploy_options, _render_quick_deploy_option
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.quick_deploy_calls: list[str] = []

    def push_quick_deploy(self, profile: str | QuickDeployProfile) -> None:
        profile_id = profile.id if isinstance(profile, QuickDeployProfile) else profile
        self.quick_deploy_calls.append(profile_id)
        self.push_screen(QuickDeployScreen(profile_id=profile))

    def action_push_deploy(self) -> None:
        pass

    def action_push_manage(self) -> None:
        pass

    def action_push_storage(self) -> None:
        pass

    def action_push_settings(self) -> None:
        pass


class MainMenuQuickDeployTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._catalog_patch = patch(
            "llm_launchpad.core.quick_deploy._read_bundled_catalog_text",
            return_value=None,
        )
        self._catalog_patch.start()
        quick_deploy._reset_quick_deploy_catalog_cache()

    def tearDown(self) -> None:
        self._catalog_patch.stop()
        quick_deploy._reset_quick_deploy_catalog_cache()

    async def test_main_menu_renders_quick_deploy_panel(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)
                subtitle = str(screen.query_one("#landing-quick-deploy-subtitle", Static).content)
                self.assertEqual(quick_list.option_count, 3)
                self.assertIn("Curated llama.cpp coding profiles", subtitle)
                self.assertIn("Qwen3.5 397B A17B[/bold] [dim]Q4XL[/dim]", str(quick_list.get_option_at_index(0).prompt))
                self.assertIn("cheap but good", str(quick_list.get_option_at_index(0).prompt))
                self.assertIn("262,144 ctx", str(quick_list.get_option_at_index(0).prompt))
                self.assertIn("GLM-5[/bold] [dim]Q4XL[/dim]", str(quick_list.get_option_at_index(1).prompt))
                self.assertIn("RTX6000x4", str(quick_list.get_option_at_index(1).prompt))
                self.assertIn("202,752 ctx", str(quick_list.get_option_at_index(1).prompt))
                self.assertIn("Kimi K2.5[/bold] [dim]Q4XL[/dim]", str(quick_list.get_option_at_index(2).prompt))
                self.assertIn("RTX6000x5", str(quick_list.get_option_at_index(2).prompt))
                self.assertIn("262,144 ctx", str(quick_list.get_option_at_index(2).prompt))

    async def test_generated_quick_deploy_option_renders_resource_tier_cleanly(self) -> None:
        profile = QuickDeployProfile(
            id="minimax-m2-7-b200-b200",
            display_name="MiniMax-M2.7",
            repo_id="unsloth/MiniMax-M2.7-GGUF",
            quant="UD-Q4_K_XL",
            gpu_type="B200",
            gpu_count=1,
            profile_label="B200",
            approx_cost_per_hour_usd=6.25,
            max_context_tokens=196608,
            instance_slug_hint="minimax-m2-7-b200",
            summary="Generated profile.",
            server_args=("--ctx-size", "196608"),
            required_vram_gb=141.0,
            resource_tier="b200",
            resource_tier_label="$$$",
        )

        rendered = _render_quick_deploy_option(profile)

        self.assertIn("MiniMax-M2.7[/bold] [dim]Q4XL[/dim] [yellow]$$$[/yellow]", rendered)
        self.assertIn("B200x1", rendered)
        self.assertIn("196,608 ctx", rendered)
        self.assertNotIn("req 141GB", rendered)
        self.assertIn("~$6.25/h", rendered)

    async def test_generated_quick_deploy_options_group_price_tiers_by_model(self) -> None:
        profiles = [
            QuickDeployProfile(
                id="kimi-k2-6-cheap-a100-80gb",
                display_name="Kimi K2.6",
                repo_id="unsloth/Kimi-K2.6-GGUF",
                quant="UD-Q4_K_XL",
                gpu_type="A100-80GB",
                gpu_count=8,
                profile_label="Slow but cheap",
                approx_cost_per_hour_usd=19.99,
                max_context_tokens=262144,
                instance_slug_hint="kimi-k2-6-cheap",
                summary="Generated profile.",
                server_args=("--ctx-size", "262144"),
                resource_tier="cheap",
                resource_tier_label="$",
                aa_coding_score=88.7,
            ),
            QuickDeployProfile(
                id="kimi-k2-6-q2xl-cheap-a100-80gb",
                display_name="Kimi K2.6",
                repo_id="unsloth/Kimi-K2.6-GGUF",
                quant="UD-Q2_K_XL",
                gpu_type="A100-80GB",
                gpu_count=4,
                profile_label="Low VRAM",
                approx_cost_per_hour_usd=9.99,
                max_context_tokens=262144,
                instance_slug_hint="kimi-k2-6-q2xl-cheap",
                summary="Generated profile.",
                server_args=("--ctx-size", "262144"),
                resource_tier="cheap",
                resource_tier_label="$/$$",
                aa_coding_score=88.7,
            ),
            QuickDeployProfile(
                id="kimi-k2-6-b200-b200",
                display_name="Kimi K2.6",
                repo_id="unsloth/Kimi-K2.6-GGUF",
                quant="UD-Q4_K_XL",
                gpu_type="B200",
                gpu_count=4,
                profile_label="B200",
                approx_cost_per_hour_usd=25.0,
                max_context_tokens=262144,
                instance_slug_hint="kimi-k2-6-b200",
                summary="Generated profile.",
                server_args=("--ctx-size", "262144"),
                resource_tier="b200",
                resource_tier_label="$$$",
                aa_coding_score=88.7,
            ),
            QuickDeployProfile(
                id="minimax-m2-7-cheap-a100-80gb",
                display_name="MiniMax-M2.7",
                repo_id="unsloth/MiniMax-M2.7-GGUF",
                quant="UD-Q4_K_XL",
                gpu_type="A100-80GB",
                gpu_count=2,
                profile_label="Slow but cheap",
                approx_cost_per_hour_usd=5.0,
                max_context_tokens=196608,
                instance_slug_hint="minimax-m2-7-cheap",
                summary="Generated profile.",
                server_args=("--ctx-size", "196608"),
                resource_tier="cheap",
                resource_tier_label="$",
                aa_coding_score=70.0,
            ),
        ]

        options = _quick_deploy_options(profiles)

        self.assertEqual(len(options), 5)
        assert options[0] is not None
        self.assertFalse(options[0].disabled)
        self.assertEqual(options[0].id, "kimi-k2-6-q2xl-cheap-a100-80gb")
        self.assertIn("Kimi K2.6[/bold] [dim]262,144 ctx · AA 88.7", str(options[0].prompt))
        self.assertIn(
            "[bold #7bf168]Q2XL[/] [dim]$[/dim][dim]/[/dim][#7bf168]$$[/]  [dim]A100x4",
            str(options[0].prompt),
        )
        assert options[1] is not None
        self.assertEqual(options[1].id, "kimi-k2-6-cheap-a100-80gb")
        self.assertIn("[dim]Q4XL[/dim] [dim]$[/dim]  [dim]A100x8", str(options[1].prompt))
        self.assertNotIn("Kimi K2.6", str(options[1].prompt))
        assert options[2] is not None
        self.assertEqual(options[2].id, "kimi-k2-6-b200-b200")
        self.assertIn("[yellow]$$$[/yellow]  [dim]B200x4", str(options[2].prompt))
        self.assertNotIn("Kimi K2.6", str(options[2].prompt))
        self.assertIsNone(options[3])
        assert options[4] is not None
        self.assertFalse(options[4].disabled)
        self.assertEqual(options[4].id, "minimax-m2-7-cheap-a100-80gb")
        self.assertIn("MiniMax-M2.7[/bold] [dim]196,608 ctx · AA 70", str(options[4].prompt))
        self.assertIn("[dim]Q4XL[/dim] [dim]$[/dim]  [dim]A100x2", str(options[4].prompt))
        self.assertNotIn("Kimi K2.6", str(options[4].prompt))

    async def test_generated_quick_deploy_options_sort_models_by_aa_coding_score(self) -> None:
        profiles = [
            QuickDeployProfile(
                id="lower-score",
                display_name="Lower Score",
                repo_id="unsloth/Lower-GGUF",
                quant="UD-Q4_K_XL",
                gpu_type="A100-80GB",
                gpu_count=1,
                profile_label="Slow but cheap",
                approx_cost_per_hour_usd=2.5,
                max_context_tokens=65536,
                instance_slug_hint="lower-score",
                summary="Generated profile.",
                server_args=("--ctx-size", "65536"),
                resource_tier="cheap",
                resource_tier_label="$",
                aa_coding_score=44.2,
            ),
            QuickDeployProfile(
                id="higher-score",
                display_name="Higher Score",
                repo_id="unsloth/Higher-GGUF",
                quant="UD-Q4_K_XL",
                gpu_type="A100-80GB",
                gpu_count=1,
                profile_label="Slow but cheap",
                approx_cost_per_hour_usd=2.5,
                max_context_tokens=131072,
                instance_slug_hint="higher-score",
                summary="Generated profile.",
                server_args=("--ctx-size", "131072"),
                resource_tier="cheap",
                resource_tier_label="$",
                aa_coding_score=91.5,
            ),
            QuickDeployProfile(
                id="higher-score-b200",
                display_name="Higher Score",
                repo_id="unsloth/Higher-GGUF",
                quant="UD-Q4_K_XL",
                gpu_type="B200",
                gpu_count=1,
                profile_label="B200",
                approx_cost_per_hour_usd=6.25,
                max_context_tokens=131072,
                instance_slug_hint="higher-score-b200",
                summary="Generated profile.",
                server_args=("--ctx-size", "131072"),
                resource_tier="b200",
                resource_tier_label="$$$",
                aa_coding_score=91.5,
            ),
        ]

        options = _quick_deploy_options(profiles)

        assert options[0] is not None
        self.assertEqual(options[0].id, "higher-score")
        self.assertIn("Higher Score[/bold] [dim]131,072 ctx · AA 91.5", str(options[0].prompt))

    async def test_tab_moves_focus_between_action_menu_and_quick_deploy(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)

                self.assertTrue(action_list.has_focus)
                await pilot.press("tab")
                await pilot.pause()
                self.assertTrue(quick_list.has_focus)

                await pilot.press("shift+tab")
                await pilot.pause()
                self.assertTrue(action_list.has_focus)

    async def test_down_from_last_main_menu_item_moves_focus_to_quick_deploy(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)

                action_list.focus()
                action_list.highlighted = action_list.option_count - 1
                await pilot.pause()

                await pilot.press("down")
                await pilot.pause()

                self.assertTrue(quick_list.has_focus)
                highlighted = quick_list.highlighted_option
                self.assertIsNotNone(highlighted)
                assert highlighted is not None
                self.assertEqual(highlighted.id, "qwen35-397b-rtxpro")

    async def test_up_from_first_quick_deploy_item_moves_focus_to_main_menu(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)

                quick_list.focus()
                quick_list.highlighted = 0
                await pilot.pause()

                await pilot.press("up")
                await pilot.pause()

                self.assertTrue(action_list.has_focus)
                highlighted = action_list.highlighted_option
                self.assertIsNotNone(highlighted)
                assert highlighted is not None
                self.assertEqual(highlighted.id, "settings")

    async def test_selecting_quick_deploy_entry_opens_detail_screen(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)
                option = quick_list.get_option_at_index(2)
                screen.on_option_list_option_selected(
                    SimpleNamespace(option=option, option_list=quick_list)
                )
                await pilot.pause()

                self.assertEqual(app.quick_deploy_calls, ["kimi25-rtxpro"])
                self.assertIsInstance(app.screen, QuickDeployScreen)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import Button, Input, Select, Static, Switch

from llm_launchpad.core import quick_deploy
from llm_launchpad.core.quick_deploy import (
    QuickDeployProfile,
    get_quick_deploy_profile,
    quick_deploy_recipe,
)
from llm_launchpad.protocol.enums import BackendType, BillingModel, ComputeProvider
from llm_launchpad.protocol.models import (
    InferencePlan,
    ModalProviderOptions,
    PrimeProviderOptions,
    ProviderQuote,
)
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.deployed_config = None

    def begin_deploy(self, config) -> None:  # type: ignore[no-untyped-def]
        self.deployed_config = config


class _StyledApp(_TestApp):
    CSS_PATH = TuiApp.CSS_PATH


class QuickDeployScreenTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_screen_renders_selected_profile_summary(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="kimi25-rtxpro"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            summary = str(screen.query_one("#quick-deploy-profile-body", Static).content)
            title = str(screen.query_one("#quick-deploy-title", Static).content)
            self.assertIn("Kimi K2.5 [dim](UD-Q4_K_XL)[/dim]", title)
            self.assertIn("[bold #7bf168]Kimi K2.5[/] [dim](UD-Q4_K_XL)[/dim]", summary)
            self.assertIn("RTX PRO 6000 96GB x5", summary)
            self.assertIn("262,144 ctx", summary)
            self.assertNotIn("Cheap but good", summary)
            self.assertIn("UD-Q4_K_XL", summary)
            self.assertIn("[bold]Provider[/bold] Modal", summary)
            self.assertIn("[bold]Billing[/bold]  Scale to zero", summary)
            self.assertIn("[bold]Hourly[/bold]   ~$15.15/hr", summary)
            self.assertIn("[bold]Monthly[/bold]  ~$909.00/mo", summary)
            self.assertIn("llama.cpp (GGUF)", summary)
            self.assertIn("unsloth/Kimi-K2.5-GGUF", summary)

    async def test_screen_renders_vllm_recipe_without_gguf_fields(self) -> None:
        profile = QuickDeployProfile(
            id="example-vllm",
            display_name="Example Model",
            repo_id="org/example",
            quant="",
            gpu_type="H100",
            gpu_count=1,
            profile_label="Fast",
            approx_cost_per_hour_usd=2.0,
            max_context_tokens=32768,
            instance_slug_hint="example-vllm",
            summary="Example vLLM profile.",
            server_args=(),
            backend=BackendType.VLLM,
            model_name="org/example",
        )

        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id=profile))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            summary = str(screen.query_one("#quick-deploy-profile-body", Static).content)

        self.assertIn("vLLM (OpenAI-compatible)", summary)
        self.assertIn("[bold]Model[/bold]    org/example", summary)
        self.assertNotIn("[bold]Quant[/bold]", summary)

    async def test_screen_moves_tier_and_vram_details_to_summary(self) -> None:
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

        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id=profile))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            summary = str(screen.query_one("#quick-deploy-profile-body", Static).content)
            self.assertIn("[bold]Tier[/bold]     $$$ B200", summary)
            self.assertIn("[bold]VRAM[/bold]     141 GB required", summary)
            self.assertIn("[bold]Max ctx[/bold]  196,608 ctx", summary)

    async def test_deploy_uses_blank_override_defaults(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="qwen35-397b-rtxpro"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            screen._deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.instance_name, "qwen35-397b-rtxpro")
        self.assertEqual(app.deployed_config.app_name, "llamacpp-qwen35-397b-rtxpro")
        self.assertEqual(app.deployed_config.gpu_type, "RTX-PRO-6000")
        self.assertEqual(app.deployed_config.gpu_count, 3)
        self.assertTrue(app.deployed_config.preload)

    async def test_advanced_options_are_hidden_by_default(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="qwen35-397b-rtxpro"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            instance_input = screen.query_one("#quick-instance-name", Input)
            self.assertTrue(instance_input.parent is not None)
            assert instance_input.parent is not None
            self.assertTrue(instance_input.parent.has_class("hidden"))
            self.assertTrue(screen.query_one("#quick-deploy-btn", Button).has_focus)

    async def test_deploy_button_stays_visible_on_compact_terminals(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(QuickDeployScreen(profile_id="kimi25-rtxpro"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            card = screen.query_one("#quick-deploy-profile-card")
            button = screen.query_one("#quick-deploy-btn", Button)
            self.assertGreaterEqual(card.region.y, 0)
            self.assertLessEqual(button.region.bottom, 24)
            self.assertTrue(button.has_focus)

    async def test_deploy_maps_override_fields_into_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="qwen35-397b-rtxpro"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            for widget in screen.query(".quick-advanced"):
                widget.remove_class("hidden")
            screen.query_one("#quick-instance-name", Input).value = "Team Alpha"
            screen.query_one("#quick-app-name", Input).value = "llamacpp-team-alpha"
            screen.query_one("#quick-warmup", Switch).value = False
            screen.query_one("#quick-debug-logs", Switch).value = True

            screen._deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.instance_name, "team-alpha")
        self.assertEqual(app.deployed_config.app_name, "llamacpp-team-alpha")
        self.assertFalse(app.deployed_config.do_warmup)
        self.assertTrue(app.deployed_config.show_debug_logs)

    async def test_prime_plan_exposes_and_maps_prime_advanced_options(self) -> None:
        profile = get_quick_deploy_profile("qwen35-397b-rtxpro")
        recipe = quick_deploy_recipe(profile)
        plan = InferencePlan(
            recipe=recipe,
            quote=ProviderQuote(
                id="prime:recipe:abc123",
                recipe_id=recipe.id,
                provider=ComputeProvider.PRIME,
                provider_reference="abc123",
                gpu_type="H100_80GB",
                gpu_count=4,
                price_per_hour_usd=8.0,
                billing_model=BillingModel.PROVISIONED,
                is_estimate=False,
                provider_options=PrimeProviderOptions(
                    offer_id="abc123",
                    region="CA",
                ),
            ),
            estimated_monthly_cost_usd=1920.0,
        )

        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id=plan))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            for widget in screen.query(".quick-advanced"):
                widget.remove_class("hidden")
            screen.query_one("#quick-prime-disk-id", Input).value = "disk-1"
            screen.query_one("#quick-prime-insecure-http", Switch).value = True
            screen.query_one("#quick-prime-keep-failed", Switch).value = True
            screen._deploy()

        config = app.deployed_config
        self.assertIsNotNone(config)
        self.assertEqual(config.provider, ComputeProvider.PRIME)
        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.app_name, "llp-prime-llamacpp-qwen35-397b-rtxpro")
        self.assertEqual(
            config.provider_options,
            PrimeProviderOptions(
                offer_id="abc123",
                region="CA",
                disk_id="disk-1",
                keep_failed_resource=True,
                allow_insecure_http=True,
            ),
        )

    async def test_compute_flow_can_change_fulfillment_during_review(self) -> None:
        profile = get_quick_deploy_profile("qwen35-397b-rtxpro")
        recipe = quick_deploy_recipe(profile)
        prime_plan = InferencePlan(
            recipe=recipe,
            quote=ProviderQuote(
                id="prime:compute:abc123",
                recipe_id=recipe.id,
                provider=ComputeProvider.PRIME,
                provider_reference="abc123",
                gpu_type="H100_80GB",
                gpu_count=4,
                price_per_hour_usd=8.0,
                billing_model=BillingModel.PROVISIONED,
                region="CA",
                is_estimate=False,
                provider_options=PrimeProviderOptions(offer_id="abc123"),
            ),
            estimated_monthly_cost_usd=1920.0,
        )
        modal_plan = InferencePlan(
            recipe=recipe,
            quote=ProviderQuote(
                id="modal:compute:h100",
                recipe_id=recipe.id,
                provider=ComputeProvider.MODAL,
                provider_reference="H100",
                gpu_type="H100",
                gpu_count=4,
                price_per_hour_usd=15.8,
                billing_model=BillingModel.SCALE_TO_ZERO,
                is_estimate=True,
                provider_options=ModalProviderOptions(),
            ),
            estimated_monthly_cost_usd=948.0,
        )

        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                QuickDeployScreen(
                    profile_id=prime_plan,
                    alternative_plans=(prime_plan, modal_plan),
                )
            )
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            selector = screen.query_one("#quick-fulfillment", Select)
            self.assertEqual(len(selector._options), 2)
            self.assertIn(
                "Prime Intellect",
                str(screen.query_one("#quick-deploy-profile-body", Static).content),
            )

            selector.value = modal_plan.quote.id
            await pilot.pause()

            summary = str(
                screen.query_one("#quick-deploy-profile-body", Static).content
            )
            self.assertIn("[bold]Provider[/bold] Modal", summary)
            screen._deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.provider, ComputeProvider.MODAL)


if __name__ == "__main__":
    unittest.main()

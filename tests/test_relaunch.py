"""Relaunching the last deployed model from the home screen."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from textual.widgets import OptionList

from llm_launchpad.core import last_launch
from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.last_launch import LastLaunch, load_last_launch, save_last_launch
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.main_menu import _action_label_tiers
from tests.test_fast_deploy_screen import _TestApp, _model, _profile


def _launch(**overrides: object) -> LastLaunch:
    base = dict(
        model_id="fits",
        display_name="Fits",
        provider="modal",
        gpu_type="B200",
        gpu_count=1,
        price_per_hour_usd=6.25,
    )
    return LastLaunch(**{**base, **overrides})  # type: ignore[arg-type]


class LastLaunchStoreTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        save_last_launch(_launch())
        loaded = load_last_launch()
        assert loaded is not None
        self.assertEqual(loaded.model_id, "fits")
        self.assertEqual(loaded.gpu_type, "B200")
        self.assertGreater(loaded.launched_at, 0)

    def test_missing_or_corrupt_file_offers_nothing(self) -> None:
        self.assertIsNone(load_last_launch())
        last_launch.LAST_LAUNCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        last_launch.LAST_LAUNCH_PATH.write_text("{not json", encoding="utf-8")
        self.assertIsNone(load_last_launch())


class HomeMenuTests(unittest.TestCase):
    def test_relaunch_heads_every_tier_only_once_something_was_deployed(self) -> None:
        plain = _action_label_tiers(None)
        self.assertTrue(all(tier[0][0] == "deploy" for tier in plain))
        tiers = _action_label_tiers(_launch())
        self.assertTrue(all(tier[0][0] == "relaunch" for tier in tiers))
        self.assertIn("Fits · B200 180GB x1", tiers[0][0][1])
        self.assertEqual([len(tier) for tier in tiers], [len(tier) + 1 for tier in plain])


class RelaunchScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_opens_on_placements_with_the_last_shape_highlighted(self) -> None:
        # Both shapes, so the highlight cannot just be the recommendation.
        for gpu in ("B200", "H100"):
            with self.subTest(gpu=gpu):
                await self._assert_highlights(gpu)

    async def _assert_highlights(self, gpu: str) -> None:
        model = _model((_profile("fits", required_vram_gb=40.0),))
        snapshot = aggregate_compute_availability(
            modal_catalog=[
                ModalGpuSpec("H100", price_per_hour_usd=4.0),
                ModalGpuSpec("B200", price_per_hour_usd=6.25),
            ]
        )
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ), patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
            return_value=snapshot,
        ):
            async with app.run_test() as pilot:
                app.push_screen(
                    FastDeployScreen(
                        initial_model_id="fits",
                        preferred_placement=("modal", gpu, 1),
                    )
                )
                for _ in range(10):
                    await pilot.pause()
                    screen = app.screen
                    if isinstance(screen, FastDeployScreen) and screen._phase == "infra":
                        break
                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                self.assertEqual(screen._phase, "infra")
                option_list = screen.query_one("#fast-deploy-list", OptionList)
                highlighted = option_list.get_option_at_index(option_list.highlighted or 0)
                row = screen._infra_rows[str(highlighted.id)]
                self.assertEqual(row.plan.quote.gpu_type, gpu)

    async def test_unknown_model_falls_back_to_the_model_list(self) -> None:
        model = _model((_profile("fits", required_vram_gb=40.0),))
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ), patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
            return_value=aggregate_compute_availability(modal_catalog=[]),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen(initial_model_id="gone"))
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                self.assertEqual(screen._phase, "models")

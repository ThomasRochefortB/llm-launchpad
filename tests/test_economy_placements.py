from dataclasses import replace
from unittest.mock import patch

from textual.widgets import Input, OptionList, Select, Static

from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.hf_models import GgufQuantMetadata, ModelCandidate
from llm_launchpad.core.modal_gpu import ModalGpuSpec, _parse_modal_gpu_pricing
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.quick_deploy import build_quick_deploy_config, list_quick_deploy_models
from llm_launchpad.core.quick_deploy_refresh import (
    _profiles_for_model, _quick_deploy_profile_from_dict, _quick_deploy_profile_to_dict,
)
from llm_launchpad.core.serving_tiers import economy_plan
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import CatalogExclusion, DeploymentConfig, ModalProviderOptions
from llm_launchpad.core.quick_deploy import QuickDeployCatalogInfo
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from tests.test_fast_deploy_screen import _StyledApp
from tests.test_serving_tiers import _plan
import unittest


def test_accessible_modal_rate_is_read_once_per_card():
    html = """
    <li>Nvidia H100 SXM5 $<span><span class="sr-only">0.001097</span>
    <span aria-hidden="true"><span>0</span>.<span>1</span></span></span> / sec</li>
    <li>Nvidia H200 SXM $<span class="sr-only">0.001261</span> / sec</li>
    """
    rates = _parse_modal_gpu_pricing(html)
    assert round(rates["H100"], 4) == 3.9492
    assert round(rates["H200"], 4) == 4.5396


def test_economy_ranks_known_total_then_precision_count_and_quote_id():
    base = _plan("4bit", price=1, single_tps=40, aggregate_tps=80)
    def offer(label, quant, count, price, once):
        return replace(base, recipe=replace(base.recipe, quant=quant), quote=replace(
            base.quote, id=label, gpu_count=count, price_per_hour_usd=price, one_time_cost_usd=once
        ))
    plans = [base, offer("transfer", "Q8_0", 1, .2, 1), offer("unpriced", "Q8_0", 1, .2, None),
             offer("unknown quant", None, 1, .2, 0), offer("small", "Q2_K", 1, .1, 0),
             offer("pair", "Q8_0", 2, 1, 0), offer("z", "Q8_0", 1, 1, 0), offer("a", "Q8_0", 1, 1, 0)]
    assert economy_plan(plans).quote.id == "a"
    assert economy_plan(list(reversed(plans))).quote.id == "a"
    assert economy_plan(plans[2:5]) is None


def test_modal_a10_capacity_is_limited_to_four_cards():
    snapshot = aggregate_compute_availability(modal_catalog=[ModalGpuSpec("A10", 1)])
    placement, = snapshot.configurations[0].placements
    assert placement.gpu_count_max == 4
    from llm_launchpad.core.quick_deploy_refresh import _select_gpu_shape

    assert _select_gpu_shape("Q4_K_M", 120, [ModalGpuSpec("A10", 1)]) is None


def test_modal_rtx_pro_6000_is_limited_to_four_cards_and_fewer_gpus_wins_close_cost():
    snapshot = aggregate_compute_availability(modal_catalog=[ModalGpuSpec("RTX-PRO-6000", 3.0312)])
    placement, = snapshot.configurations[0].placements
    assert placement.gpu_count_max == 4
    from llm_launchpad.core.quick_deploy_refresh import _select_gpu_shape

    # 8x RTX-PRO-6000 is not offered; 4x B200 is selected instead.
    assert _select_gpu_shape("UD-Q4_K_XL", 604.0, [ModalGpuSpec("RTX-PRO-6000", 3.0312)]) is None
    selected = _select_gpu_shape(
        "UD-Q4_K_XL",
        604.0,
        [ModalGpuSpec("RTX-PRO-6000", 3.0312), ModalGpuSpec("B200", 6.2496)],
    )
    assert selected is not None
    assert (selected.gpu_type, selected.gpu_count) == ("B200", 4)

    # Even among 8-card-capable GPUs, a 4-GPU shape within 5% of an 8-GPU
    # shape's hourly price wins to avoid multi-GPU scheduling stalls.
    close_cost = _select_gpu_shape(
        "Q4_K_M",
        250.0,
        [ModalGpuSpec("A100-80GB", 2.0), ModalGpuSpec("B200", 4.15)],
    )
    assert close_cost is not None
    assert (close_cost.gpu_type, close_cost.gpu_count) == ("B200", 2)


def test_modal_gpu_count_options_uses_power_of_two_slices():
    from llm_launchpad.core.modal_gpu import modal_gpu_count_options
    from llm_launchpad.core.quick_deploy_refresh import _select_gpu_shape

    assert modal_gpu_count_options("A100-80GB") == (1, 2, 4, 8)
    assert modal_gpu_count_options("H200") == (1, 2, 4, 8)
    assert modal_gpu_count_options("RTX-PRO-6000") == (1, 2, 4)
    assert modal_gpu_count_options("A10") == (1, 2, 4)

    # A 336 GiB model + 17.07 GiB per-device graph used to pick 6x A100-80GB
    # ($14.99/hr), which stalls on Modal's scheduler. Power-of-two slicing
    # skips 6x A100-80GB and picks 4x H200 ($18.16/hr) over 8x A100-80GB ($19.99/hr).
    selected = _select_gpu_shape(
        "UD-IQ1_M",
        352.97,
        [
            ModalGpuSpec("A100-80GB", 2.4984),
            ModalGpuSpec("H200", 4.5396),
            ModalGpuSpec("B200", 6.2496),
        ],
        per_device_overhead_gb=17.074,
        layer_count=40,
    )
    assert selected is not None
    assert (selected.gpu_type, selected.gpu_count) == ("H200", 4)


def test_changed_or_stale_modal_bundle_is_refused_before_deploy():
    config = DeploymentConfig(backend=BackendType.LLAMACPP, gpu_type="A10", gpu_count=2,
                              provider_options=ModalProviderOptions(2.0), do_deploy=True)
    runner = Orchestrator()
    for spec in (ModalGpuSpec("A10", 1.01), ModalGpuSpec("A10", .9, stale_since=1.0), ModalGpuSpec("A10", None)):
        with patch("llm_launchpad.core.orchestrator.fetch_modal_gpu_catalog", return_value=[spec]) as fetch, \
             patch.object(runner, "_deploy_llamacpp") as deploy:
            events = list(runner.deploy_modal_only(config))
            assert events[-1].success is False
            fetch.assert_called_once_with(force_refresh=True)
            deploy.assert_not_called()


class PickerTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_replans_reprices_and_reaches_the_deployment_config(self):
        catalog = [ModalGpuSpec("L4", 1), ModalGpuSpec("H100", 3)]
        layout = GgufQuantMetadata(["Q4_K_M"], {"Q4_K_M": 10}, architecture="llama",
                                  context_length=131072, block_count=32, embedding_length=4096,
                                  attention_head_count=32, attention_head_count_kv=8,
                                  attention_key_length=128, attention_value_length=128)
        profiles = _profiles_for_model(ModelCandidate("org/context-GGUF"), catalog,
                                       metadata=layout, prefer_mtp=False)
        cached = [_quick_deploy_profile_from_dict(_quick_deploy_profile_to_dict(p)) for p in profiles]
        self.assertEqual(profiles, cached)
        model, = list_quick_deploy_models(cached)
        app = _StyledApp()
        path = "llm_launchpad.tui.screens.fast_deploy"
        with (patch(f"{path}.get_quick_deploy_catalog_info", return_value=QuickDeployCatalogInfo("Test")),
              patch(f"{path}.list_quick_deploy_models", return_value=(model,)),
              patch(f"{path}.load_compute_availability", return_value=aggregate_compute_availability(modal_catalog=catalog))):
            async with app.run_test(size=(80, 24)) as pilot:
                screen = FastDeployScreen()
                app.push_screen(screen)
                await pilot.pause()
                self.assertIn("$2.00/hr", str(screen.query_one(OptionList).get_option_at_index(0).prompt))
                await pilot.press("enter")
                self.assertEqual(screen._context_tokens, None)
                self.assertEqual(screen.query_one("#fast-deploy-context", Select).value, 131072)
                await pilot.press("c", "enter")
                plan = app.quick_deploy_calls[-1][0]
                self.assertEqual((plan.quote.gpu_count, plan.quote.price_per_hour_usd), (1, 1))
                config = build_quick_deploy_config(app.catalog_profiles[-1], plan=plan)
                self.assertEqual(config.max_context_tokens, 32768)
                self.assertEqual(config.serving_requirements.context_tokens, 32768)
                self.assertIn("--ctx-size 32768", config.server_args)
                self.assertEqual(config.provider_options.max_hourly_cost_usd, 1)
                app.push_screen(FastDeployScreen(initial_model_id=model.id, preferred_context=8192))
                await pilot.pause()
                self.assertEqual(app.screen.query_one("#fast-deploy-context", Select).value, 8192)

    async def test_blocked_models_can_be_found_from_the_default_view(self):
        blocked = [
            CatalogExclusion("ds", "DeepSeek V4.1 Flash 1017", "org/DeepSeek-V4.1-Flash-1017-GGUF", "Unsupported deepseek41"),
            CatalogExclusion("ink", "Inkling 2 Small", "org/Inkling-2-Small-GGUF", "Unsupported inkling2"),
        ]
        info = QuickDeployCatalogInfo("Ranked catalog", exclusions=tuple(blocked))
        path = "llm_launchpad.tui.screens.fast_deploy"
        app = _StyledApp()
        with (patch(f"{path}.get_quick_deploy_catalog_info", return_value=info),
              patch(f"{path}.list_quick_deploy_models", return_value=()),
              patch(f"{path}.load_compute_availability", return_value=aggregate_compute_availability())):
            async with app.run_test(size=(80, 24)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()
                for query, exclusion in zip(("DeepSeekv4.1-flash", "inkling-2-small"), blocked, strict=True):
                    app.screen.query_one(Input).value = query
                    await pilot.pause()
                    self.assertIn(exclusion.reason, str(app.screen.query_one("#fast-deploy-detail", Static).content))
                    await pilot.press("enter")
                    self.assertEqual(app.screen.exclusions, (exclusion,))
                    await pilot.press("escape")
                self.assertFalse(app.quick_deploy_calls)

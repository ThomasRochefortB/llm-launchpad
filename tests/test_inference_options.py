from __future__ import annotations

import unittest

from llm_launchpad.core.inference_options import (
    ModalCatalogOption,
    ModalInferenceAdapter,
    PrimeInferenceAdapter,
    estimate_monthly_compute_cost,
    recommended_vllm_tool_call_parser,
    resolve_inference_plans,
)
from llm_launchpad.core.llamacpp_planner import serving_requirements, tuning_for_objective
from llm_launchpad.core.prime_backend import is_compatible_prime_offer
from llm_launchpad.core.provider_options import prime_provider_options
from llm_launchpad.core.quick_deploy import (
    QuickDeployProfile,
    build_quick_deploy_config,
    list_quick_deploy_models,
    list_quick_deploy_recipes,
    quick_deploy_profile_for_plan,
    resolve_quick_deploy_plans,
)
from llm_launchpad.protocol.enums import (
    BackendType,
    BillingModel,
    ComputeProvider,
    QuoteAvailability,
    ServingObjective,
)
from llm_launchpad.protocol.models import (
    ComputeOffer,
    DeploymentConfig,
    InferenceRecipe,
    MemoryEstimate,
    PrimeProviderOptions,
    ProviderQuote,
    WorkloadProfile,
)


def _profile(
    profile_id: str,
    *,
    gpu_type: str = "L40S",
    gpu_count: int = 1,
    price: float = 2.0,
) -> QuickDeployProfile:
    return QuickDeployProfile(
        id=profile_id,
        display_name="Example Model",
        repo_id="org/example-GGUF",
        quant="Q4_K_M",
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        profile_label="Curated",
        approx_cost_per_hour_usd=price,
        max_context_tokens=32768,
        instance_slug_hint="example",
        summary="Example summary.",
        server_args=("--ctx-size", "32768"),
        required_vram_gb=30.0,
        aa_model_id="model-1",
        aa_coding_score=42.0,
        aa_rank=3,
    )


def _planned_profile(
    profile_id: str = "example-planned",
    *,
    context_tokens: int = 262144,
    full_context_total_gb: float = 60.0,
    legacy_required_vram_gb: float = 40.0,
) -> QuickDeployProfile:
    """A catalog entry whose full-context estimate exceeds its weight size."""

    requirements = serving_requirements(context_tokens)
    tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)
    return QuickDeployProfile(
        id=profile_id,
        display_name="Planned Model",
        repo_id="org/planned-GGUF",
        quant="Q4_K_M",
        gpu_type="H100_80GB",
        gpu_count=1,
        profile_label="Curated",
        approx_cost_per_hour_usd=2.0,
        max_context_tokens=context_tokens,
        instance_slug_hint="planned",
        summary="Planned summary.",
        server_args=("--ctx-size", str(context_tokens)),
        required_vram_gb=legacy_required_vram_gb,
        gpu_memory_gb=80.0,
        llamacpp_runtime_id="runtime-1",
        serving_requirements=requirements,
        runtime_tuning=tuning,
        memory_estimate=MemoryEstimate(
            weights_gb=legacy_required_vram_gb,
            kv_cache_gb=full_context_total_gb - legacy_required_vram_gb - 2.0,
            compute_gb=2.0,
            speculative_gb=0.0,
            reserve_gb=0.0,
            total_gb=full_context_total_gb,
            per_device_required_gb=(full_context_total_gb,),
            confidence=0.82,
            source="gguf-metadata",
            total_layer_count=48,
        ),
    )


def _offer(
    offer_id: str,
    *,
    gpu_type: str,
    gpu_count: int = 1,
    price: float = 2.0,
    gpu_memory_gb: float | None = None,
) -> ComputeOffer:
    return ComputeOffer(
        id=offer_id,
        cloud_id=f"cloud-{offer_id}",
        provider_name="provider",
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        price_per_hour=price,
        security="secure_cloud",
        stock_status="Available",
        images=("ubuntu_22_cuda_12",),
    )


class _FakePrimeBackend:
    def __init__(self, offers: list[ComputeOffer]) -> None:
        self.offers = offers
        self.calls = 0

    def list_offers(self) -> list[ComputeOffer]:
        self.calls += 1
        return self.offers


class InferenceRecipeTests(unittest.TestCase):
    def test_recommends_only_known_qwen_tool_parsers(self) -> None:
        self.assertEqual(
            recommended_vllm_tool_call_parser("Qwen/Qwen3-0.6B"),
            "hermes",
        )
        self.assertEqual(
            recommended_vllm_tool_call_parser(
                "Qwen/Qwen3-Coder-30B-A3B-Instruct"
            ),
            "qwen3_xml",
        )
        self.assertIsNone(
            recommended_vllm_tool_call_parser("Qwen/Qwen3-VL-8B-Instruct")
        )
        self.assertIsNone(
            recommended_vllm_tool_call_parser("meta-llama/Llama-3.1-8B-Instruct")
        )

    def test_quick_deploy_separates_one_recipe_from_multiple_modal_quotes(self) -> None:
        profiles = (
            _profile("example-cheap", gpu_type="L40S", price=1.5),
            _profile("example-fast", gpu_type="H100", price=3.5),
        )

        recipes = list_quick_deploy_recipes(profiles)
        plans = resolve_quick_deploy_plans(profiles)
        models = list_quick_deploy_models(profiles)

        self.assertEqual(len(recipes), 1)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].recipes, recipes)
        self.assertEqual({plan.quote.id for plan in plans}, {"example-cheap", "example-fast"})
        self.assertTrue(all(plan.quote.provider == ComputeProvider.MODAL for plan in plans))

    def test_plan_builds_existing_deployment_config_through_typed_quote(self) -> None:
        profile = _profile("example-cheap")
        plan = resolve_quick_deploy_plans((profile,))[0]

        config = build_quick_deploy_config(profile, plan=plan)

        self.assertEqual(config.provider, ComputeProvider.MODAL)
        self.assertEqual(config.gpu_type, "L40S")
        self.assertEqual(config.app_name, "llamacpp-example")
        self.assertIs(config.provider_options, plan.quote.provider_options)

    def test_vllm_recipe_builds_prime_config_without_gguf_fields(self) -> None:
        profile = QuickDeployProfile(
            id="example-vllm",
            display_name="Example Model",
            repo_id="org/example",
            quant="",
            gpu_type="H100_80GB",
            gpu_count=1,
            profile_label="Curated",
            approx_cost_per_hour_usd=2.0,
            max_context_tokens=32768,
            instance_slug_hint="example-vllm",
            summary="Example vLLM recipe.",
            server_args=(),
            required_vram_gb=70.0,
            backend=BackendType.VLLM,
            model_name="org/example",
        )
        backend = _FakePrimeBackend(
            [
                ComputeOffer(
                    id="h100",
                    cloud_id="cloud-h100",
                    provider_name="provider",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    price_per_hour=2.0,
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                )
            ]
        )
        adapter = PrimeInferenceAdapter(
            backend=backend,  # type: ignore[arg-type]
            provider_options=PrimeProviderOptions(),
        )

        plan = resolve_quick_deploy_plans((profile,), adapters=(adapter,))[0]
        selected_profile = quick_deploy_profile_for_plan(plan, (profile,))
        config = build_quick_deploy_config(selected_profile, plan=plan)

        self.assertEqual(config.provider, ComputeProvider.PRIME)
        self.assertEqual(config.backend, BackendType.VLLM)
        self.assertEqual(config.model_name, "org/example")
        self.assertEqual(config.n_gpu, 1)
        self.assertEqual(config.app_name, "llp-prime-vllm-example-vllm")
        self.assertEqual(
            config.provider_options,
            PrimeProviderOptions(
                offer_id="h100",
            ),
        )

    def test_llamacpp_recipe_builds_prime_config_from_existing_bundle(self) -> None:
        profile = _profile("example-prime", gpu_type="H100_80GB", price=2.0)
        backend = _FakePrimeBackend(
            [
                ComputeOffer(
                    id="h100",
                    cloud_id="cloud-h100",
                    provider_name="provider",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    price_per_hour=1.75,
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                )
            ]
        )
        adapter = PrimeInferenceAdapter(
            backend=backend,  # type: ignore[arg-type]
            provider_options=PrimeProviderOptions(),
        )

        plan = resolve_quick_deploy_plans((profile,), adapters=(adapter,))[0]
        selected_profile = quick_deploy_profile_for_plan(plan, (profile,))
        config = build_quick_deploy_config(selected_profile, plan=plan)

        self.assertEqual(config.provider, ComputeProvider.PRIME)
        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "org/example-GGUF")
        self.assertEqual(config.quant, "Q4_K_M")
        self.assertEqual(config.app_name, "llp-prime-llamacpp-example")
        self.assertEqual(
            config.provider_options,
            PrimeProviderOptions(
                offer_id="h100",
            ),
        )


class ProviderAdapterTests(unittest.TestCase):
    def test_recommends_lowest_cost_option_for_each_recipe(self) -> None:
        recipes = (
            InferenceRecipe(
                id="recipe-a",
                model_key="a",
                display_name="A",
                backend=BackendType.VLLM,
                model_id="org/a",
            ),
            InferenceRecipe(
                id="recipe-b",
                model_key="b",
                display_name="B",
                backend=BackendType.VLLM,
                model_id="org/b",
            ),
        )
        adapter = ModalInferenceAdapter(
            [
                ModalCatalogOption("a-expensive", "recipe-a", "H100", 1, 4.0),
                ModalCatalogOption("a-cheap", "recipe-a", "L40S", 1, 1.0),
                ModalCatalogOption("b-cheap", "recipe-b", "L40S", 1, 2.0),
            ]
        )

        plans = resolve_inference_plans(recipes, (adapter,))

        recommended = {
            plan.quote.id
            for plan in plans
            if plan.recommendation_reason is not None
        }
        self.assertEqual(recommended, {"a-cheap", "b-cheap"})

    def test_prime_adapter_supports_llamacpp_recipe(self) -> None:
        backend = _FakePrimeBackend(
            [
                ComputeOffer(
                    id="h100",
                    cloud_id="cloud-h100",
                    provider_name="provider",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    price_per_hour=2.0,
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                )
            ]
        )
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]
        recipe = InferenceRecipe(
            id="gguf",
            model_key="model",
            display_name="Model",
            backend=BackendType.LLAMACPP,
            model_id="org/model-GGUF",
            required_vram_gb=70.0,
        )

        plans = resolve_inference_plans((recipe,), (adapter,))

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].quote.provider, ComputeProvider.PRIME)
        self.assertEqual(plans[0].quote.provider_reference, "h100")
        self.assertEqual(backend.calls, 1)

    def test_prime_adapter_normalizes_only_compatible_secure_fixed_offers(self) -> None:
        offers = [
            ComputeOffer(
                id="h100",
                cloud_id="cloud-h100",
                provider_name="provider",
                gpu_type="H100_80GB",
                gpu_count=1,
                price_per_hour=2.0,
                security="secure_cloud",
                stock_status="Available",
                images=("ubuntu_22_cuda_12",),
            ),
            ComputeOffer(
                id="spot",
                cloud_id="cloud-spot",
                provider_name="provider",
                gpu_type="H100_80GB",
                gpu_count=1,
                price_per_hour=0.5,
                security="secure_cloud",
                stock_status="Available",
                images=("ubuntu_22_cuda_12",),
                is_spot=True,
            ),
            ComputeOffer(
                id="small",
                cloud_id="cloud-small",
                provider_name="provider",
                gpu_type="L4",
                gpu_count=1,
                price_per_hour=0.4,
                security="secure_cloud",
                stock_status="Available",
                images=("ubuntu_22_cuda_12",),
            ),
        ]
        backend = _FakePrimeBackend(offers)
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]
        recipe = InferenceRecipe(
            id="vllm",
            model_key="model",
            display_name="Model",
            backend=BackendType.VLLM,
            model_id="org/model",
            required_vram_gb=70.0,
        )
        workload = WorkloadProfile(
            paid_hours_per_day=8,
            utilization=0.25,
            output_tokens_per_month=100_000_000,
        )

        plans = resolve_inference_plans((recipe,), (adapter,), workload)

        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.quote.provider, ComputeProvider.PRIME)
        self.assertEqual(plan.quote.availability, QuoteAvailability.AVAILABLE)
        self.assertEqual(plan.estimated_monthly_cost_usd, 480.0)
        self.assertEqual(plan.estimated_cost_per_million_output_tokens_usd, 4.8)
        self.assertIsInstance(plan.quote.provider_options, PrimeProviderOptions)
        assert isinstance(plan.quote.provider_options, PrimeProviderOptions)
        self.assertEqual(plan.quote.provider_options.offer_id, "h100")
        self.assertIsNone(plan.quote.provider_options.region)

    def test_prime_quote_keeps_display_country_out_of_api_region_filter(self) -> None:
        backend = _FakePrimeBackend(
            [
                ComputeOffer(
                    id="h100-ca",
                    cloud_id="cloud-h100-ca",
                    provider_name="provider",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    gpu_memory_gb=80,
                    price_per_hour=2.0,
                    region="north_america",
                    country="CA",
                    data_center="CANADA-1",
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                )
            ]
        )
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]
        recipe = InferenceRecipe(
            id="vllm-ca",
            model_key="model",
            display_name="Model",
            backend=BackendType.VLLM,
            model_id="org/model",
            required_vram_gb=70.0,
        )

        plan = resolve_inference_plans((recipe,), (adapter,))[0]

        self.assertEqual(plan.quote.region, "CA")
        options = plan.quote.provider_options
        self.assertIsInstance(options, PrimeProviderOptions)
        assert isinstance(options, PrimeProviderOptions)
        self.assertEqual(options.offer_id, "h100-ca")
        self.assertIsNone(options.region)

    def test_prime_adapter_reuses_one_marketplace_snapshot_across_recipes(self) -> None:
        backend = _FakePrimeBackend(
            [
                ComputeOffer(
                    id="h100",
                    cloud_id="cloud-h100",
                    provider_name="provider",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    price_per_hour=2.0,
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                )
            ]
        )
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]
        recipes = (
            InferenceRecipe(
                id="one",
                model_key="one",
                display_name="One",
                backend=BackendType.LLAMACPP,
                model_id="org/one-GGUF",
            ),
            InferenceRecipe(
                id="two",
                model_key="two",
                display_name="Two",
                backend=BackendType.VLLM,
                model_id="org/two",
            ),
        )

        plans = resolve_inference_plans(recipes, (adapter,))

        self.assertEqual(len(plans), 2)
        self.assertEqual(backend.calls, 1)

    def test_prime_adapter_excludes_cpu_and_sizes_each_model_independently(self) -> None:
        backend = _FakePrimeBackend(
            [
                ComputeOffer(
                    id="cpu",
                    cloud_id="cpu",
                    provider_name="provider",
                    gpu_type="CPU_NODE",
                    gpu_count=1,
                    gpu_memory_gb=512,
                    price_per_hour=0.05,
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                ),
                ComputeOffer(
                    id="l4",
                    cloud_id="l4",
                    provider_name="provider",
                    gpu_type="L4_24GB",
                    gpu_count=1,
                    gpu_memory_gb=24,
                    price_per_hour=0.5,
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                ),
                ComputeOffer(
                    id="h100",
                    cloud_id="h100",
                    provider_name="provider",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    gpu_memory_gb=80,
                    price_per_hour=2.0,
                    security="secure_cloud",
                    stock_status="Available",
                    images=("ubuntu_22_cuda_12",),
                ),
            ]
        )
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]
        recipes = (
            InferenceRecipe(
                id="small",
                model_key="small",
                display_name="Small",
                backend=BackendType.LLAMACPP,
                model_id="org/small-GGUF",
                required_vram_gb=20.0,
            ),
            InferenceRecipe(
                id="large",
                model_key="large",
                display_name="Large",
                backend=BackendType.LLAMACPP,
                model_id="org/large-GGUF",
                required_vram_gb=70.0,
            ),
        )

        plans = resolve_inference_plans(recipes, (adapter,))

        offers_by_recipe = {
            recipe.id: {
                plan.quote.provider_reference
                for plan in plans
                if plan.recipe.id == recipe.id
            }
            for recipe in recipes
        }
        self.assertEqual(offers_by_recipe["small"], {"l4", "h100"})
        self.assertEqual(offers_by_recipe["large"], {"h100"})
        self.assertNotIn("cpu", {plan.quote.provider_reference for plan in plans})

    def test_prime_availability_handles_human_readable_stock_statuses(self) -> None:
        from llm_launchpad.core.prime_backend import prime_stock_availability

        expected = {
            "Out of Stock": QuoteAvailability.UNAVAILABLE,
            "Out_of_Stock": QuoteAvailability.UNAVAILABLE,
            "In Stock": QuoteAvailability.AVAILABLE,
            "  Available  ": QuoteAvailability.AVAILABLE,
            "who knows": QuoteAvailability.UNKNOWN,
            None: QuoteAvailability.UNKNOWN,
        }
        for status, availability in expected.items():
            with self.subTest(stock_status=status):
                self.assertEqual(prime_stock_availability(status), availability)

    def test_billing_models_normalize_idle_time_differently(self) -> None:
        workload = WorkloadProfile(paid_hours_per_day=8, utilization=0.25)
        common = {
            "recipe_id": "recipe",
            "provider_reference": "ref",
            "gpu_type": "H100",
            "gpu_count": 1,
            "price_per_hour_usd": 2.0,
        }
        modal = ProviderQuote(
            id="modal",
            provider=ComputeProvider.MODAL,
            billing_model=BillingModel.SCALE_TO_ZERO,
            **common,  # type: ignore[arg-type]
        )
        prime = ProviderQuote(
            id="prime",
            provider=ComputeProvider.PRIME,
            billing_model=BillingModel.PROVISIONED,
            **common,  # type: ignore[arg-type]
        )

        self.assertEqual(estimate_monthly_compute_cost(modal, workload), 120.0)
        self.assertEqual(estimate_monthly_compute_cost(prime, workload), 480.0)

    def test_provider_option_accessor_rejects_cross_provider_payload(self) -> None:
        config = DeploymentConfig(provider_options=None)
        self.assertEqual(prime_provider_options(config), PrimeProviderOptions())

        modal_option = ModalInferenceAdapter(
            [ModalCatalogOption("option", "recipe", "L4", 1, 0.5)]
        ).quote(
            InferenceRecipe(
                id="recipe",
                model_key="model",
                display_name="Model",
                backend=BackendType.VLLM,
                model_id="org/model",
            ),
            WorkloadProfile(),
        )[0].provider_options
        config.provider_options = modal_option
        with self.assertRaisesRegex(ValueError, "non-Prime"):
            prime_provider_options(config)


class PrimePlacementParityTests(unittest.TestCase):
    """Prime quotes must carry the same planner evidence as Modal quotes."""

    def test_prime_quote_reports_per_gpu_memory_not_aggregate_node_memory(self) -> None:
        backend = _FakePrimeBackend(
            [_offer("h100x8", gpu_type="H100_80GB", gpu_count=8, gpu_memory_gb=640.0)]
        )
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]
        recipe = InferenceRecipe(
            id="recipe",
            model_key="model",
            display_name="Model",
            backend=BackendType.LLAMACPP,
            model_id="org/model",
        )

        quote = adapter.quote(recipe, WorkloadProfile())[0]

        self.assertEqual(quote.gpu_memory_gb, 80.0)

    def test_prime_placement_is_rejected_when_full_context_exceeds_one_gpu(self) -> None:
        profile = _planned_profile()
        backend = _FakePrimeBackend([_offer("l40s", gpu_type="L40S_48GB", price=1.1)])
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]

        # The legacy weight-only rule admits this offer: 48 GB clears the
        # 40 GB of weights plus 5% headroom. The full-context estimate does not.
        recipe = list_quick_deploy_recipes((profile,))[0]
        self.assertTrue(
            is_compatible_prime_offer(
                backend.offers[0],
                profile.required_vram_gb,
                required_image="ubuntu_22_cuda_12",
            )
        )
        self.assertEqual(len(adapter.quote(recipe, WorkloadProfile())), 1)

        self.assertEqual(resolve_quick_deploy_plans((profile,), adapters=(adapter,)), ())

    def test_prime_placement_that_holds_full_context_keeps_planner_evidence(self) -> None:
        profile = _planned_profile()
        backend = _FakePrimeBackend([_offer("h100", gpu_type="H100_80GB", price=2.4)])
        adapter = PrimeInferenceAdapter(backend=backend)  # type: ignore[arg-type]

        plans = resolve_quick_deploy_plans((profile,), adapters=(adapter,))

        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.quote.provider, ComputeProvider.PRIME)
        self.assertEqual(plan.quote.gpu_memory_gb, 80.0)
        assessment = plan.assessment
        assert assessment is not None
        self.assertTrue(assessment.fits)
        self.assertTrue(assessment.gpu_resident)
        # 60 GB of model/KV/compute plus a 5% per-device reserve on 80 GB.
        self.assertAlmostEqual(assessment.memory.total_gb, 64.0, places=3)
        # The reserve is also promised to llama.cpp's runtime fitter.
        self.assertEqual(assessment.tuning.fit_target_mib, 4096)
        self.assertIsNotNone(plan.quote.estimated_output_tokens_per_second)
        self.assertIsNotNone(plan.quote.estimated_aggregate_output_tokens_per_second)

    def test_prime_and_modal_agree_on_placement_for_the_same_topology(self) -> None:
        profile = _planned_profile()
        prime = PrimeInferenceAdapter(
            backend=_FakePrimeBackend(  # type: ignore[arg-type]
                [_offer("h100", gpu_type="H100_80GB", price=2.4)]
            )
        )
        modal = ModalInferenceAdapter(
            [
                ModalCatalogOption(
                    id=profile.id,
                    recipe_id=list_quick_deploy_recipes((profile,))[0].id,
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    price_per_hour_usd=2.4,
                    gpu_memory_gb=80.0,
                )
            ]
        )

        prime_plan = resolve_quick_deploy_plans((profile,), adapters=(prime,))[0]
        modal_plan = resolve_quick_deploy_plans((profile,), adapters=(modal,))[0]

        assert prime_plan.assessment is not None and modal_plan.assessment is not None
        self.assertEqual(
            prime_plan.assessment.fingerprint,
            modal_plan.assessment.fingerprint,
        )
        self.assertEqual(
            prime_plan.assessment.memory,
            modal_plan.assessment.memory,
        )
        self.assertEqual(prime_plan.recipe.server_args, modal_plan.recipe.server_args)


if __name__ == "__main__":
    unittest.main()

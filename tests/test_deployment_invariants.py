"""Cross-layer placement and timeout invariants for Modal deployments.

These property-style checks guard the contracts between catalog selection,
live availability sizing, memory estimation, and warmup timeouts so that
unschedulable GPU counts or self-defeating timeouts fail fast in CI ($0).
"""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.compute_availability import (
    _placement_gpu_count,
    aggregate_compute_availability,
    assessed_plans_for_compute_profile,
)
from llm_launchpad.core.deployment_lifecycle import LifecycleOptions
from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.llamacpp_planner import (
    compile_server_args,
    device_capacity_bytes,
    estimate_memory,
    gib_to_bytes,
    serving_requirements,
    tuning_for_gpu_memory,
    tuning_for_objective,
)
from llm_launchpad.core.modal_gpu import (
    MODAL_GPU_MEMORY_GB,
    ModalGpuSpec,
    modal_gpu_count_limit,
    modal_gpu_count_options,
    modal_gpu_supports_fast_deploy,
)
from llm_launchpad.core.quick_deploy import QuickDeployProfile
from llm_launchpad.core.quick_deploy_refresh import (
    _FALLBACK_GPU_PRICE_PER_HOUR,
    _select_gpu_shape,
)
from llm_launchpad.core.warmup import (
    WarmupRunner,
    _CALIBRATION_WARMUP_TIMEOUT_SECONDS,
    _MODAL_COLD_RETRY_BACKOFF_SECONDS,
    _MODAL_PROBE_TIMEOUT_SECONDS,
    _probe_timeout_seconds,
)
from llm_launchpad.protocol.enums import ComputeProvider, ServingObjective

_FULL_MODAL_CATALOG = tuple(
    ModalGpuSpec(gpu, price_per_hour_usd=_FALLBACK_GPU_PRICE_PER_HOUR.get(gpu, 2.5))
    for gpu in MODAL_GPU_MEMORY_GB
)


def _synthetic_profile(weights_gb: float, context_tokens: int, layers: int) -> QuickDeployProfile:
    metadata = GgufQuantMetadata(
        quantizations=["Q4_K_M"],
        vram_gb_by_quant={"Q4_K_M": weights_gb},
        architecture="llama",
        context_length=context_tokens,
        block_count=layers,
        embedding_length=4096,
        attention_head_count=32,
        attention_head_count_kv=8,
        attention_key_length=128,
        attention_value_length=128,
    )
    requirements = serving_requirements(context_tokens, objective=ServingObjective.GENERAL_PURPOSE)
    tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)
    memory = estimate_memory(
        metadata,
        weights_gb=weights_gb,
        requirements=requirements,
        tuning=tuning,
        gpu_count=1,
    )
    return QuickDeployProfile(
        id=f"sweep-{int(weights_gb)}gb-{context_tokens}",
        display_name=f"Sweep {weights_gb:.0f}GB",
        repo_id="org/sweep-GGUF",
        quant="Q4_K_M",
        gpu_type="H100",
        gpu_count=1,
        profile_label="Sweep",
        approx_cost_per_hour_usd=4.0,
        max_context_tokens=context_tokens,
        instance_slug_hint="sweep",
        summary="Sweep profile.",
        server_args=tuple(compile_server_args(requirements, tuning)),
        required_vram_gb=memory.total_gb,
        serving_requirements=requirements,
        runtime_tuning=tuning,
        memory_estimate=memory,
    )


class ModalPlacementInvariantTests(unittest.TestCase):
    def test_every_emitted_modal_plan_satisfies_scheduling_and_memory_contracts(self) -> None:
        snapshot = aggregate_compute_availability(modal_catalog=_FULL_MODAL_CATALOG)
        sweep = (
            (5.0, 32_768, 32),
            (18.0, 65_536, 40),
            (45.0, 131_072, 48),
            (110.0, 262_144, 60),
            (250.0, 1_048_576, 40),
            (500.0, 1_048_576, 40),
            (850.0, 1_048_576, 93),
        )
        observed_plans = 0

        for weights_gb, context_tokens, layers in sweep:
            profile = _synthetic_profile(weights_gb, context_tokens, layers)
            assert profile.memory_estimate is not None
            per_dev_overhead = (
                profile.memory_estimate.compute_gb + profile.memory_estimate.attention_scratch_gb
            )

            # Catalog selection invariant.
            selected = _select_gpu_shape(
                profile.quant,
                profile.memory_estimate.total_gb,
                _FULL_MODAL_CATALOG,
                per_device_overhead_gb=per_dev_overhead,
                layer_count=layers,
            )
            if selected is not None:
                self.assertTrue(modal_gpu_supports_fast_deploy(selected.gpu_type))
                self.assertIn(selected.gpu_count, modal_gpu_count_options(selected.gpu_type))
                self.assertLessEqual(selected.gpu_count, modal_gpu_count_limit(selected.gpu_type))

            # Live availability planning invariant across every GPU family.
            for config in snapshot.configurations:
                plans, _ = assessed_plans_for_compute_profile(config, profile)
                for plan in plans:
                    observed_plans += 1
                    quote = plan.quote
                    self.assertEqual(quote.provider, ComputeProvider.MODAL)
                    self.assertTrue(
                        modal_gpu_supports_fast_deploy(quote.gpu_type),
                        f"{quote.gpu_type} should be excluded from Fast Deploy",
                    )
                    allowed_counts = modal_gpu_count_options(quote.gpu_type)
                    self.assertIn(
                        quote.gpu_count,
                        allowed_counts,
                        f"{quote.gpu_count}x {quote.gpu_type} is not in Modal ladder {allowed_counts}",
                    )
                    self.assertLessEqual(quote.gpu_count, modal_gpu_count_limit(quote.gpu_type))

                    assert plan.assessment is not None
                    assert quote.gpu_memory_gb is not None
                    capacity = device_capacity_bytes(quote.gpu_memory_gb)
                    self.assertTrue(plan.assessment.fits)
                    self.assertTrue(plan.assessment.gpu_resident)
                    for device_gb in plan.assessment.memory.per_device_required_gb:
                        self.assertLessEqual(gib_to_bytes(device_gb), capacity)

                    # The per-device margin in server_args must match the chosen GPU size.
                    assert plan.recipe.runtime_tuning is not None
                    expected_tuning = tuning_for_gpu_memory(
                        profile.runtime_tuning,  # type: ignore[arg-type]
                        quote.gpu_memory_gb,
                    )
                    self.assertEqual(
                        plan.recipe.runtime_tuning.fit_target_mib,
                        expected_tuning.fit_target_mib,
                    )
                    args = list(plan.recipe.server_args or ())
                    self.assertEqual(
                        args[args.index("--fit-target") + 1],
                        str(expected_tuning.fit_target_mib),
                    )

        self.assertGreater(observed_plans, 25)

    def test_catalog_and_live_availability_agree_on_modal_gpu_counts(self) -> None:
        snapshot = aggregate_compute_availability(modal_catalog=_FULL_MODAL_CATALOG)
        placements_by_gpu = {
            placement.gpu_type: placement
            for config in snapshot.configurations
            for placement in config.placements
            if placement.provider == ComputeProvider.MODAL
        }

        for weights_gb, context_tokens, layers in (
            (15.0, 32_768, 32),
            (90.0, 131_072, 45),
            (250.0, 1_048_576, 40),
            (500.0, 1_048_576, 40),
        ):
            profile = _synthetic_profile(weights_gb, context_tokens, layers)
            assert profile.memory_estimate is not None
            per_dev_overhead = (
                profile.memory_estimate.compute_gb + profile.memory_estimate.attention_scratch_gb
            )
            for gpu_type, placement in placements_by_gpu.items():
                if not modal_gpu_supports_fast_deploy(gpu_type):
                    continue
                catalog_choice = _select_gpu_shape(
                    profile.quant,
                    profile.memory_estimate.total_gb,
                    _FULL_MODAL_CATALOG,
                    gpu_type=gpu_type,
                    per_device_overhead_gb=per_dev_overhead,
                    layer_count=layers,
                )
                live_count = _placement_gpu_count(
                    placement,
                    profile.memory_estimate.total_gb,
                    memory_estimate=profile.memory_estimate,
                )
                expected_count = catalog_choice.gpu_count if catalog_choice is not None else None
                self.assertEqual(
                    live_count,
                    expected_count,
                    f"Catalog vs live gpu_count drift for {weights_gb}GB on {gpu_type}",
                )

    def test_modal_gpu_count_options_are_strictly_powers_of_two(self) -> None:
        for gpu_type in MODAL_GPU_MEMORY_GB:
            options = modal_gpu_count_options(gpu_type)
            self.assertTrue(options)
            self.assertEqual(options, tuple(sorted(options)))
            for count in options:
                self.assertEqual(count & (count - 1), 0, f"{gpu_type} option {count} is not a power of two")
                self.assertLessEqual(count, modal_gpu_count_limit(gpu_type))


class TimeoutBudgetInvariantTests(unittest.TestCase):
    def test_modal_system_log_timeout_exceeds_typical_cli_round_trip(self) -> None:
        captured: list[float] = []

        def _fake_run(*_args: object, **kwargs: object) -> object:
            captured.append(float(kwargs["timeout"]))  # type: ignore[arg-type]
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        with patch("llm_launchpad.core.backend.subprocess.run", side_effect=_fake_run):
            ModalBackend.recent_system_logs("app-name", 1700000000.0)

        self.assertEqual(len(captured), 1)
        self.assertGreaterEqual(captured[0], 15.0)

    def test_modal_probe_timeout_holds_cold_starts_without_exceeding_modal_303_limit(self) -> None:
        modal_probe = _probe_timeout_seconds(ComputeProvider.MODAL, 1800, 0)
        prime_probe = _probe_timeout_seconds(ComputeProvider.PRIME, 1800, 0)

        self.assertGreaterEqual(_MODAL_PROBE_TIMEOUT_SECONDS, 30.0)
        self.assertLess(_MODAL_PROBE_TIMEOUT_SECONDS, 150.0)
        self.assertEqual(modal_probe, _MODAL_PROBE_TIMEOUT_SECONDS)
        self.assertEqual(prime_probe, 10.0)
        self.assertLessEqual(_MODAL_COLD_RETRY_BACKOFF_SECONDS, 5.0)

    def test_warmup_and_calibration_timeouts_stay_aligned(self) -> None:
        warmup_default = inspect.signature(WarmupRunner.run).parameters["timeout"].default
        self.assertEqual(LifecycleOptions().warmup_timeout_seconds, warmup_default)
        self.assertGreaterEqual(_CALIBRATION_WARMUP_TIMEOUT_SECONDS, 60.0)


if __name__ == "__main__":
    unittest.main()

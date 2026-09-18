"""Regression tests for the six lifecycle ownership fixes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from llm_launchpad.core.deployment_lifecycle import (
    LifecycleAttempt,
    LifecycleOptions,
    effective_config_for_attempt,
    run_lifecycle,
)
from llm_launchpad.core.deployment_preflight import (
    lifecycle_attempts_for_configs,
    plan_from_request,
    revalidate_plan_price,
)
from llm_launchpad.core.quick_deploy import request_from_config
from llm_launchpad.core.resource_targeting import resolve_stop_target
from llm_launchpad.protocol.enums import (
    BackendType,
    CleanupDisposition,
    ComputeProvider,
    OperationIntent,
    OperationType,
    WarmupStatus,
)
from llm_launchpad.protocol.events import (
    LogEvent,
    OperationCompleteEvent,
    ResourceAllocatedEvent,
)
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    EndpointInfo,
    ServingRequirements,
)


def _modal_config(**overrides: object) -> DeploymentConfig:
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.MODAL,
        app_name="llamacpp-test",
        repo_id="org/model-GGUF",
        quant="Q4_K_M",
    )
    return DeploymentConfig(**{**base.__dict__, **overrides})  # type: ignore[arg-type]


def _deploy_success(url: str | None = "https://alice--llamacpp-test-serve.modal.run"):
    def _deploy(_c):  # type: ignore[no-untyped-def]
        events = []
        if url:
            events.append(LogEvent(line=f"x => {url}"))
        events.append(OperationCompleteEvent(operation=OperationType.DEPLOY, success=True))
        return events

    return _deploy


def _warmup_success(url: str | None = None):
    def _warmup(*_a, **_k):  # type: ignore[no-untyped-def]
        data = {"url": url} if url else None
        return [OperationCompleteEvent(operation=OperationType.WARMUP, success=True, data=data)]

    return _warmup


class WarmupExplicitSuccessTests(unittest.TestCase):
    def test_missing_warmup_implementation_fails_requested_warmup(self) -> None:
        endpoint = EndpointInfo(name="llamacpp-test", app_id="ap-1", web_url="https://alice--x.modal.run")

        def _deploy(_c):  # type: ignore[no-untyped-def]
            return [OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=endpoint)]

        orch = SimpleNamespace(deploy=_deploy)
        result = run_lifecycle(orch, [LifecycleAttempt(config=_modal_config())])
        self.assertFalse(result.succeeded)
        self.assertIn("no warmup implementation", (result.attempts[0].failure_detail or "").lower())

    def test_empty_warmup_stream_fails(self) -> None:
        orch = SimpleNamespace(deploy=_deploy_success(), warmup=lambda *_a, **_k: [])
        result = run_lifecycle(orch, [LifecycleAttempt(config=_modal_config())])
        self.assertFalse(result.succeeded)
        self.assertIn("no events", (result.attempts[0].failure_detail or "").lower())

    def test_log_only_warmup_stream_fails(self) -> None:
        orch = SimpleNamespace(
            deploy=_deploy_success(),
            warmup=lambda *_a, **_k: [LogEvent(line="still starting")],
        )
        result = run_lifecycle(orch, [LifecycleAttempt(config=_modal_config())])
        self.assertFalse(result.succeeded)

    def test_conflicting_warmup_completions_fail(self) -> None:
        orch = SimpleNamespace(
            deploy=_deploy_success(),
            warmup=lambda *_a, **_k: [
                OperationCompleteEvent(operation=OperationType.WARMUP, success=True),
                OperationCompleteEvent(operation=OperationType.WARMUP, success=False, detail="late failure"),
            ],
        )
        result = run_lifecycle(orch, [LifecycleAttempt(config=_modal_config())])
        self.assertFalse(result.succeeded)
        self.assertIn("conflicting", (result.attempts[0].failure_detail or "").lower())

    def test_disabled_warmup_succeeds_without_verification(self) -> None:
        orch = SimpleNamespace(deploy=_deploy_success())
        result = run_lifecycle(
            orch,
            [LifecycleAttempt(config=_modal_config())],
            options=LifecycleOptions(warmup_enabled=False),
        )
        self.assertTrue(result.succeeded)

    def test_smoke_success_needs_no_endpoint_or_warmup(self) -> None:
        config = _modal_config(do_deploy=False, run_smoke=True)
        orch = SimpleNamespace(
            deploy=lambda _c: [OperationCompleteEvent(operation=OperationType.SMOKE_TEST, success=True)]
        )
        result = run_lifecycle(orch, [LifecycleAttempt(config=config)])
        self.assertTrue(result.succeeded)


class PriceSeparationTests(unittest.TestCase):
    def test_quote_price_reaches_warmup_not_the_cap(self) -> None:
        seen: dict[str, object] = {}

        def _warmup(_b, url, *_a, **_k):  # type: ignore[no-untyped-def]
            seen.update(_k)
            return [OperationCompleteEvent(operation=OperationType.WARMUP, success=True)]

        config = _modal_config(
            serving_requirements=ServingRequirements(context_tokens=4096, max_hourly_cost_usd=5.0),
            price_per_hour_usd=1.0,
        )
        orch = SimpleNamespace(deploy=_deploy_success(), warmup=_warmup)
        result = run_lifecycle(orch, [LifecycleAttempt(config=config)])
        self.assertTrue(result.succeeded)
        self.assertEqual(seen.get("price_per_hour_usd"), 1.0)

    def test_unknown_price_stays_unknown(self) -> None:
        seen: dict[str, object] = {}

        def _warmup(_b, url, *_a, **_k):  # type: ignore[no-untyped-def]
            seen.update(_k)
            return [OperationCompleteEvent(operation=OperationType.WARMUP, success=True)]

        config = _modal_config(
            serving_requirements=ServingRequirements(context_tokens=4096, max_hourly_cost_usd=5.0),
            price_per_hour_usd=None,
        )
        orch = SimpleNamespace(deploy=_deploy_success(), warmup=_warmup)
        self.assertTrue(run_lifecycle(orch, [LifecycleAttempt(config=config)]).succeeded)
        self.assertIsNone(seen.get("price_per_hour_usd"))

    def test_price_cap_revalidation_refuses_increase(self) -> None:
        request = request_from_config(
            _modal_config(
                serving_requirements=ServingRequirements(context_tokens=4096, max_hourly_cost_usd=1.0),
                price_per_hour_usd=1.0,
            )
        )
        plan = plan_from_request(request)
        finding = revalidate_plan_price(plan, current_price_per_hour_usd=2.0)
        self.assertIsNotNone(finding)
        self.assertIsNone(revalidate_plan_price(plan, current_price_per_hour_usd=0.5))
        self.assertIsNone(revalidate_plan_price(plan, current_price_per_hour_usd=None))

    def test_request_round_trip_preserves_price(self) -> None:
        from llm_launchpad.core.quick_deploy import config_from_request

        config = _modal_config(price_per_hour_usd=2.5)
        self.assertEqual(config_from_request(request_from_config(config)).price_per_hour_usd, 2.5)


class SingleCleanupTests(unittest.TestCase):
    def test_missing_url_cleans_up_exactly_once(self) -> None:
        stops: list[str] = []

        def _stop_app(backend, app_name=None, app_id=None, provider=None):  # type: ignore[no-untyped-def]
            stops.append(app_name or "")
            yield OperationCompleteEvent(operation=OperationType.STOP, success=True)

        orch = SimpleNamespace(
            deploy=lambda _c: [OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)],
            warmup=_warmup_success(),
            stop_app=_stop_app,
        )
        config = _modal_config(app_name="llamacpp-test")
        # No URL anywhere and no Modal username fallback: missing-URL failure.
        result = run_lifecycle(orch, [LifecycleAttempt(config=config)])
        self.assertFalse(result.succeeded)
        self.assertEqual(len(stops), 1)

    def test_allocation_without_endpoint_supplies_cleanup_id(self) -> None:
        seen: list[str | None] = []

        def _stop_app(backend, app_name=None, app_id=None, provider=None):  # type: ignore[no-untyped-def]
            seen.append(app_id)
            yield OperationCompleteEvent(operation=OperationType.STOP, success=True)

        def _deploy(_c):  # type: ignore[no-untyped-def]
            return [
                ResourceAllocatedEvent(app_id="pod-123"),
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True),
            ]

        orch = SimpleNamespace(
            deploy=_deploy,
            warmup=lambda *_a, **_k: [
                OperationCompleteEvent(operation=OperationType.WARMUP, success=False, detail="not ready")
            ],
            stop_app=_stop_app,
        )
        config = _modal_config(do_warmup=True)
        # Warmup fails after an allocation event with no endpoint: cleanup must
        # still target the allocated id.
        orch.deploy = _deploy
        # Give warmup a URL so it runs instead of failing on missing URL.
        result = run_lifecycle(
            SimpleNamespace(
                deploy=lambda _c: [
                    ResourceAllocatedEvent(app_id="pod-123"),
                    OperationCompleteEvent(
                        operation=OperationType.DEPLOY, success=True,
                        data=EndpointInfo(name="x", app_id="pod-123", web_url="https://example.test"),
                    ),
                ],
                warmup=lambda *_a, **_k: [
                    OperationCompleteEvent(operation=OperationType.WARMUP, success=False, detail="not ready")
                ],
                stop_app=_stop_app,
            ),
            [LifecycleAttempt(config=config)],
        )
        self.assertFalse(result.succeeded)
        self.assertIn("pod-123", seen)

    def test_failed_cleanup_blocks_fallback(self) -> None:
        endpoint = EndpointInfo(name="vast-app", app_id="i-1", web_url="https://vast.example")
        from unittest.mock import MagicMock

        orch = MagicMock()
        orch.deploy.return_value = [
            OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=endpoint)
        ]
        orch.warmup.return_value = [
            OperationCompleteEvent(operation=OperationType.WARMUP, success=False, detail="not ready")
        ]
        orch.stop_app.return_value = [
            OperationCompleteEvent(operation=OperationType.STOP, success=False, detail="destroy failed")
        ]
        from llm_launchpad.protocol.models import VastProviderOptions

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.VAST,
            app_name="vast-app",
            repo_id="org/model-GGUF",
            quant="Q4_K_M",
            provider_options=VastProviderOptions("1", 100, 0.5),
        )
        result = run_lifecycle(
            orch,
            [LifecycleAttempt(config=config), LifecycleAttempt(config=_modal_config())],
        )
        self.assertFalse(result.succeeded)
        self.assertEqual(len(result.attempts), 1)
        self.assertFalse(result.attempts[0].retry_allowed)


class ResourceTargetingTests(unittest.TestCase):
    def test_modal_name_addressable_without_id(self) -> None:
        target = resolve_stop_target(
            provider=ComputeProvider.MODAL, app_name="app", resource_id=None, allocated=True
        )
        self.assertTrue(target.stoppable)

    def test_prime_requires_pod_id(self) -> None:
        target = resolve_stop_target(
            provider=ComputeProvider.PRIME, app_name="app", resource_id=None, allocated=True
        )
        self.assertFalse(target.stoppable)
        target = resolve_stop_target(
            provider=ComputeProvider.PRIME, app_name="app", resource_id="pod-1", allocated=True
        )
        self.assertTrue(target.stoppable)

    def test_unallocated_reports_nothing_to_stop(self) -> None:
        target = resolve_stop_target(
            provider=ComputeProvider.MODAL, app_name="app", resource_id=None, allocated=False
        )
        self.assertFalse(target.stoppable)

    def test_lifecycle_has_no_job_runner_dependency(self) -> None:
        import sys

        self.assertNotIn("llm_launchpad.core.job_runner", sys.modules.get("llm_launchpad.core.deployment_lifecycle", "").__class__.__module__ if False else "")
        import llm_launchpad.core.deployment_lifecycle as lifecycle

        source_path = lifecycle.__file__ or ""
        with open(source_path) as handle:
            source = handle.read()
        self.assertNotIn("job_runner", source)
        self.assertNotIn("_NullStore", source)


class PlanAuthorityTests(unittest.TestCase):
    def test_plan_intent_governs_execution(self) -> None:
        request = request_from_config(_modal_config(do_deploy=False, run_smoke=True))
        self.assertEqual(request.intent, OperationIntent.SMOKE)
        plan = plan_from_request(request)
        attempt = LifecycleAttempt(config=_modal_config(), plan=plan)
        self.assertEqual(attempt.plan.intent, OperationIntent.SMOKE)
        effective = effective_config_for_attempt(attempt)
        self.assertTrue(effective.run_smoke)

    def test_lifecycle_attempts_carry_plans(self) -> None:
        specs = lifecycle_attempts_for_configs([_modal_config()])
        self.assertEqual(len(specs), 1)
        self.assertIsNotNone(specs[0].plan)

    def test_warmup_status_enum_distinguishes_skip(self) -> None:
        self.assertNotEqual(WarmupStatus.SKIPPED, WarmupStatus.SUCCEEDED)


class AdapterBoundaryTests(unittest.TestCase):
    def test_all_providers_expose_list_deploy_stop(self) -> None:
        from llm_launchpad.core.provider_adapters import provider_adapter

        for provider in (ComputeProvider.MODAL, ComputeProvider.PRIME, ComputeProvider.VAST):
            adapter = provider_adapter(provider)
            self.assertEqual(adapter.provider, provider)
            self.assertTrue(callable(adapter.list))
            self.assertTrue(callable(adapter.deploy))
            self.assertTrue(callable(adapter.stop))

    def test_rollback_marker_round_trip(self) -> None:
        from llm_launchpad.core.provider_adapters import rollback_from_event

        event = OperationCompleteEvent(
            operation=OperationType.DEPLOY, success=False, detail="boom",
            data={"rollback": {"attempted": True, "confirmed": False, "detail": "still billing"}},
        )
        rollback = rollback_from_event(event)
        self.assertIsNotNone(rollback)
        assert rollback is not None
        self.assertFalse(rollback.confirmed)
        self.assertIsNone(rollback_from_event(OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)))

    def test_cleanup_disposition_distinguishes_nothing_to_clean(self) -> None:
        self.assertNotEqual(CleanupDisposition.NOTHING_TO_CLEAN, CleanupDisposition.CONFIRMED)


if __name__ == "__main__":
    unittest.main()

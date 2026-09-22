"""Shared deployment lifecycle: one owner for warmup/cleanup/fallback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from llm_launchpad.core.deployment_lifecycle import (
    LifecycleAttempt,
    LifecycleCallbacks,
    run_lifecycle,
)
from llm_launchpad.protocol.enums import (
    AttemptDisposition,
    BackendType,
    ComputeProvider,
    OperationType,
)
from llm_launchpad.protocol.events import (
    LogEvent,
    OperationCompleteEvent,
)
from llm_launchpad.protocol.models import DeploymentConfig, EndpointInfo


def _modal_config(**overrides: object) -> DeploymentConfig:
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.MODAL,
        app_name="llamacpp-test",
        repo_id="org/model-GGUF",
        quant="Q4_K_M",
    )
    return DeploymentConfig(**{**base.__dict__, **overrides})  # type: ignore[arg-type]


def _resolved_plan_for(config: DeploymentConfig):
    """The plan every execution path resolves before the lifecycle runs."""
    from llm_launchpad.core.deployment_preflight import preflight_config

    result = preflight_config(config)
    assert result.plan is not None
    return result.plan


class LifecycleParityTests(unittest.TestCase):
    def test_deploy_failure_carries_provider_exit_code(self) -> None:
        orch = SimpleNamespace(
            deploy=lambda _c: [
                OperationCompleteEvent(
                    operation=OperationType.DEPLOY, success=False,
                    exit_code=9, detail="boom",
                )
            ]
        )
        result = run_lifecycle(orch, [LifecycleAttempt(config=_modal_config())])
        self.assertFalse(result.succeeded)
        self.assertEqual(result.attempts[0].failure_exit_code, 9)

    def test_the_minted_endpoint_key_reaches_the_caller(self) -> None:
        """The key is minted mid-deploy, on a config the caller does not hold.

        Callers sync OpenCode from their own config object, so a key that
        never travelled back was written into OpenCode as no key at all:
        `401 Invalid API Key` from an endpoint that had deployed, certified
        and was serving.
        """

        caller_config = _modal_config(do_warmup=False)
        plan = _resolved_plan_for(caller_config)

        def deploy(executed):
            # What every provider adapter does once it owns the config.
            executed.endpoint_api_key = "minted-during-deploy"
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=True,
                data=EndpointInfo(name="llamacpp-test", web_url="https://x/v1"),
            )

        run_lifecycle(
            SimpleNamespace(deploy=deploy),
            [LifecycleAttempt(config=caller_config, plan=plan)],
        )

        self.assertEqual(caller_config.endpoint_api_key, "minted-during-deploy")

    def test_a_caller_supplied_key_is_not_replaced(self) -> None:
        caller_config = _modal_config(do_warmup=False, endpoint_api_key="chosen")
        plan = _resolved_plan_for(caller_config)

        def deploy(executed):
            self_key = executed.endpoint_api_key
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=True,
                data=EndpointInfo(name="llamacpp-test", web_url=f"https://x/{self_key}"),
            )

        run_lifecycle(
            SimpleNamespace(deploy=deploy),
            [LifecycleAttempt(config=caller_config, plan=plan)],
        )

        self.assertEqual(caller_config.endpoint_api_key, "chosen")

    def test_smoke_success_does_not_require_serving_endpoint(self) -> None:
        config = _modal_config(do_deploy=False, run_smoke=True)
        orch = SimpleNamespace(
            deploy=lambda _c: [
                OperationCompleteEvent(operation=OperationType.SMOKE_TEST, success=True)
            ]
        )
        result = run_lifecycle(orch, [LifecycleAttempt(config=config)])
        self.assertTrue(result.succeeded)

    def test_log_scraped_modal_url_reaches_warmup(self) -> None:
        seen: list[str] = []

        def _deploy(_c):  # type: ignore[no-untyped-def]
            return [
                LogEvent(line="x => https://alice--app-serve.modal.run"),
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True),
            ]

        def _warmup(_b, url, *_a, **_k):  # type: ignore[no-untyped-def]
            seen.append(url)
            return [OperationCompleteEvent(operation=OperationType.WARMUP, success=True)]

        orch = SimpleNamespace(deploy=_deploy, warmup=_warmup)
        result = run_lifecycle(orch, [LifecycleAttempt(config=_modal_config())])
        self.assertTrue(result.succeeded)
        self.assertEqual(seen, ["https://alice--app-serve.modal.run"])
        self.assertEqual(result.url, "https://alice--app-serve.modal.run")

    def test_warmup_replacement_url_is_published(self) -> None:
        endpoint = EndpointInfo(
            name="llamacpp-test", app_id="ap-1", web_url="https://old.example"
        )

        def _deploy(_c):  # type: ignore[no-untyped-def]
            return [OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=True, data=endpoint
            )]

        def _warmup(*_a, **_k):  # type: ignore[no-untyped-def]
            return [OperationCompleteEvent(
                operation=OperationType.WARMUP, success=True,
                data={"url": "https://new.example"},
            )]

        published: list[tuple[str | None, EndpointInfo | None]] = []
        orch = SimpleNamespace(deploy=_deploy, warmup=_warmup)
        result = run_lifecycle(
            orch,
            [LifecycleAttempt(config=_modal_config(do_warmup=True))],
            callbacks=LifecycleCallbacks(
                on_connection=lambda c, url, ep: published.append((url, ep))
            ),
        )
        self.assertTrue(result.succeeded)
        self.assertEqual(result.url, "https://new.example")
        self.assertTrue(any(url == "https://new.example" for url, _ in published))

    def test_failed_cleanup_blocks_fallback(self) -> None:
        endpoint = EndpointInfo(
            name="vast-app", app_id="i-1", web_url="https://vast.example"
        )
        orch = MagicMock()
        orch.deploy.return_value = [
            OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=True, data=endpoint
            )
        ]
        orch.warmup.return_value = [
            OperationCompleteEvent(
                operation=OperationType.WARMUP, success=False, detail="not ready"
            )
        ]
        orch.stop_app.return_value = [
            OperationCompleteEvent(
                operation=OperationType.STOP, success=False, detail="destroy failed"
            )
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
        events: list[object] = []
        result = run_lifecycle(
            orch,
            [
                LifecycleAttempt(config=config),
                LifecycleAttempt(config=_modal_config()),
            ],
            callbacks=LifecycleCallbacks(on_event=events.append),
        )
        self.assertFalse(result.succeeded)
        self.assertEqual(len(result.attempts), 1)
        self.assertFalse(result.attempts[0].retry_allowed)
        self.assertEqual(orch.deploy.call_count, 1)

    def test_cancelled_cleanup_failure_is_confirmed_unknown(self) -> None:
        orch = MagicMock()
        orch.deploy.return_value = [
            OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)
        ]
        orch.stop_app.return_value = []
        config = _modal_config()
        result = run_lifecycle(
            orch,
            [LifecycleAttempt(config=config)],
            callbacks=LifecycleCallbacks(is_cancelled=lambda: True),
        )
        self.assertFalse(result.succeeded)
        self.assertEqual(
            result.attempts[0].disposition, AttemptDisposition.CANCELLED
        )


if __name__ == "__main__":
    unittest.main()

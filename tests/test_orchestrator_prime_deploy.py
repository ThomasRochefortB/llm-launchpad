"""Cover the Prime deploy path: requirements, success, and failure cleanup."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.prime_backend import PrimeApiError
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.events import (
    EndpointAvailableEvent,
    LogEvent,
    OperationCompleteEvent,
)
from llm_launchpad.protocol.models import ComputeOffer, DeploymentConfig, PrimeProviderOptions


def _prime_config(**overrides: object) -> DeploymentConfig:
    kwargs: dict[str, object] = {
        "backend": BackendType.VLLM,
        "provider": ComputeProvider.PRIME,
        "model_name": "Qwen/Qwen3-4B",
        "app_name": "llp-prime-vllm-qwen",
        "instance_name": "qwen",
        "do_warmup": False,
    }
    kwargs.update(overrides)
    return DeploymentConfig(**kwargs)  # type: ignore[arg-type]


def _offer() -> ComputeOffer:
    return ComputeOffer(
        id="offer-1",
        cloud_id="cloud-1",
        provider_name="hyperstack",
        gpu_type="H100_80GB",
        gpu_count=1,
        region="canada",
        country="CA",
        price_per_hour=1.9,
    )


class OrchestratorPrimeDeployTests(unittest.TestCase):
    def setUp(self) -> None:
        import contextlib

        self._patch_stack = contextlib.ExitStack()
        self.addCleanup(self._patch_stack.close)
        enter = self._patch_stack.enter_context
        enter(patch("llm_launchpad.core.orchestrator.discover_selected_model_reasoning", return_value=None))
        enter(patch("llm_launchpad.core.vision.prepare_vision", return_value=SimpleNamespace(
            enabled=False, message="text-only", verification="untested",
        )))

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(prime_backend=SimpleNamespace())

    def test_prime_vllm_requires_model_name(self) -> None:
        config = _prime_config(model_name="")
        events = list(self._orchestrator().deploy(config))
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertEqual(completion.exit_code, 2)
        self.assertIn("--model-name", completion.detail)

    def test_prime_llamacpp_requires_repo_id(self) -> None:
        config = _prime_config(backend=BackendType.LLAMACPP, model_name=None, repo_id="")
        events = list(self._orchestrator().deploy(config))
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("--repo-id", completion.detail)

    def test_prime_llamacpp_rejects_non_default_revision(self) -> None:
        config = _prime_config(
            backend=BackendType.LLAMACPP, model_name=None, repo_id="unsloth/phi-gguf", revision="main"
        )
        events = list(self._orchestrator().deploy(config))
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("default Hugging Face revision", completion.detail)

    def test_prime_deploy_success_emits_running_endpoint(self) -> None:
        offer = _offer()
        created = {"id": "pod-1"}
        active_pod = {"id": "pod-1", "status": "ACTIVE", "installationStatus": "FINISHED", "sshConnection": "ssh://x"}

        orchestrator = self._orchestrator()
        enter = self._patch_stack.enter_context
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_launch_spec",
                    return_value=SimpleNamespace(offer_image="img", container_image="ctr")))
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_offer_and_disk",
                    return_value=(offer, None, [])))
        backend = SimpleNamespace(
            create_pod=lambda config, offer: created,
            get_pod=lambda pod_id: dict(active_pod),
            start_bootstrap_runtime=lambda config, pod: None,
            endpoint_url=lambda pod, **kwargs: "http://10.0.0.1:8080",
            create_tunnel=lambda pod_id, name: SimpleNamespace(tunnel_id="t-1", url="https://t-1.example.com", expires_at=""),
            start_tunnel=lambda pod, tunnel: None,
            bootstrap_runtime_status=lambda pod: (True, False, "runtime ready"),
            tunnel_runtime_status=lambda pod, tunnel_id: (True, False, "tunnel ready"),
            public_endpoint_ready=lambda endpoint, api_key: (True, ""),
            get_pod_logs=lambda pod_id, tail=200: [],
            delete_pod=lambda pod_id: None,
        )
        orchestrator._prime_backend = backend  # type: ignore[assignment]

        config = _prime_config(provider_options=PrimeProviderOptions(allow_insecure_http=True))
        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.return_value = False
            events = list(orchestrator.deploy(config))

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(completion.success, [e for e in events if isinstance(e, LogEvent)][-3:])
        info = completion.data
        self.assertEqual(info.app_id, "pod-1")
        self.assertEqual(info.state, "running")
        self.assertEqual(info.provider, ComputeProvider.PRIME)
        self.assertEqual(info.web_url, "http://10.0.0.1:8080")
        self.assertTrue(any(isinstance(e, EndpointAvailableEvent) for e in events))
        self.assertTrue(any(
            isinstance(e, LogEvent) and "Selected Prime offer offer-1" in e.line for e in events
        ))

    def test_prime_deploy_tunnel_path_announces_tunnel_endpoint(self) -> None:
        offer = _offer()
        active_pod = {"id": "pod-1", "status": "ACTIVE", "installationStatus": "FINISHED", "sshConnection": "ssh://x"}
        orchestrator = self._orchestrator()
        enter = self._patch_stack.enter_context
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_launch_spec",
                    return_value=SimpleNamespace(offer_image="img", container_image="ctr")))
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_offer_and_disk",
                    return_value=(offer, None, ["disk ready"])))
        backend = SimpleNamespace(
            create_pod=lambda config, offer: {"id": "pod-1"},
            get_pod=lambda pod_id: dict(active_pod),
            start_bootstrap_runtime=lambda config, pod: None,
            create_tunnel=lambda pod_id, name: SimpleNamespace(tunnel_id="t-1", url="https://t-1.example.com", expires_at="soon"),
            start_tunnel=lambda pod, tunnel: None,
            bootstrap_runtime_status=lambda pod: (True, False, "runtime ready"),
            tunnel_runtime_status=lambda pod, tunnel_id: (True, False, "tunnel ready"),
            public_endpoint_ready=lambda endpoint, api_key: (True, ""),
            get_pod_logs=lambda pod_id, tail=200: [],
            delete_pod=lambda pod_id: None,
        )
        orchestrator._prime_backend = backend  # type: ignore[assignment]

        config = _prime_config()
        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.return_value = False
            events = list(orchestrator.deploy(config))

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(completion.success)
        self.assertEqual(completion.data.web_url, "https://t-1.example.com")
        self.assertTrue(any(
            isinstance(e, LogEvent) and "secure tunnel t-1" in e.line for e in events
        ))
        self.assertTrue(any(
            isinstance(e, LogEvent) and e.line == "disk ready" for e in events
        ))

    def test_prime_pod_provisioning_failure_terminates_pod(self) -> None:
        orchestrator = self._orchestrator()
        enter = self._patch_stack.enter_context
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_launch_spec",
                    return_value=SimpleNamespace(offer_image="img", container_image="ctr")))
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_offer_and_disk",
                    return_value=(_offer(), None, [])))
        deleted: list[str] = []
        backend = SimpleNamespace(
            create_pod=lambda config, offer: {"id": "pod-9"},
            get_pod=lambda pod_id: {"id": "pod-9", "status": "ERROR", "installationFailure": "no capacity"},
            get_pod_logs=lambda pod_id, tail=200: ["boom"],
            delete_pod=lambda pod_id: deleted.append(pod_id),
        )
        orchestrator._prime_backend = backend  # type: ignore[assignment]

        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.return_value = False
            events = list(orchestrator.deploy(_prime_config()))

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("no capacity", completion.detail)
        self.assertEqual(deleted, ["pod-9"])
        self.assertTrue(any(
            isinstance(e, LogEvent) and e.line == "boom" for e in events
        ))
        self.assertTrue(any(
            isinstance(e, LogEvent) and "Terminated failed Prime pod pod-9" in e.line for e in events
        ))

    def test_prime_failure_keeps_pod_when_keep_failed_resource(self) -> None:
        orchestrator = self._orchestrator()
        enter = self._patch_stack.enter_context
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_launch_spec",
                    return_value=SimpleNamespace(offer_image="img", container_image="ctr")))
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_offer_and_disk",
                    return_value=(_offer(), None, [])))
        deleted: list[str] = []
        backend = SimpleNamespace(
            create_pod=lambda config, offer: (_ for _ in ()).throw(PrimeApiError("launch denied")),
            get_pod_logs=lambda pod_id, tail=200: [],
            delete_pod=lambda pod_id: deleted.append(pod_id),
        )
        orchestrator._prime_backend = backend  # type: ignore[assignment]

        config = _prime_config(provider_options=PrimeProviderOptions(keep_failed_resource=True))
        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.return_value = False
            events = list(orchestrator.deploy(config))

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertEqual(deleted, [])
        # No pod id was ever allocated, so there is nothing to keep or terminate.
        self.assertFalse(any("Terminated failed Prime pod" in getattr(e, "line", "") for e in events))

    def test_prime_runtime_bootstrap_failure_reports_cause(self) -> None:
        offer = _offer()
        active_pod = {"id": "pod-1", "status": "ACTIVE", "installationStatus": "FINISHED", "sshConnection": "ssh://x"}
        orchestrator = self._orchestrator()
        enter = self._patch_stack.enter_context
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_launch_spec",
                    return_value=SimpleNamespace(offer_image="img", container_image="ctr")))
        enter(patch("llm_launchpad.core.orchestrator.resolve_prime_offer_and_disk",
                    return_value=(offer, None, [])))
        deleted: list[str] = []
        backend = SimpleNamespace(
            create_pod=lambda config, offer: {"id": "pod-1"},
            get_pod=lambda pod_id: dict(active_pod),
            start_bootstrap_runtime=lambda config, pod: None,
            endpoint_url=lambda pod, **kwargs: "http://10.0.0.1:8080",
            bootstrap_runtime_status=lambda pod: (False, True, "image pull backoff"),
            tunnel_runtime_status=lambda pod, tunnel_id: (False, False, ""),
            get_pod_logs=lambda pod_id, tail=200: [],
            delete_pod=lambda pod_id: deleted.append(pod_id),
        )
        orchestrator._prime_backend = backend  # type: ignore[assignment]

        config = _prime_config(provider_options=PrimeProviderOptions(allow_insecure_http=True))
        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.return_value = False
            events = list(orchestrator.deploy(config))

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("image pull backoff", completion.detail)
        self.assertEqual(deleted, ["pod-1"])

    def test_await_prime_pod_active_reports_state_transitions(self) -> None:
        states = iter([
            {"status": "CREATING", "installationStatus": "INSTALLING", "installationProgress": 10},
            {"status": "ACTIVE", "installationStatus": "FINISHED", "sshConnection": "ssh://x"},
        ])
        orchestrator = self._orchestrator()
        orchestrator._prime_backend = SimpleNamespace(get_pod=lambda pod_id: next(states))  # type: ignore[assignment]
        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.return_value = False
            pod = None
            logs: list[str] = []
            gen = orchestrator._await_prime_pod_active("pod-1", {"id": "pod-1"})
            try:
                while True:
                    event = next(gen)
                    if isinstance(event, LogEvent):
                        logs.append(event.line)
            except StopIteration as done:
                pod = done.value

        self.assertEqual(pod.get("status"), "ACTIVE")
        self.assertTrue(any("CREATING/INSTALLING (10%)" in line for line in logs))

    def test_await_prime_public_endpoint_blocks_until_ready(self) -> None:
        orchestrator = self._orchestrator()
        calls = {"n": 0}

        def _probe(endpoint: str, api_key: str) -> tuple[bool, str]:
            calls["n"] += 1
            return (calls["n"] >= 3, "" if calls["n"] >= 3 else "refused")

        orchestrator._prime_backend = SimpleNamespace(public_endpoint_ready=_probe)  # type: ignore[assignment]
        with patch("llm_launchpad.core.orchestrator.time.monotonic", side_effect=[0.0, 1.0, 2.0, 3.0]), patch(
            "llm_launchpad.core.orchestrator.shutdown_event"
        ) as shutdown:
            shutdown.return_value.wait.return_value = False
            orchestrator._await_prime_public_endpoint("https://x.example.com", "key")
        self.assertEqual(calls["n"], 3)


if __name__ == "__main__":
    unittest.main()

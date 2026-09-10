from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.connection_store import merge_connections
from llm_launchpad.core.vast_backend import VastApiError, VastBackend
from llm_launchpad.core.vast_deployment import VastDeploymentBackend
from llm_launchpad.core.vast_runtime import vast_runtime_image, vast_runtime_script, verify_endpoint_auth, verify_streaming
from llm_launchpad.core.vast_ssh import VastSsh, ssh_key_startup
from llm_launchpad.core.vast_state import VastState
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig, VastAuthStatus, VastInstance, VastProviderOptions
from tests.test_vast_fast_deploy import vast_offer


def config() -> DeploymentConfig:
    return DeploymentConfig(
        backend=BackendType.LLAMACPP, provider=ComputeProvider.VAST,
        app_name="llp-vast-llamacpp-test", repo_id="acme/model-GGUF", quant="Q4_K_M",
        gpu_type="RTX 4090", gpu_count=1, do_deploy=True,
        provider_options=VastProviderOptions("1001", 100, 0.42, "42"),
        gguf_architecture="llama", server_args="--ctx-size 4096 --n-gpu-layers all",
    )


class VastLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = VastState(Path(temporary.name))
        self.api = Mock(spec=VastBackend)
        self.api.auth_status.return_value = VastAuthStatus(True, account_id="12")
        self.api.account_id.return_value = "12"
        self.api.get_offer.return_value = vast_offer()
        self.remote: VastInstance | None = None
        self.api.create_instance.side_effect = self.create
        self.api.get_instance.side_effect = lambda _: self.remote
        self.api.find_instances.side_effect = lambda _: [self.remote] if self.remote else []
        self.api.destroy_instance.side_effect = self.destroy
        self.backend = VastDeploymentBackend(self.api, self.state)
        self.config = config()
        for patcher in (
            patch.object(self.backend, "preflight", return_value=(True, "12", "")),
            patch("llm_launchpad.core.vast_runtime.get_token", return_value=None),
            patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=True),
            patch("llm_launchpad.core.vast_deployment.verify_endpoint_auth"),
            patch("llm_launchpad.core.vast_deployment.is_shutting_down", return_value=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        ssh_patch = patch("llm_launchpad.core.vast_deployment.VastSsh")
        self.ssh = ssh_patch.start().return_value
        self.addCleanup(ssh_patch.stop)
        self.ssh.public_key.return_value = "ssh-ed25519 PUBLICKEY"
        self.ssh.connected.return_value = True
        # The rented host answers the inventory probe; every other command is
        # whatever a test wants the remote shell to say.
        self.gpu_inventory = "0, NVIDIA GeForce RTX 4090, 24564, 12"
        self.ssh_output = ""
        self.ssh.run.side_effect = lambda instance, command, **kwargs: (
            self.gpu_inventory if "nvidia-smi" in command else self.ssh_output
        )
        stream_patch = patch("llm_launchpad.core.vast_deployment.verify_streaming")
        self.stream = stream_patch.start()
        self.addCleanup(stream_patch.stop)
        socket_patch = patch("llm_launchpad.core.vast_deployment.socket.socket")
        listener = socket_patch.start().return_value.__enter__.return_value
        listener.getsockname.return_value = ("127.0.0.1", 48123)
        self.addCleanup(socket_patch.stop)

    def create(self, offer_id: str, **kwargs: object) -> str:
        record = self.state.load(self.config.app_name or "")
        self.assertIsNotNone(record)  # Intent precedes the billable call.
        assert record is not None
        self.assertIsNone(record.instance_id)
        self.remote = VastInstance("900", str(kwargs["label"]), "running", "42", "ssh12.vast.ai", 1234)
        return "900"

    def destroy(self, instance_id: str) -> None:
        self.assertEqual(instance_id, "900")
        self.remote = None

    def deploy(self) -> OperationCompleteEvent:
        events = list(self.backend.deploy(self.config))
        return next(event for event in reversed(events) if isinstance(event, OperationCompleteEvent))

    def test_create_verify_list_reconnect_and_destroy(self) -> None:
        event = self.deploy()
        self.assertTrue(event.success, event.detail)
        self.assertEqual(event.data.provider, ComputeProvider.VAST)
        self.assertEqual(event.data.web_url, "http://127.0.0.1:48123")
        record = self.state.load(self.config.app_name or "")
        assert record is not None
        self.assertEqual(record.instance_id, "900")
        self.assertEqual(record.state, "running")
        self.assertNotIn(record.endpoint_api_key, repr(record))
        self.assertEqual(stat.S_IMODE((self.state.directory(record.name) / "record.json").stat().st_mode), 0o600)
        self.stream.assert_called_once()
        self.api.attach_key.assert_called_once_with("900", "ssh-ed25519 PUBLICKEY")
        # No startup hook is sent: a supplied script replaces the image's own
        # /root/onstart.sh and Vast's sshd provisioning with it, after which
        # the rental refuses every key. Verified live: identical host, image
        # and code connected without the hook and never connected with it.
        self.assertNotIn("onstart", self.api.create_instance.call_args.kwargs)
        self.assertEqual(self.backend.list_deployments()[0].app_id, "900")
        self.assertEqual(self.backend.connect("900").web_url, event.data.web_url)
        self.backend.destroy(name=record.name, instance_id="900")
        self.assertIsNone(self.state.load(record.name))
        self.ssh.disconnect.assert_called_once()

    def test_price_increase_or_unknown_price_cannot_rent(self) -> None:
        for offer in (vast_offer(dph_total=0.43), vast_offer(dph_base=None, dph_total=None)):
            self.api.get_offer.return_value = offer
            event = self.deploy()
            self.assertFalse(event.success)
        self.api.create_instance.assert_not_called()

    @patch.dict("os.environ", {"LLM_LAUNCHPAD_VAST_EXPERIMENTAL": ""})
    def test_uncertified_vllm_is_refused_before_any_rental(self) -> None:
        self.config.backend = BackendType.VLLM
        self.config.model_name = "Qwen/Qwen3-0.6B"
        event = self.deploy()
        self.assertFalse(event.success)
        self.assertIn("EXPERIMENTAL=1", event.detail or "")
        self.api.create_instance.assert_not_called()
        self.api.attach_key.assert_not_called()
        self.assertEqual(self.state.records(), [])

    def test_changed_machine_or_insufficient_memory_cannot_rent(self) -> None:
        self.api.get_offer.return_value = vast_offer(machine_id=43)
        self.assertFalse(self.deploy().success)
        self.api.get_offer.return_value = vast_offer()
        self.config.required_vram_gb = 50
        self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_not_called()

    def test_incompatible_or_unknown_cuda_driver_cannot_rent(self) -> None:
        for version in (None, 12.6):
            self.api.get_offer.return_value = vast_offer(cuda_max_good=version)
            self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_not_called()

    def test_a_host_whose_topology_differs_from_the_rental_is_refused(self) -> None:
        self.api.get_offer.return_value = vast_offer(num_gpus=2)
        self.config.provider_options = replace(self.config.provider_options, gpu_count=2)
        # Rented two, host shows one.
        self.assertFalse(self.deploy().success)
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])
        self.stream.assert_not_called()

    def test_a_host_that_mixes_gpu_models_is_refused(self) -> None:
        self.api.get_offer.return_value = vast_offer(num_gpus=2)
        self.config.provider_options = replace(self.config.provider_options, gpu_count=2)
        self.gpu_inventory = (
            "0, NVIDIA GeForce RTX 4090, 24564, 12\n1, NVIDIA GeForce RTX 3090, 24564, 12"
        )
        event = self.deploy()
        self.assertFalse(event.success)
        self.assertIn("mixes GPU models", event.detail or "")
        self.api.destroy_instance.assert_called_once_with("900")

    def test_a_multi_gpu_rental_serves_once_its_devices_match(self) -> None:
        self.api.get_offer.return_value = vast_offer(num_gpus=2)
        self.config.provider_options = replace(self.config.provider_options, gpu_count=2)
        self.gpu_inventory = (
            "0, NVIDIA GeForce RTX 4090, 24564, 12\n1, NVIDIA GeForce RTX 4090, 24564, 12"
        )
        event = self.deploy()
        self.assertTrue(event.success, event.detail)
        self.assertEqual(self.config.gpu_count, 2)
        self.assertEqual(self.api.get_offer.call_args.args[1].gpu_count, 2)

    def test_a_gpu_older_than_the_runtime_cannot_rent(self) -> None:
        # The driver can be new while the silicon is too old for the image's
        # CUDA build. A Volta card still suits llama.cpp's CUDA 12 image, so
        # this uses a Kepler one; the vLLM floor is asserted separately.
        self.api.get_offer.return_value = vast_offer(compute_cap=350)
        event = self.deploy()
        self.assertFalse(event.success)
        self.assertIn("compute capability", event.detail or "")
        self.api.create_instance.assert_not_called()

    def test_an_offer_without_a_reported_architecture_cannot_rent(self) -> None:
        self.api.get_offer.return_value = vast_offer(compute_cap=None)
        self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_not_called()

    def test_stream_failure_destroys_instance_and_does_not_publish(self) -> None:
        self.stream.side_effect = RuntimeError("stream failed")
        event = self.deploy()
        self.assertFalse(event.success)
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_failed_auth_verification_destroys_instance_before_publication(self) -> None:
        with patch("llm_launchpad.core.vast_deployment.verify_endpoint_auth", side_effect=RuntimeError("unauthorized accepted")):
            self.assertFalse(self.deploy().success)
        self.api.destroy_instance.assert_called_once_with("900")
        self.stream.assert_not_called()

    def test_ssh_start_failure_destroys_instance(self) -> None:
        self.ssh.connect.side_effect = RuntimeError("port busy")
        self.assertFalse(self.deploy().success)
        self.api.destroy_instance.assert_called_once_with("900")

    def test_rejected_ssh_key_destroys_rental_without_waiting_for_ssh(self) -> None:
        self.api.attach_key.side_effect = VastApiError("Key rejected", status_code=400)
        event = self.deploy()
        self.assertFalse(event.success)
        self.assertIn("Key rejected", event.detail or "")
        self.ssh.run.assert_not_called()
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_create_timeout_reconciles_label_without_creating_again(self) -> None:
        def uncertain(*args, **kwargs):
            self.create(*args, **kwargs)
            raise VastApiError("request timed out")
        self.api.create_instance.side_effect = uncertain
        self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_called_once()
        self.api.find_instances.assert_called_once()
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_uncertain_create_blocks_repeat_and_disables_fallback(self) -> None:
        self.api.create_instance.side_effect = VastApiError("timeout")
        self.config.fallback_configs = (config(),)
        self.assertFalse(self.deploy().success)
        self.assertEqual(self.config.fallback_configs, ())
        self.assertEqual(len(self.state.records()), 1)
        self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_called_once()

    def test_failed_cleanup_retains_recovery_record(self) -> None:
        self.stream.side_effect = RuntimeError("stream failed")
        self.api.destroy_instance.side_effect = VastApiError("provider unavailable")
        self.assertFalse(self.deploy().success)
        self.assertEqual(self.state.records()[0].instance_id, "900")

    def test_interrupted_generator_cleans_up_after_instance_creation(self) -> None:
        events = self.backend.deploy(self.config)
        next(events)  # queued rental
        next(events)  # server startup, with remote identity already persisted
        events.close()
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_cancel_before_rental_leaves_no_uncertain_intent(self) -> None:
        events = self.backend.deploy(self.config)
        next(events)
        events.close()
        self.assertEqual(self.state.records(), [])
        self.api.create_instance.assert_not_called()

    def test_existing_rental_blocks_fallback_and_external_destruction_is_visible(self) -> None:
        self.assertTrue(self.deploy().success)
        self.config.fallback_configs = (config(),)
        self.assertFalse(self.deploy().success)
        self.assertEqual(self.config.fallback_configs, ())
        self.remote = None
        row = self.backend.list_deployments()[0]
        self.assertEqual(row.state, "destroyed")
        self.assertIsNone(row.web_url)
        self.backend.destroy(instance_id="900")
        self.assertEqual(self.state.records(), [])

    def test_disconnected_tunnel_is_not_restored_from_cached_metadata(self) -> None:
        self.assertTrue(self.deploy().success)
        self.ssh.connected.return_value = False
        rows = self.backend.list_deployments()
        with patch("llm_launchpad.core.connection_store.load_connection_entries", return_value={
            rows[0].name: {"base_url": "http://127.0.0.1:48123/v1", "provider": "vast", "resource_id": "900"},
        }), patch("llm_launchpad.core.connection_store._backfill_legacy_reasoning"):
            merge_connections(rows)
        self.assertIsNone(rows[0].web_url)

    def test_account_or_label_mismatch_refuses_destroy(self) -> None:
        self.assertTrue(self.deploy().success)
        self.api.account_id.return_value = "other"
        with self.assertRaises(VastApiError):
            self.backend.destroy(instance_id="900")
        self.api.account_id.return_value = "12"
        assert self.remote is not None
        self.remote = replace(self.remote, label="someone-else")
        with self.assertRaises(VastApiError):
            self.backend.destroy(instance_id="900")
        self.api.destroy_instance.assert_not_called()

    def test_logs_redact_credentials(self) -> None:
        self.assertTrue(self.deploy().success)
        self.ssh_output = f"{self.config.endpoint_api_key} hf_private123\nloaded"
        lines = self.backend.logs("900")
        self.assertEqual(lines, ["[redacted] [redacted]", "loaded"])

    def test_orchestrator_management_uses_vast_backend(self) -> None:
        self.assertTrue(self.deploy().success)
        orchestrator = Orchestrator(vast_backend=self.backend)
        events = list(orchestrator.list_deployments(ComputeProvider.VAST))
        self.assertEqual(events[-1].data[0].app_id, "900")
        events = list(orchestrator.stop_app(BackendType.LLAMACPP, app_id="900", provider=ComputeProvider.VAST))
        self.assertTrue(events[-1].success)
        self.assertEqual(events[-1].operation, OperationType.STOP)


class VastTransportTests(unittest.TestCase):
    def test_tunnel_is_local_persistent_and_rejects_changed_host_keys(self) -> None:
        with TemporaryDirectory() as directory:
            ssh = VastSsh(Path(directory))
            instance = VastInstance("900", "label", "running", "42", "ssh12.vast.ai", 1234)
            with patch.object(ssh, "_run", return_value=Mock(stdout="")) as run:
                self.assertEqual(ssh.connect(instance, 48123), 48123)
            args = run.call_args.args[0]
            self.assertIn("127.0.0.1:48123:127.0.0.1:8000", args)
            self.assertIn("ControlPersist=yes", args)
            self.assertIn("StrictHostKeyChecking=accept-new", args)
            self.assertIn("ExitOnForwardFailure=yes", args)
            self.assertNotIn("-g", args)
            for host in ("-oProxyCommand=bad", "localhost", "ssh12.vast.ai;bad"):
                with self.assertRaises(ValueError):
                    ssh.args(replace(instance, ssh_host=host))

    def test_ssh_errors_never_include_secret_stdin(self) -> None:
        with patch("llm_launchpad.core.vast_ssh.subprocess.run", side_effect=subprocess.TimeoutExpired(["ssh"], 30, output="secret")):
            with self.assertRaisesRegex(RuntimeError, "could not complete") as caught:
                VastSsh._run(["ssh"], input_text="secret")
        self.assertNotIn("secret", str(caught.exception))

    def test_pinned_runtime_and_secret_script_do_not_expose_a_public_port(self) -> None:
        candidate = config()
        candidate.endpoint_api_key = "private-key"
        self.assertIn("@sha256:", vast_runtime_image(candidate))
        with patch("llm_launchpad.core.vast_runtime.get_token", return_value="hf_private"):
            script = vast_runtime_script(candidate)
        self.assertIn("--host 127.0.0.1", script)
        self.assertIn("umask 077", script)
        self.assertIn("--no-mmproj", script)
        self.assertNotIn("--api-key", script)
        self.assertIn("export LLAMA_API_KEY=private-key", script)
        self.assertIn("export LD_LIBRARY_PATH=/app:", script)
        # Multi-GPU rentals share the pinned image; only shapes Vast cannot
        # bundle are refused.
        candidate.gpu_count = 2
        self.assertIn("@sha256:", vast_runtime_image(candidate))
        candidate.gpu_count = 9
        with self.assertRaises(ValueError):
            vast_runtime_image(candidate)

    def test_auth_verifier_requires_both_missing_and_wrong_keys_to_be_rejected(self) -> None:
        with patch("llm_launchpad.core.vast_runtime.requests.Session") as session_type:
            session = session_type.return_value.__enter__.return_value
            response = session.post.return_value.__enter__.return_value
            response.status_code = 401
            verify_endpoint_auth("http://127.0.0.1:48123", "model")
            self.assertEqual(session.post.call_count, 2)
            self.assertFalse(session.trust_env)
            response.status_code = 200
            with self.assertRaisesRegex(RuntimeError, "unauthorized"):
                verify_endpoint_auth("http://127.0.0.1:48123", "model")

    def test_stream_requires_sse_json_and_done(self) -> None:
        with patch("llm_launchpad.core.vast_runtime.requests.Session") as session_type:
            session = session_type.return_value.__enter__.return_value
            response = session.post.return_value.__enter__.return_value
            response.headers = {"Content-Type": "text/event-stream"}
            response.iter_lines.return_value = [b'data: {"choices":[{"delta":{"content":"OK"}}]}', b"data: [DONE]"]
            verify_streaming("http://127.0.0.1:48123", "key", "model")
            self.assertFalse(session.trust_env)
            for lines in ([b"data: [DONE]"], [b"data: malformed"], [b'data: {"choices":[]}']):
                response.iter_lines.return_value = lines
                with self.assertRaises(RuntimeError):
                    verify_streaming("http://127.0.0.1:48123", "key", "model")


class VastVllmRecordTests(unittest.TestCase):
    """A vLLM rental must not be filed as llama.cpp."""

    def test_record_backend_shapes_the_published_endpoint(self) -> None:
        from llm_launchpad.protocol.models import VastDeploymentRecord

        record = VastDeploymentRecord(
            name="llp-vast-vllm-test", label="llp-vast-x", account_id="12", offer_id="1",
            machine_id="42", repo_id="", quant="", served_model_name="model",
            endpoint_api_key="k", instance_id="900", backend="vllm", model_name="acme/model",
        )
        endpoint = VastDeploymentBackend._endpoint(record, "running", "http://127.0.0.1:1")
        self.assertEqual(endpoint.backend, BackendType.VLLM)

    def test_a_record_written_before_vllm_support_still_loads_as_llamacpp(self) -> None:
        from llm_launchpad.protocol.models import VastDeploymentRecord

        with TemporaryDirectory() as directory:
            state = VastState(Path(directory))
            record = VastDeploymentRecord(
                name="legacy", label="llp-vast-y", account_id="12", offer_id="1",
                machine_id="42", repo_id="acme/model-GGUF", quant="Q4_K_M",
                served_model_name="model", endpoint_api_key="k", instance_id="901",
            )
            state.save(record)
            loaded = state.load("legacy")
            assert loaded is not None
            self.assertEqual(loaded.backend, "llamacpp")
            self.assertEqual(
                VastDeploymentBackend._endpoint(loaded, "running", None).backend,
                BackendType.LLAMACPP,
            )

    def test_an_unknown_recorded_backend_degrades_instead_of_crashing(self) -> None:
        from llm_launchpad.protocol.models import VastDeploymentRecord

        record = VastDeploymentRecord(
            name="odd", label="llp-vast-z", account_id="12", offer_id="1", machine_id="42",
            repo_id="", quant="", served_model_name="m", endpoint_api_key="k",
            instance_id="902", backend="sglang",
        )
        self.assertEqual(
            VastDeploymentBackend._endpoint(record, "running", None).backend,
            BackendType.LLAMACPP,
        )


class VastThrottlingTests(VastLifecycleTests):
    """A paid rental must survive throttling it did not cause."""

    def test_a_rate_limited_status_check_waits_instead_of_destroying(self) -> None:
        calls = {"n": 0}
        real = self.api.get_instance.side_effect

        def throttle_once(instance_id: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise VastApiError("Vast rate limit reached.", status_code=429)
            return real(instance_id)

        self.api.get_instance.side_effect = throttle_once
        event = self.deploy()
        self.assertTrue(event.success, event.detail)
        self.api.destroy_instance.assert_not_called()

    def test_other_api_failures_still_stop_the_deployment(self) -> None:
        calls = {"n": 0}
        real = self.api.get_instance.side_effect

        def fail_once(instance_id: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise VastApiError("gone", status_code=500)
            return real(instance_id)

        self.api.get_instance.side_effect = fail_once
        self.assertFalse(self.deploy().success)
        self.api.destroy_instance.assert_called_once_with("900")


class VastTeardownThrottlingTests(VastLifecycleTests):
    """An unconfirmed destroy leaves a rental billing, so it must not give up."""

    def test_teardown_retries_account_and_instance_lookups_before_destroy(self) -> None:
        self.deploy()
        self.api.account_id.side_effect = [VastApiError("throttled", status_code=429), "12"]
        self.api.get_instance.side_effect = [
            VastApiError("throttled", status_code=429), self.remote, None,
        ]
        with patch("llm_launchpad.core.vast_deployment.time.sleep"):
            self.backend.destroy(instance_id="900")
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_uncertain_create_retries_throttled_label_reconciliation(self) -> None:
        def uncertain(*args, **kwargs):
            self.create(*args, **kwargs)
            self.api.find_instances.side_effect = [
                VastApiError("throttled", status_code=429), [self.remote],
            ]
            raise VastApiError("request timed out")

        self.api.create_instance.side_effect = uncertain
        with patch("llm_launchpad.core.vast_deployment.time.sleep"):
            self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_called_once()
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_persistent_identity_throttle_preserves_rental_record(self) -> None:
        self.deploy()
        self.api.get_instance.side_effect = VastApiError("throttled", status_code=429)
        with patch("llm_launchpad.core.vast_deployment.time.sleep"):
            with self.assertRaises(VastApiError):
                self.backend.destroy(instance_id="900")
        self.api.destroy_instance.assert_not_called()
        self.assertIsNotNone(self.state.load(self.config.app_name or ""))


    def test_destroy_retries_through_a_rate_limit(self) -> None:
        self.deploy()
        record = self.state.load(self.config.app_name or "")
        assert record is not None
        calls = {"n": 0}
        real = self.api.destroy_instance.side_effect

        def throttle_once(instance_id: str) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise VastApiError("Vast rate limit reached.", status_code=429)
            real(instance_id)

        self.api.destroy_instance.side_effect = throttle_once
        with patch("llm_launchpad.core.vast_deployment.time.sleep"):
            self.backend.destroy(name=record.name, instance_id="900")
        self.assertIsNone(self.state.load(record.name))
        self.assertIsNone(self.remote)

    def test_a_persistent_rate_limit_keeps_the_recovery_record(self) -> None:
        self.deploy()
        record = self.state.load(self.config.app_name or "")
        assert record is not None
        self.api.destroy_instance.side_effect = VastApiError("nope", status_code=429)
        with patch("llm_launchpad.core.vast_deployment.time.sleep"):
            with self.assertRaises(VastApiError):
                self.backend.destroy(name=record.name, instance_id="900")
        # The record must survive so the rental can still be reclaimed.
        self.assertIsNotNone(self.state.load(record.name))


class VastSshDiagnosticsTests(unittest.TestCase):
    """Transport errors stay silent unless someone asks to see them."""

    def _failing(self) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ssh"], 255, "", "Permission denied (publickey).")

    def test_stderr_is_withheld_by_default(self) -> None:
        with patch("llm_launchpad.core.vast_ssh.subprocess.run", return_value=self._failing()):
            with patch.dict("os.environ", {}, clear=False):
                import os as _os
                _os.environ.pop("LLM_LAUNCHPAD_SSH_DEBUG", None)
                with self.assertRaises(RuntimeError) as caught:
                    VastSsh._run(["ssh", "host"])
        self.assertNotIn("publickey", str(caught.exception))

    def test_stderr_is_surfaced_when_explicitly_enabled(self) -> None:
        with patch("llm_launchpad.core.vast_ssh.subprocess.run", return_value=self._failing()):
            with patch.dict("os.environ", {"LLM_LAUNCHPAD_SSH_DEBUG": "1"}):
                with self.assertRaises(RuntimeError) as caught:
                    VastSsh._run(["ssh", "host"])
        self.assertIn("Permission denied", str(caught.exception))


class VastSshStartupTests(unittest.TestCase):
    def test_startup_installs_key_once_preserving_existing_keys_and_permissions(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "rental's home"
            directory = root / ".ssh"
            directory.mkdir(parents=True)
            authorized = directory / "authorized_keys"
            existing = "ssh-ed25519 EXISTING"
            authorized.write_text(existing)  # Missing trailing newline.
            root.chmod(0o777)
            directory.chmod(0o777)
            authorized.chmod(0o666)
            # Shell metacharacters in a comment must stay literal.
            public_key = "ssh-ed25519 PUBLICKEY user'; $(exit 17) `exit 18`"
            script = ssh_key_startup(public_key, root=str(root))
            for _ in range(2):
                subprocess.run(["sh", "-c", script], check=True, capture_output=True)
            self.assertEqual(authorized.read_text().splitlines(), [existing, public_key])
            self.assertEqual(stat.S_IMODE(root.stat().st_mode) & 0o022, 0)
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(authorized.stat().st_mode), 0o600)

    def test_startup_creates_missing_ssh_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            script = ssh_key_startup("ssh-ed25519 PUBLICKEY", root=temporary)
            subprocess.run(["sh", "-c", script], check=True, capture_output=True)
            self.assertEqual(
                (Path(temporary) / ".ssh/authorized_keys").read_text().strip(),
                "ssh-ed25519 PUBLICKEY",
            )

    def test_startup_refuses_missing_or_multiple_keys(self) -> None:
        for key in ("", "private key", "ssh-ed25519 KEY\nssh-ed25519 OTHER"):
            with self.assertRaises(ValueError):
                ssh_key_startup(key)

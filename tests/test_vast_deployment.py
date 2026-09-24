from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.connection_store import merge_connections
from llm_launchpad.core.vast_backend import VastApiError, VastBackend
from llm_launchpad.core.vast_deployment import (
    VAST_PULL_STALL_SECONDS, VAST_READY_DEADLINE_SECONDS, VAST_SSH_STALL_SECONDS,
    VAST_KEY_GRACE_SECONDS,
    VAST_STARTUP_HEARTBEAT_SECONDS,
    VastDeploymentBackend, vast_progress_key, vast_startup_detail,
)
from llm_launchpad.core.vast_runtime import vast_runtime_image, vast_runtime_script, verify_endpoint_auth, verify_streaming
from llm_launchpad.core.vast_ssh import (
    SSH_REFUSED, SSH_REJECTED, SSH_THROTTLED, VastSsh, VastSshError, ssh_failure_reason,
    ssh_key_startup,
)
from llm_launchpad.core.vast_state import VastState
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig, VastAuthStatus, VastInstance, VastProviderOptions
from tests.test_vast_fast_deploy import vast_offer


class InstantEvent(threading.Event):
    """A cancellation event whose timed waits return immediately.

    Vast's provisioning loop backs off with ``cancellation.wait(...)`` between
    polls -- 15s after a throttled status check, 3s otherwise. Those backoffs
    are exactly what the throttling and stall tests drive, so the loop has to
    keep running them; waiting them out in real time just costs wall clock.
    """

    def wait(self, timeout: float | None = None) -> bool:
        return super().wait(0)


def config() -> DeploymentConfig:
    return DeploymentConfig(
        backend=BackendType.LLAMACPP, provider=ComputeProvider.VAST,
        app_name="llp-vast-llamacpp-test", repo_id="acme/model-GGUF", quant="Q4_K_M",
        gpu_type="RTX 4090", gpu_count=1, do_deploy=True,
        provider_options=VastProviderOptions("1001", 100, 0.42, "42"),
        gguf_architecture="llama", server_args="--ctx-size 4096 --n-gpu-layers all",
    )


class SshFailureReasonTests(unittest.TestCase):
    """A refused port and a denied key need opposite decisions, so they differ."""

    def test_a_host_that_has_not_started_sshd_reads_as_refused(self) -> None:
        for stderr in (
            "ssh: connect to host ssh9.vast.ai port 29814: Connection refused",
            "kex_exchange_identification: Connection closed by remote host",
            "Connection reset by 1.2.3.4 port 29814",
        ):
            with self.subTest(stderr=stderr):
                self.assertEqual(ssh_failure_reason(stderr), SSH_REFUSED)

    def test_a_host_that_has_decided_reads_as_rejected(self) -> None:
        for stderr in (
            "root@ssh9.vast.ai: Permission denied (publickey).",
            "No supported authentication methods available",
        ):
            with self.subTest(stderr=stderr):
                self.assertEqual(ssh_failure_reason(stderr), SSH_REJECTED)

    def test_an_attempt_limit_is_about_our_knocking_not_about_the_key(self) -> None:
        """The proxy's rate limit is caused by this loop, so it decides nothing.

        The deploy path already notes that hundreds of failed authentications
        look like an attack to Vast's shared SSH proxy. Reading the limit that
        provokes as a rejected key blames the rental for our own polling.
        """
        for stderr in (
            "Received disconnect from 1.2.3.4: Too many authentication failures",
            "Maximum authentication attempts exceeded for root",
        ):
            with self.subTest(stderr=stderr):
                self.assertEqual(ssh_failure_reason(stderr), SSH_THROTTLED)
                self.assertNotEqual(ssh_failure_reason(stderr), SSH_REJECTED)

    def test_a_changed_host_key_is_its_own_reason_not_a_denial(self) -> None:
        # The rental is reached through a shared proxy that answers before the
        # container's sshd, so the key seen first is not always the final one.
        # A denial-shaped reading of that ends a healthy rental in seconds.
        for stderr in (
            "Host key verification failed.",
            "@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@",
            "Offending ECDSA key for IP in /x/known_hosts:1",
        ):
            with self.subTest(stderr=stderr):
                self.assertNotEqual(ssh_failure_reason(stderr), SSH_REJECTED)

    def test_anything_unrecognised_is_not_guessed_into_a_decision(self) -> None:
        # Unreachable keeps waiting, like a refusal: only a positive match on
        # a denial is allowed to end a rental early.
        self.assertEqual(ssh_failure_reason(""), "unreachable")
        self.assertEqual(ssh_failure_reason("ssh: Could not resolve hostname"), "unreachable")

    def test_the_raised_error_carries_the_reason_without_the_stderr(self) -> None:
        error = VastSshError("Vast SSH command failed.", SSH_REJECTED)
        self.assertEqual(error.reason, SSH_REJECTED)
        self.assertNotIn("publickey", str(error))
        # Callers that only know RuntimeError keep catching it.
        self.assertIsInstance(error, RuntimeError)


class StartupDetailTests(unittest.TestCase):
    """What the startup heartbeat says, given what the host could be asked."""

    def test_known_total_reports_percentage_rate_and_eta(self) -> None:
        self.assertEqual(
            vast_startup_detail(9_000_000_000, 3_000_000_000, 30, "", total=15_000_000_000),
            "downloading weights (60%), 9.0 / 15.0 GB at 200 MB/s, ~30s remaining",
        )

    def test_active_download_cannot_claim_completion(self) -> None:
        self.assertIn("(99%)", vast_startup_detail(1000, 500, 30, "", total=1000))
        self.assertIn("(100%)", vast_startup_detail(1000, 500, 30, "", total=1000, active=False))

    def test_known_total_starts_at_zero(self) -> None:
        self.assertIn("(0%)", vast_startup_detail(0, 0, 30, "stale", total=1000))

    def test_the_first_sample_reports_size_without_inventing_a_rate(self) -> None:
        self.assertEqual(
            vast_startup_detail(3_000_000_000, 0, 31.0, "stale log line"),
            "downloading weights, 3.0 GB fetched",
        )

    def test_a_later_sample_reports_the_rate_it_measured(self) -> None:
        self.assertEqual(
            vast_startup_detail(9_000_000_000, 3_000_000_000, 30.0, ""),
            "downloading weights, 9.0 GB fetched at 200 MB/s",
        )

    def test_a_download_that_gained_nothing_reports_size_only(self) -> None:
        self.assertEqual(
            vast_startup_detail(9_000_000_000, 9_000_000_000, 30.0, ""),
            "downloading weights, 9.0 GB fetched",
        )

    def test_once_the_weights_land_the_server_log_takes_over(self) -> None:
        self.assertEqual(
            vast_startup_detail(0, 17_000_000_000, 30.0, "load_tensors: offloaded 63/63"),
            "load_tensors: offloaded 63/63",
        )

    def test_nothing_measurable_says_so_rather_than_showing_an_empty_line(self) -> None:
        self.assertEqual(vast_startup_detail(0, 0, 30.0, ""), "no progress reported yet")


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
            patch("llm_launchpad.core.vast_deployment.fetch_download_files", return_value=()),
            patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=True),
            patch("llm_launchpad.core.vast_deployment.verify_endpoint_auth"),
            patch("llm_launchpad.core.vast_deployment.is_shutting_down", return_value=False),
            # Provisioning and teardown both back off between polls. The waits
            # are what the retry tests are about, but serving them out in real
            # time costs the suite more wall clock than every other Vast test
            # combined.
            patch("llm_launchpad.core.vast_deployment.time.sleep"),
            patch("llm_launchpad.core.vast_deployment.threading.Event", InstantEvent),
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
        # Retained so a test can assert on what the deploy reported, not only
        # on how it ended.
        self.events = list(self.backend.deploy(self.config))
        return next(
            event for event in reversed(self.events)
            if isinstance(event, OperationCompleteEvent)
        )

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

    def test_vllm_image_input_rents_like_every_other_supported_path(self) -> None:
        """Vast gates no runtime behind an opt-in that Modal and Prime lack."""
        from llm_launchpad.protocol.enums import VisionMode

        self.config.backend = BackendType.VLLM
        self.config.model_name = "Qwen/Qwen3-0.6B"
        self.config.vision_mode = VisionMode.ON
        # The vLLM image's own CUDA floor is 13.0, above llama.cpp's.
        self.api.get_offer.return_value = vast_offer(cuda_max_good=13.0)
        event = self.deploy()
        self.assertTrue(event.success, event.detail)
        self.api.create_instance.assert_called_once()

    def test_changed_machine_or_insufficient_memory_cannot_rent(self) -> None:
        self.api.get_offer.return_value = vast_offer(machine_id=43)
        self.assertFalse(self.deploy().success)
        self.api.get_offer.return_value = vast_offer()
        self.config.required_vram_gb = 50
        self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_not_called()

    def test_a_reporting_host_that_goes_silent_is_given_back(self) -> None:
        """A host that reports a step and never leaves it is stuck, not slow.

        Vast leaves ``actual_status`` on "loading" for the whole pull and
        provisioning, so only ``status_msg`` distinguishes the two. This is the
        live GTX 1650 wedge: one BuildKit line for 20 billed minutes.
        """
        def stalled_create(offer_id: str, **kwargs: object) -> str:
            self.remote = VastInstance(
                "900", str(kwargs["label"]), "loading", "42",
                status_msg="#6 1.0 Get:8 http://archive.ubuntu.com noble InRelease",
            )
            return "900"

        self.api.create_instance.side_effect = stalled_create
        # 100s of provisioning time per clock read closes the 900s pull window
        # within a few passes, short of the 1800s deadline.
        clock = iter(range(0, 1_000_000, 100))
        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=lambda: next(clock)):
            event = self.deploy()

        self.assertFalse(event.success)
        self.assertIn("stopped making progress", event.detail or "")
        # Name the step it never left: without it the report says only that
        # something was silent, which is what made the first one undiagnosable.
        self.assertIn("Get:8 http://archive.ubuntu.com noble InRelease", event.detail or "")
        # The rental is handed back rather than billed out to the deadline.
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_a_running_host_that_will_not_answer_ssh_is_given_back(self) -> None:
        """A running host that refuses SSH advances no work, so it is stuck.

        Live evidence: a rental sat in ``running`` with its image up for the
        full 1800s deadline, answering status polls but never SSH, billing
        $0.015 for nothing. The installer had finished but the daemon never
        started, and no further wait would have changed that. The stall clock
        must therefore fire while the loop is knocking on a running host,
        while a host still pulling its image keeps the full deadline.
        """
        def running_create(offer_id: str, **kwargs: object) -> str:
            self.remote = VastInstance(
                "900", str(kwargs["label"]), "running", "42",
                ssh_host="ssh12.vast.ai", ssh_port=1234,
                status_msg="success, running ghcr.io/ggml-org/llama.cpp",
            )
            return "900"

        self.api.create_instance.side_effect = running_create
        self.ssh.run.side_effect = RuntimeError("connection refused")
        # The SSH knock's own backoff doubles to 60s, so a 100s clock step
        # sails past both the stall window and the deadline in one jump and
        # the loop reports the deadline instead. Step 10s at a time so the
        # stall branch wins while the deadline is still far away.
        clock = iter(range(0, 1_000_000, 10))
        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=lambda: next(clock)):
            event = self.deploy()

        self.assertFalse(event.success)
        self.assertIn("stopped making progress", event.detail or "")
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_a_rental_that_stops_says_what_vast_reported(self) -> None:
        """"Stopped before the runtime became ready" alone was undiagnosable.

        Live evidence (2026-09-23): an RTX 3060 host pulled the image and the
        container exited before SSH was ever up -- a host fault, not the
        runtime -- but the report could not say which.
        """
        def exited_create(offer_id: str, **kwargs: object) -> str:
            self.remote = VastInstance(
                "900", str(kwargs["label"]), "exited", "42",
                status_msg="Error response from daemon: failed to create task",
            )
            return "900"

        self.api.create_instance.side_effect = exited_create
        event = self.deploy()

        self.assertFalse(event.success)
        self.assertIn("state: exited", event.detail or "")
        self.assertIn("failed to create task", event.detail or "")
        self.api.destroy_instance.assert_called_once_with("900")

    def test_a_host_still_installing_sshd_is_not_given_back_at_the_old_window(self) -> None:
        """The wait that was killing healthy rentals, at the length that killed one.

        A live RTX PRO 6000 refused SSH for 378s while it was still installing
        its SSH server and was destroyed for it, which cost the rental and the
        eight minutes of the retry. The fastest success on record answered at
        386.6s, so the old 360s window closed on working hosts. Refuse for
        longer than that here and then answer: the loop has to still be there.
        """
        def running_create(offer_id: str, **kwargs: object) -> str:
            self.remote = VastInstance(
                "900", str(kwargs["label"]), "running", "42",
                ssh_host="ssh12.vast.ai", ssh_port=1234,
                status_msg="success, running ghcr.io/ggml-org/llama.cpp",
            )
            return "900"

        self.api.create_instance.side_effect = running_create
        clock = [0.0]
        refusals = [0]

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if command == "true":
                refusals[0] += 1
                # 400s of refusals: past the window that destroyed the live
                # rental, short of the one the evidence supports.
                if clock[0] < 400:
                    raise VastSshError("refused", SSH_REFUSED)
                return ""
            return self.gpu_inventory if "nvidia-smi" in command else ""

        self.ssh.run.side_effect = remote

        def tick() -> float:
            clock[0] += 5.0
            return clock[0]

        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=tick):
            event = self.deploy()

        self.assertTrue(event.success, event.detail)
        self.assertGreater(refusals[0], 1)
        self.api.destroy_instance.assert_not_called()

    def _running_host(self) -> None:
        def running_create(offer_id: str, **kwargs: object) -> str:
            self.remote = VastInstance(
                "900", str(kwargs["label"]), "running", "42",
                ssh_host="ssh12.vast.ai", ssh_port=1234,
                status_msg="success, running ghcr.io/ggml-org/llama.cpp",
            )
            return "900"

        self.api.create_instance.side_effect = running_create

    def test_a_key_denied_before_the_key_is_installed_is_not_an_answer(self) -> None:
        """Vast attaches the key asynchronously, so the first denial means nothing.

        Live evidence: an RTX PRO 6000 WS rental was destroyed 49.7s into its
        deploy, one second after Vast reported the host running, because the
        first knock was denied while authorized_keys was still being written.
        Inside the grace window a denial is provisioning, not a decision.
        """
        self._running_host()
        clock = [0.0]
        knocks = [0]

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if command == "true":
                knocks[0] += 1
                if clock[0] < VAST_KEY_GRACE_SECONDS - 30:
                    raise VastSshError("denied", SSH_REJECTED)
                return ""
            return self.gpu_inventory if "nvidia-smi" in command else ""

        self.ssh.run.side_effect = remote

        def tick() -> float:
            clock[0] += 5.0
            return clock[0]

        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=tick):
            event = self.deploy()

        self.assertTrue(event.success, event.detail)
        self.assertGreater(knocks[0], 1)
        self.api.destroy_instance.assert_not_called()

    def test_a_throttled_knock_never_ends_the_rental(self) -> None:
        """An attempt limit must outlast the grace window without deciding."""
        self._running_host()
        clock = [0.0]

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if command == "true":
                # Well past the key grace window: a denial here would end it.
                if clock[0] < VAST_KEY_GRACE_SECONDS * 2:
                    raise VastSshError("throttled", SSH_THROTTLED)
                return ""
            return self.gpu_inventory if "nvidia-smi" in command else ""

        self.ssh.run.side_effect = remote

        def tick() -> float:
            clock[0] += 5.0
            return clock[0]

        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=tick):
            event = self.deploy()

        self.assertTrue(event.success, event.detail)
        self.api.destroy_instance.assert_not_called()

    def test_a_key_still_denied_after_the_grace_window_is_given_back(self) -> None:
        """A denial that outlives key installation is decided, so it ends the rental.

        This is the case the SSH stall window was really aimed at, and timing
        it at 900s was the wrong instrument: the host has answered, so the
        answer is available long before the window that watches silence.
        """
        self._running_host()
        clock = [0.0]

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if command == "true":
                raise VastSshError("denied", SSH_REJECTED)
            return self.gpu_inventory if "nvidia-smi" in command else ""

        self.ssh.run.side_effect = remote

        def tick() -> float:
            clock[0] += 5.0
            return clock[0]

        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=tick):
            event = self.deploy()

        self.assertFalse(event.success)
        self.assertIn("refused the deployment key", event.detail or "")
        # Decided well inside the window that watches a silent host.
        self.assertLess(clock[0], VAST_SSH_STALL_SECONDS)
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_a_changed_host_key_while_provisioning_is_not_a_denied_key(self) -> None:
        """The proxy answers before the rental's own sshd, so the key can change.

        Reading that as an authentication decision ends a healthy rental in
        the first seconds, which is what a denial-shaped classification did.
        """
        self.assertEqual(
            ssh_failure_reason(
                "@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\n"
                "Host key verification failed."
            ),
            "host-key",
        )
        self._running_host()
        clock = [0.0]

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if command == "true":
                if clock[0] < 300:
                    raise VastSshError("host key changed", "host-key")
                return ""
            return self.gpu_inventory if "nvidia-smi" in command else ""

        self.ssh.run.side_effect = remote

        def tick() -> float:
            clock[0] += 5.0
            return clock[0]

        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=tick):
            event = self.deploy()

        self.assertTrue(event.success, event.detail)
        self.api.destroy_instance.assert_not_called()

    def test_a_host_that_reports_nothing_is_never_called_stalled(self) -> None:
        """An empty status line is no signal, so it cannot be a stall signal.

        Vast leaves ``status_msg`` empty for stretches of a healthy pull -- the
        certified rental logged a docker line at 32s and nothing at 64s -- and
        a sibling rental reached SSH only at 386.6s. Reading that silence as a
        wedge destroyed hosts that were still minutes from working, so a host
        reporting nothing keeps the full deadline and only the deadline.
        """
        clock = [0.0]
        polls = 0

        def silent_create(offer_id: str, **kwargs: object) -> str:
            self.remote = VastInstance("900", str(kwargs["label"]), "loading", "42")
            return "900"

        def silent_get(_: str) -> VastInstance | None:
            # One clock step per poll, so the loop's own progress is the only
            # thing that advances time and the silence below is exact.
            nonlocal polls
            polls += 1
            clock[0] += 200.0
            assert self.remote is not None
            if polls >= 7:  # 1400s in: past the 900s window, inside the 1800s deadline.
                self.remote = VastInstance(
                    "900", self.remote.label, "running", "42", "ssh12.vast.ai", 1234,
                )
            return self.remote

        self.api.create_instance.side_effect = silent_create
        self.api.get_instance.side_effect = silent_get
        with patch("llm_launchpad.core.vast_deployment.time.monotonic", side_effect=lambda: clock[0]):
            event = self.deploy()

        self.assertTrue(event.success, event.detail)
        self.assertGreater(polls, 5)  # It really did sit out the pull window.
        self.api.destroy_instance.assert_not_called()

    def test_a_running_host_still_pulling_keeps_the_full_deadline(self) -> None:
        """A loading host is not knocking on anything, so it keeps the deadline.

        The same silent status key that means "stuck" on a running host is
        normal while the image is still being pulled: no SSH address exists
        yet, so there is nothing to knock on.
        """
        seen: list[str] = []
        running: VastInstance | None = None

        def flapping_create(offer_id: str, **kwargs: object) -> str:
            nonlocal running
            label = str(kwargs["label"])
            self.remote = VastInstance(
                "900", label, "loading", "42",
                status_msg="#6 1.0 Get:8 http://archive.ubuntu.com noble InRelease",
            )
            # The running twin carries this rental's own label: the identity
            # check refuses a host whose label changed mid-flight.
            running = VastInstance(
                "900", label, "running", "42",
                ssh_host="ssh12.vast.ai", ssh_port=1234,
                status_msg="success, running ghcr.io/ggml-org/llama.cpp",
            )
            return "900"

        self.api.create_instance.side_effect = flapping_create

        def flapping_get(_: str) -> VastInstance | None:
            assert self.remote is not None
            seen.append(self.remote.state)
            # Once the image is up SSH answers. Flipping on the second poll
            # proves the loop survives a silent loading state without
            # tripping the stall clock.
            if len(seen) > 1:
                assert running is not None
                self.remote = running
            return self.remote

        self.api.get_instance.side_effect = flapping_get
        event = self.deploy()

        self.assertTrue(event.success, event.detail)
        self.assertEqual(seen[0], "loading")

    def test_the_model_startup_wait_reports_what_the_runtime_is_doing(self) -> None:
        """The longest wait of a deploy has to keep saying it is a wait.

        Weights arrive with the server log holding one stale line, so a silent
        loop here showed "Rented GPUs" until the endpoint answered: ten quiet
        minutes that read as a hang and got a paying rental killed by hand.
        """
        clock = {"now": 0.0}
        fetched = iter([3_000_000_000, 9_000_000_000, 17_000_000_000])
        polls: list[int] = []

        def healthy(*_: object) -> bool:
            polls.append(len(polls))
            if len(polls) > 3:
                return True
            # Each poll carries the wait past one heartbeat window.
            clock["now"] += VAST_STARTUP_HEARTBEAT_SECONDS + 1
            return False

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if "nvidia-smi" in command:
                return self.gpu_inventory
            if "downloadInProgress" in command:
                return f"BYTES {next(fetched)}\nLOG waiting on weights\n"
            return ""

        self.ssh.run.side_effect = remote
        with (
            patch("llm_launchpad.core.vast_deployment.endpoint_healthy", healthy),
            patch(
                "llm_launchpad.core.vast_deployment.time.monotonic",
                lambda: clock["now"],
            ),
        ):
            events = list(self.backend.deploy(self.config))

        lines = [
            event.line for event in events
            if getattr(event, "line", "").startswith("Vast model starting:")
        ]
        self.assertEqual(lines, [
            "Vast model starting: downloading weights, 3.0 GB fetched (waiting 31s)",
            "Vast model starting: downloading weights, 9.0 GB fetched at 194 MB/s"
            " (waiting 1m02s)",
            "Vast model starting: downloading weights, 17.0 GB fetched at 258 MB/s"
            " (waiting 1m33s)",
        ])
        # The probe rides the tunnel the deploy already opened rather than
        # dialling Vast's shared SSH proxy once per heartbeat.
        probes = [
            call for call in self.ssh.run.call_args_list
            if "downloadInProgress" in call.args[1]
        ]
        self.assertEqual(len(probes), 3)
        self.assertTrue(all(call.kwargs.get("multiplex") for call in probes))

    def test_a_probe_the_host_cannot_answer_never_ends_a_paid_deploy(self) -> None:
        clock = {"now": 0.0}
        polls: list[int] = []

        def healthy(*_: object) -> bool:
            polls.append(len(polls))
            if len(polls) > 1:
                return True
            clock["now"] += VAST_STARTUP_HEARTBEAT_SECONDS + 1
            return False

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if "nvidia-smi" in command:
                return self.gpu_inventory
            if "downloadInProgress" in command:
                raise RuntimeError("Vast SSH command failed.")
            return ""

        self.ssh.run.side_effect = remote
        with (
            patch("llm_launchpad.core.vast_deployment.endpoint_healthy", healthy),
            patch(
                "llm_launchpad.core.vast_deployment.time.monotonic",
                lambda: clock["now"],
            ),
        ):
            events = list(self.backend.deploy(self.config))

        completion = next(
            event for event in reversed(events)
            if isinstance(event, OperationCompleteEvent)
        )
        self.assertTrue(completion.success, completion.detail)
        self.assertIn(
            "Vast model starting: still starting (waiting 31s)",
            [getattr(event, "line", "") for event in events],
        )

    def test_shard_progress_completes_once_then_reports_loading(self) -> None:
        from llm_launchpad.core.vast_download import DownloadFile

        clock = {"now": 0.0}
        samples = iter([
            "FILE 500 8 models/blobs/a.downloadInProgress",
            "FILE 1000 8 models/blobs/a\nFILE 500 8 models/blobs/b.downloadInProgress",
            "FILE 1000 8 models/blobs/a\nFILE 1000 8 models/blobs/b",
            "FILE 1000 8 models/blobs/a\nFILE 1000 8 models/blobs/b",
        ])

        def healthy(*_: object) -> bool:
            clock["now"] += 31
            return clock["now"] > 124

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if "nvidia-smi" in command:
                return self.gpu_inventory
            if "downloadInProgress" in command:
                return next(samples) + "\nLOG load_tensors: loading model tensors\n"
            return ""

        self.ssh.run.side_effect = remote
        with (
            patch("llm_launchpad.core.vast_deployment.fetch_download_files", return_value=(
                DownloadFile("models/blobs/a", 1000, "q4"),
                DownloadFile("models/blobs/b", 1000, "q4"),
            )),
            patch("llm_launchpad.core.vast_deployment.endpoint_healthy", healthy),
            patch("llm_launchpad.core.vast_deployment.time.monotonic", lambda: clock["now"]),
        ):
            events = list(self.backend.deploy(self.config))
        lines = [event.line for event in events if getattr(event, "line", "").startswith("Vast model starting:")]
        self.assertEqual(len(lines), 4)
        for line, pct in zip(lines[:3], (25, 75, 100), strict=True):
            self.assertIn(f"({pct}%)", line)
        self.assertIn("load_tensors: loading model tensors", lines[3])
        self.assertTrue(next(event for event in reversed(events) if isinstance(event, OperationCompleteEvent)).success)

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
        from llm_launchpad.core.provider_adapters import rollback_from_event

        self.api.create_instance.side_effect = VastApiError("timeout")
        self.config.fallback_configs = (config(),)
        events = list(self.backend.deploy(self.config))
        failure = next(event for event in reversed(events) if isinstance(event, OperationCompleteEvent))
        self.assertFalse(failure.success)
        # Provider no longer mutates the approved ladder; it reports an
        # unconfirmed rollback so the lifecycle blocks the next rental.
        self.assertEqual(len(self.config.fallback_configs), 1)
        rollback = rollback_from_event(failure)
        self.assertIsNotNone(rollback)
        assert rollback is not None
        self.assertFalse(rollback.confirmed)
        self.assertEqual(len(self.state.records()), 1)
        self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_called_once()

    def test_a_refused_create_frees_the_name_instead_of_blocking_it(self) -> None:
        # Live: Vast answered PUT /asks/<id>/ with HTTP 400 and no rental was
        # made. Cleanup reconciled by label, found zero instances, and read
        # that as "creation remains uncertain" -- so the record survived, and
        # the name was then refused by deploy ("already recorded") *and* by
        # destroy ("uncertain"), with no way out through the product.
        self.api.create_instance.side_effect = VastApiError(
            "Vast request failed (HTTP 400).", status_code=400
        )

        event = self.deploy()

        self.assertFalse(event.success)
        self.assertNotIn("Cleanup remains pending", event.detail or "")
        self.assertEqual(self.state.records(), [])
        self.api.destroy_instance.assert_not_called()
        # The name is free, so the obvious next move actually works.
        self.api.create_instance.side_effect = self.create
        self.assertTrue(self.deploy().success)

    def test_an_unanswered_create_still_retains_the_record(self) -> None:
        # A transport failure leaves it genuinely unknown whether Vast acted,
        # and a rental that lands a moment after the label scan would bill
        # unattended. That record is kept on purpose.
        self.api.create_instance.side_effect = VastApiError("request timed out")

        self.assertFalse(self.deploy().success)

        self.assertEqual(len(self.state.records()), 1)

    def test_a_throttled_create_is_not_treated_as_a_refusal(self) -> None:
        # 429 means Vast declined to answer yet, not that it refused the
        # rental, so it keeps the conservative path.
        self.api.create_instance.side_effect = VastApiError(
            "Vast rate limit reached.", status_code=429
        )

        self.assertFalse(self.deploy().success)

        self.assertEqual(len(self.state.records()), 1)

    def test_a_dead_runtime_reports_more_than_the_heartbeat_window(self) -> None:
        """vLLM's "See root cause above" is useless without the above.

        The heartbeat reads 25 lines because it runs every few seconds; a
        Python traceback is longer than that, so the line naming the cause is
        pushed out by the line that points at it. A live Vast rental failed
        exactly this way and reported nothing but the pointer.
        """
        probe = (
            f"{self.gpu_inventory}\n"
            "BYTES 0\n"
            "LOG RuntimeError: Engine core initialization failed. See root cause above.\n"
            "RUNTIME 0\n"
        )
        post_mortem = "\n".join(
            ["torch.OutOfMemoryError: CUDA out of memory allocating 12.00 GiB"]
            + [f"  frame {index}" for index in range(60)]
            + ["RuntimeError: Engine core initialization failed. See root cause above."]
        )

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if "nvidia-smi" in command:
                return self.gpu_inventory
            if command.startswith("tail -c 200000"):
                return post_mortem
            return probe

        self.ssh.run.side_effect = remote
        with patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=False), patch(
            "llm_launchpad.core.vast_deployment.VAST_STARTUP_HEARTBEAT_SECONDS", 0
        ), patch("llm_launchpad.core.vast_deployment.VAST_SERVE_DEADLINE_SECONDS", 20):
            event = self.deploy()

        self.assertFalse(event.success)
        logged = "\n".join(line.line for line in self.events if isinstance(line, LogEvent))
        self.assertIn("CUDA out of memory", logged)

    def test_a_post_mortem_read_that_fails_keeps_the_heartbeat_lines(self) -> None:
        probe = (
            f"{self.gpu_inventory}\n"
            "BYTES 0\n"
            "LOG E srv: exiting due to model loading error\n"
            "RUNTIME 0\n"
        )

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if "nvidia-smi" in command:
                return self.gpu_inventory
            if command.startswith("tail -c 200000"):
                raise RuntimeError("ssh closed")
            return probe

        self.ssh.run.side_effect = remote
        with patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=False), patch(
            "llm_launchpad.core.vast_deployment.VAST_STARTUP_HEARTBEAT_SECONDS", 0
        ), patch("llm_launchpad.core.vast_deployment.VAST_SERVE_DEADLINE_SECONDS", 20):
            event = self.deploy()

        self.assertFalse(event.success)
        logged = "\n".join(line.line for line in self.events if isinstance(line, LogEvent))
        self.assertIn("exiting due to model loading error", logged)

    def test_a_runtime_that_exited_fails_now_with_the_reason_it_printed(self) -> None:
        # Live: a malformed export killed the startup script in its first
        # second. The watcher read only the server log, which simply stopped
        # changing, so the rental billed for the full 30-minute deadline and
        # then reported "startup timed out" -- while the real message sat
        # unread in server.log the whole time.
        probe = (
            f"{self.gpu_inventory}\n"
            "BYTES 0\n"
            "LOG load_tensors: offloading 64 repeating layers to GPU\n"
            "LOG ggml_backend_cuda_buffer_type_alloc_buffer: allocating 21474 MiB on device 0: out of memory\n"
            "LOG E srv llama_server: exiting due to model loading error\n"
            "RUNTIME 0\n"
        )
        # The post-mortem re-reads server.log directly, with a window a whole
        # traceback fits in; the heartbeat's own 25 lines do not.
        post_mortem = (
            "load_tensors: offloading 64 repeating layers to GPU\n"
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 21474 MiB on device 0: out of memory\n"
            "E srv llama_server: exiting due to model loading error\n"
        )

        def remote(instance: object, command: str, **kwargs: object) -> str:
            if "nvidia-smi" in command:
                return self.gpu_inventory
            if command.startswith("tail -c 200000"):
                return post_mortem
            return probe

        self.ssh.run.side_effect = remote
        # Probe on the first pass, and bound the deadline so a regression that
        # stops watching the runtime fails here in seconds rather than spinning
        # out the real half hour the way the paid rental did.
        with patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=False), patch(
            "llm_launchpad.core.vast_deployment.VAST_STARTUP_HEARTBEAT_SECONDS", 0
        ), patch("llm_launchpad.core.vast_deployment.VAST_SERVE_DEADLINE_SECONDS", 20):
            event = self.deploy()

        self.assertFalse(event.success)
        self.assertIn("runtime exited", event.detail or "")
        self.assertIn("exiting due to model loading error", event.detail or "")
        # The last line names the symptom; the cause is the line before it, so
        # the run-up is reported rather than left on a destroyed host.
        logged = "\n".join(
            line.line for line in self.events if isinstance(line, LogEvent)
        )
        self.assertIn("out of memory", logged)
        self.assertIn("offloading 64 repeating layers", logged)
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_one_probe_that_missed_the_process_does_not_end_the_rental(self) -> None:
        # A single reading is not evidence: staging spends minutes inside one
        # quiet `snapshot_download`, and a probe that failed to see it would
        # otherwise destroy a healthy rental mid-download.
        alive = f"{self.gpu_inventory}\nBYTES 0\nLOG staging weights\nRUNTIME 1\n"
        blip = f"{self.gpu_inventory}\nBYTES 0\nLOG staging weights\nRUNTIME 0\n"
        self.ssh.run.side_effect = [alive, alive, blip, *([alive] * 40)]

        with patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=False), patch(
            "llm_launchpad.core.vast_deployment.VAST_STARTUP_HEARTBEAT_SECONDS", 0
        ), patch("llm_launchpad.core.vast_deployment.VAST_SERVE_DEADLINE_SECONDS", 3):
            event = self.deploy()

        self.assertFalse(event.success)
        # However this deploy ends, it must not end by declaring the runtime
        # dead on the strength of one reading that disagreed with its
        # neighbours.
        self.assertNotIn("runtime exited", event.detail or "")

    def test_a_runtime_that_has_left_no_trace_yet_is_not_called_dead(self) -> None:
        # The seconds before the startup script runs look exactly like a script
        # that has exited: no process, no log. Acting on that destroyed a
        # healthy rental 74s into a deploy, so absence needs corroboration.
        self.ssh.run.side_effect = None
        self.ssh.run.return_value = f"{self.gpu_inventory}\nBYTES 0\nRUNTIME 0\n"

        with patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=False), patch(
            "llm_launchpad.core.vast_deployment.VAST_STARTUP_HEARTBEAT_SECONDS", 0
        ), patch("llm_launchpad.core.vast_deployment.VAST_SERVE_DEADLINE_SECONDS", 3):
            event = self.deploy()

        self.assertFalse(event.success)
        # It waits out the deadline rather than claiming an exit it cannot see.
        self.assertIn("timed out", event.detail or "")
        self.assertNotIn("runtime exited", event.detail or "")

    def test_a_runtime_seen_running_and_then_gone_is_called_dead(self) -> None:
        alive = f"{self.gpu_inventory}\nBYTES 0\nLOG loading model\nRUNTIME 1\n"
        gone = f"{self.gpu_inventory}\nBYTES 0\nLOG loading model\nRUNTIME 0\n"
        self.ssh.run.side_effect = [alive, alive, *([gone] * 40)]

        with patch("llm_launchpad.core.vast_deployment.endpoint_healthy", return_value=False), patch(
            "llm_launchpad.core.vast_deployment.VAST_STARTUP_HEARTBEAT_SECONDS", 0
        ), patch("llm_launchpad.core.vast_deployment.VAST_SERVE_DEADLINE_SECONDS", 30):
            event = self.deploy()

        self.assertFalse(event.success)
        self.assertIn("runtime exited", event.detail or "")

    def test_a_probe_that_cannot_answer_is_not_read_as_a_dead_runtime(self) -> None:
        from llm_launchpad.core.vast_runtime import startup_runtime_stopped

        # Only an explicit marker is an answer; silence is a busy host.
        self.assertFalse(startup_runtime_stopped("BYTES 0\nLOG loading model"))
        self.assertFalse(startup_runtime_stopped(""))
        self.assertFalse(startup_runtime_stopped("RUNTIME 1"))
        self.assertTrue(startup_runtime_stopped("RUNTIME 0"))

    def test_the_liveness_probe_cannot_match_its_own_command_line(self) -> None:
        import re

        from llm_launchpad.core.vast_runtime import VAST_STARTUP_PROBE_COMMAND

        # The probe runs as a shell command whose own /proc entry would match a
        # naive pattern, so every rental would look alive forever.
        patterns = re.findall(r"-e '([^']+)'", VAST_STARTUP_PROBE_COMMAND)
        self.assertTrue(patterns)
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, VAST_STARTUP_PROBE_COMMAND))
        # It still matches the processes it is looking for.
        self.assertTrue(any(re.search(p, "sh /root/.llm-launchpad/runtime.sh") for p in patterns))
        self.assertTrue(any(re.search(p, "/app/llama-server --host 127.0.0.1") for p in patterns))

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
        from llm_launchpad.core.provider_adapters import rollback_from_event

        self.assertTrue(self.deploy().success)
        self.config.fallback_configs = (config(),)
        events = list(self.backend.deploy(self.config))
        failure = next(event for event in reversed(events) if isinstance(event, OperationCompleteEvent))
        self.assertFalse(failure.success)
        # The provider reports the conflict; the lifecycle (not a mutation of
        # the approved ladder) owns blocking the retry.
        self.assertEqual(len(self.config.fallback_configs), 1)
        rollback = rollback_from_event(failure)
        self.assertIsNotNone(rollback)
        assert rollback is not None
        self.assertFalse(rollback.confirmed)
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
        self.assertFalse(self.deploy().success)
        self.api.create_instance.assert_called_once()
        self.api.destroy_instance.assert_called_once_with("900")
        self.assertEqual(self.state.records(), [])

    def test_persistent_identity_throttle_preserves_rental_record(self) -> None:
        self.deploy()
        self.api.get_instance.side_effect = VastApiError("throttled", status_code=429)
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
        self.backend.destroy(name=record.name, instance_id="900")
        self.assertIsNone(self.state.load(record.name))
        self.assertIsNone(self.remote)

    def test_a_persistent_rate_limit_keeps_the_recovery_record(self) -> None:
        self.deploy()
        record = self.state.load(self.config.app_name or "")
        assert record is not None
        self.api.destroy_instance.side_effect = VastApiError("nope", status_code=429)
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


class VastProvisioningProgressTests(unittest.TestCase):
    """A rental that is slow is not a rental that is stuck.

    Vast reports provisioning in ``status_msg`` while ``actual_status`` sits on
    "loading" throughout, so the deploy loop reads progress from the message
    rather than waiting out a fixed clock.
    """

    def test_a_multi_line_status_arrives_as_one_readable_line(self) -> None:
        from llm_launchpad.core.vast_backend import _parse_instance

        raw = {
            "id": 900, "label": "l", "actual_status": "loading", "machine_id": 42,
            "status_msg": "404ad8037495: Verifying Checksum\n404ad8037495: Download complete\n",
        }
        instance = _parse_instance(raw)
        # Dropping the newline outright ran the two lines into one word.
        self.assertEqual(
            instance.status_msg,
            "404ad8037495: Verifying Checksum 404ad8037495: Download complete",
        )

    def test_a_ticking_buildkit_counter_is_not_progress(self) -> None:
        # Same step, same work, later timestamp: the host has not advanced.
        first = VastInstance("1", "l", "loading", "42", status_msg="#6 62.25 Get:8 http://archive.ubuntu.com noble InRelease")
        later = VastInstance("1", "l", "loading", "42", status_msg="#6 127.1 Get:8 http://archive.ubuntu.com noble InRelease")
        self.assertEqual(vast_progress_key(first), vast_progress_key(later))

    def test_a_new_step_or_state_is_progress(self) -> None:
        base = VastInstance("1", "l", "loading", "42", status_msg="#6 62.25 Get:8 noble InRelease")
        self.assertNotEqual(
            vast_progress_key(base),
            vast_progress_key(replace(base, status_msg="#6 63.10 Get:9 noble/main Packages")),
        )
        self.assertNotEqual(vast_progress_key(base), vast_progress_key(replace(base, state="running")))

    def test_both_runtimes_get_the_same_allowance(self) -> None:
        # The old limits were 900s for llama.cpp and 1800s for vLLM, scaled on
        # image size. Nothing in the wait scales with the image.
        self.assertEqual(VAST_READY_DEADLINE_SECONDS, 1800)
        self.assertLess(VAST_PULL_STALL_SECONDS, VAST_READY_DEADLINE_SECONDS)
        self.assertLess(VAST_SSH_STALL_SECONDS, VAST_READY_DEADLINE_SECONDS)
        # A rental that reached SSH at 386.6s is the measurement both windows
        # have to clear: inside it, a healthy host gets handed back. The SSH
        # window used to sit at 360s, under that success rather than over it,
        # and destroyed a live rental after 378s of refusals.
        self.assertGreater(VAST_PULL_STALL_SECONDS, 386.6)
        self.assertGreater(VAST_SSH_STALL_SECONDS, 386.6)

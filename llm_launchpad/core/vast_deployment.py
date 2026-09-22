"""Recoverable Vast rentals, serving startup, and local endpoint management."""

from typing import Any
from collections.abc import Callable, Generator
import math
import os
import re
import secrets
import shutil
import socket
import threading
import time
import uuid

from ..protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from ..protocol.events import BaseEvent, LogEvent, OperationCompleteEvent, ResourceAllocatedEvent, StateChangeEvent
from ..protocol.models import DeploymentConfig, EndpointInfo, VastDeploymentRecord, VastInstance, VastOfferQuery, VastProviderOptions
from .deploy_log_summary import format_elapsed
from .diagnostics import log_exception
from .naming import infer_instance_from_app_name
from .operation_events import fail_operation
from .shutdown import is_shutting_down
from .vast_backend import VastApiError, VastBackend
from .vast_download import fetch_download_files, measure_download
from .vast_runtime import (
    GPU_INVENTORY_COMMAND,
    VAST_FAILURE_LOG_COMMAND,
    VAST_RUNTIME_DIR,
    VAST_STARTUP_PROBE_COMMAND,
    endpoint_healthy,
    parse_gpu_inventory,
    parse_startup_probe,
    startup_log_tail,
    startup_runtime_stopped,
    vast_runtime,
    vast_runtime_script,
    verify_endpoint_auth,
    verify_gpu_topology,
    verify_streaming,
)
from .vast_startup_history import record_vast_startup
from .vast_ssh import SSH_REFUSED, SSH_REJECTED, SSH_THROTTLED, VastSsh, VastSshError
from .vast_state import VastState

_CANCELLATIONS: dict[str, threading.Event] = {}
_CANCELLATIONS_LOCK = threading.Lock()

# How long a rental may take to answer SSH, and how long it may show no sign of
# progress before that is called stuck. The old limits were scaled on image
# size (900s llama.cpp, 1800s vLLM), but the pull is the smallest part of the
# wait: 2.59 GB and 8.67 GB compressed are seconds to a minute on these hosts,
# while the host still has to unpack them and run Vast's own provisioning,
# which installs the SSH server the pinned upstream images do not carry.
# Neither scales with the image, so both runtimes get the same allowance and a
# stalled host is caught by its silence instead of by a clock.
VAST_READY_DEADLINE_SECONDS = 1800
# After SSH: staging the weights and loading them onto the GPUs. vLLM also
# compiles and captures CUDA graphs on a cold run, which llama.cpp has no
# equivalent of, so it gets twice the allowance. Named rather than inlined so a
# regression that stops watching the runtime fails a test instead of spinning
# out the full half hour the way a paid rental did.
VAST_SERVE_DEADLINE_SECONDS = 1800
VAST_VLLM_SERVE_DEADLINE_SECONDS = 3600
# The two waits before SSH are watched differently because only one of them is
# measured. Once the host is running and the loop is knocking, the knock is
# itself the signal: a daemon that has refused this long has finished
# provisioning without starting sshd and will not change its mind.
#
# 360s was that judgement set to the only measurement available, and the
# measurement turned out to be the healthy case rather than the wedge: the
# fastest recorded success answered SSH 386.6s after creation, so the window
# closed on the population it was meant to keep. A live RTX PRO 6000 rental
# was then destroyed after 378s of refusals while it was still installing its
# SSH server, and the retry it forced cost another rental and another eight
# minutes. The wedge this catches billed the full 1800s deadline, so the two
# are separated anywhere between; 900s sits clear of every success on record
# and still hands a wedged host back at half the deadline. A refusal is also
# no longer the only signal here -- a denied key is now told apart below and
# fails in seconds, which is the case that never needed a clock at all.
VAST_SSH_STALL_SECONDS = 900
# How long a key denial is treated as provisioning rather than as an answer.
# Vast reports "running" about a second into the container and installs the
# attached key asynchronously, so the first knocks can land on an sshd whose
# authorized_keys is not written yet. Failing on the first denial destroyed a
# healthy RTX PRO 6000 WS rental 49.7s into its deploy, one second after the
# host came up. A denial that survives this window is still decisive, and
# still ends the rental far inside the stall window above.
VAST_KEY_GRACE_SECONDS = 150
# Before SSH exists the only signal is Vast's own status line, which is a much
# weaker one, so its window has to clear the normal spread rather than sit in
# the middle of it: a rental measured at 386.6s from creation to SSH would have
# been handed back by a 360s window while it was still working. The one wedge
# actually observed held a single BuildKit line for ~20 minutes, so the window
# only has to separate those two.
VAST_PULL_STALL_SECONDS = 900

# How often the model-startup wait reports in. Matches the rental loop's
# cadence so the summary view collapses both onto one row at the same rate.
VAST_STARTUP_HEARTBEAT_SECONDS = 30

# BuildKit prefixes every progress line with its step number and an elapsed
# counter, so the raw string keeps changing even while a step is wedged.
_BUILDKIT_STEP_PREFIX = re.compile(r"^#\d+\s+\d+(?:\.\d+)?\s+")


def vast_startup_detail(
    downloaded: int, previous: int, interval: float, last_log: str,
    *, total: int | None = None, active: bool = True,
) -> str:
    """Say what a starting runtime is doing, in the order the user can use.

    Measured bytes beat the server log while weights are arriving: llama.cpp
    draws its download as a carriage-returned bar that a redirected log never
    receives, so the log holds one stale line for the whole download while the
    partial file on disk grows. Once nothing is downloading the log is the
    only thing left that knows about loading and warmup.
    """
    if downloaded > 0 or total:
        gained = downloaded - previous
        rate = (
            f" at {gained / interval / 1e6:.0f} MB/s"
            if previous > 0 and gained > 0 and interval > 0
            else ""
        )
        if total is not None and total > 0:
            pct = min(99 if active else 100, downloaded * 100 // total)
            eta = ""
            if rate and downloaded < total:
                eta = f", ~{format_elapsed((total - downloaded) * interval / gained)} remaining"
            return f"downloading weights ({pct}%), {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB{rate}{eta}"
        return f"downloading weights, {downloaded / 1e9:.1f} GB fetched{rate}"
    return last_log or "no progress reported yet"


def vast_progress_key(instance: VastInstance) -> str:
    """Reduce a status line to what changes only when work actually advances."""
    return f"{instance.state}|{_BUILDKIT_STEP_PREFIX.sub('', instance.status_msg).strip()}"


def _create_was_refused(exc: BaseException) -> bool:
    """Whether Vast answered a rental request by refusing it.

    A 4xx other than 429 is the provider stating it did not act: the request
    was rejected, so no rental was made and the recovery record is only noise.
    A timeout, a 5xx or a transport error leave that genuinely unknown -- Vast
    may act moments after the reply was lost -- so those keep the record and
    the name blocked until the label has been reconciled by hand.
    """

    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429


class VastDeploymentBackend:
    """Keep remote identities durable and publish only tested SSH endpoints."""

    def __init__(self, api: VastBackend | None = None, state: VastState | None = None) -> None:
        self.api = api or VastBackend()
        self.state = state or VastState()

    def preflight(self) -> tuple[bool, str, str]:
        if os.name != "posix" or not shutil.which("ssh") or not shutil.which("ssh-keygen"):
            return False, "", "Vast deployment requires OpenSSH on a POSIX system."
        status = self.api.auth_status()
        return status.authenticated, status.account_id or "", status.error or ""

    def _account(self, record: VastDeploymentRecord) -> None:
        account_id = self._retry_while_throttled(self.api.account_id)
        if account_id != record.account_id:
            raise VastApiError("Use the Vast account that owns this recorded deployment.")

    def _instance(self, record: VastDeploymentRecord) -> VastInstance | None:
        if record.instance_id:
            instance = self.api.get_instance(record.instance_id)
        else:
            matches = self.api.find_instances(record.label)
            if len(matches) != 1:
                raise VastApiError("Rental creation remains uncertain. Inspect the recorded Vast label before retrying.")
            instance = matches[0]
            record.instance_id = instance.id
            self.state.save(record)
        if instance is not None and (instance.label != record.label or instance.machine_id != record.machine_id):
            raise VastApiError("Vast instance identity differs from the recorded rental; refusing to manage it.")
        return instance

    @staticmethod
    def _retry_while_throttled(call: Callable[[], Any], attempts: int = 6) -> Any:
        """Keep trying through rate limits when giving up would cost money.

        Teardown is the one place a 429 must never win: an unconfirmed destroy
        leaves a rental billing, which is worse than any delay here.
        """

        delay = 2.0
        for remaining in range(attempts - 1, -1, -1):
            try:
                return call()
            except VastApiError as exc:
                if exc.status_code != 429 or not remaining:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise VastApiError("Vast remained rate limited.")

    def _destroy(self, record: VastDeploymentRecord) -> None:
        instance = self._retry_while_throttled(lambda: self._instance(record))
        if instance is not None:
            record.state = "destroying"
            self.state.save(record)
            self._retry_while_throttled(lambda: self.api.destroy_instance(instance.id))
            for attempt in range(10):
                if self._retry_while_throttled(lambda: self.api.get_instance(instance.id)) is None:
                    break
                time.sleep(1 + attempt)
            else:
                raise VastApiError("Vast destruction is not yet confirmed. The recovery record has been retained.")
            try:
                VastSsh(self.state.directory(record.name)).disconnect(instance)
            except (ValueError, RuntimeError):
                pass  # Remote absence is confirmed even if the local tunnel has already died.
        self.state.remove(record.name)

    @staticmethod
    def _start_idle_watchdog(
        ssh: VastSsh, instance: Any, config: DeploymentConfig, endpoint_api_key: str,
    ) -> Generator[BaseEvent, None, None]:
        """Let the rental delete itself once idle, even with this computer asleep."""
        from .idle_watchdog import (
            WATCHDOG_SCRIPT_NAME,
            WatchdogSpec,
            describe_idle_shutdown,
            resolve_idle_shutdown,
            start_command,
            vast_destroy_command,
            watchdog_script,
        )

        idle = resolve_idle_shutdown(config)
        if idle <= 0:
            yield LogEvent(line="Idle shutdown is off: this rental bills until you stop it.")
            return
        script = watchdog_script(WatchdogSpec(
            metrics_url="http://127.0.0.1:8000/metrics",
            endpoint_api_key=endpoint_api_key,
            idle_seconds=idle,
            destroy_command=vast_destroy_command(),
            runtime_dir=VAST_RUNTIME_DIR,
        ))
        try:
            ssh.run(instance, f"cat > {VAST_RUNTIME_DIR}/{WATCHDOG_SCRIPT_NAME}", input_text=script)
            ssh.run(instance, start_command(VAST_RUNTIME_DIR))
        except (RuntimeError, ValueError) as exc:
            yield LogEvent(
                line=f"Warning: could not start idle shutdown ({exc}). This rental bills until you stop it.",
                is_milestone=True,
            )
            return
        yield LogEvent(line=f"Idle shutdown: {describe_idle_shutdown(idle)}.", is_milestone=True)

    def deploy(self, config: DeploymentConfig) -> Generator[BaseEvent, None, None]:
        """Rent once, bootstrap, verify streaming, and clean up failures."""
        name = config.app_name or ""
        options = config.provider_options
        if not name or not isinstance(options, VastProviderOptions):
            yield from fail_operation(OperationType.DEPLOY, "Vast requires a named deployment and a selected offer with a price limit.", recoverable=False)
            return
        cancellation = threading.Event()
        with _CANCELLATIONS_LOCK:
            already_running = name in _CANCELLATIONS
            if not already_running:
                _CANCELLATIONS[name] = cancellation
        if already_running:
            yield from fail_operation(
                OperationType.DEPLOY, "This Vast deployment is already in progress.",
                recoverable=False,
                data={"rollback": {"attempted": False, "confirmed": False, "detail": "already in progress"}},
            )
            return
        try:
            with self.state.locked(name):
                yield from self._deploy_locked(config, options, cancellation)
        finally:
            with _CANCELLATIONS_LOCK:
                _CANCELLATIONS.pop(name, None)

    def _deploy_locked(
        self, config: DeploymentConfig, options: VastProviderOptions, cancellation: threading.Event,
    ) -> Generator[BaseEvent, None, None]:
        record: VastDeploymentRecord | None = None
        completed = False
        try:
            runtime = vast_runtime(config)
            image = runtime.image
            if not math.isfinite(options.max_hourly_cost_usd) or options.max_hourly_cost_usd <= 0:
                raise ValueError("Vast requires a positive maximum hourly price.")
            ok, account, error = self.preflight()
            if not ok:
                raise ValueError(error)
            name = config.app_name or ""
            if self.state.load(name) is not None:
                raise ValueError("A Vast rental is already recorded for this name. Connect to or destroy it before deploying again.")
            ssh = VastSsh(self.state.directory(name))
            public_key = ssh.public_key()
            offer = self.api.get_offer(options.offer_id, VastOfferQuery(
                gpu_type=config.gpu_type, gpu_count=options.gpu_count, disk_gb=options.disk_gb,
            ))
            if options.machine_id and offer.machine_id != options.machine_id:
                raise ValueError("The selected Vast machine has changed. Refresh offers.")
            if offer.cuda_max_good is None or offer.cuda_max_good < runtime.min_cuda_version:
                raise ValueError(f"The pinned Vast runtime requires a host reporting CUDA {runtime.min_cuda_version} or newer.")
            if offer.compute_capability is None or offer.compute_capability < runtime.min_compute_capability:
                raise ValueError(
                    f"This GPU's compute capability {offer.compute_capability} is older than the "
                    f"{runtime.min_compute_capability} the pinned runtime was built for."
                )
            price = offer.costs.total_per_hour_usd
            if price is None or price > options.max_hourly_cost_usd + 1e-9:
                raise ValueError("The Vast hourly total is unknown or exceeds the approved price. Refresh offers.")
            config.gpu_count = offer.gpu_count
            assessment = config.placement_assessment
            required = (
                max(assessment.memory.per_device_required_gb, default=assessment.memory.total_gb)
                if assessment
                else (config.required_vram_gb or 0) * 1.05 / max(1, offer.gpu_count)
            )
            if required > offer.gpu_memory_gib:
                raise ValueError(
                    f"The selected Vast GPU no longer fits the model's memory requirements: "
                    f"{required:.1f} GiB required per GPU, {offer.gpu_memory_gib:.1f} GiB "
                    f"available on each of {offer.gpu_count} GPUs."
                )
            config.endpoint_api_key = config.endpoint_api_key or secrets.token_urlsafe(32)
            source = config.model_name if config.backend == BackendType.VLLM else config.repo_id
            config.served_model_name = config.served_model_name or (source or "model").rsplit("/", 1)[-1]
            script = vast_runtime_script(config)
            download_files = fetch_download_files(config)
            yield StateChangeEvent(current=DeploymentState.DEPLOYING, operation=OperationType.DEPLOY, detail=f"Renting Vast offer {offer.id} at ${price:.4f}/hr including disk")
            if cancellation.is_set() or is_shutting_down():
                raise RuntimeError("Vast deployment cancelled before rental.")
            record = VastDeploymentRecord(
                name=name, label="llp-vast-" + uuid.uuid4().hex, account_id=account,
                offer_id=offer.id, machine_id=offer.machine_id, repo_id=config.repo_id or "",
                quant=config.quant or "", served_model_name=config.served_model_name,
                endpoint_api_key=config.endpoint_api_key, max_context_tokens=config.max_context_tokens,
                backend=config.backend.value, model_name=config.model_name or "",
            )
            self.state.save(record)  # Write intent before the billable request.
            # No onstart hook: a supplied script displaces the image's own
            # /root/onstart.sh, and with it Vast's sshd provisioning, so the
            # rental then refuses every key. Vast's own key attachment below
            # installs it correctly.
            record.instance_id = self.api.create_instance(
                offer.id, image=image, disk_gb=options.disk_gb, label=record.label,
            )
            self.state.save(record)  # Persist identity before yielding control.
            yield ResourceAllocatedEvent(app_id=str(record.instance_id))
            self.api.attach_key(record.instance_id, public_key)
            # A rental cannot answer SSH until its image is pulled, and the
            # vLLM image is several times the size of the llama.cpp one. On a
            # modest link that is minutes of legitimate waiting, not a failure.
            rental_started = time.monotonic()
            ssh_ready_at: float | None = None
            started = rental_started
            deadline = started + VAST_READY_DEADLINE_SECONDS
            instance = None
            last_state = ""
            progress_key = ""
            progress_at = started
            reported_at = started
            ssh_reported = False
            ssh_first_try = time.monotonic()
            ssh_backoff = 5.0
            knocking_since: float | None = None
            while time.monotonic() < deadline:
                if cancellation.is_set() or is_shutting_down():
                    raise RuntimeError("Vast deployment cancelled.")
                try:
                    instance = self._instance(record)
                except VastApiError as exc:
                    # The rental is already paid for and this loop polls every
                    # few seconds. Throttling says nothing about the instance,
                    # so wait it out rather than destroying a live host.
                    if exc.status_code != 429:
                        raise
                    yield LogEvent(line="Vast is throttling status checks; still waiting for the rental.")
                    cancellation.wait(15)
                    continue
                if instance is None or instance.state in {"error", "exited", "offline", "destroyed"}:
                    raise RuntimeError("Vast instance stopped before the runtime became ready.")
                if instance.state != last_state:
                    # Pulling a large image can take many minutes. Say which
                    # step is slow instead of reporting a bare timeout.
                    last_state = instance.state
                    yield LogEvent(line=f"Vast instance state: {instance.state}")
                current_key = vast_progress_key(instance)
                # Once the host is up and addressable the loop is knocking on
                # SSH, and the knock is its own progress signal.
                knocking = bool(
                    instance.state == "running" and instance.ssh_host and instance.ssh_port
                )
                if current_key != progress_key:
                    progress_key, progress_at = current_key, time.monotonic()
                if knocking and knocking_since is None:
                    knocking_since = time.monotonic()
                # The two waits are timed from their own starts. Vast freezes
                # status_msg at "success, running <image>" the moment the pull
                # finishes and never writes another line, so timing the SSH
                # wait from the last status change was really timing it from
                # the end of the pull -- and charging sshd for however long the
                # pull happened to take. The knock is timed from the first
                # knock instead, which is what the recorded successes measure.
                if knocking and knocking_since is not None:
                    stalled_for, window = time.monotonic() - knocking_since, VAST_SSH_STALL_SECONDS
                    reason = f"refused SSH for {int(stalled_for)}s after its image came up"
                else:
                    stalled_for, window = time.monotonic() - progress_at, VAST_PULL_STALL_SECONDS
                    reason = f"silent for {int(stalled_for)}s"
                # A host that reports nothing at all is not stalled, it is
                # unobserved, and the two must not be read the same way: Vast
                # leaves status_msg empty for stretches of a healthy pull (a
                # certified rental logged a docker line at 32s and nothing at
                # 64s), so calling that a wedge hands back hosts that were
                # minutes from answering SSH. Where the status line does
                # report, a step it has not left inside the window is the
                # wedge that billed 20 minutes for one BuildKit line.
                if stalled_for > window and (knocking or instance.status_msg):
                    # Silence means the host stopped working, not that it is
                    # slow. Give the money back sooner; the loop's own deadline
                    # remains the backstop for everything this cannot see.
                    raise RuntimeError(
                        "Vast instance stopped making progress while provisioning "
                        f"(last state: {last_state}, {reason}, last reported: "
                        f"{instance.status_msg[:120] or 'nothing'})."
                    )
                # A minutes-long wait with one log line reads as a hang. Report
                # what the host is doing without echoing every BuildKit line,
                # in the heartbeat shape the summary view already knows how to
                # collapse onto a single row.
                if time.monotonic() - reported_at >= 30:
                    reported_at = time.monotonic()
                    detail = instance.status_msg[:120] or "no progress reported yet"
                    elapsed = format_elapsed(time.monotonic() - started)
                    phase = "waiting for SSH" if instance.state == "running" else "rental preparing"
                    yield LogEvent(line=f"Vast {phase}: {detail} (waiting {elapsed})")
                if knocking:
                    try:
                        ssh.run(instance, "true")
                        ssh_ready_at = time.monotonic()
                        yield LogEvent(line="Vast SSH ready")
                        break
                    except RuntimeError as exc:
                        # A refused port and a denied key arrive at the same
                        # place and mean opposite things. The port is refused
                        # by a host that has not installed its SSH server yet,
                        # which is most of this wait; the key is denied by one
                        # that has finished and decided, so waiting out the
                        # window buys nothing but rental minutes. The denial
                        # only counts once the host has had time to install
                        # the key Vast attached, which it does asynchronously.
                        rejected = (
                            isinstance(exc, VastSshError) and exc.reason == SSH_REJECTED
                        )
                        if isinstance(exc, VastSshError) and exc.reason == SSH_THROTTLED:
                            # The proxy is rate-limiting this loop's own
                            # knocking. Stop knocking for a while rather than
                            # reading it as an answer about the key.
                            ssh_backoff = 60.0
                        if rejected and knocking_since is not None and (
                            time.monotonic() - knocking_since > VAST_KEY_GRACE_SECONDS
                        ):
                            raise RuntimeError(
                                "Vast rental refused the deployment key rather than the "
                                "connection, so no further waiting would make it answer."
                            ) from None
                        if time.monotonic() - ssh_first_try > 120 and not ssh_reported:
                            ssh_reported = True
                            # The generic failure text tells the reader to
                            # check the saved host key, which is wrong advice
                            # for the case this almost always is: the host has
                            # not installed its SSH server yet. Say that where
                            # the refusal says it, and keep the prefix, which
                            # is what the summary view matches on.
                            detail = (
                                "still installing its SSH server"
                                if isinstance(exc, VastSshError) and exc.reason == SSH_REFUSED
                                else str(exc)
                            )
                            yield LogEvent(line=f"Vast host is running but not accepting SSH yet: {detail}")
                        # Back off rather than retrying every few seconds. A
                        # host that is up but refusing keys will not change its
                        # mind quickly, and hundreds of failed authentications
                        # look like an attack to Vast's SSH proxy.
                        cancellation.wait(ssh_backoff)
                        ssh_backoff = min(ssh_backoff * 2, 60)
                        continue
                cancellation.wait(3)
            else:
                raise RuntimeError(
                    f"Vast instance did not provide SSH before the deadline (last state: {last_state})."
                )
            assert instance is not None
            devices = parse_gpu_inventory(ssh.run(instance, GPU_INVENTORY_COMMAND))
            yield LogEvent(line="Rented GPUs: " + (", ".join(
                f"{device.index}:{device.name} {device.memory_free_gib:.1f} GiB free" for device in devices
            ) or "none reported"))
            verify_gpu_topology(devices, gpu_count=offer.gpu_count, per_device_required_gb=required)
            ssh.run(instance, f"umask 077; mkdir -p {VAST_RUNTIME_DIR}; cat > {VAST_RUNTIME_DIR}/runtime.sh", input_text=script)
            ssh.run(instance, f"nohup sh {VAST_RUNTIME_DIR}/runtime.sh > {VAST_RUNTIME_DIR}/server.log 2>&1 < /dev/null &")
            yield from self._start_idle_watchdog(ssh, instance, config, record.endpoint_api_key)
            # Save the selected port before opening a detached SSH process.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                record.local_port = listener.getsockname()[1]
            self.state.save(record)
            yield LogEvent(line="Vast opening secure endpoint")
            ssh.connect(instance, record.local_port)
            url = f"http://127.0.0.1:{record.local_port}"
            yield StateChangeEvent(current=DeploymentState.DEPLOYING, operation=OperationType.DEPLOY, detail="Waiting for the Vast model through the local SSH tunnel")
            deadline = time.monotonic() + (
                VAST_VLLM_SERVE_DEADLINE_SECONDS
                if config.backend == BackendType.VLLM
                else VAST_SERVE_DEADLINE_SECONDS
            )
            # The longest stretch of a Vast deploy sits here, and until now it
            # was the only one that said nothing: the rental loop above
            # heartbeats every 30s, then the weights download in silence for
            # as long as tens of GB take. The last line on screen stayed
            # "Rented GPUs" for ten minutes of healthy work, which reads as a
            # hang, so a paid deploy got killed and retried for being quiet.
            startup_started = time.monotonic()
            startup_reported = startup_started
            downloaded = 0
            download_complete = False
            runtime_seen_alive = False
            runtime_gone_streak = 0
            while not endpoint_healthy(url, record.endpoint_api_key):
                if cancellation.is_set() or is_shutting_down():
                    raise RuntimeError("Vast deployment cancelled.")
                if time.monotonic() >= deadline or not ssh.connected(instance):
                    raise RuntimeError("Vast model startup timed out or its SSH tunnel disconnected.")
                if time.monotonic() - startup_reported >= VAST_STARTUP_HEARTBEAT_SECONDS:
                    interval = time.monotonic() - startup_reported
                    startup_reported = time.monotonic()
                    previous, downloaded = downloaded, 0
                    runtime_gone = False
                    log_tail: tuple[str, ...] = ()
                    try:
                        output = ssh.run(instance, VAST_STARTUP_PROBE_COMMAND, multiplex=True)
                        downloaded, last_log = parse_startup_probe(output)
                        runtime_gone = startup_runtime_stopped(output)
                        log_tail = startup_log_tail(output)
                        progress = measure_download(output, download_files)
                        if progress is not None:
                            downloaded = progress.downloaded
                            if progress.active or (progress.total and not download_complete):
                                detail = vast_startup_detail(
                                    downloaded, previous, interval, last_log,
                                    total=progress.total, active=progress.active,
                                )
                            else:
                                detail = last_log or "loading model"
                            download_complete = not progress.active
                        else:
                            detail = vast_startup_detail(downloaded, previous, interval, last_log)
                    except (RuntimeError, ValueError):
                        # The probe is commentary on a rental that is already
                        # billing and already watched by the deadline above.
                        # A host too busy to answer it is still starting.
                        detail = "still starting"
                    if not runtime_gone:
                        runtime_seen_alive = True
                    # Raised outside the probe's own handler, which exists to
                    # forgive an unreadable answer rather than a conclusive one.
                    #
                    # Two independent guards, because this ends a rental the
                    # user is paying for and a wrong verdict destroys a healthy
                    # one. "No process and nothing logged" is not evidence of an
                    # exit -- it is also what the seconds before the script
                    # starts look like -- so the runtime must have left a trace
                    # first. And a single reading is not enough either: a
                    # process that has genuinely exited is still gone 30s later,
                    # whereas one probe that failed to see a live process is
                    # not. A dead runtime therefore costs one extra heartbeat
                    # instead of the full half-hour deadline.
                    if runtime_gone and (runtime_seen_alive or bool(last_log)):
                        runtime_gone_streak += 1
                    else:
                        runtime_gone_streak = 0
                    if runtime_gone_streak >= 2 and not endpoint_healthy(
                        url, record.endpoint_api_key
                    ):
                        # Nothing is left to wait for, and the reason is in
                        # the log. Waiting out the deadline instead bills the
                        # rental for half an hour and then reports it as a
                        # timeout. The heartbeat's own window is 25 lines,
                        # which a Python traceback overflows -- vLLM's "See
                        # root cause above" lands inside it and the cause does
                        # not -- so the post-mortem re-reads a longer tail.
                        for line in self._failure_log(record, instance, log_tail):
                            yield LogEvent(line=f"Vast runtime log: {line}")
                        raise RuntimeError(
                            "The Vast runtime exited before the endpoint came up: "
                            + (last_log or "server.log recorded no output.")
                        )
                    elapsed = format_elapsed(time.monotonic() - startup_started)
                    yield LogEvent(line=f"Vast model starting: {detail} (waiting {elapsed})")
                cancellation.wait(3)
            verify_endpoint_auth(url, record.served_model_name)
            verify_streaming(url, record.endpoint_api_key, record.served_model_name)
            if cancellation.is_set() or is_shutting_down():
                raise RuntimeError("Vast deployment cancelled.")
            record.state = "running"
            self.state.save(record)
            completed = True
            # Measured evidence for future host selection: advertised inet_down
            # bandwidth predicts this wait poorly, so store what this machine
            # actually took from rental to SSH and to a healthy endpoint.
            try:
                record_vast_startup(
                    machine_id=offer.machine_id or "",
                    offer_id=offer.id,
                    ssh_seconds=(
                        ssh_ready_at - rental_started
                        if ssh_ready_at is not None
                        else None
                    ),
                    healthy_seconds=time.monotonic() - rental_started,
                )
            except Exception:
                pass
            yield LogEvent(line="Vast streaming chat verified. The endpoint is local to this computer; stop destroys the rental and its disk.", operation=OperationType.DEPLOY)
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=self._endpoint(record, "running", url))
        except Exception as exc:
            detail = str(exc)
            rollback: dict[str, object] | None = None
            if record is None and (
                "already recorded" in detail or "already in progress" in detail
            ):
                # Pre-allocation refusal for a name that may still bill: block
                # the next rental via an explicit unconfirmed rollback rather
                # than mutating the caller's approved ladder.
                rollback = {"attempted": False, "confirmed": False, "detail": detail}
            if record is not None:
                try:
                    machine_id = getattr(record, "machine_id", "") or ""
                except Exception:
                    machine_id = ""
                try:
                    record_vast_startup(
                        machine_id=machine_id,
                        offer_id=getattr(record, "offer_id", "") or "",
                        ssh_seconds=None,
                        healthy_seconds=None,
                        failed=True,
                    )
                except Exception:
                    pass
                try:
                    if not record.instance_id and _create_was_refused(exc):
                        # Vast answered the rental request with a refusal, so
                        # nothing exists under this label and there is nothing
                        # to reconcile. Going through _destroy would scan by
                        # label, find zero instances, and report that as
                        # uncertainty -- which left the record in place and the
                        # name refused from both sides: deploy called it
                        # "already recorded", destroy called it "uncertain",
                        # and the product offered no way out.
                        self.state.remove(record.name)
                    else:
                        self._destroy(record)
                    record = None
                    rollback = {"attempted": True, "confirmed": True, "detail": ""}
                except Exception as cleanup:
                    # The rental may still be billing: report the unconfirmed
                    # rollback so the lifecycle blocks the next rental instead
                    # of mutating the approved ladder in place.
                    detail += f" Cleanup remains pending: {cleanup}"
                    rollback = {"attempted": True, "confirmed": False, "detail": str(cleanup)}
            yield from fail_operation(
                OperationType.DEPLOY, detail, recoverable=False,
                data={"rollback": rollback} if rollback is not None else None,
            )
        finally:
            if record is not None and not completed:
                # Also runs on GeneratorExit / KeyboardInterrupt, with no yields.
                try:
                    self._destroy(record)
                except Exception:
                    pass

    @staticmethod
    def _endpoint(record: VastDeploymentRecord, state: str, url: str | None) -> EndpointInfo:
        try:
            backend = BackendType(record.backend)
        except ValueError:
            backend = BackendType.LLAMACPP
        return EndpointInfo(
            name=record.name, app_id=record.instance_id or "", backend=backend,
            provider=ComputeProvider.VAST, state=state, web_url=url,
            instance_name=infer_instance_from_app_name(record.name, backend),
            repo_id=record.repo_id, quant=record.quant, served_model_name=record.served_model_name,
            endpoint_api_key=record.endpoint_api_key, max_context_tokens=record.max_context_tokens,
        )

    def list_deployments(self) -> list[EndpointInfo]:
        records = self.state.records()
        if not records:
            return []
        status = self.api.auth_status()
        if not status.authenticated:
            raise VastApiError(status.error or "Vast authentication failed.")
        rows = []
        for record in records:
            if record.account_id != status.account_id:
                continue
            # Listing must not rewrite a deployment while its worker owns it.
            instance = self.api.get_instance(record.instance_id) if record.instance_id else None
            state = instance.state if instance else ("destroyed" if record.instance_id else record.state)
            url = None
            if instance is not None and record.local_port and VastSsh(self.state.directory(record.name)).connected(instance):
                url = f"http://127.0.0.1:{record.local_port}"
            rows.append(self._endpoint(record, state, url))
        return rows

    def _record(self, name: str | None, instance_id: str | None) -> VastDeploymentRecord:
        records = self.state.records()
        matches = [row for row in records if (row.instance_id == instance_id if instance_id else row.name == name)]
        if len(matches) != 1:
            raise ValueError("No unique Launchpad-managed Vast rental matches this target.")
        return matches[0]

    def destroy(self, *, name: str | None = None, instance_id: str | None = None) -> None:
        if name:
            with _CANCELLATIONS_LOCK:
                cancellation = _CANCELLATIONS.get(name)
                if cancellation:
                    cancellation.set()
            with self.state.locked(name):
                if self.state.load(name) is None:
                    return
        record = self._record(name, instance_id)
        with self.state.locked(record.name):
            record = self.state.load(record.name)
            if record is None:
                return
            self._account(record)
            self._destroy(record)

    def connect(self, instance_id: str) -> EndpointInfo:
        record = self._record(None, instance_id)
        with self.state.locked(record.name):
            self._account(record)
            instance = self._instance(record)
            if instance is None or instance.state != "running":
                raise ValueError("The Vast rental is not running.")
            ssh = VastSsh(self.state.directory(record.name))
            if not record.local_port:
                raise ValueError("This rental never completed startup; destroy it and deploy again.")
            ssh.connect(instance, record.local_port)
            url = f"http://127.0.0.1:{record.local_port}"
            verify_endpoint_auth(url, record.served_model_name)
            verify_streaming(url, record.endpoint_api_key, record.served_model_name)
            return self._endpoint(record, "running", url)

    def _failure_log(
        self, record: Any, instance: Any, fallback: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Re-read enough of the runtime log to carry a whole traceback."""
        try:
            output = VastSsh(self.state.directory(record.name)).run(
                instance, VAST_FAILURE_LOG_COMMAND
            )
        except Exception:
            log_exception(f"Could not read the Vast runtime log for {record.name}")
            return fallback
        lines = tuple(
            cleaned
            for line in output.splitlines()
            if (cleaned := "".join(char for char in line if char.isprintable()).strip())
        )
        return lines or fallback

    def logs(self, instance_id: str) -> list[str]:
        record = self._record(None, instance_id)
        self._account(record)
        instance = self._instance(record)
        if instance is None:
            raise ValueError("The Vast instance no longer exists.")
        output = VastSsh(self.state.directory(record.name)).run(instance, f"tail -n 200 {VAST_RUNTIME_DIR}/server.log")
        output = output.replace(record.endpoint_api_key, "[redacted]")
        output = re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", output)
        return ["".join(char for char in line if char.isprintable()) for line in output.splitlines()]

"""Recoverable Vast rentals, serving startup, and local endpoint management."""

from collections.abc import Generator
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
from ..protocol.events import BaseEvent, LogEvent, OperationCompleteEvent, StateChangeEvent
from ..protocol.models import DeploymentConfig, EndpointInfo, VastDeploymentRecord, VastInstance, VastOfferQuery, VastProviderOptions
from .naming import infer_instance_from_app_name
from .operation_events import fail_operation
from .shutdown import is_shutting_down
from .vast_backend import VastApiError, VastBackend
from .vast_runtime import (
    GPU_INVENTORY_COMMAND,
    VAST_RUNTIME_DIR,
    endpoint_healthy,
    parse_gpu_inventory,
    vast_runtime,
    vast_runtime_script,
    verify_endpoint_auth,
    verify_gpu_topology,
    verify_streaming,
)
from .vast_ssh import VastSsh
from .vast_state import VastState

_CANCELLATIONS: dict[str, threading.Event] = {}
_CANCELLATIONS_LOCK = threading.Lock()


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
        status = self.api.auth_status()
        if not status.authenticated or status.account_id != record.account_id:
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

    def _destroy(self, record: VastDeploymentRecord) -> None:
        instance = self._instance(record)
        if instance is not None:
            record.state = "destroying"
            self.state.save(record)
            self.api.destroy_instance(instance.id)
            for _ in range(10):
                if self.api.get_instance(instance.id) is None:
                    break
                time.sleep(1)
            else:
                raise VastApiError("Vast destruction is not yet confirmed. The recovery record has been retained.")
            try:
                VastSsh(self.state.directory(record.name)).disconnect(instance)
            except (ValueError, RuntimeError):
                pass  # Remote absence is confirmed even if the local tunnel has already died.
        self.state.remove(record.name)

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
            config.fallback_configs = ()
            yield from fail_operation(OperationType.DEPLOY, "This Vast deployment is already in progress.", recoverable=False)
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
                config.fallback_configs = ()
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
                raise ValueError("The selected Vast GPU no longer fits the model's memory requirements.")
            config.endpoint_api_key = config.endpoint_api_key or secrets.token_urlsafe(32)
            source = config.model_name if config.backend == BackendType.VLLM else config.repo_id
            config.served_model_name = config.served_model_name or (source or "model").rsplit("/", 1)[-1]
            script = vast_runtime_script(config)
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
            record.instance_id = self.api.create_instance(offer.id, image=image, disk_gb=options.disk_gb, label=record.label)
            self.state.save(record)  # Persist identity before yielding control.
            self.api.attach_key(record.instance_id, public_key)
            deadline = time.monotonic() + 900
            instance = None
            last_state = ""
            ssh_reported = False
            ssh_first_try = time.monotonic()
            ssh_backoff = 5.0
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
                if instance.state == "running" and instance.ssh_host and instance.ssh_port:
                    try:
                        ssh.run(instance, "true")
                        break
                    except RuntimeError as exc:
                        if time.monotonic() - ssh_first_try > 120 and not ssh_reported:
                            ssh_reported = True
                            yield LogEvent(line=f"Vast host is running but not accepting SSH yet: {exc}")
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
                    f"Vast instance did not provide SSH within 15 minutes (last state: {last_state})."
                )
            assert instance is not None
            devices = parse_gpu_inventory(ssh.run(instance, GPU_INVENTORY_COMMAND))
            yield LogEvent(line="Rented GPUs: " + (", ".join(
                f"{device.index}:{device.name} {device.memory_free_gib:.1f} GiB free" for device in devices
            ) or "none reported"))
            verify_gpu_topology(devices, gpu_count=offer.gpu_count, per_device_required_gb=required)
            ssh.run(instance, f"umask 077; mkdir -p {VAST_RUNTIME_DIR}; cat > {VAST_RUNTIME_DIR}/runtime.sh", input_text=script)
            ssh.run(instance, f"nohup sh {VAST_RUNTIME_DIR}/runtime.sh > {VAST_RUNTIME_DIR}/server.log 2>&1 < /dev/null &")
            # Save the selected port before opening a detached SSH process.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                record.local_port = listener.getsockname()[1]
            self.state.save(record)
            ssh.connect(instance, record.local_port)
            url = f"http://127.0.0.1:{record.local_port}"
            yield StateChangeEvent(current=DeploymentState.DEPLOYING, operation=OperationType.DEPLOY, detail="Waiting for the Vast model through the local SSH tunnel")
            deadline = time.monotonic() + (3600 if config.backend == BackendType.VLLM else 1800)
            while not endpoint_healthy(url, record.endpoint_api_key):
                if cancellation.is_set() or is_shutting_down():
                    raise RuntimeError("Vast deployment cancelled.")
                if time.monotonic() >= deadline or not ssh.connected(instance):
                    raise RuntimeError("Vast model startup timed out or its SSH tunnel disconnected.")
                cancellation.wait(3)
            verify_endpoint_auth(url, record.served_model_name)
            verify_streaming(url, record.endpoint_api_key, record.served_model_name)
            if cancellation.is_set() or is_shutting_down():
                raise RuntimeError("Vast deployment cancelled.")
            record.state = "running"
            self.state.save(record)
            completed = True
            yield LogEvent(line="Vast streaming chat verified. The endpoint is local to this computer; stop destroys the rental and its disk.", operation=OperationType.DEPLOY)
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=self._endpoint(record, "running", url))
        except Exception as exc:
            detail = str(exc)
            if record is not None:
                try:
                    self._destroy(record)
                    record = None
                except Exception as cleanup:
                    # Prevent the TUI recovery ladder from renting another GPU
                    # while the first rental may still be billing.
                    config.fallback_configs = ()
                    detail += f" Cleanup remains pending: {cleanup}"
            yield from fail_operation(OperationType.DEPLOY, detail, recoverable=False)
        finally:
            if record is not None and not completed:
                # Also runs on GeneratorExit / KeyboardInterrupt, with no yields.
                try:
                    self._destroy(record)
                except Exception:
                    config.fallback_configs = ()

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

"""Prime Intellect provider deployment execution runner."""

from __future__ import annotations

from collections.abc import Callable, Generator
import secrets
import time
from typing import Any

from ..protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    EndpointAvailableEvent,
    LogEvent,
    OperationCompleteEvent,
    ResourceAllocatedEvent,
    StateChangeEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo, LaunchpadSettings
from .config import ConfigStore
from .deploy_log_summary import format_elapsed
from .diagnostics import log_exception
from .operation_events import fail_operation
from .prime_backend import (
    PrimeApiError,
    PrimeBackend,
    resolve_prime_launch_spec,
)
from .prime_disks import bind_prime_disk, resolve_prime_offer_and_disk
from .provider_options import prime_provider_options
from .shutdown import shutdown_event

EventStream = Generator[BaseEvent, None, None]

PRIME_PROGRESS_REPORT_SECONDS = 30.0


def prime_endpoint_info(
    config: DeploymentConfig,
    pod_id: str,
    endpoint: str,
) -> EndpointInfo:
    """Build the OpenCode-facing endpoint descriptor for a Prime pod."""
    return EndpointInfo(
        name=config.app_name or "",
        app_id=pod_id,
        state="running",
        backend=config.backend,
        instance_name=config.instance_name,
        web_url=endpoint,
        served_model_name=config.served_model_name,
        model_name=config.model_name,
        repo_id=config.repo_id,
        quant=config.quant,
        provider=ComputeProvider.PRIME,
        endpoint_api_key=config.endpoint_api_key,
        max_context_tokens=config.max_context_tokens,
        max_output_tokens=config.max_output_tokens,
        reasoning=config.reasoning,
    )


def prime_launch_requirement_error(config: DeploymentConfig) -> str | None:
    """Return why this config cannot launch on Prime, or ``None`` when it can."""
    if config.backend == BackendType.VLLM and not (config.model_name or "").strip():
        return "Prime vLLM deployment requires --model-name."
    if config.backend == BackendType.LLAMACPP and not (config.repo_id or "").strip():
        return "Prime llama.cpp deployment requires --repo-id."
    if config.backend == BackendType.LLAMACPP and (config.revision or "").strip():
        return "Prime llama.cpp currently supports only the default Hugging Face revision."
    return None


class PrimeDeploymentRunner:
    """Provision a Prime pod, publish its endpoint, and handle rollback on failure."""

    def __init__(
        self,
        prime_backend: PrimeBackend | None = None,
        config_store: ConfigStore | None = None,
        *,
        launch_spec_resolver: Callable[[DeploymentConfig], Any] = resolve_prime_launch_spec,
        offer_disk_resolver: Callable[..., Any] = resolve_prime_offer_and_disk,
        shutdown_event_fn: Callable[[], Any] = shutdown_event,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.prime_backend = prime_backend or PrimeBackend()
        self.config_store = config_store or ConfigStore()
        self._resolve_launch_spec = launch_spec_resolver
        self._resolve_offer_and_disk = offer_disk_resolver
        self._shutdown_event = shutdown_event_fn
        self._monotonic = monotonic_fn

    def await_pod_active(
        self, pod_id: str, created: dict[str, Any]
    ) -> Generator[BaseEvent, None, dict[str, Any]]:
        """Poll until the Prime pod reports ACTIVE with SSH reachable."""
        waiting_started = self._monotonic()
        deadline = waiting_started + 1800
        reported_at = waiting_started
        last_state = ""
        pod = created
        while self._monotonic() < deadline:
            pod = self.prime_backend.get_pod(pod_id)
            state = str(pod.get("status") or "unknown").upper()
            install = str(pod.get("installationStatus") or "").upper()
            progress = pod.get("installationProgress")
            state_detail = f"{state}/{install or '-'}"
            if progress is not None:
                state_detail += f" ({progress}%)"
            if state_detail != last_state:
                yield LogEvent(
                    line=f"Prime pod state: {state_detail}",
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )
                last_state = state_detail
                reported_at = self._monotonic()
            if state in {"ERROR", "TERMINATED"} or install == "FAILED":
                failure = str(pod.get("installationFailure") or state_detail)
                raise PrimeApiError(f"Prime pod provisioning failed: {failure}")
            ssh_ready = bool(pod.get("sshConnection"))
            if state == "ACTIVE" and install in {"", "FINISHED"} and ssh_ready:
                break
            if (
                last_state
                and self._monotonic() - reported_at >= PRIME_PROGRESS_REPORT_SECONDS
            ):
                reported_at = self._monotonic()
                elapsed = format_elapsed(self._monotonic() - waiting_started)
                yield LogEvent(
                    line=f"Prime pod state: {last_state} (waiting {elapsed})",
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )
            if self._shutdown_event().wait(timeout=5):
                raise PrimeApiError("Prime deployment cancelled during provisioning.")
        else:
            raise PrimeApiError("Timed out waiting for the Prime pod to become active.")

        return pod

    def establish_endpoint(
        self,
        config: DeploymentConfig,
        pod: dict[str, Any],
        pod_id: str,
        options: Any,
    ) -> Generator[BaseEvent, None, tuple[str, EndpointInfo | None]]:
        """Publish the pod's endpoint and wait for the runtime to serve it."""
        endpoint = ""
        announced: EndpointInfo | None = None
        tunnel_id = ""
        tunnel_ready = False
        if options.allow_insecure_http:
            endpoint = self.prime_backend.endpoint_url(
                pod,
                allow_insecure_http=True,
                allow_direct_ip=True,
            )
            yield LogEvent(
                line="Prime networking: direct HTTP fallback enabled",
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )
            announced = prime_endpoint_info(config, pod_id, endpoint)
            yield LogEvent(
                line=f"Prime endpoint URL ready: {endpoint}",
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )
            yield EndpointAvailableEvent(
                endpoint=announced,
                operation=OperationType.DEPLOY,
            )
            tunnel_ready = True
        else:
            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail="Creating a secure Prime Tunnel endpoint",
            )
            tunnel = self.prime_backend.create_tunnel(
                pod_id,
                name=config.app_name or f"llm-launchpad-{pod_id}",
            )
            self.prime_backend.start_tunnel(pod, tunnel)
            tunnel_id = tunnel.tunnel_id
            endpoint = tunnel.url
            tunnel_expiry = (
                f"; registration expires {tunnel.expires_at}"
                if tunnel.expires_at
                else ""
            )
            yield LogEvent(
                line=f"Prime networking: secure tunnel {tunnel.tunnel_id}{tunnel_expiry}",
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )

        runtime_ready = False
        last_runtime_detail = ""
        last_tunnel_detail = ""
        waiting_started = self._monotonic()
        reported_at = waiting_started
        wait_deadline = waiting_started + 1800
        while self._monotonic() < wait_deadline:
            if not runtime_ready:
                try:
                    ready, failed, detail = self.prime_backend.bootstrap_runtime_status(pod)
                except RuntimeError as exc:
                    ready, failed, detail = False, False, f"waiting for pod SSH ({exc})"
                if detail and detail != last_runtime_detail:
                    yield LogEvent(
                        line=f"Prime runtime: {detail}",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                    )
                    last_runtime_detail = detail
                    reported_at = self._monotonic()
                if failed:
                    raise PrimeApiError(f"Prime runtime bootstrap failed: {detail}")
                if ready:
                    runtime_ready = True
            if not tunnel_ready:
                try:
                    ready, failed, detail = self.prime_backend.tunnel_runtime_status(
                        pod, tunnel_id
                    )
                except RuntimeError as exc:
                    ready, failed, detail = False, False, f"waiting for tunnel SSH ({exc})"
                if detail and detail != last_tunnel_detail:
                    yield LogEvent(
                        line=f"Prime Tunnel: {detail}",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                    )
                    last_tunnel_detail = detail
                if failed:
                    raise PrimeApiError(f"Prime Tunnel failed: {detail}")
                if ready:
                    tunnel_ready = True
                    announced = prime_endpoint_info(config, pod_id, endpoint)
                    yield LogEvent(
                        line=f"Prime endpoint URL ready: {endpoint}",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                    )
                    yield EndpointAvailableEvent(
                        endpoint=announced,
                        operation=OperationType.DEPLOY,
                    )
            if runtime_ready and tunnel_ready:
                break
            if (
                not runtime_ready
                and last_runtime_detail
                and self._monotonic() - reported_at >= PRIME_PROGRESS_REPORT_SECONDS
            ):
                reported_at = self._monotonic()
                elapsed = format_elapsed(self._monotonic() - waiting_started)
                yield LogEvent(
                    line=f"Prime runtime: {last_runtime_detail} (waiting {elapsed})",
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )
            if self._shutdown_event().wait(timeout=5):
                raise PrimeApiError("Prime deployment cancelled during runtime startup.")
        else:
            if not runtime_ready:
                raise PrimeApiError("Timed out waiting for the Prime runtime to load.")
            raise PrimeApiError("Timed out waiting for Prime Tunnel to connect.")

        return endpoint, announced

    def await_public_endpoint(self, endpoint: str, api_key: str) -> None:
        """Block until the published URL answers, raising ``PrimeApiError`` on timeout."""
        public_deadline = self._monotonic() + 120
        public_error = "endpoint did not respond"
        while self._monotonic() < public_deadline:
            public_ready, public_error = self.prime_backend.public_endpoint_ready(
                endpoint, api_key
            )
            if public_ready:
                break
            if self._shutdown_event().wait(timeout=5):
                raise PrimeApiError("Prime deployment cancelled during endpoint startup.")
        else:
            raise PrimeApiError(
                "Prime runtime is healthy inside the pod, but its public endpoint "
                f"is unreachable: {public_error}"
            )

    def cleanup_failed_pod(
        self, pod_id: str, options: Any, message: str
    ) -> Generator[BaseEvent, None, tuple[str, dict[str, object] | None]]:
        """Drain the failed pod's logs and release it."""
        rollback: dict[str, object] | None = None
        if pod_id:
            try:
                for line in self.prime_backend.get_pod_logs(pod_id):
                    yield LogEvent(line=line, operation=OperationType.DEPLOY)
            except Exception:
                log_exception(f"Could not read logs from failed Prime pod {pod_id}")
            if options.keep_failed_resource:
                yield LogEvent(
                    line=f"Keeping failed Prime pod {pod_id}; billing may continue.",
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )
                rollback = {
                    "attempted": False,
                    "confirmed": False,
                    "detail": "kept for inspection",
                }
            else:
                try:
                    self.prime_backend.delete_pod(pod_id)
                    yield LogEvent(
                        line=f"Terminated failed Prime pod {pod_id}.",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                    )
                    rollback = {"attempted": True, "confirmed": True, "detail": ""}
                except Exception as cleanup_exc:
                    message = f"{message} Cleanup also failed: {cleanup_exc}"
                    rollback = {
                        "attempted": True,
                        "confirmed": False,
                        "detail": str(cleanup_exc),
                    }
        return message, rollback

    def start_idle_shutdown(
        self,
        config: DeploymentConfig,
        pod: dict[str, Any],
        pod_id: str,
        endpoint: str,
    ) -> EventStream:
        """Arrange for an idle pod to be deleted: on the pod if allowed, else locally."""
        from .idle_watchdog import describe_idle_shutdown, resolve_idle_shutdown
        from .local_watchdog import spawn_local_watchdog

        try:
            settings = self.config_store.load()
        except Exception:
            settings = LaunchpadSettings()
        idle = resolve_idle_shutdown(config, settings)
        if idle <= 0:
            yield LogEvent(
                line="Idle shutdown is off: this pod bills until you stop it.",
                operation=OperationType.DEPLOY,
            )
            return
        described = describe_idle_shutdown(idle)
        if settings.prime_self_terminate:
            try:
                self.prime_backend.start_idle_watchdog(
                    pod, pod_id, config.endpoint_api_key or "", idle
                )
            except Exception as exc:
                log_exception("Failed to start the on-pod idle watchdog")
                yield LogEvent(
                    line=(
                        f"Could not start idle shutdown on the pod ({exc}); "
                        "watching from this computer instead."
                    ),
                    operation=OperationType.DEPLOY,
                )
            else:
                yield LogEvent(
                    line=f"Idle shutdown: {described} (runs on the pod).",
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )
                return
        try:
            spawn_local_watchdog(
                app_name=config.app_name or "",
                backend=config.backend.value,
                pod_id=pod_id,
                endpoint_url=endpoint,
                endpoint_api_key=config.endpoint_api_key or "",
                idle_seconds=idle,
            )
        except Exception as exc:
            log_exception("Failed to start the local idle watchdog")
            yield LogEvent(
                line=(
                    f"Warning: could not start idle shutdown ({exc}). "
                    "This pod bills until you stop it."
                ),
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )
            return
        yield LogEvent(
            line=(
                f"Idle shutdown: {described}, while this computer is awake. "
                "To cover sleep too, allow Prime pods to stop themselves in Settings."
            ),
            operation=OperationType.DEPLOY,
            is_milestone=True,
        )

    def deploy(self, config: DeploymentConfig) -> EventStream:
        """Provision a Prime pod and resolve its public inference endpoint."""
        options = prime_provider_options(config)
        requirement_error = prime_launch_requirement_error(config)
        if requirement_error is None:
            from .deployment_preflight import preflight_config

            findings = preflight_config(config).findings
            blocking = [finding for finding in findings if finding.blocking]
            if blocking:
                requirement_error = blocking[0].message
        if requirement_error:
            yield from fail_operation(
                OperationType.DEPLOY, requirement_error, exit_code=2, recoverable=False
            )
            return

        config.endpoint_api_key = config.endpoint_api_key or secrets.token_urlsafe(32)
        pod_id = ""
        completed = False
        cleaned_up = False
        try:
            launch = self._resolve_launch_spec(config)
            yield StateChangeEvent(
                current=DeploymentState.QUEUED,
                operation=OperationType.DEPLOY,
                detail="Fetching Prime GPU availability",
            )
            offer, disk_id, disk_messages = self._resolve_offer_and_disk(
                self.prime_backend,
                config,
                required_image=launch.offer_image,
            )
            config.gpu_count = offer.gpu_count
            if offer.gpu_type:
                config.gpu_type = offer.gpu_type
            config.price_per_hour_usd = offer.price_per_hour
            bind_prime_disk(config, disk_id)
            options = prime_provider_options(config)
            price = (
                f"${offer.price_per_hour:.2f}/hr"
                if offer.price_per_hour is not None
                else "price unavailable"
            )
            yield LogEvent(
                line=(
                    f"Selected Prime offer {offer.id}: {offer.gpu_count}x {offer.gpu_type} "
                    f"via {offer.provider_name} ({offer.country or offer.region or '-'}, {price})"
                ),
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )
            for line in disk_messages:
                yield LogEvent(
                    line=line,
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )
            yield LogEvent(
                line=f"Prime runtime: portable bootstrap on {launch.offer_image}",
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )
            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail=f"Provisioning Prime pod {config.app_name or ''}".strip(),
            )
            created = self.prime_backend.create_pod(config, offer)
            pod_id = str(created.get("id") or "").strip()
            if not pod_id:
                raise PrimeApiError("Prime did not return a pod ID.")
            yield ResourceAllocatedEvent(app_id=pod_id)
            yield LogEvent(
                line=f"Prime pod created: {pod_id}",
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )

            pod = yield from self.await_pod_active(pod_id, created)

            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail="Installing the portable Prime inference runtime",
            )
            self.prime_backend.start_bootstrap_runtime(config, pod)

            endpoint, announced = yield from self.establish_endpoint(
                config, pod, pod_id, options
            )

            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail="Waiting for the public inference endpoint",
            )
            self.await_public_endpoint(endpoint, config.endpoint_api_key or "")
            yield from self.start_idle_shutdown(config, pod, pod_id, endpoint)
            info = announced or prime_endpoint_info(config, pod_id, endpoint)
            info.web_url = endpoint
            info.state = "running"
            completed = True
            yield LogEvent(
                line=f"Prime endpoint: {endpoint}",
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY,
                success=True,
                data=info,
            )
        except Exception as exc:
            cleaned_up = True
            message, rollback = yield from self.cleanup_failed_pod(
                pod_id, options, str(exc)
            )
            yield from fail_operation(
                OperationType.DEPLOY,
                message,
                data={"rollback": rollback} if rollback is not None else None,
            )
        finally:
            if pod_id and not completed and not cleaned_up and not options.keep_failed_resource:
                try:
                    self.prime_backend.delete_pod(pod_id)
                except Exception:
                    log_exception(f"Could not delete Prime pod {pod_id} during abrupt exit")

"""High-level workflow orchestrator that yields protocol events.

All long-running operations (deploy, warmup, logs, etc.) are implemented
as generators that yield protocol events. The TUI workers and headless
CLI both consume these generators.
"""

from __future__ import annotations

from .shutdown import is_shutting_down, shutdown_event

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import secrets
import time
from typing import Any, Generator, List, Optional

from ..protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    EndpointAvailableEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo
from ..protocol.models import BenchmarkConcurrencyResult, BenchmarkConfig
from ..protocol.models import StoredModelInfo, StorageSnapshot

from .benchmark import (
    aiperf_metrics_have_successful_requests,
    aiperf_cli_path,
    build_aiperf_command,
    build_run_summary,
    expected_export_paths,
    format_benchmark_summary,
    default_benchmark_run_dir,
    parse_aiperf_summary,
    write_run_summary,
)
from .backend import ModalBackend
from .config import ConfigStore
from .hf_models import fetch_gguf_quant_metadata
from .modal_auth import get_modal_auth_status
from .naming import legacy_app_name
from .naming import default_llamacpp_served_model_name
from .naming import random_function_slug
from .paths import MODAL_LLAMACPP_SCRIPT, MODAL_VLLM_SCRIPT
from .prime_backend import (
    default_prime_container_image,
    PrimeApiError,
    PrimeBackend,
    resolve_prime_launch_spec,
)
from .prime_disks import bind_prime_disk, resolve_prime_offer_and_disk
from .provider_options import prime_provider_options
from .runtime_support import (
    DEFAULT_LLAMACPP_IMAGE_REF,
    RuntimeCompatibility,
    RuntimeCompatibilityDecision,
    evaluate_llamacpp_architecture,
)
from .warmup import WarmupRunner
from .warmup import modal_gpu_scheduling_hint as _modal_gpu_scheduling_hint
from .warmup import probe_response_is_ready as _probe_response_is_ready
from .warmup import status_probe_url as _status_probe_url

# Type alias for event generators
EventStream = Generator[BaseEvent, None, None]

_GGUF_QUANT_RE = re.compile(r"(Q\d(?:_[A-Z0-9]+)+|IQ\d+_[A-Z0-9_]+)", flags=re.IGNORECASE)
_SIZE_TOKEN_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)?\s*$")
_LLAMACPP_MIN_PLAUSIBLE_GGUF_BYTES = 1024 * 1024
_LLAMACPP_STORAGE_JSON_BEGIN = "LLM_LAUNCHPAD_STORAGE_JSON_BEGIN"
_LLAMACPP_STORAGE_JSON_END = "LLM_LAUNCHPAD_STORAGE_JSON_END"


def _tag_operation(events: EventStream, operation: OperationType) -> EventStream:
    """Attach the owning workflow to events from a generic subprocess stream."""
    for event in events:
        if isinstance(event, (LogEvent, ErrorEvent, OperationCompleteEvent, StateChangeEvent)):
            yield replace(event, operation=operation)
        else:
            yield event


def _prime_endpoint_info(
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
    )


def _is_historical_modal_app_state(state: str) -> bool:
    """Return True for terminal/inactive Modal app rows that commonly accumulate."""
    normalized = (state or "").strip().lower()
    return normalized in {"stopped", "stopping", "terminated", "archived"}


def _dedupe_launchpad_apps(rows: list[EndpointInfo]) -> list[EndpointInfo]:
    """Collapse repeated historical rows while preserving concurrent live apps.

    Modal app list often includes many stopped revisions for one logical app.
    Those should not crowd the UI, but two active rows with the same app name
    can represent distinct live jobs and must both remain visible.
    """
    active_keys = {
        (
            row.backend.value if row.backend else "",
            (row.instance_name or "").strip(),
            (row.name or "").strip(),
        )
        for row in rows
        if not _is_historical_modal_app_state(row.state)
    }
    kept_historical_keys: set[tuple[str, str, str]] = set()
    deduped: list[EndpointInfo] = []

    for row in rows:
        key = (
            row.backend.value if row.backend else "",
            (row.instance_name or "").strip(),
            (row.name or "").strip(),
        )
        if not _is_historical_modal_app_state(row.state):
            deduped.append(row)
            continue
        if key in active_keys or key in kept_historical_keys:
            continue
        kept_historical_keys.add(key)
        deduped.append(row)

    return deduped


class Orchestrator:
    """Coordinate high-level deployment flows.

    Every public method is a generator yielding protocol events so callers
    can stream output in real-time (TUI workers, headless printers, etc.).
    """

    _LLAMACPP_STORAGE_VOLUME = "huggingface-cache"
    _VLLM_STORAGE_VOLUME = "huggingface-cache"

    def __init__(
        self,
        config_store: ConfigStore | None = None,
        backend: ModalBackend | None = None,
        prime_backend: PrimeBackend | None = None,
    ) -> None:
        self.config_store = config_store or ConfigStore()
        self.backend = backend or ModalBackend()
        self._prime_backend = prime_backend

    @property
    def prime_backend(self) -> PrimeBackend:
        """Lazily construct the Prime client so Modal-only use needs no Prime config."""
        if self._prime_backend is None:
            self._prime_backend = PrimeBackend()
        return self._prime_backend

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def preflight(
        self,
        provider: ComputeProvider = ComputeProvider.MODAL,
    ) -> tuple[bool, str, str]:
        """Check the selected provider's local authentication.

        Returns ``(ok, username, error_message)``.
        """
        if provider == ComputeProvider.PRIME:
            ok, error = self.prime_backend.preflight()
            return ok, self.prime_backend.config.user_id or "", error
        if not ModalBackend.is_cli_available():
            return False, "", "Modal CLI not found. Reinstall llm-launchpad, then run: modal setup"
        status = get_modal_auth_status()
        if not status.authenticated:
            return False, "", "Modal authentication missing. Run: modal setup"
        return True, status.profile or "", ""

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def deploy(self, config: DeploymentConfig) -> EventStream:
        """Run a full deploy workflow (optional preload + deploy + warmup)."""
        if config.backend == BackendType.LLAMACPP and not (config.served_model_name or "").strip():
            config.served_model_name = default_llamacpp_served_model_name(
                config.repo_id,
                config.quant,
            )
        if config.backend == BackendType.LLAMACPP:
            compatibility, lookup_error = self._llamacpp_compatibility(config)
            if compatibility.status == RuntimeCompatibility.UNSUPPORTED:
                message = (
                    f"Cannot deploy {config.repo_id or config.preset or 'this GGUF'}: "
                    f"{compatibility.message} Select a GGUF whose general.architecture "
                    "is present in llm_launchpad/data/llamacpp_runtime_support.json, "
                    "or explicitly configure a llama.cpp image that supports this architecture."
                )
                yield ErrorEvent(
                    message=message,
                    operation=OperationType.DEPLOY,
                    recoverable=False,
                )
                yield OperationCompleteEvent(
                    operation=OperationType.DEPLOY,
                    success=False,
                    exit_code=2,
                    detail=message,
                )
                return
            if compatibility.status == RuntimeCompatibility.SUPPORTED:
                yield LogEvent(
                    line=f"Compatibility preflight: {compatibility.message}",
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )
            else:
                detail = compatibility.message
                if lookup_error:
                    detail = f"{detail} Metadata lookup failed: {lookup_error}"
                yield LogEvent(
                    line=f"Compatibility preflight warning: {detail} Continuing unverified.",
                    operation=OperationType.DEPLOY,
                )
        if config.provider == ComputeProvider.PRIME:
            yield from self._deploy_prime(config)
            return
        if config.do_deploy and config.backend != BackendType.VLLM and not config.function_slug:
            config.function_slug = random_function_slug()
        settings = self.config_store.load()
        env = ModalBackend.build_full_env(settings, config)

        if config.backend == BackendType.VLLM:
            yield from self._deploy_vllm(config, env)
        else:
            yield from self._deploy_llamacpp(config, env)

    def _llamacpp_compatibility(
        self,
        config: DeploymentConfig,
    ) -> tuple[RuntimeCompatibilityDecision, str | None]:
        """Resolve GGUF metadata and check it before allocating remote compute."""

        architecture = (config.gguf_architecture or "").strip() or None
        lookup_error: str | None = None
        if architecture is None and (config.repo_id or "").strip():
            try:
                metadata = fetch_gguf_quant_metadata(
                    config.repo_id or "",
                    revision=config.revision,
                )
                architecture = metadata.architecture
                config.gguf_architecture = architecture
            except Exception as exc:
                lookup_error = str(exc)

        if config.provider == ComputeProvider.PRIME:
            image_ref = default_prime_container_image(BackendType.LLAMACPP)
        else:
            image_ref = (
                os.environ.get("LLAMA_CPP_IMAGE_REF", "").strip()
                or DEFAULT_LLAMACPP_IMAGE_REF
            )
        decision = evaluate_llamacpp_architecture(
            architecture,
            image_ref=image_ref,
        )
        config.llamacpp_runtime_id = decision.runtime_id
        return decision, lookup_error

    def _deploy_prime(self, config: DeploymentConfig) -> EventStream:
        """Provision a Prime pod and resolve its public inference endpoint."""
        options = prime_provider_options(config)
        if config.backend == BackendType.VLLM and not (config.model_name or "").strip():
            message = "Prime vLLM deployment requires --model-name."
            yield ErrorEvent(message=message, operation=OperationType.DEPLOY, recoverable=False)
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=False, exit_code=2, detail=message
            )
            return
        if config.backend == BackendType.LLAMACPP and not (config.repo_id or "").strip():
            message = "Prime llama.cpp deployment requires --repo-id."
            yield ErrorEvent(message=message, operation=OperationType.DEPLOY, recoverable=False)
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=False, exit_code=2, detail=message
            )
            return
        if config.backend == BackendType.LLAMACPP and (config.revision or "").strip():
            message = "Prime llama.cpp currently supports only the default Hugging Face revision."
            yield ErrorEvent(message=message, operation=OperationType.DEPLOY, recoverable=False)
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY, success=False, exit_code=2, detail=message
            )
            return

        config.endpoint_api_key = config.endpoint_api_key or secrets.token_urlsafe(32)
        pod_id = ""
        try:
            launch = resolve_prime_launch_spec(config)
            yield StateChangeEvent(
                current=DeploymentState.QUEUED,
                operation=OperationType.DEPLOY,
                detail="Fetching Prime GPU availability",
            )
            offer, disk_id, disk_messages = resolve_prime_offer_and_disk(
                self.prime_backend,
                config,
                required_image=launch.offer_image,
            )
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
            yield LogEvent(
                line=f"Prime pod created: {pod_id}",
                operation=OperationType.DEPLOY,
                is_milestone=True,
            )

            deadline = time.monotonic() + 1800
            last_state = ""
            pod = created
            while time.monotonic() < deadline:
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
                if state in {"ERROR", "TERMINATED"} or install == "FAILED":
                    failure = str(pod.get("installationFailure") or state_detail)
                    raise PrimeApiError(f"Prime pod provisioning failed: {failure}")
                ssh_ready = bool(pod.get("sshConnection"))
                if (
                    state == "ACTIVE"
                    and install in {"", "FINISHED"}
                    and ssh_ready
                ):
                    break
                if shutdown_event().wait(timeout=5):
                    raise PrimeApiError("Prime deployment cancelled during provisioning.")
            else:
                raise PrimeApiError("Timed out waiting for the Prime pod to become active.")

            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail="Installing the portable Prime inference runtime",
            )
            self.prime_backend.start_bootstrap_runtime(config, pod)

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
                announced = _prime_endpoint_info(config, pod_id, endpoint)
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
                    line=(
                        f"Prime networking: secure tunnel {tunnel.tunnel_id}"
                        f"{tunnel_expiry}"
                    ),
                    operation=OperationType.DEPLOY,
                    is_milestone=True,
                )

            runtime_ready = False
            last_runtime_detail = ""
            last_tunnel_detail = ""
            wait_deadline = time.monotonic() + 1800
            while time.monotonic() < wait_deadline:
                if not runtime_ready:
                    ready, failed, detail = self.prime_backend.bootstrap_runtime_status(pod)
                    if detail and detail != last_runtime_detail:
                        yield LogEvent(
                            line=f"Prime runtime: {detail}",
                            operation=OperationType.DEPLOY,
                            is_milestone=True,
                        )
                        last_runtime_detail = detail
                    if failed:
                        raise PrimeApiError(f"Prime runtime bootstrap failed: {detail}")
                    if ready:
                        runtime_ready = True
                if not tunnel_ready:
                    ready, failed, detail = self.prime_backend.tunnel_runtime_status(
                        pod,
                        tunnel_id,
                    )
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
                        announced = _prime_endpoint_info(config, pod_id, endpoint)
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
                if shutdown_event().wait(timeout=5):
                    raise PrimeApiError("Prime deployment cancelled during runtime startup.")
            else:
                if not runtime_ready:
                    raise PrimeApiError("Timed out waiting for the Prime runtime to load.")
                raise PrimeApiError("Timed out waiting for Prime Tunnel to connect.")

            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail="Waiting for the public inference endpoint",
            )
            public_deadline = time.monotonic() + 120
            public_error = "endpoint did not respond"
            while time.monotonic() < public_deadline:
                public_ready, public_error = self.prime_backend.public_endpoint_ready(
                    endpoint,
                    config.endpoint_api_key,
                )
                if public_ready:
                    break
                if shutdown_event().wait(timeout=5):
                    raise PrimeApiError("Prime deployment cancelled during endpoint startup.")
            else:
                raise PrimeApiError(
                    "Prime runtime is healthy inside the pod, but its public endpoint "
                    f"is unreachable: {public_error}"
                )
            info = announced or _prime_endpoint_info(config, pod_id, endpoint)
            info.web_url = endpoint
            info.state = "running"
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
            message = str(exc)
            if pod_id:
                try:
                    for line in self.prime_backend.get_pod_logs(pod_id):
                        yield LogEvent(line=line, operation=OperationType.DEPLOY)
                except Exception:
                    pass
                if options.keep_failed_resource:
                    yield LogEvent(
                        line=f"Keeping failed Prime pod {pod_id}; billing may continue.",
                        operation=OperationType.DEPLOY,
                        is_milestone=True,
                    )
                else:
                    try:
                        self.prime_backend.delete_pod(pod_id)
                        yield LogEvent(
                            line=f"Terminated failed Prime pod {pod_id}.",
                            operation=OperationType.DEPLOY,
                            is_milestone=True,
                        )
                    except Exception as cleanup_exc:
                        message = f"{message} Cleanup also failed: {cleanup_exc}"
            yield ErrorEvent(message=message, operation=OperationType.DEPLOY)
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY,
                success=False,
                exit_code=1,
                detail=message,
            )

    def _deploy_vllm(
        self, config: DeploymentConfig, env: dict[str, str]
    ) -> EventStream:
        if config.do_deploy:
            cmd = ModalBackend.build_deploy_command(config.backend, app_name=config.app_name)
            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail=" ".join(cmd),
            )
            yield LogEvent(line=f"Running: {' '.join(cmd)}")
            if env:
                yield LogEvent(line=f"  env: {', '.join(f'{k}={v}' for k, v in env.items())}")
            yield from _tag_operation(
                ModalBackend.run_streaming(cmd, env=env),
                OperationType.DEPLOY,
            )
        elif config.run_smoke:
            cmd = ["modal", "run", BackendType.VLLM.script]
            yield StateChangeEvent(
                current=DeploymentState.RUNNING,
                operation=OperationType.SMOKE_TEST,
                detail=" ".join(cmd),
            )
            yield LogEvent(line=f"Running: {' '.join(cmd)}")
            yield from _tag_operation(
                ModalBackend.run_streaming(cmd, env=env),
                OperationType.SMOKE_TEST,
            )

    def _deploy_llamacpp(
        self, config: DeploymentConfig, env: dict[str, str]
    ) -> EventStream:
        def _run_step(
            cmd: list[str],
            *,
            state: DeploymentState,
            emit_env: bool,
        ) -> Generator[BaseEvent, None, tuple[bool, int, str]]:
            yield StateChangeEvent(
                current=state,
                operation=OperationType.DEPLOY,
                detail=" ".join(cmd),
            )
            yield LogEvent(line=f"Running: {' '.join(cmd)}")
            if emit_env and env:
                yield LogEvent(line=f"  env: {', '.join(f'{k}={v}' for k, v in env.items())}")

            saw_completion = False
            success = True
            exit_code = 0
            detail = ""
            for event in ModalBackend.run_streaming(cmd, env=env):
                if isinstance(event, OperationCompleteEvent):
                    saw_completion = True
                    success = event.success
                    exit_code = event.exit_code
                    detail = event.detail
                    continue
                if isinstance(event, ErrorEvent):
                    yield ErrorEvent(
                        message=event.message,
                        operation=OperationType.DEPLOY,
                        exit_code=event.exit_code,
                        recoverable=event.recoverable,
                    )
                    return False, event.exit_code or 1, event.message
                if isinstance(event, LogEvent):
                    yield LogEvent(
                        line=event.line,
                        stream=event.stream,
                        operation=OperationType.DEPLOY,
                    )
                    continue
                yield event

            if not saw_completion:
                return False, 1, "Command finished without completion event."
            return success, exit_code, detail

        if not config.do_deploy:
            cmd = ModalBackend.build_run_command(config)
            ok, exit_code, detail = yield from _run_step(
                cmd,
                state=DeploymentState.RUNNING,
                emit_env=True,
            )
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY,
                success=ok,
                exit_code=exit_code,
                detail=detail,
            )
            return

        # Split llama.cpp flow into:
        # 1) configure + optional preload (`modal run ...::main` without nested deploy)
        # 2) explicit `modal deploy`
        prep_config = replace(config, do_deploy=False)
        prep_cmd = ModalBackend.build_run_command(prep_config)
        prep_ok, prep_exit_code, prep_detail = yield from _run_step(
            prep_cmd,
            state=DeploymentState.DEPLOYING,
            emit_env=True,
        )
        if not prep_ok:
            yield OperationCompleteEvent(
                operation=OperationType.DEPLOY,
                success=False,
                exit_code=prep_exit_code,
                detail=prep_detail or "llama.cpp preparation step failed",
            )
            return

        deploy_cmd = ModalBackend.build_deploy_command(config.backend, app_name=config.app_name)
        deploy_ok, deploy_exit_code, deploy_detail = yield from _run_step(
            deploy_cmd,
            state=DeploymentState.DEPLOYING,
            emit_env=False,
        )
        yield OperationCompleteEvent(
            operation=OperationType.DEPLOY,
            success=deploy_ok,
            exit_code=deploy_exit_code,
            detail=deploy_detail,
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(
        self,
        backend: BackendType,
        server_url: str,
        timeout: int = 1800,
        tail_logs: bool = True,
        app_name: Optional[str] = None,
        served_model_name: Optional[str] = None,
        provider: ComputeProvider = ComputeProvider.MODAL,
        api_key: Optional[str] = None,
        pod_id: Optional[str] = None,
    ) -> EventStream:
        """Probe endpoint readiness and optionally tail logs."""
        yield from WarmupRunner().run(
            backend=backend,
            server_url=server_url,
            timeout=timeout,
            tail_logs=tail_logs,
            app_name=app_name,
            served_model_name=served_model_name,
            provider=provider,
            api_key=api_key,
            pod_id=pod_id,
            prime_backend=self.prime_backend if provider == ComputeProvider.PRIME else None,
        )

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def tail_logs(
        self,
        backend: BackendType,
        follow: bool = True,
        app_name: Optional[str] = None,
        app_id: Optional[str] = None,
        provider: ComputeProvider = ComputeProvider.MODAL,
    ) -> EventStream:
        """Tail logs for the selected provider and backend."""
        if provider == ComputeProvider.PRIME:
            pod_id = (app_id or "").strip()
            if not pod_id:
                message = "Prime log retrieval requires a pod ID."
                yield ErrorEvent(message=message, operation=OperationType.LOGS)
                yield OperationCompleteEvent(
                    operation=OperationType.LOGS, success=False, exit_code=2, detail=message
                )
                return
            seen: list[str] = []
            yield StateChangeEvent(
                current=DeploymentState.RUNNING,
                operation=OperationType.LOGS,
                detail=f"Reading Prime pod logs for {pod_id}",
            )
            while True:
                try:
                    lines = self.prime_backend.get_pod_logs(pod_id, tail=500)
                except Exception as exc:
                    yield ErrorEvent(message=str(exc), operation=OperationType.LOGS)
                    yield OperationCompleteEvent(
                        operation=OperationType.LOGS, success=False, exit_code=1, detail=str(exc)
                    )
                    return
                common = 0
                limit = min(len(seen), len(lines))
                for count in range(limit, -1, -1):
                    if count == 0 or seen[-count:] == lines[:count]:
                        common = count
                        break
                for line in lines[common:]:
                    yield LogEvent(line=line, operation=OperationType.LOGS)
                seen = lines
                if not follow:
                    yield OperationCompleteEvent(operation=OperationType.LOGS, success=True)
                    return
                if shutdown_event().wait(timeout=2):
                    return

        target_app_name = (app_id or app_name or legacy_app_name(backend)).strip()
        cmd: List[str] = ["modal", "app", "logs"]
        if follow:
            cmd.extend(ModalBackend.logs_follow_args())
        cmd.append(target_app_name)

        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.LOGS,
            detail=" ".join(cmd),
        )
        yield from _tag_operation(
            ModalBackend.run_streaming(cmd),
            OperationType.LOGS,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def check_status(
        self,
        backend: BackendType,
        server_url: str,
        timeout: int = 60,
        served_model_name: Optional[str] = None,
        provider: ComputeProvider = ComputeProvider.MODAL,
        api_key: Optional[str] = None,
        pod_id: Optional[str] = None,
    ) -> EventStream:
        """Probe endpoint health with backoff."""
        is_vllm = backend == BackendType.VLLM
        probe_url = _status_probe_url(backend, server_url)

        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STATUS,
            detail=f"Checking {probe_url}",
        )
        yield LogEvent(line=f"Checking endpoint status at: {probe_url}")

        try:
            import requests
        except ImportError:
            yield ErrorEvent(
                message="'requests' required", operation=OperationType.STATUS, recoverable=False
            )
            yield OperationCompleteEvent(
                operation=OperationType.STATUS,
                success=False,
                exit_code=1,
            )
            return

        import json as _json

        payload = {
            "model": (served_model_name or "").strip() or "default",
            "prompt": "ping",
            "max_tokens": 1,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        start = time.time()
        backoff = 2.0
        max_backoff = 15.0
        last_err: Optional[str] = None
        last_scheduling_hint: Optional[str] = None

        while True:
            if is_shutting_down():
                return

            if time.time() - start > timeout:
                yield ErrorEvent(
                    message=f"Unhealthy (timed out). Last: {last_err}",
                    operation=OperationType.STATUS,
                )
                yield OperationCompleteEvent(
                    operation=OperationType.STATUS, success=False, exit_code=1
                )
                return

            try:
                if provider == ComputeProvider.PRIME and pod_id:
                    pod = self.prime_backend.get_pod(pod_id)
                    pod_state = str(pod.get("status") or "unknown").upper()
                    if pod_state in {"ERROR", "TERMINATED"}:
                        last_err = f"Prime pod state: {pod_state}"
                        raise RuntimeError(last_err)
                if is_vllm:
                    resp = requests.get(probe_url, headers=headers, timeout=10)
                else:
                    resp = requests.post(
                        probe_url, headers=headers, data=_json.dumps(payload), timeout=10
                    )
                body = resp.text or ""
                ready_ok, readiness_reason = _probe_response_is_ready(
                    backend, resp.status_code, body
                )
                if ready_ok:
                    curl_cmd = ModalBackend.test_curl_command(
                        backend,
                        server_url,
                        served_model_name=served_model_name,
                        api_key=api_key,
                    )
                    if api_key:
                        curl_cmd = curl_cmd.replace(api_key, "$LLM_LAUNCHPAD_API_KEY")
                    yield LogEvent(
                        line=f"Status: healthy (backend={backend.value}, url={server_url})"
                    )
                    yield LogEvent(line=f"Test command:\n{curl_cmd}")
                    yield StateChangeEvent(
                        current=DeploymentState.HEALTHY, operation=OperationType.STATUS
                    )
                    yield OperationCompleteEvent(
                        operation=OperationType.STATUS, success=True
                    )
                    return
                if 200 <= resp.status_code < 300 and readiness_reason:
                    last_err = f"HTTP {resp.status_code} (not ready): {readiness_reason}; body={body[:200]}"
                else:
                    last_err = f"HTTP {resp.status_code}: {body[:200]}"
                scheduling_hint = _modal_gpu_scheduling_hint(body)
                if scheduling_hint:
                    if scheduling_hint != last_scheduling_hint:
                        yield StateChangeEvent(
                            current=DeploymentState.QUEUED,
                            operation=OperationType.STATUS,
                            detail=scheduling_hint,
                        )
                        yield LogEvent(line=scheduling_hint, operation=OperationType.STATUS)
                        last_scheduling_hint = scheduling_hint
                elif last_scheduling_hint is not None:
                    yield StateChangeEvent(
                        current=DeploymentState.RUNNING,
                        operation=OperationType.STATUS,
                        detail="GPU allocated; continuing health check",
                    )
                    yield LogEvent(
                        line="GPU allocated; continuing health check.",
                        operation=OperationType.STATUS,
                    )
                    last_scheduling_hint = None
            except Exception as exc:
                last_err = str(exc)

            # Interruptible sleep: wakes immediately on shutdown.
            if shutdown_event().wait(timeout=backoff):
                return
            backoff = min(max_backoff, backoff * 1.5)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, config: BenchmarkConfig) -> EventStream:
        """Run an AIPerf benchmark sweep against a deployed endpoint."""
        executable = aiperf_cli_path()
        if not executable:
            message = (
                "AIPerf CLI not found. Install benchmark support with: "
                'uv tool install "llm-launchpad[benchmark]". '
                "For a local checkout, run: uv sync --extra benchmark "
                "or launch with: uv run --extra benchmark llm-launchpad"
            )
            yield ErrorEvent(
                message=message,
                operation=OperationType.BENCHMARK,
                recoverable=False,
            )
            yield OperationCompleteEvent(
                operation=OperationType.BENCHMARK,
                success=False,
                exit_code=1,
                detail=message,
            )
            return

        app_name = (config.app_name or config.instance_name or "endpoint").strip()
        run_dir = Path(config.output_dir).expanduser() if config.output_dir else default_benchmark_run_dir(app_name)
        run_dir.mkdir(parents=True, exist_ok=True)

        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.BENCHMARK,
            detail=f"Benchmarking {config.server_url}",
        )
        yield LogEvent(
            line=(
                f"Benchmark target: app={config.app_name or '-'} "
                f"backend={config.backend.value} url={config.server_url} "
                f"model={config.model_name}"
            ),
            operation=OperationType.BENCHMARK,
        )
        yield LogEvent(line=f"Artifact root: {run_dir}", operation=OperationType.BENCHMARK)

        results: list[BenchmarkConcurrencyResult] = []
        for concurrency in config.concurrency:
            if is_shutting_down():
                return
            artifact_dir = run_dir / f"c{concurrency}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            try:
                cmd = build_aiperf_command(
                    config,
                    concurrency=concurrency,
                    artifact_dir=artifact_dir,
                    executable=executable,
                )
            except Exception as exc:
                detail = str(exc)
                yield ErrorEvent(
                    message=detail,
                    operation=OperationType.BENCHMARK,
                    exit_code=1,
                    recoverable=False,
                )
                results.append(
                    BenchmarkConcurrencyResult(
                        concurrency=concurrency,
                        command=[],
                        artifact_dir=str(artifact_dir),
                        exit_code=1,
                        success=False,
                        detail=detail,
                    )
                )
                continue

            yield StateChangeEvent(
                current=DeploymentState.RUNNING,
                operation=OperationType.BENCHMARK,
                detail=f"concurrency={concurrency}",
            )
            yield LogEvent(
                line=f"Running AIPerf concurrency={concurrency}: {' '.join(cmd)}",
                operation=OperationType.BENCHMARK,
            )

            exit_code = 0
            success = True
            detail = ""
            saw_completion = False
            benchmark_env = {"OPENAI_API_KEY": config.api_key} if config.api_key else None
            benchmark_events = (
                ModalBackend.run_streaming(cmd, env=benchmark_env)
                if benchmark_env
                else ModalBackend.run_streaming(cmd)
            )
            for event in benchmark_events:
                if isinstance(event, LogEvent):
                    yield LogEvent(
                        line=event.line,
                        stream=event.stream,
                        operation=OperationType.BENCHMARK,
                    )
                elif isinstance(event, ErrorEvent):
                    success = False
                    exit_code = event.exit_code or 1
                    detail = event.message
                    yield ErrorEvent(
                        message=event.message,
                        operation=OperationType.BENCHMARK,
                        exit_code=event.exit_code,
                        recoverable=event.recoverable,
                    )
                    break
                elif isinstance(event, OperationCompleteEvent):
                    saw_completion = True
                    success = event.success
                    exit_code = event.exit_code
                    detail = event.detail

            if not saw_completion and success:
                success = False
                exit_code = 1
                detail = "AIPerf command finished without a completion event."

            json_path, csv_path = expected_export_paths(artifact_dir)
            metrics: dict[str, Optional[float]] = {}
            if success:
                try:
                    metrics, parsed_path = parse_aiperf_summary(json_path, csv_path)
                    yield LogEvent(
                        line=f"Parsed AIPerf export: {parsed_path}",
                        operation=OperationType.BENCHMARK,
                    )
                    if not aiperf_metrics_have_successful_requests(metrics):
                        success = False
                        exit_code = 1
                        detail = "AIPerf export did not contain completed request metrics."
                        yield ErrorEvent(
                            message=detail,
                            operation=OperationType.BENCHMARK,
                            exit_code=exit_code,
                            recoverable=True,
                        )
                except Exception as exc:
                    success = False
                    exit_code = 1
                    detail = f"AIPerf finished but export parsing failed: {exc}"
                    yield ErrorEvent(
                        message=detail,
                        operation=OperationType.BENCHMARK,
                        exit_code=exit_code,
                        recoverable=True,
                    )

            results.append(
                BenchmarkConcurrencyResult(
                    concurrency=concurrency,
                    command=cmd,
                    artifact_dir=str(artifact_dir),
                    exit_code=exit_code,
                    success=success,
                    detail=detail,
                    json_export_path=str(json_path) if json_path.exists() else None,
                    csv_export_path=str(csv_path) if csv_path.exists() else None,
                    metrics=metrics,
                )
            )

        summary = build_run_summary(config, run_dir, results)
        summary_path = write_run_summary(summary)
        for line in format_benchmark_summary(summary):
            yield LogEvent(line=line, operation=OperationType.BENCHMARK)
        yield OperationCompleteEvent(
            operation=OperationType.BENCHMARK,
            success=summary.success,
            exit_code=0 if summary.success else 1,
            detail="" if summary.success else f"One or more benchmark runs failed. See {summary_path}",
            data=summary,
        )

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_deployments(
        self,
        provider: ComputeProvider = ComputeProvider.MODAL,
    ) -> EventStream:
        """List launchpad deployments for one compute provider."""
        yield StateChangeEvent(
            current=DeploymentState.RUNNING, operation=OperationType.LIST
        )

        if provider == ComputeProvider.PRIME:
            try:
                rows = self.prime_backend.list_deployments()
            except Exception as exc:
                message = f"Failed to query Prime pods: {exc}"
                yield ErrorEvent(message=message, operation=OperationType.LIST)
                yield OperationCompleteEvent(
                    operation=OperationType.LIST,
                    success=False,
                    exit_code=1,
                    detail=message,
                )
                return
            if not rows:
                yield LogEvent(line="No Prime launchpad deployments found.")
            else:
                yield LogEvent(line="Prime launchpad deployments:")
                for info in rows:
                    yield LogEvent(
                        line=(
                            f"  provider=prime backend="
                            f"{info.backend.value if info.backend else 'unknown'} "
                            f"instance={info.instance_name or '-'} "
                            f"pod={info.name} state={info.state} ({info.app_id})"
                        )
                    )
            yield OperationCompleteEvent(operation=OperationType.LIST, success=True, data=rows)
            return

        apps_result = ModalBackend.list_apps_result()
        if apps_result.success and apps_result.rows is not None:
            apps = apps_result.rows
            launchpad = [a for a in apps if a.backend is not None]
            visible_launchpad = _dedupe_launchpad_apps(launchpad)
            if not launchpad:
                yield LogEvent(line="No launchpad deployments found.")
            else:
                yield LogEvent(line="Launchpad deployments:")
                for info in visible_launchpad:
                    suffix = f" ({info.app_id})" if info.app_id else ""
                    bk = info.backend.value if info.backend else "unknown"
                    inst = info.instance_name or "-"
                    yield LogEvent(
                        line=f"  backend={bk}  instance={inst}  app={info.name}  state={info.state}{suffix}"
                    )
                hidden_count = len(launchpad) - len(visible_launchpad)
                if hidden_count > 0:
                    yield LogEvent(
                        line=f"  [hidden {hidden_count} historical duplicate app row{'s' if hidden_count != 1 else ''}]"
                    )
            yield OperationCompleteEvent(
                operation=OperationType.LIST, success=True, data=visible_launchpad
            )
            return

        # Fallback to raw text
        raw_result = ModalBackend.list_apps_raw_result()
        if raw_result.success and raw_result.output:
            raw = raw_result.output
            names = {legacy_app_name(bt) for bt in BackendType}
            lines = raw.splitlines()
            matches = [
                line
                for line in lines
                if any(name in line for name in names) or "vllm-" in line or "llamacpp-" in line
            ]
            if matches:
                yield LogEvent(line="Launchpad deployments:")
                for line in matches:
                    yield LogEvent(line=f"  {line.strip()}")
            else:
                yield LogEvent(line="No launchpad deployments found.")
            yield OperationCompleteEvent(operation=OperationType.LIST, success=True)
        else:
            details = []
            if apps_result.error is not None:
                details.append(apps_result.error.message)
            if raw_result.error is not None:
                details.append(raw_result.error.message)
            message = "Failed to query Modal app list."
            if details:
                message = f"{message} {' '.join(details)}"
            yield ErrorEvent(
                message=message,
                operation=OperationType.LIST,
            )
            yield OperationCompleteEvent(
                operation=OperationType.LIST, success=False, exit_code=1, detail=message
            )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def list_storage(self) -> EventStream:
        """List cached model artifacts for both backends."""
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STORAGE_LIST,
            detail="Scanning Modal volumes for cached models",
        )
        scan_started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                llamacpp_future = executor.submit(self._list_llamacpp_models)
                vllm_future = executor.submit(self._list_vllm_models)
                snapshot = StorageSnapshot(
                    llamacpp_models=llamacpp_future.result(),
                    vllm_models=vllm_future.result(),
                )
        except Exception as exc:
            yield ErrorEvent(
                message=f"Failed to list storage: {exc}",
                operation=OperationType.STORAGE_LIST,
                recoverable=True,
            )
            yield OperationCompleteEvent(
                operation=OperationType.STORAGE_LIST,
                success=False,
                exit_code=1,
                detail=str(exc),
            )
            return

        scan_elapsed_ms = int((time.perf_counter() - scan_started) * 1000)
        yield LogEvent(
            line=(
                f"Storage snapshot: llama.cpp={len(snapshot.llamacpp_models)} models, "
                f"vLLM={len(snapshot.vllm_models)} models, "
                f"total={snapshot.total_models}, bytes={snapshot.total_size_bytes}, "
                f"scan_ms={scan_elapsed_ms}"
            ),
            operation=OperationType.STORAGE_LIST,
        )
        yield OperationCompleteEvent(
            operation=OperationType.STORAGE_LIST,
            success=True,
            data=snapshot,
        )

    def predownload_model(
        self,
        backend: BackendType,
        model_id: str,
        quant: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> EventStream:
        """Pre-download a model for a backend without deploying."""
        if backend == BackendType.LLAMACPP:
            script = MODAL_LLAMACPP_SCRIPT
            entrypoint = "predownload_model"
            args = ["--repo-id", model_id]
            if quant:
                args.extend(["--quant", quant])
            if revision:
                args.extend(["--revision", revision])
        else:
            script = MODAL_VLLM_SCRIPT
            entrypoint = "predownload_model"
            args = ["--repo-id", model_id]
            if revision:
                args.extend(["--revision", revision])

        cmd = ModalBackend.build_modal_entrypoint_command(script, entrypoint, args)
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STORAGE_PREDOWNLOAD,
            detail=" ".join(cmd),
        )
        yield LogEvent(
            line=f"Running: {' '.join(cmd)}",
            operation=OperationType.STORAGE_PREDOWNLOAD,
        )

        for event in ModalBackend.run_modal_script_entrypoint(script, entrypoint, args=args):
            if isinstance(event, OperationCompleteEvent):
                yield OperationCompleteEvent(
                    operation=OperationType.STORAGE_PREDOWNLOAD,
                    success=event.success,
                    exit_code=event.exit_code,
                    detail=event.detail,
                    data=event.data,
                )
            elif isinstance(event, ErrorEvent):
                yield ErrorEvent(
                    message=event.message,
                    operation=OperationType.STORAGE_PREDOWNLOAD,
                    exit_code=event.exit_code,
                    recoverable=event.recoverable,
                )
            elif isinstance(event, LogEvent):
                yield LogEvent(
                    line=event.line,
                    stream=event.stream,
                    operation=OperationType.STORAGE_PREDOWNLOAD,
                )
            else:
                yield event

    def delete_stored_model(self, model: StoredModelInfo) -> EventStream:
        """Delete cached storage artifacts for a selected model."""
        targets = self._storage_delete_targets(model)
        if not targets:
            yield ErrorEvent(
                message=f"No removable paths found for {model.model_id}",
                operation=OperationType.STORAGE_DELETE,
                recoverable=True,
            )
            yield OperationCompleteEvent(
                operation=OperationType.STORAGE_DELETE,
                success=False,
                exit_code=1,
                detail="No paths resolved for deletion.",
            )
            return

        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STORAGE_DELETE,
            detail=f"Deleting {model.model_id}",
        )
        for volume_name, remote_path, recursive in targets:
            yield LogEvent(
                line=f"Deleting {remote_path} from {volume_name}",
                operation=OperationType.STORAGE_DELETE,
            )
            for event in ModalBackend.run_volume_remove(volume_name, remote_path, recursive=recursive):
                if isinstance(event, OperationCompleteEvent):
                    if not event.success:
                        yield OperationCompleteEvent(
                            operation=OperationType.STORAGE_DELETE,
                            success=False,
                            exit_code=event.exit_code,
                            detail=event.detail or f"Failed removing {remote_path}",
                        )
                        return
                    continue
                if isinstance(event, ErrorEvent):
                    yield ErrorEvent(
                        message=event.message,
                        operation=OperationType.STORAGE_DELETE,
                        exit_code=event.exit_code,
                        recoverable=event.recoverable,
                    )
                    yield OperationCompleteEvent(
                        operation=OperationType.STORAGE_DELETE,
                        success=False,
                        exit_code=event.exit_code or 1,
                        detail=event.message,
                    )
                    return
                if isinstance(event, LogEvent):
                    yield LogEvent(
                        line=event.line,
                        stream=event.stream,
                        operation=OperationType.STORAGE_DELETE,
                    )
                else:
                    yield event

        yield OperationCompleteEvent(
            operation=OperationType.STORAGE_DELETE,
            success=True,
            detail=f"Deleted storage for {model.model_id}",
        )

    def _list_llamacpp_models(self) -> list[StoredModelInfo]:
        backend_rows = self._list_llamacpp_models_via_backend()
        if backend_rows is not None:
            return backend_rows
        return self._list_llamacpp_models_from_volume()

    def _list_llamacpp_models_via_backend(self) -> list[StoredModelInfo] | None:
        captured = ModalBackend.run_modal_script_entrypoint_capture(
            MODAL_LLAMACPP_SCRIPT,
            "list_downloaded_models_json",
        )
        if not captured:
            return None
        returncode, stdout, _stderr = captured
        if returncode != 0 or not stdout.strip():
            return None

        lines = stdout.splitlines()
        try:
            start = lines.index(_LLAMACPP_STORAGE_JSON_BEGIN)
            end = lines.index(_LLAMACPP_STORAGE_JSON_END, start + 1)
        except ValueError:
            return None
        if end <= start + 1:
            return []

        payload_text = "\n".join(lines[start + 1 : end]).strip()
        if not payload_text:
            return []
        try:
            payload = json.loads(payload_text)
        except Exception:
            return None
        if not isinstance(payload, list):
            return None

        rows: list[StoredModelInfo] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("model_id", "")).strip()
            if not model_id:
                continue
            revision_raw = item.get("revision")
            quant_raw = item.get("quant")
            source_volume_raw = item.get("source_volume")
            paths_raw = item.get("paths")
            rows.append(
                StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id=model_id,
                    revision=str(revision_raw).strip() if revision_raw not in (None, "") else None,
                    quant=str(quant_raw).strip() if quant_raw not in (None, "") else None,
                    size_bytes=max(0, int(item.get("size_bytes", 0) or 0)),
                    file_count=max(0, int(item.get("file_count", 0) or 0)),
                    source_volume=(
                        str(source_volume_raw).strip()
                        if source_volume_raw not in (None, "")
                        else self._LLAMACPP_STORAGE_VOLUME
                    ),
                    paths=[str(p) for p in paths_raw if isinstance(p, str)]
                    if isinstance(paths_raw, list)
                    else [],
                    incomplete=False,
                )
            )
        return sorted(rows, key=lambda row: (row.model_id, row.revision or ""))

    def _list_llamacpp_models_from_volume(self) -> list[StoredModelInfo]:
        grouped: dict[tuple[str, Optional[str]], StoredModelInfo] = {}
        blob_files: dict[str, set[str]] = {}
        incomplete_blob_files: dict[str, set[str]] = {}
        snapshot_gguf_sizes: dict[str, list[int]] = {}
        hub_files = self._walk_volume_files(self._LLAMACPP_STORAGE_VOLUME, "/hub")
        for row in hub_files:
            path = str(row.get("path", "")).strip()
            size = int(row.get("size_bytes", 0) or 0)
            rel = path.removeprefix("/hub/").strip("/")
            if not rel.startswith("models--"):
                continue
            if not row.get("is_file", False):
                continue
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            model_dir = parts[0]
            encoded = model_dir[len("models--") :]
            model_id = encoded.replace("--", "/")
            lower_path = path.lower()
            if "/blobs/" in lower_path:
                blob_name = path.rsplit("/", 1)[-1].strip()
                if blob_name.endswith(".incomplete"):
                    final_blob_name = blob_name[: -len(".incomplete")]
                    if final_blob_name:
                        incomplete_blob_files.setdefault(model_id, set()).add(final_blob_name)
                elif blob_name:
                    blob_files.setdefault(model_id, set()).add(blob_name)
            if not lower_path.endswith(".gguf"):
                continue
            snapshot_gguf_sizes.setdefault(model_id, []).append(size)
            key = (model_id, None)
            entry = grouped.get(key)
            if entry is None:
                entry = StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id=model_id,
                    revision=None,
                    quant=None,
                    size_bytes=0,
                    file_count=0,
                    source_volume=self._LLAMACPP_STORAGE_VOLUME,
                    paths=[],
                )
                grouped[key] = entry
            entry.size_bytes += size
            entry.file_count += 1
            if entry.paths is not None and path not in entry.paths:
                entry.paths.append(path)
            if entry.quant is None:
                quant_match = _GGUF_QUANT_RE.search(path.upper())
                if quant_match:
                    entry.quant = quant_match.group(1).upper()

        for (model_id, _revision), entry in grouped.items():
            known_blobs = blob_files.get(model_id, set())
            pending_blobs = incomplete_blob_files.get(model_id, set())
            unresolved_incomplete = any(blob_name not in known_blobs for blob_name in pending_blobs)
            gguf_sizes = snapshot_gguf_sizes.get(model_id, [])
            tiny_snapshot_without_blob = (
                bool(gguf_sizes)
                and all(size < _LLAMACPP_MIN_PLAUSIBLE_GGUF_BYTES for size in gguf_sizes)
                and not known_blobs
            )
            if unresolved_incomplete or tiny_snapshot_without_blob:
                entry.incomplete = True

        return sorted(grouped.values(), key=lambda row: (row.model_id, row.revision or ""))

    def _list_vllm_models(self) -> list[StoredModelInfo]:
        files = self._walk_volume_files(self._VLLM_STORAGE_VOLUME, "/hub")
        grouped: dict[str, StoredModelInfo] = {}
        blob_files: dict[str, set[str]] = {}
        incomplete_blob_files: dict[str, set[str]] = {}
        snapshot_file_counts: dict[str, int] = {}
        ref_file_counts: dict[str, int] = {}
        gguf_models: set[str] = set()
        for row in files:
            path = str(row.get("path", "")).strip()
            size = int(row.get("size_bytes", 0) or 0)
            rel = path.removeprefix("/hub/").strip("/")
            if not rel.startswith("models--"):
                continue
            parts = rel.split("/")
            if not parts:
                continue
            model_dir = parts[0]
            encoded = model_dir[len("models--") :]
            model_id = encoded.replace("--", "/")
            entry = grouped.get(model_id)
            if entry is None:
                entry = StoredModelInfo(
                    backend=BackendType.VLLM,
                    model_id=model_id,
                    revision=None,
                    quant=None,
                    size_bytes=0,
                    file_count=0,
                    source_volume=self._VLLM_STORAGE_VOLUME,
                    paths=[],
                )
                grouped[model_id] = entry
            if row.get("is_file", False):
                entry.size_bytes += size
                entry.file_count += 1
                lower_path = path.lower()
                if "/snapshots/" in lower_path:
                    snapshot_file_counts[model_id] = snapshot_file_counts.get(model_id, 0) + 1
                if "/refs/" in lower_path:
                    ref_file_counts[model_id] = ref_file_counts.get(model_id, 0) + 1
                if "/blobs/" in lower_path:
                    blob_name = path.rsplit("/", 1)[-1].strip()
                    if blob_name.endswith(".incomplete"):
                        final_blob_name = blob_name[: -len(".incomplete")]
                        if final_blob_name:
                            incomplete_blob_files.setdefault(model_id, set()).add(final_blob_name)
                    elif blob_name:
                        blob_files.setdefault(model_id, set()).add(blob_name)
                if lower_path.endswith(".gguf"):
                    gguf_models.add(model_id)
                if entry.paths is not None and path not in entry.paths:
                    entry.paths.append(path)

        for model_id in list(grouped.keys()):
            if model_id in gguf_models:
                del grouped[model_id]

        if not grouped:
            return []

        for model_id, entry in grouped.items():
            known_blobs = blob_files.get(model_id, set())
            pending_blobs = incomplete_blob_files.get(model_id, set())
            snapshot_count = snapshot_file_counts.get(model_id, 0)
            ref_count = ref_file_counts.get(model_id, 0)
            # Align with huggingface_hub cache semantics:
            # incomplete blobs are resumable partial downloads, but once the
            # final blob exists, download logic treats that file as complete.
            # Keep stale orphan .incomplete blobs from old attempts from marking
            # a model incomplete when a usable snapshot already exists.
            unresolved_incomplete = any(blob_name not in known_blobs for blob_name in pending_blobs)
            if unresolved_incomplete and snapshot_count == 0:
                entry.incomplete = True
            # Refs without snapshots indicate metadata without an accessible
            # snapshot payload for this repo.
            if ref_count > 0 and snapshot_count == 0:
                entry.incomplete = True
        return sorted(grouped.values(), key=lambda row: row.model_id)

    def _walk_volume_files(
        self,
        volume_name: str,
        root: str,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        pending = [root]
        files: list[dict[str, Any]] = []
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            rows = ModalBackend.list_volume(volume_name, current) or []
            for row in rows:
                path = self._volume_entry_path(current, row)
                if not path:
                    continue
                is_dir = self._volume_entry_is_dir(row, path)
                if is_dir:
                    if path == current or not recursive:
                        continue
                    pending.append(path)
                else:
                    files.append(
                        {
                            "path": path,
                            "size_bytes": Orchestrator._parse_size_bytes(
                                Orchestrator._entry_value(
                                    row, "size", "size_bytes", "bytes", "filesize"
                                )
                            ),
                            "is_file": True,
                        }
                    )
        return files

    def _storage_delete_targets(
        self, model: StoredModelInfo
    ) -> list[tuple[str, str, bool]]:
        targets: list[tuple[str, str, bool]] = []
        if model.backend == BackendType.LLAMACPP:
            encoded = model.model_id.replace("/", "--")
            targets.append(
                (
                    model.source_volume or self._LLAMACPP_STORAGE_VOLUME,
                    f"/hub/models--{encoded}",
                    True,
                )
            )
            return targets

        if model.backend == BackendType.VLLM:
            encoded = model.model_id.replace("/", "--")
            targets.append(
                (
                    model.source_volume or self._VLLM_STORAGE_VOLUME,
                    f"/hub/models--{encoded}",
                    True,
                )
            )
        return targets

    @staticmethod
    def _entry_value(row: dict[str, Any], *keys: str) -> Any:
        """Case-insensitive lookup for JSON fields from Modal CLI."""
        lowered = {str(key).strip().lower(): value for key, value in row.items()}
        for key in keys:
            needle = key.strip().lower()
            if needle in lowered:
                return lowered[needle]
        return None

    @staticmethod
    def _volume_entry_path(parent: str, row: dict[str, Any]) -> str:
        for key in ("path", "name", "file", "filename", "entry"):
            value = Orchestrator._entry_value(row, key)
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if candidate.startswith("/"):
                    return candidate
                parent_clean = parent.strip("/")
                if parent_clean and candidate.startswith(f"{parent_clean}/"):
                    return f"/{candidate.lstrip('/')}"
                if parent == "/" and candidate:
                    return f"/{candidate.lstrip('/')}"
                if parent.endswith("/"):
                    return f"{parent}{candidate}"
                return f"{parent}/{candidate}"
        return ""

    @staticmethod
    def _volume_entry_is_dir(row: dict[str, Any], path: str) -> bool:
        is_dir = Orchestrator._entry_value(row, "is_dir", "isdir", "directory")
        if isinstance(is_dir, bool):
            return is_dir
        entry_type = str(Orchestrator._entry_value(row, "type", "kind") or "").strip().lower()
        if entry_type in {"dir", "directory", "folder"}:
            return True
        size_hint = Orchestrator._entry_value(row, "size", "size_bytes", "bytes")
        if size_hint in (None, "", 0, "0"):
            # If no file-size signal is present, treat slash-terminated entries as directories.
            if path.endswith("/"):
                return True
        return path.endswith("/")

    @staticmethod
    def _parse_size_bytes(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return 0
        clean = text.replace(",", "")
        if clean.isdigit():
            return int(clean)
        match = _SIZE_TOKEN_RE.match(clean)
        if not match:
            return 0
        number = float(match.group("num"))
        unit = (match.group("unit") or "B").upper()
        factors = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
            "KIB": 1024,
            "MIB": 1024**2,
            "GIB": 1024**3,
            "TIB": 1024**4,
        }
        factor = factors.get(unit, 1)
        return int(number * factor)

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop_app(
        self,
        backend: BackendType,
        app_name: Optional[str] = None,
        app_id: Optional[str] = None,
        provider: ComputeProvider = ComputeProvider.MODAL,
    ) -> EventStream:
        """Stop a deployed app."""
        if provider == ComputeProvider.PRIME:
            pod_id = (app_id or "").strip()
            if not pod_id:
                message = "Prime termination requires a pod ID."
                yield ErrorEvent(message=message, operation=OperationType.STOP)
                yield OperationCompleteEvent(
                    operation=OperationType.STOP, success=False, exit_code=2, detail=message
                )
                return
            yield StateChangeEvent(
                current=DeploymentState.RUNNING,
                operation=OperationType.STOP,
                detail=f"Terminating Prime pod {pod_id}",
            )
            try:
                self.prime_backend.delete_pod(pod_id)
            except Exception as exc:
                yield ErrorEvent(message=str(exc), operation=OperationType.STOP)
                yield OperationCompleteEvent(
                    operation=OperationType.STOP, success=False, exit_code=1, detail=str(exc)
                )
                return
            yield LogEvent(
                line=f"Terminated Prime pod: {pod_id}",
                operation=OperationType.STOP,
            )
            yield StateChangeEvent(
                current=DeploymentState.STOPPED,
                operation=OperationType.STOP,
            )
            yield OperationCompleteEvent(operation=OperationType.STOP, success=True)
            return
        target_app_name = (app_id or app_name or legacy_app_name(backend)).strip()
        cmd = ["modal", "app", "stop", "--yes", target_app_name]
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STOP,
            detail=f"Stopping {target_app_name}",
        )
        yield LogEvent(
            line=f"Stopping app: {target_app_name}",
            operation=OperationType.STOP,
        )
        yield from _tag_operation(
            ModalBackend.run_streaming(cmd),
            OperationType.STOP,
        )

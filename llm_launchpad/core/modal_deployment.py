"""Modal provider deployment execution runner."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import replace
import secrets
from typing import Any

from ..protocol.enums import BackendType, DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import DeploymentConfig, ModalProviderOptions
from .backend import ModalBackend
from .config import ConfigStore
from .deployment_preflight import price_cap_finding
from .modal_gpu import fetch_modal_gpu_catalog
from .naming import random_function_slug
from .operation_events import fail_operation

EventStream = Generator[BaseEvent, None, None]


def _tag_operation(events: EventStream, operation: OperationType) -> EventStream:
    """Attach the owning workflow to events from a generic subprocess stream."""
    for event in events:
        if isinstance(event, (LogEvent, ErrorEvent, OperationCompleteEvent, StateChangeEvent)):
            yield replace(event, operation=operation)
        else:
            yield event


_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _format_env_for_log(env: dict[str, str]) -> str:
    """Format environment key/value pairs for logs with secrets redacted."""
    formatted: list[str] = []
    for key, value in env.items():
        upper = key.upper()
        display = "[redacted]" if any(marker in upper for marker in _SENSITIVE_ENV_MARKERS) else value
        formatted.append(f"{key}={display}")
    return ", ".join(formatted)


def _deploy_vllm(config: DeploymentConfig, env: dict[str, str]) -> EventStream:
    if config.do_deploy:
        cmd = ModalBackend.build_deploy_command(config.backend, app_name=config.app_name)
        yield StateChangeEvent(
            current=DeploymentState.DEPLOYING,
            operation=OperationType.DEPLOY,
            detail=" ".join(cmd),
        )
        yield LogEvent(line=f"Running: {' '.join(cmd)}")
        if env:
            yield LogEvent(line=f"  env: {_format_env_for_log(env)}")
        yield from _tag_operation(
            ModalBackend.run_streaming(cmd, env=env),
            OperationType.DEPLOY,
        )
    elif config.run_smoke:
        cmd = ModalBackend.build_run_command(config)
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.SMOKE_TEST,
            detail=" ".join(cmd),
        )
        yield LogEvent(line=f"Running: {' '.join(cmd)}")
        if env:
            yield LogEvent(line=f"  env: {_format_env_for_log(env)}")
        yield from _tag_operation(
            ModalBackend.run_streaming(cmd, env=env),
            OperationType.SMOKE_TEST,
        )
    elif config.preload:
        cmd = [*ModalBackend.build_run_command(config), "--preload"]
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.DEPLOY,
            detail=" ".join(cmd),
        )
        yield LogEvent(line=f"Running: {' '.join(cmd)}")
        if env:
            yield LogEvent(line=f"  env: {_format_env_for_log(env)}")
        yield from _tag_operation(
            ModalBackend.run_streaming(cmd, env=env),
            OperationType.DEPLOY,
        )
    else:
        yield from fail_operation(
            OperationType.DEPLOY,
            "Modal vLLM deployment requires deploy, smoke test, or preload intent.",
        )


def _deploy_llamacpp(config: DeploymentConfig, env: dict[str, str]) -> EventStream:
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
            yield LogEvent(line=f"  env: {_format_env_for_log(env)}")

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


def deploy_modal(
    config: DeploymentConfig,
    *,
    config_store: ConfigStore | None = None,
    slug_factory: Callable[[], str] = random_function_slug,
    gpu_catalog_fetcher: Callable[..., Any] = fetch_modal_gpu_catalog,
    vllm_deployer: Callable[[DeploymentConfig, dict[str, str]], EventStream] = _deploy_vllm,
    llamacpp_deployer: Callable[[DeploymentConfig, dict[str, str]], EventStream] = _deploy_llamacpp,
) -> EventStream:
    """Execute the Modal-specific deploy step for a prepared config."""
    options = config.provider_options
    if (
        config.do_deploy
        and isinstance(options, ModalProviderOptions)
        and options.max_hourly_cost_usd is not None
    ):
        try:
            rates = gpu_catalog_fetcher(force_refresh=True)
            target_gpu = (config.gpu_type or "").strip().upper()
            unit = next(
                (
                    r.price_per_hour_usd
                    for r in rates
                    if r.value.upper() == target_gpu and r.stale_since is None
                ),
                None,
            )
            current = unit * (config.gpu_count or 1) if unit is not None else None
            finding = price_cap_finding(options.max_hourly_cost_usd, current)
        except Exception as exc:
            yield from fail_operation(
                OperationType.DEPLOY, f"Could not verify the Modal GPU rate: {exc}"
            )
            return
        if finding:
            yield from fail_operation(OperationType.DEPLOY, finding.message)
            return
        config.price_per_hour_usd = current

    if config.do_deploy and config.backend != BackendType.VLLM and not config.function_slug:
        config.function_slug = slug_factory()
    if config.backend == BackendType.VLLM and config.do_deploy:
        config.endpoint_api_key = config.endpoint_api_key or secrets.token_urlsafe(32)

    store = config_store or ConfigStore()
    settings = store.load()
    env = ModalBackend.build_full_env(settings, config)

    if config.backend == BackendType.VLLM:
        yield from vllm_deployer(config, env)
    else:
        yield from llamacpp_deployer(config, env)

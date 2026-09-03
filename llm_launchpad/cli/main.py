"""CLI dispatcher: the default command launches the TUI, with headless
commands available for automation.

This replaces the monolithic ``llm_launchpad.cli`` module while keeping
all non-interactive commands available for automation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
import json
from typing import Annotated, Optional

import typer

try:
    from rich.console import Console
except Exception:
    Console = None

from ..core.backend import ModalBackend
from ..core.benchmark import (
    benchmark_config_from_endpoint,
    merge_cached_benchmark_connections,
    parse_concurrency_values,
)
from ..core.config import ConfigStore
from ..core.connection_store import (
    merge_connections,
    remove_connection,
    rows_from_connection_cache,
    save_connection,
)
from ..core.diagnostics import log_exception, setup_logging
from ..core.deploy_log_summary import DeployLogSummarizer, beautify_summary_line
from ..core.hf_models import fetch_gguf_quant_metadata
from ..core.modal_gpu import fetch_modal_gpu_types
from ..tui import mouse as mouse_defaults
from ..core.naming import (
    auto_instance_name_for_backend,
    build_deployment_name,
    legacy_app_name,
    random_function_slug,
    slugify_instance_name,
)
from ..core.prime_backend import PrimeBackend
from ..core.prime_auth import get_prime_auth_status
from ..core.provider_options import prime_provider_options
from ..core.opencode import (
    resolve_connection_for_app,
    resolve_connections_for_rows,
    sync_opencode_config,
    visible_launchpad_rows,
)
from ..core.orchestrator import Orchestrator
from ..core.runtime_support import (
    evaluate_llamacpp_architecture,
    load_llamacpp_support_manifest,
)
from ..protocol.enums import BackendType, ComputeProvider, OperationType
from ..protocol.events import (
    EndpointAvailableEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo, PrimeProviderOptions

app = typer.Typer(
    help="llm-launchpad CLI - configure and deploy LLM backends on Modal.",
    invoke_without_command=True,
)
opencode_app = typer.Typer(help="OpenCode integration commands.")
app.add_typer(opencode_app, name="opencode")
aai_auth_app = typer.Typer(help="Manage the stored Artificial Analysis API key.")
app.add_typer(aai_auth_app, name="aai-auth")


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        from importlib.metadata import version as _package_version

        pkg_version = _package_version("llm-launchpad")
    except Exception:
        pkg_version = "unknown"
    typer.echo(f"llm-launchpad {pkg_version}")
    raise typer.Exit()


@app.callback()
def _default(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = None,
) -> None:
    """Launch the TUI when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        tui()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _print_event(
    event: object,
    summarizer: DeployLogSummarizer | None = None,
) -> None:
    """Print a protocol event to stdout/stderr for headless mode."""
    if summarizer is not None:
        if isinstance(event, LogEvent):
            for line in summarizer.transform(event.line, event.operation):
                typer.echo(beautify_summary_line(line))
            return
        if isinstance(event, StateChangeEvent):
            for line in summarizer.transform_state(event.detail, event.operation):
                typer.echo(beautify_summary_line(line))
            return
        if isinstance(event, ErrorEvent):
            typer.echo(beautify_summary_line(f"Error: {event.message}"), err=True)
            return
        if isinstance(event, OperationCompleteEvent):
            if event.success:
                typer.echo(beautify_summary_line(f"Done ({event.operation.value})."))
            else:
                typer.echo(
                    beautify_summary_line(
                        f"Failed ({event.operation.value}): {event.detail}"
                    ),
                    err=True,
                )
            return
    if isinstance(event, LogEvent):
        typer.echo(event.line)
    elif isinstance(event, StateChangeEvent):
        typer.echo(f"[{event.current.value}] {event.detail}")
    elif isinstance(event, ErrorEvent):
        typer.echo(f"Error: {event.message}", err=True)
    elif isinstance(event, OperationCompleteEvent):
        if event.success:
            typer.echo(f"Done ({event.operation.value}).")
        else:
            typer.echo(f"Failed ({event.operation.value}): {event.detail}", err=True)


def _ensure(ok: bool, msg: str) -> None:
    if not ok:
        typer.echo(msg, err=True)
        raise typer.Exit(code=1)


def _raise_on_failed_completion(event: object) -> None:
    """Exit when an operation-complete event reports failure."""
    if isinstance(event, OperationCompleteEvent) and not event.success:
        raise typer.Exit(code=event.exit_code or 1)


def _preflight(
    provider: ComputeProvider = ComputeProvider.MODAL,
) -> tuple[Orchestrator, str]:
    orch = Orchestrator()
    ok, username, err = orch.preflight(provider)
    _ensure(ok, err)
    return orch, username


def _parse_bool_env(name: str, default: bool) -> bool:
    return mouse_defaults.parse_bool_env(name, default)


def _is_ssh_session() -> bool:
    return mouse_defaults.is_ssh_session()


def _likely_remote_clipboard_supported() -> bool:
    return mouse_defaults.likely_remote_clipboard_supported()


def _default_tui_mouse_enabled() -> bool:
    return mouse_defaults.default_tui_mouse_enabled()


def _ensure_tui_runtime() -> None:
    """Fail fast with visible CLI errors before handing off to Textual."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo(
            "Error: llm-launchpad TUI requires an interactive terminal (TTY).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not ModalBackend.is_cli_available() and not get_prime_auth_status().authenticated:
        typer.echo(
            "Error: no compute provider is configured. Run `modal setup` or `prime login`.",
            err=True,
        )
        raise typer.Exit(code=1)


def _resolve_deploy_target(
    backend: BackendType,
    model_hint: Optional[str],
    instance_name: Optional[str],
    app_name: Optional[str],
    provider: ComputeProvider = ComputeProvider.MODAL,
) -> tuple[str, str]:
    """Resolve (instance_name, app_name) for deployment."""
    explicit_app = (app_name or "").strip()
    if explicit_app:
        explicit_instance = (instance_name or "").strip() or slugify_instance_name(explicit_app)
        return explicit_instance, explicit_app

    explicit_instance = (instance_name or "").strip()
    if explicit_instance:
        slug = slugify_instance_name(explicit_instance)
        return slug, build_deployment_name(provider, backend, slug)

    auto = auto_instance_name_for_backend(backend, model_hint)
    return auto, build_deployment_name(provider, backend, auto)


def _backend_instances(backend: BackendType) -> list[EndpointInfo]:
    rows = ModalBackend.list_apps() or []
    return [row for row in rows if row.backend == backend]


def _provider_instances(
    backend: BackendType,
    provider: ComputeProvider,
) -> list[EndpointInfo]:
    if provider == ComputeProvider.MODAL:
        return _backend_instances(backend)
    rows = PrimeBackend().list_deployments()
    return [row for row in rows if row.backend == backend]


def _resolve_manage_app_name(
    backend: BackendType,
    app_name: Optional[str],
    instance_name: Optional[str],
    provider: ComputeProvider = ComputeProvider.MODAL,
) -> str:
    explicit_app = (app_name or "").strip()
    if explicit_app:
        return explicit_app
    explicit_instance = (instance_name or "").strip()
    if explicit_instance:
        return build_deployment_name(provider, backend, explicit_instance)

    matches = (
        _backend_instances(backend)
        if provider == ComputeProvider.MODAL
        else _provider_instances(backend, provider)
    )
    if len(matches) == 1:
        return matches[0].name
    if len(matches) > 1:
        typer.echo(
            f"Multiple '{backend.value}' instances found. "
            "Specify --instance-name or --app-name. Use `llm-launchpad list` to inspect instances.",
            err=True,
        )
        raise typer.Exit(code=1)
    if provider == ComputeProvider.MODAL:
        return legacy_app_name(backend)
    return build_deployment_name(provider, backend, "default")


def _resolve_manage_target(
    backend: BackendType,
    provider: ComputeProvider,
    app_name: Optional[str],
    instance_name: Optional[str],
) -> EndpointInfo:
    """Resolve a management target and hydrate its locally stored connection."""
    target_name = _resolve_manage_app_name(backend, app_name, instance_name, provider)
    rows = _provider_instances(backend, provider)
    merge_connections(rows)
    row = next((item for item in rows if item.name == target_name), None)
    if row is not None:
        return row
    fallback = EndpointInfo(name=target_name, backend=backend, provider=provider)
    merge_connections([fallback])
    return fallback


def _print_banner() -> None:
    if Console is None:
        typer.echo("llm-launchpad")
        return
    try:
        from rich.panel import Panel
        from importlib.metadata import version
    except Exception:
        typer.echo("llm-launchpad")
        return
    try:
        v = version("llm-launchpad")
        subtitle = f"v{v}  ·  Modal LLM backends"
    except Exception:
        subtitle = "Modal LLM backends"
    console = Console()
    console.print(Panel.fit("llm-launchpad", subtitle=subtitle, border_style="cyan"))


def _connected_compute_providers() -> tuple[ComputeProvider, ...]:
    providers: list[ComputeProvider] = []
    if ModalBackend.is_cli_available():
        providers.append(ComputeProvider.MODAL)
    try:
        if get_prime_auth_status().authenticated:
            providers.append(ComputeProvider.PRIME)
    except Exception:
        log_exception("Prime auth probe failed while listing connected providers")
    return tuple(providers)


def _load_visible_launchpad_rows(
    provider: ComputeProvider = ComputeProvider.MODAL,
) -> list[EndpointInfo] | None:
    rows = (
        ModalBackend.list_apps()
        if provider == ComputeProvider.MODAL
        else PrimeBackend().list_deployments()
    )
    if rows is None:
        return None
    return visible_launchpad_rows(merge_connections(rows))


def _list_provider_deployments(provider: ComputeProvider) -> list[EndpointInfo] | None:
    """List deployments for one provider; ``None`` when the listing is unavailable."""
    try:
        rows = (
            ModalBackend.list_apps()
            if provider == ComputeProvider.MODAL
            else PrimeBackend().list_deployments()
        )
    except Exception:
        return None
    return rows


def _load_rows_for_opencode_sync(
    provider: ComputeProvider | None = None,
    *,
    persist_backfill: bool = True,
) -> tuple[list[EndpointInfo] | None, tuple[ComputeProvider, ...], list[str]]:
    """Load deployments for OpenCode sync and the providers that were listed.

    Returns ``(rows, live_listed_providers, fallback_notes)``. When a provider
    listing is unavailable its cached connection summaries are used instead so
    known deployments stay in scope; only successfully listed providers are
    returned as prune scope. ``None`` rows mean no deployment source was found.
    """
    connected = _connected_compute_providers()
    providers = (
        (provider,)
        if provider is not None
        else (connected or (ComputeProvider.MODAL, ComputeProvider.PRIME))
    )
    collected: list[EndpointInfo] = []
    listed: list[ComputeProvider] = []
    notes: list[str] = []
    for compute_provider in providers:
        rows = _list_provider_deployments(compute_provider)
        if rows is not None:
            collected.extend(rows)
            listed.append(compute_provider)
            continue
        cached_rows = [
            row
            for row in rows_from_connection_cache(
                persist_backfill=persist_backfill,
            )
            if row.provider == compute_provider
        ]
        if cached_rows:
            collected.extend(cached_rows)
            notes.append(f"{compute_provider.value.title()} listing failed; using cached connection summaries.")
        else:
            notes.append(
                f"{compute_provider.value.title()} listing failed and no cached connections are available."
            )
    if not listed and not collected:
        return None, (), notes
    return (
        visible_launchpad_rows(
            merge_connections(
                collected,
                persist_backfill=persist_backfill,
            )
        ),
        tuple(listed),
        notes,
    )


def _sync_opencode_cli(
    *,
    target_app_name: Optional[str] = None,
    target_url: Optional[str] = None,
    target_config: Optional[DeploymentConfig] = None,
    current_rows: list[EndpointInfo] | None = None,
    remove_app_names: list[str] | None = None,
    prune_providers: tuple[ComputeProvider, ...] | None = None,
    username: str = "",
    dry_run: bool = False,
    fail_on_error: bool = False,
) -> None:
    target = None
    targets = None
    if target_app_name:
        target = resolve_connection_for_app(
            target_app_name,
            rows=current_rows,
            username=username,
            fallback_config=target_config,
            fallback_server_url=target_url,
        )
    elif current_rows is not None:
        targets = resolve_connections_for_rows(current_rows, username=username)
    try:
        result = sync_opencode_config(
            target=target,
            targets=targets,
            current_rows=current_rows,
            remove_app_names=remove_app_names,
            prune_providers=prune_providers,
            dry_run=dry_run,
        )
    except Exception as exc:
        message = f"OpenCode sync failed: {exc}"
        if fail_on_error:
            typer.echo(message, err=True)
            raise typer.Exit(code=1)
        typer.echo(message, err=True)
        return

    for line in result.messages:
        typer.echo(line)


def _deploy_and_maybe_warmup(
    orch: Orchestrator,
    *,
    username: str,
    backend: BackendType,
    config: DeploymentConfig,
    server_url: Optional[str],
    do_warmup: bool,
    timeout: int,
    tail_logs: bool,
    debug_logs: bool = False,
) -> None:
    """Run deploy, optional warmup, and OpenCode sync for CLI commands."""
    deployed_web_url: Optional[str] = None
    deployed_endpoint: Optional[EndpointInfo] = None
    deploy_succeeded = False
    opencode_synced = False
    summarizer = None if debug_logs else DeployLogSummarizer(backend)
    for event in orch.deploy(config):
        if isinstance(event, LogEvent):
            maybe_url = ModalBackend.extract_modal_web_url(event.line)
            if maybe_url:
                deployed_web_url = maybe_url
        elif isinstance(event, EndpointAvailableEvent):
            deployed_endpoint = event.endpoint
            deployed_web_url = event.endpoint.web_url or deployed_web_url
            if deployed_web_url:
                save_connection(config, event.endpoint)
                _sync_opencode_cli(
                    target_app_name=config.app_name,
                    target_url=deployed_web_url,
                    target_config=config,
                    current_rows=_load_visible_launchpad_rows(config.provider),
                    prune_providers=(config.provider,),
                    username=username,
                )
                opencode_synced = True
        elif isinstance(event, OperationCompleteEvent) and event.operation == OperationType.DEPLOY:
            deploy_succeeded = event.success
            if event.success and isinstance(event.data, EndpointInfo):
                deployed_endpoint = event.data
                deployed_web_url = event.data.web_url or deployed_web_url
            elif not event.success and opencode_synced:
                remove_connection(config.app_name or "")
                _sync_opencode_cli(
                    current_rows=_load_visible_launchpad_rows(config.provider),
                    remove_app_names=[config.app_name or ""],
                    prune_providers=(config.provider,),
                    username=username,
                )
                opencode_synced = False
        _print_event(event, summarizer=summarizer)
        _raise_on_failed_completion(event)

    if deploy_succeeded and deployed_endpoint and deployed_web_url:
        # Persist provider credentials immediately. A later warmup failure must
        # not strand a live, billable pod without its generated bearer key.
        save_connection(config, deployed_endpoint)

    warmup_succeeded = False
    final_sync_url: Optional[str] = None
    if do_warmup:
        url = server_url or deployed_web_url
        if not url and config.provider == ComputeProvider.MODAL:
            url = ModalBackend.default_server_url(
                username,
                app_name=config.app_name,
                function_slug=config.function_slug,
            )
        if not url:
            typer.echo("Error: the provider did not return a usable endpoint URL.", err=True)
            raise typer.Exit(code=1)
        warmup_events = (
            orch.warmup(
                backend,
                url,
                timeout,
                tail_logs,
                app_name=config.app_name,
                served_model_name=config.served_model_name,
            )
            if config.provider == ComputeProvider.MODAL
            else orch.warmup(
                backend,
                url,
                timeout,
                tail_logs,
                app_name=config.app_name,
                served_model_name=config.served_model_name,
                provider=config.provider,
                api_key=config.endpoint_api_key,
                pod_id=deployed_endpoint.app_id if deployed_endpoint else None,
            )
        )
        for event in warmup_events:
            if (
                isinstance(event, OperationCompleteEvent)
                and event.success
                and event.operation == OperationType.WARMUP
            ):
                warmup_succeeded = True
                final_sync_url = url
                if isinstance(event.data, dict):
                    maybe_url = event.data.get("url")
                    if isinstance(maybe_url, str) and maybe_url.strip():
                        final_sync_url = maybe_url.strip()
            _print_event(event, summarizer=summarizer)
            if (
                isinstance(event, OperationCompleteEvent)
                and not event.success
                and event.operation == OperationType.WARMUP
                and config.provider == ComputeProvider.PRIME
                and deployed_endpoint is not None
                and not prime_provider_options(config).keep_failed_resource
            ):
                typer.echo(
                    f"Warmup failed; terminating Prime pod {deployed_endpoint.app_id}."
                )
                for cleanup_event in orch.stop_app(
                    backend,
                    app_name=config.app_name,
                    app_id=deployed_endpoint.app_id,
                    provider=ComputeProvider.PRIME,
                ):
                    _print_event(cleanup_event, summarizer=summarizer)
                remove_connection(config.app_name or "")
                _sync_opencode_cli(
                    current_rows=_load_visible_launchpad_rows(config.provider),
                    remove_app_names=[config.app_name or ""],
                    prune_providers=(config.provider,),
                    username=username,
                )
                opencode_synced = False
            _raise_on_failed_completion(event)
    elif deploy_succeeded:
        final_sync_url = server_url or deployed_web_url
        if not final_sync_url and config.provider == ComputeProvider.MODAL:
            final_sync_url = ModalBackend.default_server_url(
                username,
                app_name=config.app_name,
                function_slug=config.function_slug,
            )

    if opencode_synced:
        return
    if (do_warmup and warmup_succeeded and final_sync_url) or (not do_warmup and final_sync_url):
        endpoint = deployed_endpoint or EndpointInfo(
            name=config.app_name or "",
            backend=config.backend,
            instance_name=config.instance_name,
            provider=config.provider,
        )
        endpoint.web_url = final_sync_url
        endpoint.endpoint_api_key = config.endpoint_api_key
        save_connection(config, endpoint)
        _sync_opencode_cli(
            target_app_name=config.app_name,
            target_url=final_sync_url,
            target_config=config,
            current_rows=_load_visible_launchpad_rows(config.provider),
            prune_providers=(config.provider,),
            username=username,
        )


# -----------------------------------------------------------------------
# TUI
# -----------------------------------------------------------------------


@app.command("tui")
def tui(
    mouse: Annotated[
        Optional[bool],
        typer.Option(
            "--mouse/--no-mouse",
            help=(
                "Enable Textual mouse support. "
                "Use --no-mouse to let the terminal handle native text selection/copy."
            ),
        ),
    ] = None,
) -> None:
    """Interactive terminal UI for deploying and managing LLM backends."""
    try:
        from ..tui.app import TuiApp
    except ImportError as exc:
        typer.echo(
            f"Error: Textual is required for the TUI. Install with: pip install textual\n({exc})",
            err=True,
        )
        raise typer.Exit(code=1)

    _ensure_tui_runtime()
    app_instance = TuiApp(mouse_enabled=mouse)
    try:
        app_instance.run(mouse=app_instance.mouse_enabled)
    finally:
        from ..core.backend import ModalBackend
        ModalBackend.terminate_all()


# -----------------------------------------------------------------------
# Headless commands
# -----------------------------------------------------------------------


@app.command("doctor")
def doctor() -> None:
    """Check local prerequisites and print a fix hint for each failure."""
    from ..core.doctor import doctor_exit_code, run_doctor_checks

    checks = run_doctor_checks()
    for check in checks:
        if check.ok:
            marker = "ok"
        elif check.required:
            marker = "FAIL"
        else:
            marker = "warn"
        suffix = f": {check.detail}" if check.detail else ""
        typer.echo(f"[{marker}] {check.name}{suffix}")
        if not check.ok and check.hint:
            typer.echo(f"       fix: {check.hint}")
    failed = [check for check in checks if not check.ok and check.required]
    if failed:
        typer.echo(
            f"{len(failed)} required check(s) failed; see hints above.",
            err=True,
        )
        raise typer.Exit(code=doctor_exit_code(checks))


@app.command("gpu-types")
def gpu_types(
    timeout: int = typer.Option(10, min=1, help="Modal docs request timeout in seconds"),
) -> None:
    """Fetch Modal GPU type values from the Modal docs page."""
    try:
        values = fetch_modal_gpu_types(timeout=float(timeout))
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    for value in values:
        typer.echo(value)


@app.command("llamacpp-support")
def llamacpp_support(
    repo_id: Optional[str] = typer.Option(
        None,
        help="Hugging Face GGUF repo to check against the pinned llama.cpp runtime",
    ),
    revision: Optional[str] = typer.Option(None, help="Optional Hugging Face revision"),
    architecture: Optional[str] = typer.Option(
        None,
        help="GGUF general.architecture value to check directly",
    ),
    output_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List supported llama.cpp architectures or preflight one GGUF repo."""

    manifest = load_llamacpp_support_manifest()
    resolved_architecture = architecture
    if repo_id:
        try:
            metadata = fetch_gguf_quant_metadata(repo_id, revision=revision)
        except Exception as exc:
            typer.echo(f"Error: could not inspect {repo_id}: {exc}", err=True)
            raise typer.Exit(code=1)
        resolved_architecture = metadata.architecture

    if repo_id or architecture:
        decision = evaluate_llamacpp_architecture(resolved_architecture, manifest=manifest)
        payload = {
            "repo_id": repo_id,
            "architecture": decision.architecture,
            "status": decision.status.value,
            "runtime_id": decision.runtime_id,
            "image_ref": decision.image_ref,
            "message": decision.message,
        }
        if output_json:
            typer.echo(json.dumps(payload, indent=2))
        else:
            if repo_id:
                typer.echo(f"repo: {repo_id}")
            typer.echo(f"architecture: {decision.architecture or 'unknown'}")
            typer.echo(f"status: {decision.status.value}")
            typer.echo(f"runtime: {decision.runtime_id}")
            typer.echo(decision.message)
        return

    payload = {
        "runtime_id": manifest.runtime_id,
        "runtime_build": manifest.runtime_build,
        "image_ref": manifest.image_ref,
        "image_digest": manifest.image_digest,
        "source_revision": manifest.source_revision,
        "architectures": sorted(manifest.architectures),
    }
    if output_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"runtime: {manifest.runtime_id}")
    typer.echo(f"image: {manifest.image_ref}@{manifest.image_digest}")
    typer.echo("supported general.architecture values:")
    for value in sorted(manifest.architectures):
        typer.echo(value)


@app.command("offers")
def prime_offers(
    gpu_type: Optional[str] = typer.Option(None, help="GPU type filter"),
    gpu_count: Optional[int] = typer.Option(None, min=1, help="GPU count filter"),
    region: Optional[str] = typer.Option(None, help="Region or country filter"),
    disk_id: Optional[str] = typer.Option(None, help="Only offers compatible with a Prime disk"),
    secure_only: bool = typer.Option(True, help="Show only secure-cloud offers"),
    on_demand_only: bool = typer.Option(True, help="Hide spot offers"),
    output_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List current Prime Intellect GPU availability and offer IDs."""
    orch, _ = _preflight(ComputeProvider.PRIME)
    try:
        rows = orch.prime_backend.list_offers(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            region=region,
            disk_id=disk_id,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    if secure_only:
        rows = [row for row in rows if (row.security or "").casefold() == "secure_cloud"]
    if on_demand_only:
        rows = [row for row in rows if not row.is_spot]
    rows.sort(
        key=lambda row: (
            row.price_per_hour is None,
            row.price_per_hour if row.price_per_hour is not None else float("inf"),
            row.id,
        )
    )
    if output_json:
        typer.echo(json.dumps([asdict(row) for row in rows], indent=2))
        return
    if not rows:
        typer.echo("No Prime GPU offers matched the requested filters.")
        return
    typer.echo("ID      GPU                 Location              Security       Price/hr")
    for row in rows:
        gpu = f"{row.gpu_count}x {row.gpu_type}"
        location = row.country or row.region or row.data_center or "-"
        price = f"${row.price_per_hour:.3f}" if row.price_per_hour is not None else "-"
        typer.echo(f"{row.id:<7} {gpu:<19} {location:<21} {(row.security or '-'):<14} {price}")


@app.command()
def deploy(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
    backend: BackendType = typer.Option(
        BackendType.LLAMACPP,
        help="Backend: llamacpp or vllm",
    ),
    do_warmup: bool = typer.Option(False, help="Verify readiness after deploy"),
    preset: Optional[str] = typer.Option(None, help="llama.cpp preset name (Modal only)"),
    repo_id: Optional[str] = typer.Option(None, help="llama.cpp Hugging Face GGUF repo ID"),
    quant: Optional[str] = typer.Option(None, help="llama.cpp GGUF quantization"),
    revision: Optional[str] = typer.Option(None, help="llama.cpp HF revision (Modal only)"),
    server_args: Optional[str] = typer.Option(None, help="Additional llama-server arguments"),
    n_gpu_layers: Optional[int] = typer.Option(None, help="llama.cpp GPU layers (default: auto)"),
    model_name: Optional[str] = typer.Option(None, help="vLLM MODEL_NAME"),
    model_revision: Optional[str] = typer.Option(None, help="vLLM MODEL_REVISION"),
    served_model_name: Optional[str] = typer.Option(None, help="vLLM SERVED_MODEL_NAME"),
    fast_boot: Optional[bool] = typer.Option(None, help="vLLM FAST_BOOT"),
    n_gpu: Optional[int] = typer.Option(None, help="vLLM N_GPU"),
    gpu_type: Optional[str] = typer.Option(None, help="Prime GPU type filter"),
    gpu_count: Optional[int] = typer.Option(None, min=1, help="Prime GPU count filter"),
    prime_offer_id: Optional[str] = typer.Option(
        None, help="Exact six-character Prime availability offer ID"
    ),
    prime_region: Optional[str] = typer.Option(None, help="Prime region or country filter"),
    prime_disk_id: Optional[str] = typer.Option(None, help="Existing Prime disk ID to attach"),
    prime_disk: bool = typer.Option(
        True,
        "--prime-disk/--no-prime-disk",
        help="Auto-attach a persistent Prime disk so model weights survive redeploys",
    ),
    keep_failed_pod: bool = typer.Option(
        False, help="Keep a failed Prime pod instead of terminating it"
    ),
    allow_insecure_http: bool = typer.Option(
        False, help="Bypass Prime Tunnel and use a direct HTTP endpoint"
    ),
    trust_remote_code: Optional[bool] = typer.Option(
        None,
        help="vLLM TRUST_REMOTE_CODE (allow model custom code from Hugging Face)",
    ),
    reasoning_parser: Optional[str] = typer.Option(
        None,
        help="vLLM reasoning parser (e.g. qwen3, deepseek_r1, granite)",
    ),
    tool_call_parser: Optional[str] = typer.Option(
        None,
        help="vLLM tool call parser (e.g. hermes, qwen3_xml, llama3_json)",
    ),
    default_chat_template_kwargs: Optional[str] = typer.Option(
        None,
        help=(
            "vLLM default chat template kwargs JSON "
            "(e.g. '{\"enable_thinking\": false}' or '{\"thinking\": true}')"
        ),
    ),
    instance_name: Optional[str] = typer.Option(
        None, help="Instance name (auto-generated from model when omitted)"
    ),
    app_name: Optional[str] = typer.Option(None, help="Explicit deployment name override"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    timeout: int = typer.Option(1800, help="Warmup timeout seconds"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
    debug_logs: bool = typer.Option(
        False,
        "--debug-logs/--summary-logs",
        help="Show raw backend logs instead of the concise progress view",
    ),
) -> None:
    """Deploy a server to Modal or Prime Intellect."""
    compute_provider = ComputeProvider(provider)
    orch, username = _preflight(compute_provider)
    _print_banner()

    from ..protocol.models import DeploymentConfig

    bt = BackendType(backend)
    if compute_provider == ComputeProvider.PRIME and bt == BackendType.LLAMACPP and not repo_id:
        typer.echo("Error: Prime llama.cpp requires --repo-id.", err=True)
        raise typer.Exit(code=2)
    model_hint = model_name if bt == BackendType.VLLM else (repo_id or preset)
    resolved_instance, resolved_app_name = _resolve_deploy_target(
        bt,
        model_hint=model_hint,
        instance_name=instance_name,
        app_name=app_name,
        provider=compute_provider,
    )

    config = DeploymentConfig(
        backend=bt,
        provider=compute_provider,
        do_deploy=True,
        preset=preset,
        repo_id=repo_id,
        quant=quant,
        revision=revision,
        server_args=server_args,
        n_gpu_layers=n_gpu_layers,
        model_name=model_name,
        model_revision=model_revision,
        served_model_name=served_model_name,
        fast_boot=fast_boot,
        n_gpu=n_gpu,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        trust_remote_code=trust_remote_code,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        instance_name=resolved_instance,
        app_name=resolved_app_name,
        provider_options=(
            PrimeProviderOptions(
                offer_id=prime_offer_id,
                region=prime_region,
                disk_id=prime_disk_id,
                keep_failed_resource=keep_failed_pod,
                allow_insecure_http=allow_insecure_http,
                auto_disk=prime_disk,
            )
            if compute_provider == ComputeProvider.PRIME
            else None
        ),
    )
    typer.echo(
        f"Deploy target: backend={bt.value} provider={compute_provider.value} "
        f"instance={resolved_instance} app={resolved_app_name}"
    )

    _deploy_and_maybe_warmup(
        orch,
        username=username,
        backend=bt,
        config=config,
        server_url=server_url,
        do_warmup=do_warmup,
        timeout=timeout,
        tail_logs=tail_logs,
        debug_logs=debug_logs,
    )
    raise typer.Exit(code=0)


@app.command()
def warmup(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
    backend: BackendType = typer.Option(
        BackendType.LLAMACPP,
        help="Backend: llamacpp or vllm",
    ),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    served_model_name: Optional[str] = typer.Option(None, help="Served model name for llama.cpp probes"),
    function_slug: Optional[str] = typer.Option(
        None, help="Modal function slug suffix used in endpoint URL"
    ),
    timeout: int = typer.Option(1800, help="Seconds to wait"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
    debug_logs: bool = typer.Option(
        False,
        "--debug-logs/--summary-logs",
        help="Show raw backend logs instead of the concise progress view",
    ),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target deployment name"),
) -> None:
    """Cold start the container by probing the server."""
    compute_provider = ComputeProvider(provider)
    orch, username = _preflight(compute_provider)
    _print_banner()
    bt = BackendType(backend)
    target = _resolve_manage_target(bt, compute_provider, app_name, instance_name)
    url = server_url or os.environ.get("SERVER_URL")
    if (
        not url
        and compute_provider == ComputeProvider.MODAL
        and (function_slug or os.environ.get("MODAL_FUNCTION_SLUG"))
    ):
        url = ModalBackend.default_server_url(
            username,
            app_name=target.name,
            function_slug=function_slug or os.environ.get("MODAL_FUNCTION_SLUG"),
        )
    url = url or target.web_url
    if not url and compute_provider == ComputeProvider.MODAL:
        url = ModalBackend.default_server_url(
            username,
            app_name=target.name,
            function_slug=function_slug or os.environ.get("MODAL_FUNCTION_SLUG"),
        )
    if not url:
        typer.echo("Error: could not resolve the endpoint URL for this deployment.", err=True)
        raise typer.Exit(code=1)
    warmup_events = (
        orch.warmup(
            bt,
            url,
            timeout,
            tail_logs,
            app_name=target.name,
            served_model_name=served_model_name,
        )
        if compute_provider == ComputeProvider.MODAL
        else orch.warmup(
            bt,
            url,
            timeout,
            tail_logs,
            app_name=target.name,
            served_model_name=served_model_name,
            provider=compute_provider,
            api_key=target.endpoint_api_key,
            pod_id=target.app_id,
        )
    )
    summarizer = None if debug_logs else DeployLogSummarizer(bt)
    for event in warmup_events:
        _print_event(event, summarizer=summarizer)
        _raise_on_failed_completion(event)


@app.command("list")
def list_apps(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
) -> None:
    """List launchpad deployments for a compute provider."""
    compute_provider = ComputeProvider(provider)
    orch, username = _preflight(compute_provider)
    _print_banner()
    visible_rows: list[EndpointInfo] | None = None
    list_events = (
        orch.list_deployments()
        if compute_provider == ComputeProvider.MODAL
        else orch.list_deployments(compute_provider)
    )
    for event in list_events:
        if (
            isinstance(event, OperationCompleteEvent)
            and event.success
            and isinstance(event.data, list)
        ):
            visible_rows = merge_connections(
                [row for row in event.data if isinstance(row, EndpointInfo)]
            )
        _print_event(event)
    if visible_rows is not None:
        _sync_opencode_cli(
            current_rows=visible_rows,
            username=username,
            prune_providers=(compute_provider,),
        )


@app.command()
def status(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
    backend: BackendType = typer.Option(
        BackendType.LLAMACPP,
        help="Backend: llamacpp or vllm",
    ),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    served_model_name: Optional[str] = typer.Option(None, help="Served model name for llama.cpp probes"),
    function_slug: Optional[str] = typer.Option(
        None, help="Modal function slug suffix used in endpoint URL"
    ),
    timeout: int = typer.Option(60, help="Timeout seconds"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target deployment name"),
) -> None:
    """Check endpoint readiness."""
    compute_provider = ComputeProvider(provider)
    orch, username = _preflight(compute_provider)
    _print_banner()
    bt = BackendType(backend)
    target = _resolve_manage_target(bt, compute_provider, app_name, instance_name)
    url = server_url or os.environ.get("SERVER_URL")
    if (
        not url
        and compute_provider == ComputeProvider.MODAL
        and (function_slug or os.environ.get("MODAL_FUNCTION_SLUG"))
    ):
        url = ModalBackend.default_server_url(
            username,
            app_name=target.name,
            function_slug=function_slug or os.environ.get("MODAL_FUNCTION_SLUG"),
        )
    url = url or target.web_url
    if not url and compute_provider == ComputeProvider.MODAL:
        url = ModalBackend.default_server_url(
            username,
            app_name=target.name,
            function_slug=function_slug or os.environ.get("MODAL_FUNCTION_SLUG"),
        )
    if not url:
        typer.echo("Error: could not resolve the endpoint URL for this deployment.", err=True)
        raise typer.Exit(code=1)
    status_events = (
        orch.check_status(bt, url, timeout, served_model_name=served_model_name)
        if compute_provider == ComputeProvider.MODAL
        else orch.check_status(
            bt,
            url,
            timeout,
            served_model_name=served_model_name,
            provider=compute_provider,
            api_key=target.endpoint_api_key,
            pod_id=target.app_id,
        )
    )
    for event in status_events:
        _print_event(event)
        if isinstance(event, OperationCompleteEvent) and not event.success:
            raise typer.Exit(code=1)


@app.command()
def benchmark(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
    backend: BackendType = typer.Option(
        BackendType.LLAMACPP,
        help="Backend: llamacpp or vllm",
    ),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    model: Optional[str] = typer.Option(None, "--model", help="Served model name to benchmark"),
    function_slug: Optional[str] = typer.Option(
        None, help="Modal function slug suffix used in endpoint URL fallback"
    ),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target deployment name"),
    concurrency: str = typer.Option(
        "1,2,4,8,16",
        help="Comma or space separated concurrency sweep values",
    ),
    request_count: Optional[int] = typer.Option(
        None,
        min=1,
        help="Requests per concurrency run (default: max(24, concurrency * 4))",
    ),
    input_tokens: int = typer.Option(550, min=1, help="Synthetic input token mean"),
    output_tokens: int = typer.Option(256, min=1, help="Synthetic output token mean"),
    tokenizer: str = typer.Option("gpt2", help="AIPerf tokenizer identifier"),
    request_timeout_seconds: int = typer.Option(
        300,
        min=1,
        help="Per-request timeout passed to AIPerf",
    ),
    output_dir: Optional[str] = typer.Option(None, help="Benchmark run output directory"),
    aiperf_arg: Annotated[
        Optional[list[str]],
        typer.Option(
            "--aiperf-arg",
            help="Extra argument passed through to `aiperf profile`; repeat for multiple args",
        ),
    ] = None,
) -> None:
    """Benchmark a deployed OpenAI-compatible backend with AIPerf."""
    compute_provider = ComputeProvider(provider)
    orch, username = _preflight(compute_provider)
    _print_banner()
    bt = BackendType(backend)
    target = _resolve_manage_target(bt, compute_provider, app_name, instance_name)
    target_app_name = target.name
    rows = _load_visible_launchpad_rows(compute_provider) or []
    merge_cached_benchmark_connections(rows)
    target_row = next((row for row in rows if (row.name or "").strip() == target_app_name), None)
    if server_url is None and target_row is None and compute_provider == ComputeProvider.MODAL:
        server_url = ModalBackend.default_server_url(
            username,
            app_name=target_app_name,
            function_slug=function_slug or os.environ.get("MODAL_FUNCTION_SLUG"),
        )

    try:
        concurrency_values = parse_concurrency_values(concurrency)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    config = benchmark_config_from_endpoint(
        target_row,
        backend=bt,
        provider=compute_provider,
        username=username,
        app_name=target_app_name,
        instance_name=instance_name,
        server_url=server_url,
        model_name=model,
        concurrency=concurrency_values,
        request_count=request_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokenizer=tokenizer,
        request_timeout_seconds=request_timeout_seconds,
        output_dir=output_dir,
        aiperf_args=list(aiperf_arg or []),
        api_key=(target_row.endpoint_api_key if target_row else target.endpoint_api_key),
    )
    if not (config.server_url or "").strip():
        typer.echo("Error: could not resolve benchmark server URL.", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"Benchmark target: backend={bt.value} app={target_app_name} "
        f"url={config.server_url} model={config.model_name}"
    )
    for event in orch.benchmark(config):
        _print_event(event)
        _raise_on_failed_completion(event)


@app.command()
def logs(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
    backend: BackendType = typer.Option(
        BackendType.LLAMACPP,
        help="Backend: llamacpp or vllm",
    ),
    follow: bool = typer.Option(True, help="Follow log stream"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target deployment name"),
) -> None:
    """Show logs for a deployed backend."""
    compute_provider = ComputeProvider(provider)
    orch, _ = _preflight(compute_provider)
    _print_banner()
    bt = BackendType(backend)
    target = _resolve_manage_target(bt, compute_provider, app_name, instance_name)
    log_events = (
        orch.tail_logs(bt, follow, app_name=target.name)
        if compute_provider == ComputeProvider.MODAL
        else orch.tail_logs(
            bt,
            follow,
            app_name=target.name,
            provider=compute_provider,
            app_id=target.app_id,
        )
    )
    for event in log_events:
        _print_event(event)
        _raise_on_failed_completion(event)


@app.command()
def stop(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
    backend: BackendType = typer.Option(
        BackendType.LLAMACPP,
        help="Backend: llamacpp or vllm",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target deployment name"),
) -> None:
    """Stop a deployed backend app."""
    compute_provider = ComputeProvider(provider)
    orch, username = _preflight(compute_provider)
    _print_banner()
    bt = BackendType(backend)
    target = _resolve_manage_target(bt, compute_provider, app_name, instance_name)
    target_app_name = target.name
    if not yes:
        confirmed = typer.confirm(f"Stop deployed app '{target_app_name}'?")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=1)
    stop_succeeded = False
    stop_events = (
        orch.stop_app(bt, app_name=target_app_name)
        if compute_provider == ComputeProvider.MODAL
        else orch.stop_app(
            bt,
            app_name=target_app_name,
            provider=compute_provider,
            app_id=target.app_id,
        )
    )
    for event in stop_events:
        if isinstance(event, OperationCompleteEvent) and event.operation == OperationType.STOP:
            stop_succeeded = event.success
        _print_event(event)
        _raise_on_failed_completion(event)
    if stop_succeeded:
        remove_connection(target_app_name)
        _sync_opencode_cli(
            current_rows=_load_visible_launchpad_rows(compute_provider),
            remove_app_names=[target_app_name],
            prune_providers=(compute_provider,),
            username=username,
        )


@app.command()
def switch(
    provider: ComputeProvider = typer.Option(
        ComputeProvider.MODAL,
        help="Compute provider: modal or prime",
    ),
    backend: BackendType = typer.Option(
        BackendType.LLAMACPP,
        help="Backend: llamacpp or vllm",
    ),
    preset: Optional[str] = typer.Option(None, help="Preset name"),
    repo_id: Optional[str] = typer.Option(None, help="HF repo id"),
    quant: Optional[str] = typer.Option(None, help="Quant pattern"),
    revision: Optional[str] = typer.Option(None, help="HF revision"),
    model_name: Optional[str] = typer.Option(None, help="vLLM MODEL_NAME"),
    model_revision: Optional[str] = typer.Option(None, help="vLLM MODEL_REVISION"),
    served_model_name: Optional[str] = typer.Option(None, help="vLLM SERVED_MODEL_NAME"),
    fast_boot: Optional[bool] = typer.Option(None, help="vLLM FAST_BOOT"),
    n_gpu: Optional[int] = typer.Option(None, help="vLLM N_GPU"),
    gpu_type: Optional[str] = typer.Option(None, help="Prime GPU type filter"),
    gpu_count: Optional[int] = typer.Option(None, min=1, help="Prime GPU count filter"),
    prime_offer_id: Optional[str] = typer.Option(None, help="Exact Prime availability offer ID"),
    prime_region: Optional[str] = typer.Option(None, help="Prime region or country filter"),
    prime_disk_id: Optional[str] = typer.Option(None, help="Existing Prime disk ID to attach"),
    prime_disk: bool = typer.Option(
        True,
        "--prime-disk/--no-prime-disk",
        help="Auto-attach a persistent Prime disk so model weights survive redeploys",
    ),
    keep_failed_pod: bool = typer.Option(False, help="Keep a failed Prime pod"),
    allow_insecure_http: bool = typer.Option(
        False,
        help="Bypass Prime Tunnel and use a direct HTTP endpoint",
    ),
    trust_remote_code: Optional[bool] = typer.Option(
        None,
        help="vLLM TRUST_REMOTE_CODE (allow model custom code from Hugging Face)",
    ),
    reasoning_parser: Optional[str] = typer.Option(
        None,
        help="vLLM reasoning parser (e.g. qwen3, deepseek_r1, granite)",
    ),
    tool_call_parser: Optional[str] = typer.Option(
        None,
        help="vLLM tool call parser (e.g. hermes, qwen3_xml, llama3_json)",
    ),
    default_chat_template_kwargs: Optional[str] = typer.Option(
        None,
        help=(
            "vLLM default chat template kwargs JSON "
            "(e.g. '{\"enable_thinking\": false}' or '{\"thinking\": true}')"
        ),
    ),
    instance_name: Optional[str] = typer.Option(
        None, help="Instance name (auto-generated from model when omitted)"
    ),
    app_name: Optional[str] = typer.Option(None, help="Explicit deployment name override"),
    preload: bool = typer.Option(True, help="Preload weights"),
    redeploy: bool = typer.Option(True, help="Redeploy after switching"),
    do_warmup: bool = typer.Option(True, help="Verify readiness after redeploy"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    timeout: int = typer.Option(1800, help="Warmup timeout"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
) -> None:
    """Switch model and optionally redeploy."""
    compute_provider = ComputeProvider(provider)
    orch, username = _preflight(compute_provider)
    _print_banner()
    bt = BackendType(backend)

    from ..protocol.models import DeploymentConfig

    config = DeploymentConfig(
        backend=bt,
        provider=compute_provider,
        preset=preset,
        repo_id=repo_id,
        quant=quant,
        revision=revision,
        preload=preload,
        do_deploy=False,  # first run just switches model
        model_name=model_name,
        model_revision=model_revision,
        served_model_name=served_model_name,
        fast_boot=fast_boot,
        n_gpu=n_gpu,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        trust_remote_code=trust_remote_code,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        provider_options=(
            PrimeProviderOptions(
                offer_id=prime_offer_id,
                region=prime_region,
                disk_id=prime_disk_id,
                keep_failed_resource=keep_failed_pod,
                allow_insecure_http=allow_insecure_http,
                auto_disk=prime_disk,
            )
            if compute_provider == ComputeProvider.PRIME
            else None
        ),
    )
    model_hint = model_name if bt == BackendType.VLLM else (repo_id or preset)
    resolved_instance, resolved_app_name = _resolve_deploy_target(
        bt,
        model_hint=model_hint,
        instance_name=instance_name,
        app_name=app_name,
        provider=compute_provider,
    )
    config.instance_name = resolved_instance
    config.app_name = resolved_app_name

    if compute_provider == ComputeProvider.PRIME:
        if bt == BackendType.LLAMACPP and not repo_id:
            typer.echo("Prime llama.cpp requires --repo-id.", err=True)
            raise typer.Exit(code=2)
        if redeploy:
            config.do_deploy = True
            _deploy_and_maybe_warmup(
                orch,
                username=username,
                backend=bt,
                config=config,
                server_url=server_url,
                do_warmup=do_warmup,
                timeout=timeout,
                tail_logs=tail_logs,
            )
        else:
            typer.echo("No deploy performed. Prime model changes require --redeploy.")
        raise typer.Exit(code=0)

    if bt == BackendType.VLLM:
        if preload:
            typer.echo("Note: --preload is only used by llama.cpp and is ignored for vLLM.")
        if redeploy:
            config.do_deploy = True
            _deploy_and_maybe_warmup(
                orch,
                username=username,
                backend=bt,
                config=config,
                server_url=server_url,
                do_warmup=do_warmup,
                timeout=timeout,
                tail_logs=tail_logs,
            )
        else:
            typer.echo("No deploy performed. Use --redeploy to apply vLLM model changes.")
        raise typer.Exit(code=0)

    if not any([preset, repo_id]):
        typer.echo("Provide --preset or --repo-id to switch.", err=True)
        raise typer.Exit(code=1)

    # Run model switch (no deploy)
    for event in orch.deploy(config):
        _print_event(event)
        _raise_on_failed_completion(event)

    if redeploy:
        deploy_config = DeploymentConfig(backend=bt, do_deploy=True)
        deploy_config.function_slug = random_function_slug()
        deploy_config.instance_name = resolved_instance
        settings = ConfigStore().load()
        deploy_config.app_name = resolved_app_name
        env = ModalBackend.build_full_env(settings, deploy_config)
        code = ModalBackend.run_blocking(
            ModalBackend.build_deploy_command(bt, app_name=resolved_app_name), env=env
        )
        if code != 0:
            raise typer.Exit(code=code)
        final_sync_url = server_url or ModalBackend.default_server_url(
            username,
            app_name=resolved_app_name,
            function_slug=deploy_config.function_slug,
        )
        warmup_succeeded = False
        if do_warmup:
            for event in orch.warmup(
                bt,
                final_sync_url,
                timeout,
                tail_logs,
                app_name=resolved_app_name,
            ):
                if (
                    isinstance(event, OperationCompleteEvent)
                    and event.success
                    and event.operation == OperationType.WARMUP
                ):
                    warmup_succeeded = True
                    if isinstance(event.data, dict):
                        maybe_url = event.data.get("url")
                        if isinstance(maybe_url, str) and maybe_url.strip():
                            final_sync_url = maybe_url.strip()
                _print_event(event)
                _raise_on_failed_completion(event)
        if (do_warmup and warmup_succeeded) or not do_warmup:
            _sync_opencode_cli(
                target_app_name=resolved_app_name,
                target_url=final_sync_url,
                target_config=config,
                current_rows=_load_visible_launchpad_rows(compute_provider),
                prune_providers=(compute_provider,),
                username=username,
            )

    raise typer.Exit(code=0)


# -----------------------------------------------------------------------
# OpenCode
# -----------------------------------------------------------------------


@opencode_app.command("sync")
def opencode_sync(
    provider: Optional[ComputeProvider] = typer.Option(
        None,
        help="Limit to modal or prime. Default: every connected compute provider.",
    ),
    backend: Optional[BackendType] = typer.Option(
        None,
        help="Backend: llamacpp or vllm",
    ),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target deployment name"),
    dry_run: bool = typer.Option(False, help="Print the intended sync changes without writing files"),
) -> None:
    """Sync Launchpad-managed deployments into OpenCode config."""
    compute_provider = ComputeProvider(provider) if provider else None
    if compute_provider is not None:
        _, username = _preflight(compute_provider)
    else:
        try:
            username = ModalBackend.get_username() or ""
        except Exception:
            username = ""
    _print_banner()

    current_rows, listed_providers, fallback_notes = _load_rows_for_opencode_sync(
        compute_provider,
        persist_backfill=not dry_run,
    )
    for note in fallback_notes:
        typer.echo(note)
    target_app_name = (app_name or "").strip() or None
    if target_app_name is None and instance_name:
        if not backend:
            typer.echo("Specify --backend when using --instance-name.", err=True)
            raise typer.Exit(code=1)
        backend_type = BackendType(backend)
        wanted_instance = instance_name.strip()
        if compute_provider is not None:
            target_app_name = _resolve_manage_app_name(
                backend_type, None, wanted_instance, compute_provider
            )
        else:
            matches = [
                row
                for row in (current_rows or [])
                if row.backend == backend_type
                and (row.instance_name or "").strip() == wanted_instance
            ]
            if len(matches) == 1:
                target_app_name = matches[0].name
            elif len(matches) > 1:
                typer.echo(
                    "Multiple matching instances found. Specify --provider or --app-name.",
                    err=True,
                )
                raise typer.Exit(code=1)
            else:
                typer.echo(
                    f"OpenCode sync target instance '{wanted_instance}' was not found.",
                    err=True,
                )
                raise typer.Exit(code=1)

    if target_app_name:
        if current_rows is None:
            typer.echo("Could not load deployments to resolve the requested sync target.", err=True)
            raise typer.Exit(code=1)
        if not any((row.name or "").strip() == target_app_name for row in current_rows):
            typer.echo(f"OpenCode sync target '{target_app_name}' was not found.", err=True)
            raise typer.Exit(code=1)
    elif current_rows is None:
        typer.echo(
            "Could not load deployments for OpenCode sync. Check provider auth (`modal setup`, `prime login`) and retry.",
            err=True,
        )
        raise typer.Exit(code=1)

    _sync_opencode_cli(
        target_app_name=target_app_name,
        current_rows=current_rows,
        # An empty tuple is meaningful here: cached rows are useful for
        # upserting connections, but they are not evidence that a provider's
        # live deployment list was successfully queried. Passing ``None``
        # would make sync_opencode_config infer a prune scope from those
        # cached rows and could remove valid entries while the provider is
        # temporarily unavailable.
        prune_providers=listed_providers,
        username=username,
        dry_run=dry_run,
        fail_on_error=True,
    )


# -----------------------------------------------------------------------
# Artificial Analysis
# -----------------------------------------------------------------------


@aai_auth_app.command("login")
def aai_auth_login(
    api_key: Optional[str] = typer.Argument(
        None, help="Artificial Analysis API key (prompts when omitted)"
    ),
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Store the key without validating it"),
    ] = False,
) -> None:
    """Store an Artificial Analysis API key for llm-launchpad."""
    from ..core.artificial_analysis_auth import save_artificial_analysis_api_key
    from ..core.quick_deploy_refresh import get_artificial_analysis_auth_status

    key = (api_key or "").strip()
    if not key and sys.stdin.isatty():
        key = typer.prompt("Artificial Analysis API key", hide_input=True).strip()
    if not key:
        typer.echo("Error: an Artificial Analysis API key is required.", err=True)
        raise typer.Exit(code=1)

    if not no_verify:
        status = get_artificial_analysis_auth_status(api_key=key)
        if not status.authenticated:
            typer.echo(
                f"Error: {status.error or 'Artificial Analysis auth check failed'}",
                err=True,
            )
            raise typer.Exit(code=1)
    saved_path = save_artificial_analysis_api_key(key)
    if no_verify:
        typer.echo(f"Artificial Analysis API key stored at {saved_path} without verification.")
        return
    tier_suffix = f" ({status.tier} tier)" if status.tier else ""
    typer.echo(f"Artificial Analysis API key stored at {saved_path}. Authenticated{tier_suffix}.")


@aai_auth_app.command("status")
def aai_auth_status() -> None:
    """Show the Artificial Analysis API key source and auth status."""
    from ..core.artificial_analysis_auth import (
        AAI_API_KEY_ENV,
        load_saved_artificial_analysis_api_key,
    )
    from ..core.quick_deploy_refresh import get_artificial_analysis_auth_status

    env_key = os.getenv(AAI_API_KEY_ENV, "").strip()
    stored_key = load_saved_artificial_analysis_api_key()
    if env_key:
        source = "environment"
    elif stored_key:
        source = "stored"
    else:
        typer.echo(
            "No Artificial Analysis API key configured. "
            "Run `llm-launchpad aai-auth login` to store one.",
            err=True,
        )
        raise typer.Exit(code=1)
    status = get_artificial_analysis_auth_status()
    if not status.authenticated:
        typer.echo(
            f"Artificial Analysis auth check failed for the {source} key: "
            f"{status.error or 'unknown error'}",
            err=True,
        )
        raise typer.Exit(code=1)
    tier_suffix = f" ({status.tier} tier)" if status.tier else ""
    typer.echo(f"AAI API key source: {source}. Authenticated{tier_suffix}.")


@aai_auth_app.command("clear")
def aai_auth_clear() -> None:
    """Remove the stored Artificial Analysis API key."""
    from ..core.artificial_analysis_auth import clear_artificial_analysis_api_key

    if not clear_artificial_analysis_api_key():
        typer.echo("No stored Artificial Analysis API key to remove.", err=True)
        raise typer.Exit(code=1)
    typer.echo("Stored Artificial Analysis API key removed.")


# -----------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------


def main() -> None:
    """Console script entrypoint."""
    setup_logging()
    if len(sys.argv) == 1:
        sys.argv.append("tui")
    app()


if __name__ == "__main__":
    main()

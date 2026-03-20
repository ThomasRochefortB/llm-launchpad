"""CLI dispatcher: the default command launches the TUI, with headless
commands available for automation.

This replaces the monolithic ``llm_launchpad.cli`` module while keeping
all non-interactive commands available for automation.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Optional

import typer

try:
    from rich.console import Console
except Exception:
    Console = None  # type: ignore

from ..core.backend import ModalBackend
from ..core.config import ConfigStore
from ..core.modal_gpu import fetch_modal_gpu_types
from ..core.naming import (
    auto_instance_name_for_backend,
    build_app_name,
    legacy_app_name,
    random_function_slug,
    slugify_instance_name,
)
from ..core.opencode import resolve_connection_for_app, sync_opencode_config, visible_launchpad_rows
from ..core.orchestrator import Orchestrator
from ..protocol.enums import BackendType
from ..protocol.enums import OperationType
from ..protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent, StateChangeEvent
from ..protocol.models import DeploymentConfig, EndpointInfo

app = typer.Typer(
    help="llm-launchpad CLI - configure and deploy LLM backends on Modal.",
    invoke_without_command=True,
)
opencode_app = typer.Typer(help="OpenCode integration commands.")
app.add_typer(opencode_app, name="opencode")


@app.callback()
def _default(ctx: typer.Context) -> None:
    """Launch the TUI when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        tui()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _print_event(event: object) -> None:
    """Print a protocol event to stdout/stderr for headless mode."""
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


def _preflight() -> tuple[Orchestrator, str]:
    orch = Orchestrator()
    ok, username, err = orch.preflight()
    _ensure(ok, err)
    return orch, username


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _is_ssh_session() -> bool:
    return bool(os.getenv("SSH_CONNECTION") or os.getenv("SSH_TTY"))


def _likely_remote_clipboard_supported() -> bool:
    """Best-effort guess for whether remote clipboard writes can reach the local terminal."""
    marker_values = [
        os.getenv("TERM", ""),
        os.getenv("TERM_PROGRAM", ""),
        os.getenv("LC_TERMINAL", ""),
    ]
    normalized_markers = " ".join(value.strip().lower() for value in marker_values if value)

    if any(
        os.getenv(name)
        for name in (
            "ITERM_SESSION_ID",
            "KITTY_WINDOW_ID",
            "KITTY_PUBLIC_KEY",
            "WEZTERM_EXECUTABLE",
            "GHOSTTY_RESOURCES_DIR",
        )
    ):
        return True

    if any(marker in normalized_markers for marker in ("iterm", "wezterm", "kitty", "ghostty", "vscode")):
        return True

    if "apple_terminal" in normalized_markers or "apple terminal" in normalized_markers:
        return False

    # Over SSH, default to terminal-native selection unless we recognize a terminal
    # that is likely to accept clipboard escape sequences from the remote app.
    return False


def _default_tui_mouse_enabled() -> bool:
    default = True
    if _is_ssh_session():
        default = _likely_remote_clipboard_supported()
    return _parse_bool_env("LLM_LAUNCHPAD_TUI_MOUSE", default)


def _ensure_tui_runtime() -> None:
    """Fail fast with visible CLI errors before handing off to Textual."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo(
            "Error: llm-launchpad TUI requires an interactive terminal (TTY).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not ModalBackend.is_cli_available():
        typer.echo(
            "Error: Modal CLI not found in PATH. Install it and run `modal setup`.",
            err=True,
        )
        raise typer.Exit(code=1)


def _resolve_deploy_target(
    backend: BackendType,
    model_hint: Optional[str],
    instance_name: Optional[str],
    app_name: Optional[str],
) -> tuple[str, str]:
    """Resolve (instance_name, app_name) for deployment."""
    explicit_app = (app_name or "").strip()
    if explicit_app:
        explicit_instance = (instance_name or "").strip() or slugify_instance_name(explicit_app)
        return explicit_instance, explicit_app

    explicit_instance = (instance_name or "").strip()
    if explicit_instance:
        slug = slugify_instance_name(explicit_instance)
        return slug, build_app_name(backend, slug)

    auto = auto_instance_name_for_backend(backend, model_hint)
    return auto, build_app_name(backend, auto)


def _backend_instances(backend: BackendType) -> list[EndpointInfo]:
    rows = ModalBackend.list_apps() or []
    return [row for row in rows if row.backend == backend]


def _resolve_manage_app_name(
    backend: BackendType,
    app_name: Optional[str],
    instance_name: Optional[str],
) -> str:
    explicit_app = (app_name or "").strip()
    if explicit_app:
        return explicit_app
    explicit_instance = (instance_name or "").strip()
    if explicit_instance:
        return build_app_name(backend, explicit_instance)

    matches = _backend_instances(backend)
    if len(matches) == 1:
        return matches[0].name
    if len(matches) > 1:
        typer.echo(
            f"Multiple '{backend.value}' instances found. "
            "Specify --instance-name or --app-name. Use `llm-launchpad list` to inspect instances.",
            err=True,
        )
        raise typer.Exit(code=1)
    return legacy_app_name(backend)


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


def _load_visible_launchpad_rows() -> list[EndpointInfo] | None:
    rows = ModalBackend.list_apps()
    if rows is None:
        return None
    return visible_launchpad_rows(rows)


def _sync_opencode_cli(
    *,
    target_app_name: Optional[str] = None,
    target_url: Optional[str] = None,
    target_config: Optional[DeploymentConfig] = None,
    current_rows: list[EndpointInfo] | None = None,
    remove_app_names: list[str] | None = None,
    username: str = "",
    dry_run: bool = False,
    fail_on_error: bool = False,
) -> None:
    target = None
    if target_app_name:
        target = resolve_connection_for_app(
            target_app_name,
            rows=current_rows,
            username=username,
            fallback_config=target_config,
            fallback_server_url=target_url,
        )
    try:
        result = sync_opencode_config(
            target=target,
            current_rows=current_rows,
            remove_app_names=remove_app_names,
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
    resolved_mouse = _default_tui_mouse_enabled() if mouse is None else mouse
    app_instance = TuiApp(mouse_enabled=resolved_mouse)
    try:
        app_instance.run(mouse=resolved_mouse)
    finally:
        from ..core.backend import ModalBackend
        ModalBackend.terminate_all()


# -----------------------------------------------------------------------
# Headless commands
# -----------------------------------------------------------------------


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


@app.command()
def deploy(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    do_warmup: bool = typer.Option(False, help="Verify readiness after deploy"),
    model_name: Optional[str] = typer.Option(None, help="vLLM MODEL_NAME"),
    model_revision: Optional[str] = typer.Option(None, help="vLLM MODEL_REVISION"),
    served_model_name: Optional[str] = typer.Option(None, help="vLLM SERVED_MODEL_NAME"),
    fast_boot: Optional[bool] = typer.Option(None, help="vLLM FAST_BOOT"),
    n_gpu: Optional[int] = typer.Option(None, help="vLLM N_GPU"),
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
    app_name: Optional[str] = typer.Option(None, help="Explicit Modal app name override"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    timeout: int = typer.Option(1800, help="Warmup timeout seconds"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
) -> None:
    """Deploy the server to Modal."""
    orch, username = _preflight()
    _print_banner()

    from ..protocol.models import DeploymentConfig

    bt = BackendType(backend)
    model_hint = model_name if bt == BackendType.VLLM else None
    resolved_instance, resolved_app_name = _resolve_deploy_target(
        bt, model_hint=model_hint, instance_name=instance_name, app_name=app_name
    )

    config = DeploymentConfig(
        backend=bt,
        do_deploy=True,
        model_name=model_name,
        model_revision=model_revision,
        served_model_name=served_model_name,
        fast_boot=fast_boot,
        n_gpu=n_gpu,
        trust_remote_code=trust_remote_code,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        instance_name=resolved_instance,
        app_name=resolved_app_name,
    )
    typer.echo(
        f"Deploy target: backend={bt.value} instance={resolved_instance} app={resolved_app_name}"
    )

    deployed_web_url: Optional[str] = None
    deploy_succeeded = False
    for event in orch.deploy(config):
        if isinstance(event, LogEvent):
            maybe_url = ModalBackend.extract_modal_web_url(event.line)
            if maybe_url:
                deployed_web_url = maybe_url
        elif isinstance(event, OperationCompleteEvent) and event.operation == OperationType.DEPLOY:
            deploy_succeeded = event.success
        _print_event(event)
        _raise_on_failed_completion(event)

    warmup_succeeded = False
    final_sync_url: Optional[str] = None
    if do_warmup:
        url = server_url or deployed_web_url or ModalBackend.default_server_url(
            username,
            app_name=resolved_app_name,
            function_slug=config.function_slug,
        )
        for event in orch.warmup(
            bt,
            url,
            timeout,
            tail_logs,
            app_name=resolved_app_name,
            served_model_name=config.served_model_name,
        ):
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
            _print_event(event)
            _raise_on_failed_completion(event)
    elif deploy_succeeded:
        final_sync_url = server_url or deployed_web_url or ModalBackend.default_server_url(
            username,
            app_name=resolved_app_name,
            function_slug=config.function_slug,
        )

    if (do_warmup and warmup_succeeded and final_sync_url) or (not do_warmup and final_sync_url):
        _sync_opencode_cli(
            target_app_name=resolved_app_name,
            target_url=final_sync_url,
            target_config=config,
            current_rows=_load_visible_launchpad_rows(),
            username=username,
        )

    raise typer.Exit(code=0)


@app.command()
def warmup(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    served_model_name: Optional[str] = typer.Option(None, help="Served model name for llama.cpp probes"),
    function_slug: Optional[str] = typer.Option(
        None, help="Modal function slug suffix used in endpoint URL"
    ),
    timeout: int = typer.Option(1800, help="Seconds to wait"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target Modal app name"),
) -> None:
    """Cold start the container by probing the server."""
    orch, username = _preflight()
    _print_banner()
    bt = BackendType(backend)
    target_app_name = _resolve_manage_app_name(bt, app_name, instance_name)
    url = (
        server_url
        or os.environ.get("SERVER_URL")
        or ModalBackend.default_server_url(
            username,
            app_name=target_app_name,
            function_slug=function_slug or os.environ.get("MODAL_FUNCTION_SLUG"),
        )
    )
    for event in orch.warmup(
        bt,
        url,
        timeout,
        tail_logs,
        app_name=target_app_name,
        served_model_name=served_model_name,
    ):
        _print_event(event)
        _raise_on_failed_completion(event)


@app.command("list")
def list_apps() -> None:
    """List launchpad Modal apps."""
    orch, _ = _preflight()
    _print_banner()
    visible_rows: list[EndpointInfo] | None = None
    for event in orch.list_deployments():
        if (
            isinstance(event, OperationCompleteEvent)
            and event.success
            and isinstance(event.data, list)
        ):
            visible_rows = [row for row in event.data if isinstance(row, EndpointInfo)]
        _print_event(event)
    _sync_opencode_cli(current_rows=visible_rows)


@app.command()
def status(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    served_model_name: Optional[str] = typer.Option(None, help="Served model name for llama.cpp probes"),
    function_slug: Optional[str] = typer.Option(
        None, help="Modal function slug suffix used in endpoint URL"
    ),
    timeout: int = typer.Option(60, help="Timeout seconds"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target Modal app name"),
) -> None:
    """Check endpoint readiness."""
    orch, username = _preflight()
    _print_banner()
    bt = BackendType(backend)
    target_app_name = _resolve_manage_app_name(bt, app_name, instance_name)
    url = (
        server_url
        or os.environ.get("SERVER_URL")
        or ModalBackend.default_server_url(
            username,
            app_name=target_app_name,
            function_slug=function_slug or os.environ.get("MODAL_FUNCTION_SLUG"),
        )
    )
    for event in orch.check_status(bt, url, timeout, served_model_name=served_model_name):
        _print_event(event)
        if isinstance(event, OperationCompleteEvent) and not event.success:
            raise typer.Exit(code=1)


@app.command()
def logs(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    follow: bool = typer.Option(True, help="Follow log stream"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target Modal app name"),
) -> None:
    """Show logs for a deployed backend."""
    orch, _ = _preflight()
    _print_banner()
    bt = BackendType(backend)
    target_app_name = _resolve_manage_app_name(bt, app_name, instance_name)
    for event in orch.tail_logs(bt, follow, app_name=target_app_name):
        _print_event(event)


@app.command()
def stop(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target Modal app name"),
) -> None:
    """Stop a deployed backend app."""
    orch, _ = _preflight()
    _print_banner()
    bt = BackendType(backend)
    target_app_name = _resolve_manage_app_name(bt, app_name, instance_name)
    if not yes:
        confirmed = typer.confirm(f"Stop deployed app '{target_app_name}'?")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=1)
    stop_succeeded = False
    for event in orch.stop_app(bt, app_name=target_app_name):
        if isinstance(event, OperationCompleteEvent) and event.operation == OperationType.STOP:
            stop_succeeded = event.success
        _print_event(event)
    if stop_succeeded:
        _sync_opencode_cli(
            current_rows=_load_visible_launchpad_rows(),
            remove_app_names=[target_app_name],
        )


@app.command()
def switch(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    preset: Optional[str] = typer.Option(None, help="Preset name"),
    repo_id: Optional[str] = typer.Option(None, help="HF repo id"),
    quant: Optional[str] = typer.Option(None, help="Quant pattern"),
    revision: Optional[str] = typer.Option(None, help="HF revision"),
    model_name: Optional[str] = typer.Option(None, help="vLLM MODEL_NAME"),
    model_revision: Optional[str] = typer.Option(None, help="vLLM MODEL_REVISION"),
    served_model_name: Optional[str] = typer.Option(None, help="vLLM SERVED_MODEL_NAME"),
    fast_boot: Optional[bool] = typer.Option(None, help="vLLM FAST_BOOT"),
    n_gpu: Optional[int] = typer.Option(None, help="vLLM N_GPU"),
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
    app_name: Optional[str] = typer.Option(None, help="Explicit Modal app name override"),
    preload: bool = typer.Option(True, help="Preload weights"),
    redeploy: bool = typer.Option(True, help="Redeploy after switching"),
    do_warmup: bool = typer.Option(True, help="Verify readiness after redeploy"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    timeout: int = typer.Option(1800, help="Warmup timeout"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
) -> None:
    """Switch model and optionally redeploy."""
    orch, username = _preflight()
    _print_banner()
    bt = BackendType(backend)

    from ..protocol.models import DeploymentConfig

    config = DeploymentConfig(
        backend=bt,
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
        trust_remote_code=trust_remote_code,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )
    model_hint = model_name if bt == BackendType.VLLM else (repo_id or preset)
    resolved_instance, resolved_app_name = _resolve_deploy_target(
        bt, model_hint=model_hint, instance_name=instance_name, app_name=app_name
    )
    config.instance_name = resolved_instance
    config.app_name = resolved_app_name

    if bt == BackendType.VLLM:
        if preload:
            typer.echo("Note: --preload is only used by llama.cpp and is ignored for vLLM.")
        if redeploy:
            deployed_web_url: Optional[str] = None
            deploy_succeeded = False
            config.do_deploy = True
            for event in orch.deploy(config):
                if isinstance(event, LogEvent):
                    maybe_url = ModalBackend.extract_modal_web_url(event.line)
                    if maybe_url:
                        deployed_web_url = maybe_url
                elif isinstance(event, OperationCompleteEvent) and event.operation == OperationType.DEPLOY:
                    deploy_succeeded = event.success
                _print_event(event)
                _raise_on_failed_completion(event)
            warmup_succeeded = False
            final_sync_url: Optional[str] = None
            if do_warmup:
                url = server_url or deployed_web_url or ModalBackend.default_server_url(
                    username,
                    app_name=resolved_app_name,
                    function_slug=config.function_slug,
                )
                for event in orch.warmup(
                    bt,
                    url,
                    timeout,
                    tail_logs,
                    app_name=resolved_app_name,
                    served_model_name=config.served_model_name,
                ):
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
                    _print_event(event)
                    _raise_on_failed_completion(event)
            elif deploy_succeeded:
                final_sync_url = server_url or deployed_web_url or ModalBackend.default_server_url(
                    username,
                    app_name=resolved_app_name,
                    function_slug=config.function_slug,
                )
            if (do_warmup and warmup_succeeded and final_sync_url) or (not do_warmup and final_sync_url):
                _sync_opencode_cli(
                    target_app_name=resolved_app_name,
                    target_url=final_sync_url,
                    target_config=config,
                    current_rows=_load_visible_launchpad_rows(),
                    username=username,
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
                current_rows=_load_visible_launchpad_rows(),
                username=username,
            )

    raise typer.Exit(code=0)


# -----------------------------------------------------------------------
# OpenCode
# -----------------------------------------------------------------------


@opencode_app.command("sync")
def opencode_sync(
    backend: Optional[str] = typer.Option(None, help="Backend: llamacpp or vllm"),
    instance_name: Optional[str] = typer.Option(None, help="Target instance name"),
    app_name: Optional[str] = typer.Option(None, help="Target Modal app name"),
    dry_run: bool = typer.Option(False, help="Print the intended sync changes without writing files"),
) -> None:
    """Sync Launchpad-managed deployments into OpenCode config."""
    _, username = _preflight()
    _print_banner()

    target_app_name = (app_name or "").strip() or None
    if target_app_name is None and instance_name:
        if not backend:
            typer.echo("Specify --backend when using --instance-name.", err=True)
            raise typer.Exit(code=1)
        target_app_name = _resolve_manage_app_name(BackendType(backend), None, instance_name)

    current_rows = _load_visible_launchpad_rows()
    if target_app_name:
        if current_rows is None:
            typer.echo("Could not load Modal app list to resolve the requested sync target.", err=True)
            raise typer.Exit(code=1)
        if not any((row.name or "").strip() == target_app_name for row in current_rows):
            typer.echo(f"OpenCode sync target '{target_app_name}' was not found.", err=True)
            raise typer.Exit(code=1)

    _sync_opencode_cli(
        target_app_name=target_app_name,
        current_rows=current_rows,
        username=username,
        dry_run=dry_run,
        fail_on_error=True,
    )


# -----------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------


def main() -> None:
    """Console script entrypoint."""
    if len(sys.argv) == 1:
        sys.argv.append("tui")
    app()


if __name__ == "__main__":
    main()

"""CLI dispatcher: ``wizard`` launches TUI, other commands route through Core.

This replaces the monolithic ``llm_launchpad.cli`` module while keeping
all non-interactive commands available for automation.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import typer

try:
    from rich.console import Console
except Exception:
    Console = None  # type: ignore

from ..core.backend import ModalBackend
from ..core.config import ConfigStore
from ..core.modal_gpu import fetch_modal_gpu_types
from ..core.orchestrator import Orchestrator
from ..protocol.enums import BackendType, OperationType
from ..protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent, StateChangeEvent

app = typer.Typer(
    help="llm-launchpad CLI - configure and deploy LLM backends on Modal.",
    invoke_without_command=True,
)


@app.callback()
def _default(ctx: typer.Context) -> None:
    """Launch the TUI wizard when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        wizard()


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


def _preflight() -> tuple[Orchestrator, str]:
    orch = Orchestrator()
    ok, username, err = orch.preflight()
    _ensure(ok, err)
    return orch, username


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


# -----------------------------------------------------------------------
# Wizard (TUI)
# -----------------------------------------------------------------------


@app.command()
def wizard() -> None:
    """Interactive TUI wizard for deploying and managing LLM backends."""
    try:
        from ..tui.app import WizardApp
    except ImportError as exc:
        typer.echo(
            f"Error: Textual is required for the wizard. Install with: pip install textual\n({exc})",
            err=True,
        )
        raise typer.Exit(code=1)

    tui = WizardApp()
    tui.run()


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
    do_warmup: bool = typer.Option(False, help="Warm up after deploy"),
    model_name: Optional[str] = typer.Option(None, help="vLLM MODEL_NAME"),
    model_revision: Optional[str] = typer.Option(None, help="vLLM MODEL_REVISION"),
    served_model_name: Optional[str] = typer.Option(None, help="vLLM SERVED_MODEL_NAME"),
    fast_boot: Optional[bool] = typer.Option(None, help="vLLM FAST_BOOT"),
    n_gpu: Optional[int] = typer.Option(None, help="vLLM N_GPU"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    timeout: int = typer.Option(1800, help="Warmup timeout seconds"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
) -> None:
    """Deploy the server to Modal."""
    orch, username = _preflight()
    _print_banner()

    from ..protocol.models import DeploymentConfig

    bt = BackendType(backend)
    config = DeploymentConfig(
        backend=bt,
        do_deploy=True,
        model_name=model_name,
        model_revision=model_revision,
        served_model_name=served_model_name,
        fast_boot=fast_boot,
        n_gpu=n_gpu,
    )

    for event in orch.deploy(config):
        _print_event(event)
        if isinstance(event, OperationCompleteEvent) and not event.success:
            raise typer.Exit(code=event.exit_code or 1)

    if do_warmup:
        url = server_url or ModalBackend.default_server_url(username, bt)
        for event in orch.warmup(bt, url, timeout, tail_logs):
            _print_event(event)

    raise typer.Exit(code=0)


@app.command()
def warmup(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    timeout: int = typer.Option(1800, help="Seconds to wait"),
    tail_logs: bool = typer.Option(True, help="Tail logs during warmup"),
) -> None:
    """Cold start the container by probing the server."""
    orch, username = _preflight()
    _print_banner()
    bt = BackendType(backend)
    url = server_url or os.environ.get("SERVER_URL") or ModalBackend.default_server_url(username, bt)
    for event in orch.warmup(bt, url, timeout, tail_logs):
        _print_event(event)


@app.command("list")
def list_apps() -> None:
    """List launchpad Modal apps."""
    orch, _ = _preflight()
    _print_banner()
    for event in orch.list_deployments():
        _print_event(event)


@app.command()
def status(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    server_url: Optional[str] = typer.Option(None, help="Deployed web URL"),
    timeout: int = typer.Option(60, help="Timeout seconds"),
) -> None:
    """Check endpoint readiness."""
    orch, username = _preflight()
    _print_banner()
    bt = BackendType(backend)
    url = server_url or os.environ.get("SERVER_URL") or ModalBackend.default_server_url(username, bt)
    for event in orch.check_status(bt, url, timeout):
        _print_event(event)
        if isinstance(event, OperationCompleteEvent) and not event.success:
            raise typer.Exit(code=1)


@app.command()
def logs(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    follow: bool = typer.Option(True, help="Follow log stream"),
) -> None:
    """Show logs for a deployed backend."""
    orch, _ = _preflight()
    _print_banner()
    bt = BackendType(backend)
    for event in orch.tail_logs(bt, follow):
        _print_event(event)


@app.command()
def stop(
    backend: str = typer.Option("llamacpp", help="Backend: llamacpp or vllm"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """Stop a deployed backend app."""
    orch, _ = _preflight()
    _print_banner()
    bt = BackendType(backend)
    if not yes:
        confirmed = typer.confirm(f"Stop deployed app '{bt.app_name}'?")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=1)
    for event in orch.stop_app(bt):
        _print_event(event)


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
    preload: bool = typer.Option(True, help="Preload weights"),
    redeploy: bool = typer.Option(True, help="Redeploy after switching"),
    do_warmup: bool = typer.Option(True, help="Warm up after redeploy"),
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
    )

    if bt == BackendType.VLLM:
        if preload:
            typer.echo("Note: --preload is only used by llama.cpp and is ignored for vLLM.")
        if redeploy:
            config.do_deploy = True
            for event in orch.deploy(config):
                _print_event(event)
                if isinstance(event, OperationCompleteEvent) and not event.success:
                    raise typer.Exit(code=event.exit_code or 1)
            if do_warmup:
                url = server_url or ModalBackend.default_server_url(username, bt)
                for event in orch.warmup(bt, url, timeout, tail_logs):
                    _print_event(event)
        else:
            typer.echo("No deploy performed. Use --redeploy to apply vLLM model changes.")
        raise typer.Exit(code=0)

    if not any([preset, repo_id]):
        typer.echo("Provide --preset or --repo-id to switch.", err=True)
        raise typer.Exit(code=1)

    # Run model switch (no deploy)
    for event in orch.deploy(config):
        _print_event(event)
        if isinstance(event, OperationCompleteEvent) and not event.success:
            raise typer.Exit(code=event.exit_code or 1)

    if redeploy:
        deploy_config = DeploymentConfig(backend=bt, do_deploy=True)
        settings = ConfigStore().load()
        env = settings.to_env()
        code = ModalBackend.run_blocking(ModalBackend.build_deploy_command(bt), env=env)
        if code != 0:
            raise typer.Exit(code=code)
        if do_warmup:
            url = server_url or ModalBackend.default_server_url(username, bt)
            for event in orch.warmup(bt, url, timeout, tail_logs):
                _print_event(event)

    raise typer.Exit(code=0)


# -----------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------


def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()

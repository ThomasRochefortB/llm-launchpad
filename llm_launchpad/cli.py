from __future__ import annotations

import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional

import typer

try:
    # Pretty output if rich is available
    from rich import box
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    box = None  # type: ignore

from .presets import PRESETS


app = typer.Typer(help="llm-launchpad CLI - configure, preload, and deploy llama.cpp on Modal.")


def _ensure_modal_cli_available() -> None:
    if shutil.which("modal") is None:
        typer.echo("Error: Modal CLI not found. Install with: pip install modal && modal setup", err=True)
        raise typer.Exit(code=1)


def _print_banner() -> None:
    """Render a simple banner using rich if available, else plain text."""
    if not Console:
        typer.echo("llm-launchpad")
        return
    try:
        # Lazy imports here to keep rich optional
        from rich.panel import Panel  # type: ignore
        from importlib.metadata import version  # type: ignore
    except Exception:
        typer.echo("llm-launchpad")
        return

    try:
        pkg_version = version("llm-launchpad")
        subtitle = f"v{pkg_version}  •  Modal + llama.cpp"
    except Exception:
        subtitle = "Modal + llama.cpp"

    console = Console()
    console.print(
        Panel.fit(
            "🧠  llm-launchpad",
            subtitle=subtitle,
            border_style="cyan",
        )
    )


@contextmanager
def _app_screen():
    """Enter an alternate screen to give an app-like experience, then restore."""
    if Console:
        try:
            console = Console()
            with console.screen():
                yield
                return
        except Exception:
            pass
    # Fallback: ANSI alt screen
    sys.stdout.write("\033[?1049h\033[H")
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write("\033[?1049l")
        sys.stdout.flush()


def _build_modal_run_args(
    preset: Optional[str],
    repo_id: Optional[str],
    quant: Optional[str],
    revision: Optional[str],
    preload: bool,
    deploy: bool,
    server_args: Optional[str],
    host: Optional[str],
    port: Optional[int],
    n_gpu_layers: Optional[int],
) -> List[str]:
    args: List[str] = [
        "modal",
        "run",
        "modal-llamacpp.py::main",
    ]

    if preset:
        args += ["--preset", preset]
    if repo_id:
        args += ["--repo-id", repo_id]
    if quant:
        args += ["--quant", quant]
    if revision:
        args += ["--revision", revision]
    if preload:
        args += ["--preload"]
    else:
        args += ["--no-preload"]
    if deploy:
        args += ["--deploy"]
    if server_args:
        args += ["--server_args", server_args]
    if host:
        args += ["--host", host]
    if port is not None:
        args += ["--port", str(port)]
    if n_gpu_layers is not None:
        args += ["--n_gpu_layers", str(n_gpu_layers)]

    return args


def _run_command(command: List[str]) -> int:
    process = subprocess.run(command, text=True)
    return process.returncode


@app.command()
def wizard() -> None:
    """Interactive setup: choose a preset or custom model, preload, and deploy."""
    _ensure_modal_cli_available()
    # Variables to share across alternate screen scope
    preset: Optional[str] = None
    repo_id: Optional[str] = None
    quant: Optional[str] = None
    revision: Optional[str] = None
    preload: bool = True
    deploy: bool = False
    server_args: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    n_gpu_layers: Optional[int] = None

    with _app_screen():
        _print_banner()

        # Import here so non-interactive commands don't require this dependency at import time
        try:
            from InquirerPy import inquirer  # type: ignore
        except Exception:
            typer.echo("Error: InquirerPy is required for interactive mode. Install with: uv pip install InquirerPy", err=True)
            raise typer.Exit(code=1)

        preset_names = list(PRESETS.keys())
        choices = []
        for name in preset_names:
            entry = PRESETS[name]
            label = f"{name}  →  {entry.get('repo_id','')}  [{entry.get('quant','')}]"
            choices.append({"name": label, "value": name})
        choices.append({"name": "custom (enter repo-id and quant)", "value": "__custom__"})

        selection = inquirer.select(
            message="Choose a preset",
            choices=choices,
            default=preset_names[0] if preset_names else "__custom__",
            cycle=True,
        ).execute()

        if selection == "__custom__":
            repo_id = inquirer.text(
                message="Hugging Face repo-id (e.g., Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)",
                validate=lambda x: len(x.strip()) > 0,
                invalid_message="Repo-id is required.",
            ).execute()
            quant = inquirer.text(message="Quant pattern (e.g., Q4_K_M)", default="Q4_K_M").execute()
            rev_in = inquirer.text(message="HF revision (optional)", default="").execute()
            revision = rev_in or None
        else:
            preset = str(selection)

        preload = inquirer.confirm(message="Preload/download weights now?", default=True).execute()
        deploy = inquirer.confirm(message="Deploy the server when finished?", default=True).execute()

        tweak = inquirer.confirm(message="Advanced options (server args, host/port, n_gpu_layers)?", default=False).execute()
        if tweak:
            server_args_in = inquirer.text(message="Server args (e.g., --ctx-size 65536 --threads 24)", default="").execute()
            server_args = server_args_in or None
            host_in = inquirer.text(message="Host", default="0.0.0.0").execute()
            host = host_in or None
            port_in = inquirer.text(message="Port", default="8080").execute()
            try:
                port = int(port_in)
            except ValueError:
                typer.echo("Port must be an integer.", err=True)
                raise typer.Exit(code=1)
            n_gpu_layers_in = inquirer.text(message="n_gpu_layers (press Enter for auto)", default="").execute()
            n_gpu_layers = int(n_gpu_layers_in) if n_gpu_layers_in.strip() else None

    args = _build_modal_run_args(
        preset=preset,
        repo_id=repo_id,
        quant=quant,
        revision=revision,
        preload=preload,
        deploy=deploy,
        server_args=server_args,
        host=host,
        port=port,
        n_gpu_layers=n_gpu_layers,
    )

    typer.echo("\nRunning:")
    typer.echo(" "+" ".join(args))
    code = _run_command(args)
    if code != 0:
        raise typer.Exit(code=code)

    if not deploy:
        typer.echo("\nNext: Deploy with 'modal deploy modal-llamacpp.py' when ready.")


@app.command()
def deploy() -> None:
    """Deploy the server to Modal."""
    _ensure_modal_cli_available()
    _print_banner()
    code = _run_command(["modal", "deploy", "modal-llamacpp.py"])
    raise typer.Exit(code=code)


@app.command()
def switch(
    preset: Optional[str] = typer.Option(None, help="Preset name to switch to"),
    repo_id: Optional[str] = typer.Option(None, help="Hugging Face repo id"),
    quant: Optional[str] = typer.Option(None, help="Quantization pattern (e.g., Q4_K_M)"),
    revision: Optional[str] = typer.Option(None, help="HF revision"),
    preload: bool = typer.Option(True, help="Preload/download weights immediately"),
    redeploy: bool = typer.Option(True, help="Redeploy after switching"),
) -> None:
    """Switch model (preset or custom), optionally preload and redeploy."""
    _ensure_modal_cli_available()
    _print_banner()

    if not any([preset, repo_id]):
        typer.echo("Provide --preset or --repo-id to switch.", err=True)
        raise typer.Exit(code=1)

    args = _build_modal_run_args(
        preset=preset,
        repo_id=repo_id,
        quant=quant,
        revision=revision,
        preload=preload,
        deploy=False,
        server_args=None,
        host=None,
        port=None,
        n_gpu_layers=None,
    )
    code = _run_command(args)
    if code != 0:
        raise typer.Exit(code=code)

    if redeploy:
        code = _run_command(["modal", "deploy", "modal-llamacpp.py"])
        raise typer.Exit(code=code)


def main() -> None:  # console script entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()



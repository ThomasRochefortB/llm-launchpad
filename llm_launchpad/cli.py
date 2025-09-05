from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

import typer

try:
    # Pretty output if rich is available
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore

from .presets import PRESETS


app = typer.Typer(help="llm-launchpad CLI - configure, preload, and deploy llama.cpp on Modal.")


# --- Settings persistence (local)
SETTINGS_DIR = Path.home() / ".llm_launchpad"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

# --- ASCII banner placeholder (replace the contents to customize)
ASCII_BANNER = r"""
                                                                 
▗▖   ▗▖   ▗▄ ▄▖     ▗▖     ▄  ▗▖ ▗▖▗▄ ▗▖  ▄▄ ▗▖ ▗▖▗▄▄▖   ▄  ▗▄▄  
▐▌   ▐▌   ▐█ █▌     ▐▌    ▐█▌ ▐▌ ▐▌▐█ ▐▌ █▀▀▌▐▌ ▐▌▐▛▀▜▖ ▐█▌ ▐▛▀█ 
▐▌   ▐▌   ▐███▌     ▐▌    ▐█▌ ▐▌ ▐▌▐▛▌▐▌▐▛   ▐▌ ▐▌▐▌ ▐▌ ▐█▌ ▐▌ ▐▌
▐▌   ▐▌   ▐▌█▐▌     ▐▌    █ █ ▐▌ ▐▌▐▌█▐▌▐▌   ▐███▌▐██▛  █ █ ▐▌ ▐▌
▐▌   ▐▌   ▐▌▀▐▌     ▐▌    ███ ▐▌ ▐▌▐▌▐▟▌▐▙   ▐▌ ▐▌▐▌    ███ ▐▌ ▐▌
▐▙▄▄▖▐▙▄▄▖▐▌ ▐▌     ▐▙▄▄▖▗█ █▖▝█▄█▘▐▌ █▌ █▄▄▌▐▌ ▐▌▐▌   ▗█ █▖▐▙▄█ 
▝▀▀▀▘▝▀▀▀▘▝▘ ▝▘     ▝▀▀▀▘▝▘ ▝▘ ▝▀▘ ▝▘ ▀▘  ▀▀ ▝▘ ▝▘▝▘   ▝▘ ▝▘▝▀▀  
                                                                 
               ▀▀▀▀▀        
----------------------------------------------------------------                                     
"""


def _load_settings() -> Dict[str, Any]:
    """Load persisted settings from the user's home directory.

    Returns an empty dict if the settings file is missing or invalid.
    """
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_settings(settings: Dict[str, Any]) -> None:
    """Persist settings to the user's home directory. Failures are ignored."""
    try:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    except Exception:
        pass


def _ensure_modal_cli_available() -> None:
    """Exit with error if the Modal CLI is not installed on PATH."""
    if shutil.which("modal") is None:
        typer.echo("Error: Modal CLI not found. Install with: pip install modal && modal setup", err=True)
        raise typer.Exit(code=1)


def _print_banner() -> None:
    """Render a simple banner using rich if available, else plain text."""
    if not Console:
        if ASCII_BANNER.strip():
            typer.echo(ASCII_BANNER)
        typer.echo("llm-launchpad")
        return
    try:
        # Lazy imports here to keep rich optional
        from rich.panel import Panel  # type: ignore
        from importlib.metadata import version  # type: ignore
    except Exception:
        if ASCII_BANNER.strip():
            typer.echo(ASCII_BANNER)
        typer.echo("llm-launchpad")
        return

    try:
        pkg_version = version("llm-launchpad")
        subtitle = f"v{pkg_version}  •  Modal + llama.cpp"
    except Exception:
        subtitle = "Modal + llama.cpp"

    console = Console()
    if ASCII_BANNER.strip():
        console.print(ASCII_BANNER)
    console.print(Panel.fit("🧠  llm-launchpad", subtitle=subtitle, border_style="cyan"))


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
    """Translate CLI options into a `modal run` command invocation list."""
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


def _run_command(command: List[str], env: Optional[Dict[str, str]] = None) -> int:
    """Run a subprocess command with optional environment overrides.

    Returns the process return code.
    """
    merged_env = None
    if env is not None:
        merged_env = {**os.environ, **env}
    process = subprocess.run(command, text=True, env=merged_env)
    return process.returncode


def _env_for_modal(settings: Dict[str, Any]) -> Dict[str, str]:
    """Derive environment variables for Modal from saved settings."""
    env: Dict[str, str] = {}
    gpu_cfg = settings.get("GPU_CONFIG")
    if isinstance(gpu_cfg, str) and gpu_cfg.strip():
        env["GPU_CONFIG"] = gpu_cfg.strip()
    scaledown = settings.get("SCALEDOWN_WINDOW")
    if isinstance(scaledown, int) and scaledown > 0:
        env["SCALEDOWN_WINDOW"] = str(scaledown)
    return env


def _ensure_modal_authenticated() -> str:
    """Verify Modal auth by checking current profile. Warn and exit if missing. Returns username."""
    try:
        res = subprocess.run(
            ["modal", "profile", "current"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        typer.echo("Failed to invoke Modal CLI. Run: modal setup", err=True)
        raise typer.Exit(code=1)
    username = (res.stdout or "").strip()
    if res.returncode != 0 or not username:
        typer.echo("Modal authentication missing. Run: modal setup", err=True)
        raise typer.Exit(code=1)
    return username


@app.command()
def wizard() -> None:
    """Interactive setup: choose a preset or custom model, preload, and deploy."""
    _ensure_modal_cli_available()
    username = _ensure_modal_authenticated()
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

        # First menu: choose between deploy flow and settings
        auth_status = f"Authenticated on Modal as {username}"
        action = inquirer.select(
            message="Choose action",
            choices=[
                {"name": "🚀 deploy", "value": "deploy"},
                {"name": "⚙️  settings", "value": "settings"},
                {"name": "─" * 50, "value": "__separator__", "disabled": True},
                {"name": f"✓ {auth_status}", "value": "__auth_status__", "disabled": True},
            ],
            default="deploy",
            cycle=True,
        ).execute()

        if action == "settings":
            settings = _load_settings()
            current_gpu = settings.get("GPU_CONFIG", "A100-80GB:1")
            current_scaledown = str(settings.get("SCALEDOWN_WINDOW", 30 * 60))
            new_gpu = inquirer.text(message="GPU_CONFIG (e.g., A100-80GB:1)", default=str(current_gpu)).execute()
            new_scaledown = inquirer.text(message="scaledown_window seconds", default=current_scaledown).execute()
            try:
                new_scaledown_int = int(new_scaledown)
            except ValueError:
                typer.echo("scaledown_window must be an integer (seconds).", err=True)
                raise typer.Exit(code=1)
            settings["GPU_CONFIG"] = new_gpu
            settings["SCALEDOWN_WINDOW"] = new_scaledown_int
            _save_settings(settings)
            # After saving, go back to first menu
            return wizard()

        # Deploy flow: choose preset or custom
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
        warm_up = False
        if deploy:
            warm_up = inquirer.confirm(
                message="Warm up after deploy (tail logs until ready)?",
                default=True,
            ).execute()

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

    settings = _load_settings()
    env = _env_for_modal(settings)

    typer.echo("\nRunning:")
    typer.echo(" "+" ".join(args))
    if env:
        typer.echo(" with env: " + ", ".join([f"{k}={v}" for k, v in env.items()]))
    code = _run_command(args, env=env)
    if code != 0:
        raise typer.Exit(code=code)

    if not deploy:
        typer.echo("\nNext: Deploy with 'modal deploy modal-llamacpp.py' when ready.")
    else:
        # Optionally warm up the server by probing the public URL and tailing logs
        try:
            # Reuse the warmup command implementation
            if warm_up:
                typer.echo("\nStarting warmup...")
                # Defer URL prompt to warmup command if not provided
                warmup(server_url=None, timeout=1800, tail_logs=True)  # type: ignore
        except Exception:
            pass


@app.command()
def deploy(
    do_warmup: bool = typer.Option(False, help="After deploy, warm up and tail logs until ready"),
    server_url: Optional[str] = typer.Option(
        None, help="Deployed web URL, e.g., https://<user>--llamacpp-server-serve.modal.run"
    ),
    timeout: int = typer.Option(1800, help="Seconds to wait for readiness during warmup"),
    tail_logs: bool = typer.Option(True, help="Tail serve logs during warmup"),
) -> None:
    """Deploy the server to Modal."""
    _ensure_modal_cli_available()
    _ensure_modal_authenticated()
    _print_banner()
    settings = _load_settings()
    env = _env_for_modal(settings)
    code = _run_command(["modal", "deploy", "modal-llamacpp.py"], env=env)
    if code != 0:
        raise typer.Exit(code=code)
    if do_warmup:
        warmup(server_url=server_url, timeout=timeout, tail_logs=tail_logs)
    raise typer.Exit(code=0)


@app.command()
def switch(
    preset: Optional[str] = typer.Option(None, help="Preset name to switch to"),
    repo_id: Optional[str] = typer.Option(None, help="Hugging Face repo id"),
    quant: Optional[str] = typer.Option(None, help="Quantization pattern (e.g., Q4_K_M)"),
    revision: Optional[str] = typer.Option(None, help="HF revision"),
    preload: bool = typer.Option(True, help="Preload/download weights immediately"),
    redeploy: bool = typer.Option(True, help="Redeploy after switching"),
    do_warmup: bool = typer.Option(True, help="Warm up after redeploy and tail logs until ready"),
    server_url: Optional[str] = typer.Option(
        None, help="Deployed web URL, e.g., https://<user>--llamacpp-server-serve.modal.run"
    ),
    timeout: int = typer.Option(1800, help="Seconds to wait for readiness during warmup"),
    tail_logs: bool = typer.Option(True, help="Tail serve logs during warmup"),
) -> None:
    """Switch model (preset or custom), optionally preload and redeploy."""
    _ensure_modal_cli_available()
    _ensure_modal_authenticated()
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
        settings = _load_settings()
        env = _env_for_modal(settings)
        code = _run_command(["modal", "deploy", "modal-llamacpp.py"], env=env)
        if code != 0:
            raise typer.Exit(code=code)
        if do_warmup:
            warmup(server_url=server_url, timeout=timeout, tail_logs=tail_logs)
    raise typer.Exit(code=0)


@app.command()
def warmup(
    server_url: Optional[str] = typer.Option(
        None, help="Deployed web URL, e.g., https://<user>--llamacpp-server-serve.modal.run"
    ),
    timeout: int = typer.Option(1800, help="Seconds to wait for readiness (default 30m)"),
    tail_logs: bool = typer.Option(True, help="Tail serve logs during warmup"),
) -> None:
    """Cold start the container by probing the server and tail logs until ready."""
    _ensure_modal_cli_available()
    _ensure_modal_authenticated()

    _print_banner()

    # Resolve server URL
    if not server_url:
        # Try environment variable first for convenience
        env_url = os.environ.get("SERVER_URL")
        if env_url:
            server_url = env_url
        else:
            server_url = typer.prompt(
                "Server URL (e.g., https://<user>--llamacpp-server-serve.modal.run)",
            )

    probe_url = server_url.rstrip("/") + "/v1/completions"

    # Tail logs from Modal in background
    logs_process: Optional[subprocess.Popen] = None
    if tail_logs:
        try:
            logs_process = subprocess.Popen(
                ["modal", "app", "logs", "-f", "llamacpp-server.serve"],
            )
        except Exception as exc:
            typer.echo(f"Warning: failed to start log tailing: {exc}")

    # Probe readiness by calling the OpenAI-compatible completions endpoint
    try:
        import time
        import json as _json
        import requests  # type: ignore
    except Exception:
        typer.echo("Error: 'requests' is required. Install with: uv pip install requests", err=True)
        if logs_process:
            try:
                logs_process.terminate()
            except Exception:
                pass
        raise typer.Exit(code=1)

    start_time = time.time()
    backoff_seconds = 2.0
    max_backoff_seconds = 30.0
    last_error_message: Optional[str] = None

    payload = {
        "model": "default",
        "prompt": "ping",
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}

    typer.echo(f"Probing readiness at: {probe_url}")

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            typer.echo("Timed out waiting for readiness.", err=True)
            if last_error_message:
                typer.echo(f"Last error: {last_error_message}", err=True)
            if logs_process:
                try:
                    logs_process.terminate()
                except Exception:
                    pass
            raise typer.Exit(code=1)
        try:
            response = requests.post(probe_url, headers=headers, data=_json.dumps(payload), timeout=10)
            if 200 <= response.status_code < 300:
                typer.echo("\n✅ Server is ready.")
                if logs_process:
                    try:
                        logs_process.terminate()
                    except Exception:
                        pass
                return
            else:
                last_error_message = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error_message = str(exc)

        time.sleep(backoff_seconds)
        backoff_seconds = min(max_backoff_seconds, backoff_seconds * 1.5)

def main() -> None:  # console script entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()



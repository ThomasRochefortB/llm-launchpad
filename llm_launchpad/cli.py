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


app = typer.Typer(help="llm-launchpad CLI - configure and deploy LLM backends on Modal.")


# --- Settings persistence (local)
SETTINGS_DIR = Path.home() / ".llm_launchpad"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

BACKEND_LLAMA = "llamacpp"
BACKEND_VLLM = "vllm"

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
        subtitle = f"v{pkg_version}  •  Modal LLM backends"
    except Exception:
        subtitle = "Modal LLM backends"

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


def _backend_script(backend: str) -> str:
    if backend == BACKEND_VLLM:
        return "modal-vllm.py"
    return "modal-llamacpp.py"


def _backend_app_name(backend: str) -> str:
    if backend == BACKEND_VLLM:
        return "vllm-server"
    return "llamacpp-server"


def _launchpad_apps() -> Dict[str, str]:
    return {
        BACKEND_LLAMA: _backend_app_name(BACKEND_LLAMA),
        BACKEND_VLLM: _backend_app_name(BACKEND_VLLM),
    }


def _ensure_launchpad_backend(backend: str) -> str:
    if backend not in _launchpad_apps():
        allowed = ", ".join(sorted(_launchpad_apps().keys()))
        typer.echo(f"Invalid backend '{backend}'. Choose one of: {allowed}", err=True)
        raise typer.Exit(code=1)
    return backend


def _backend_server_example(backend: str) -> str:
    app_name = _backend_app_name(backend)
    return f"https://<user>--{app_name}-serve.modal.run"


def _default_server_url(username: str, backend: str) -> str:
    return f"https://{username}--{_backend_app_name(backend)}-serve.modal.run"


def _env_for_backend(
    backend: str,
    model_name: Optional[str],
    model_revision: Optional[str],
    served_model_name: Optional[str],
    fast_boot: Optional[bool],
    n_gpu: Optional[int],
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if backend != BACKEND_VLLM:
        return env
    if model_name:
        env["MODEL_NAME"] = model_name
    if model_revision:
        env["MODEL_REVISION"] = model_revision
    if served_model_name:
        env["SERVED_MODEL_NAME"] = served_model_name
    if fast_boot is not None:
        env["FAST_BOOT"] = "true" if fast_boot else "false"
    if n_gpu is not None and n_gpu > 0:
        env["N_GPU"] = str(n_gpu)
    return env


def _build_modal_run_args(
    backend: str,
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
    if backend == BACKEND_VLLM:
        return ["modal", "run", _backend_script(BACKEND_VLLM)]

    args: List[str] = [
        "modal",
        "run",
        f"{_backend_script(BACKEND_LLAMA)}::main",
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


def _modal_logs_follow_args() -> List[str]:
    """Return the best available follow flag for `modal app logs`.

    Modal CLI versions differ (`-f`, `--follow`, or neither). We inspect
    `modal app logs --help` and choose a compatible option.
    """
    try:
        help_result = subprocess.run(
            ["modal", "app", "logs", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        help_text = (help_result.stdout or "") + (help_result.stderr or "")
        if "--follow" in help_text:
            return ["--follow"]
        if "-f" in help_text:
            return ["-f"]
    except Exception:
        pass
    return []


def _extract_modal_app_rows(payload: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("apps"), list):
            payload = payload.get("apps")
        elif isinstance(payload.get("data"), list):
            payload = payload.get("data")
    if not isinstance(payload, list):
        return rows

    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(
            item.get("name")
            or item.get("app_name")
            or item.get("app")
            or item.get("label")
            or item.get("description")
            or item.get("Description")
            or ""
        ).strip()
        app_id = str(
            item.get("app_id")
            or item.get("id")
            or item.get("appId")
            or item.get("App ID")
            or ""
        ).strip()
        state = str(
            item.get("state")
            or item.get("status")
            or item.get("phase")
            or item.get("State")
            or "unknown"
        ).strip()
        rows.append({"name": name, "app_id": app_id, "state": state})
    return rows


def _modal_app_list_rows() -> Optional[List[Dict[str, str]]]:
    try:
        result = subprocess.run(
            ["modal", "app", "list", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "[]")
    except Exception:
        return None
    rows = _extract_modal_app_rows(payload)
    return rows if rows else None


def _print_test_request_command(backend: str, server_url: str) -> None:
    """Print a copy/paste curl command to test the deployed endpoint."""
    base_url = server_url.rstrip("/")
    if backend == BACKEND_VLLM:
        model_name = os.environ.get("SERVED_MODEL_NAME", "llm")
        typer.echo("\nTest command:")
        typer.echo(
            "curl -s -X POST "
            f"{base_url}/v1/chat/completions "
            "-H 'Content-Type: application/json' "
            "-d '{\"model\":\""
            f"{model_name}"
            "\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}]}'"
        )
        return

    typer.echo("\nTest command:")
    typer.echo(
        "curl -s -X POST "
        f"{base_url}/v1/completions "
        "-H 'Content-Type: application/json' "
        "-d '{\"model\":\"default\",\"prompt\":\"Say hello in one short sentence.\",\"max_tokens\":32}'"
    )


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
    backend: str = BACKEND_LLAMA
    server_args: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    served_model_name: Optional[str] = None
    fast_boot: Optional[bool] = None
    n_gpu: Optional[int] = None
    action: str = "deploy"
    endpoint_action: Optional[str] = None
    endpoint_backend: Optional[str] = None
    endpoint_server_url: Optional[str] = None
    endpoint_timeout: int = 60
    endpoint_follow: bool = True
    endpoint_confirm_stop: bool = False
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
                {"name": "🛠️  manage endpoints", "value": "manage_endpoints"},
                {"name": "⚙️  settings", "value": "settings"},
            ],
            default="deploy",
            cycle=True,
            instruction=f"✓ {auth_status}",
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

        if action == "manage_endpoints":
            endpoint_action = inquirer.select(
                message="Endpoint action",
                choices=[
                    {"name": "list deployments", "value": "list"},
                    {"name": "status check", "value": "status"},
                    {"name": "tail logs", "value": "logs"},
                    {"name": "stop deployment", "value": "stop"},
                ],
                default="list",
                cycle=True,
            ).execute()
            if endpoint_action != "list":
                endpoint_backend = inquirer.select(
                    message="Choose backend",
                    choices=[
                        {"name": "llama.cpp (llamacpp-server)", "value": BACKEND_LLAMA},
                        {"name": "vLLM (vllm-server)", "value": BACKEND_VLLM},
                    ],
                    default=BACKEND_LLAMA,
                    cycle=True,
                ).execute()
            if endpoint_action == "status":
                server_url_in = inquirer.text(
                    message="Server URL override (optional)",
                    default="",
                ).execute()
                endpoint_server_url = server_url_in.strip() or None
                timeout_in = inquirer.text(message="Status timeout seconds", default="60").execute()
                try:
                    endpoint_timeout = int(str(timeout_in).strip())
                except ValueError:
                    typer.echo("Status timeout must be an integer.", err=True)
                    raise typer.Exit(code=1)
            elif endpoint_action == "logs":
                endpoint_follow = inquirer.confirm(
                    message="Follow logs stream?",
                    default=True,
                ).execute()
            elif endpoint_action == "stop" and endpoint_backend is not None:
                endpoint_confirm_stop = inquirer.confirm(
                    message=f"Stop deployed app '{_backend_app_name(endpoint_backend)}'?",
                    default=False,
                ).execute()

    if action == "manage_endpoints":
        if endpoint_action == "list":
            try:
                list_apps()
            except typer.Exit as exc:
                if exc.exit_code not in (None, 0):
                    raise
            # Return to the wizard after listing deployments.
            return wizard()
        if endpoint_backend is None:
            typer.echo("Backend selection is required.", err=True)
            raise typer.Exit(code=1)
        if endpoint_action == "status":
            status(
                backend=endpoint_backend,
                server_url=endpoint_server_url,
                timeout=endpoint_timeout,
            )
            return
        if endpoint_action == "logs":
            logs(backend=endpoint_backend, follow=endpoint_follow)
            return
        if endpoint_action == "stop":
            if not endpoint_confirm_stop:
                typer.echo("Aborted.")
                raise typer.Exit(code=1)
            stop(backend=endpoint_backend, yes=True)
            return
        typer.echo("Unknown endpoint action.", err=True)
        raise typer.Exit(code=1)

    # Deploy flow: choose preset or custom
    backend = inquirer.select(
        message="Choose backend",
        choices=[
            {"name": "llama.cpp (GGUF)", "value": BACKEND_LLAMA},
            {"name": "vLLM (OpenAI-compatible)", "value": BACKEND_VLLM},
        ],
        default=BACKEND_LLAMA,
        cycle=True,
    ).execute()

    if backend == BACKEND_VLLM:
        model_name_in = inquirer.text(
            message="Model name",
            default="Qwen/Qwen3-4B-Thinking-2507-FP8",
        ).execute()
        model_name = model_name_in.strip() or None
        model_revision_in = inquirer.text(
            message="Model revision (optional, recommended pin)",
            default="953532f942706930ec4bb870569932ef63038fdf",
        ).execute()
        model_revision = model_revision_in.strip() or None
        served_model_name_in = inquirer.text(
            message="Served model alias",
            default="llm",
        ).execute()
        served_model_name = served_model_name_in.strip() or None
        fast_boot = inquirer.confirm(
            message="Fast boot (enforce eager mode)?",
            default=True,
        ).execute()
        n_gpu_in = inquirer.text(message="Number of GPUs (tensor parallel size)", default="1").execute()
        try:
            n_gpu = int(n_gpu_in.strip())
        except ValueError:
            typer.echo("N_GPU must be an integer.", err=True)
            raise typer.Exit(code=1)

        deploy = inquirer.confirm(message="Deploy the server now?", default=True).execute()
        run_smoke = False
        if not deploy:
            run_smoke = inquirer.confirm(
                message="Run smoke test now (modal run modal-vllm.py)?",
                default=True,
            ).execute()
        warm_up = False
        if deploy:
            warm_up = inquirer.confirm(
                message="Warm up after deploy (tail logs until ready)?",
                default=True,
            ).execute()

        settings = _load_settings()
        env = _env_for_modal(settings)
        env.update(
            _env_for_backend(
                backend=backend,
                model_name=model_name,
                model_revision=model_revision,
                served_model_name=served_model_name,
                fast_boot=fast_boot,
                n_gpu=n_gpu,
            )
        )

        if deploy:
            command = ["modal", "deploy", _backend_script(backend)]
            typer.echo("\nRunning:")
            typer.echo(" " + " ".join(command))
            if env:
                typer.echo(" with env: " + ", ".join([f"{k}={v}" for k, v in env.items()]))
            code = _run_command(command, env=env)
            if code != 0:
                raise typer.Exit(code=code)
            if warm_up:
                warmup(
                    backend=backend,
                    server_url=_default_server_url(username, backend),
                    timeout=1800,
                    tail_logs=True,
                )  # type: ignore
        elif run_smoke:
            command = ["modal", "run", _backend_script(backend)]
            typer.echo("\nRunning:")
            typer.echo(" " + " ".join(command))
            if env:
                typer.echo(" with env: " + ", ".join([f"{k}={v}" for k, v in env.items()]))
            code = _run_command(command, env=env)
            if code != 0:
                raise typer.Exit(code=code)
        return

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
        backend=backend,
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
        typer.echo(f"\nNext: Deploy with 'modal deploy {_backend_script(backend)}' when ready.")
    else:
        # Optionally warm up the server by probing the public URL and tailing logs
        try:
            # Reuse the warmup command implementation
            if warm_up:
                typer.echo("\nStarting warmup...")
                # Defer URL prompt to warmup command if not provided
                warmup(
                    backend=backend,
                    server_url=_default_server_url(username, backend),
                    timeout=1800,
                    tail_logs=True,
                )  # type: ignore
        except Exception:
            pass


@app.command()
def deploy(
    backend: str = typer.Option(
        BACKEND_LLAMA,
        help="Backend to deploy: llamacpp or vllm",
    ),
    do_warmup: bool = typer.Option(False, help="After deploy, warm up and tail logs until ready"),
    model_name: Optional[str] = typer.Option(None, help="vLLM MODEL_NAME"),
    model_revision: Optional[str] = typer.Option(None, help="vLLM MODEL_REVISION"),
    served_model_name: Optional[str] = typer.Option(None, help="vLLM SERVED_MODEL_NAME"),
    fast_boot: Optional[bool] = typer.Option(None, help="vLLM FAST_BOOT"),
    n_gpu: Optional[int] = typer.Option(None, help="vLLM N_GPU (tensor parallel size)"),
    server_url: Optional[str] = typer.Option(
        None, help="Deployed web URL, e.g., https://<user>--<app>-serve.modal.run"
    ),
    timeout: int = typer.Option(1800, help="Seconds to wait for readiness during warmup"),
    tail_logs: bool = typer.Option(True, help="Tail serve logs during warmup"),
) -> None:
    """Deploy the server to Modal."""
    _ensure_modal_cli_available()
    username = _ensure_modal_authenticated()
    _print_banner()
    settings = _load_settings()
    env = _env_for_modal(settings)
    env.update(
        _env_for_backend(
            backend=backend,
            model_name=model_name,
            model_revision=model_revision,
            served_model_name=served_model_name,
            fast_boot=fast_boot,
            n_gpu=n_gpu,
        )
    )
    code = _run_command(["modal", "deploy", _backend_script(backend)], env=env)
    if code != 0:
        raise typer.Exit(code=code)
    if do_warmup:
        warmup(
            backend=backend,
            server_url=server_url or _default_server_url(username, backend),
            timeout=timeout,
            tail_logs=tail_logs,
        )
    raise typer.Exit(code=0)


@app.command()
def switch(
    backend: str = typer.Option(
        BACKEND_LLAMA,
        help="Backend to switch: llamacpp or vllm",
    ),
    preset: Optional[str] = typer.Option(None, help="Preset name to switch to"),
    repo_id: Optional[str] = typer.Option(None, help="Hugging Face repo id"),
    quant: Optional[str] = typer.Option(None, help="Quantization pattern (e.g., Q4_K_M)"),
    revision: Optional[str] = typer.Option(None, help="HF revision"),
    model_name: Optional[str] = typer.Option(None, help="vLLM MODEL_NAME"),
    model_revision: Optional[str] = typer.Option(None, help="vLLM MODEL_REVISION"),
    served_model_name: Optional[str] = typer.Option(None, help="vLLM SERVED_MODEL_NAME"),
    fast_boot: Optional[bool] = typer.Option(None, help="vLLM FAST_BOOT"),
    n_gpu: Optional[int] = typer.Option(None, help="vLLM N_GPU (tensor parallel size)"),
    preload: bool = typer.Option(True, help="Preload/download weights immediately"),
    redeploy: bool = typer.Option(True, help="Redeploy after switching"),
    do_warmup: bool = typer.Option(True, help="Warm up after redeploy and tail logs until ready"),
    server_url: Optional[str] = typer.Option(
        None, help="Deployed web URL, e.g., https://<user>--<app>-serve.modal.run"
    ),
    timeout: int = typer.Option(1800, help="Seconds to wait for readiness during warmup"),
    tail_logs: bool = typer.Option(True, help="Tail serve logs during warmup"),
) -> None:
    """Switch model (preset or custom), optionally preload and redeploy."""
    _ensure_modal_cli_available()
    username = _ensure_modal_authenticated()
    _print_banner()

    if backend == BACKEND_VLLM:
        settings = _load_settings()
        env = _env_for_modal(settings)
        env.update(
            _env_for_backend(
                backend=backend,
                model_name=model_name,
                model_revision=model_revision,
                served_model_name=served_model_name,
                fast_boot=fast_boot,
                n_gpu=n_gpu,
            )
        )
        if preload:
            typer.echo("Note: --preload is only used by llama.cpp and is ignored for vLLM.")
        if redeploy:
            code = _run_command(["modal", "deploy", _backend_script(backend)], env=env)
            if code != 0:
                raise typer.Exit(code=code)
            if do_warmup:
                warmup(
                    backend=backend,
                    server_url=server_url or _default_server_url(username, backend),
                    timeout=timeout,
                    tail_logs=tail_logs,
                )
        else:
            typer.echo("No deploy performed. Use --redeploy to apply vLLM model changes.")
        raise typer.Exit(code=0)

    if not any([preset, repo_id]):
        typer.echo("Provide --preset or --repo-id to switch.", err=True)
        raise typer.Exit(code=1)

    args = _build_modal_run_args(
        backend=backend,
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
        code = _run_command(["modal", "deploy", _backend_script(backend)], env=env)
        if code != 0:
            raise typer.Exit(code=code)
        if do_warmup:
            warmup(
                backend=backend,
                server_url=server_url or _default_server_url(username, backend),
                timeout=timeout,
                tail_logs=tail_logs,
            )
    raise typer.Exit(code=0)


@app.command()
def warmup(
    backend: str = typer.Option(
        BACKEND_LLAMA,
        help="Backend to warm up: llamacpp or vllm",
    ),
    server_url: Optional[str] = typer.Option(
        None, help="Deployed web URL, e.g., https://<user>--<app>-serve.modal.run"
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
                f"Server URL (e.g., {_backend_server_example(backend)})",
            )
    is_vllm = backend == BACKEND_VLLM
    probe_url = server_url.rstrip("/") + ("/health" if is_vllm else "/v1/completions")

    # Tail logs from Modal in background
    logs_process: Optional[subprocess.Popen] = None
    if tail_logs:
        try:
            follow_args = _modal_logs_follow_args()
            logs_process = subprocess.Popen(
                ["modal", "app", "logs", *follow_args, _backend_app_name(backend)],
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
            if is_vllm:
                response = requests.get(probe_url, timeout=10)
            else:
                response = requests.post(probe_url, headers=headers, data=_json.dumps(payload), timeout=10)
            if 200 <= response.status_code < 300:
                typer.echo("\n✅ Server is ready.")
                _print_test_request_command(backend, server_url)
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


@app.command("list")
def list_apps() -> None:
    """List launchpad Modal apps (llamacpp-server and vllm-server)."""
    _ensure_modal_cli_available()
    _ensure_modal_authenticated()
    _print_banner()

    launchpad_by_name = {name: backend for backend, name in _launchpad_apps().items()}
    rows = _modal_app_list_rows()
    if rows:
        filtered = [row for row in rows if row.get("name") in launchpad_by_name]
        if not filtered:
            typer.echo("No launchpad deployments found in Modal app list.")
            raise typer.Exit(code=0)
        typer.echo("Launchpad deployments:")
        for row in filtered:
            backend = launchpad_by_name.get(row.get("name", ""), "unknown")
            app_name = row.get("name", "")
            state = row.get("state", "unknown")
            app_id = row.get("app_id", "")
            suffix = f" ({app_id})" if app_id else ""
            typer.echo(f"- backend={backend} app={app_name} state={state}{suffix}")
        raise typer.Exit(code=0)

    # Fallback for older Modal versions that may not support `--json`.
    try:
        result = subprocess.run(
            ["modal", "app", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        typer.echo(f"Failed to run `modal app list`: {exc}", err=True)
        raise typer.Exit(code=1)
    if result.returncode != 0:
        typer.echo(result.stderr.strip() or "Failed to query Modal app list.", err=True)
        raise typer.Exit(code=result.returncode)

    lines = (result.stdout or "").splitlines()
    matches = [line for line in lines if any(name in line for name in launchpad_by_name)]
    if not matches:
        typer.echo("No launchpad deployments found in Modal app list.")
    else:
        typer.echo("Launchpad deployments:")
        for line in matches:
            typer.echo(f"- {line.strip()}")
    raise typer.Exit(code=0)


@app.command()
def status(
    backend: str = typer.Option(
        BACKEND_LLAMA,
        help="Backend to check: llamacpp or vllm",
    ),
    server_url: Optional[str] = typer.Option(
        None, help="Deployed web URL, e.g., https://<user>--<app>-serve.modal.run"
    ),
    timeout: int = typer.Option(60, help="Seconds to wait for readiness check"),
) -> None:
    """Check endpoint readiness for a deployed launchpad backend."""
    _ensure_modal_cli_available()
    username = _ensure_modal_authenticated()
    _print_banner()

    backend = _ensure_launchpad_backend(backend)
    resolved_server_url = server_url or os.environ.get("SERVER_URL") or _default_server_url(username, backend)
    is_vllm = backend == BACKEND_VLLM
    probe_url = resolved_server_url.rstrip("/") + ("/health" if is_vllm else "/v1/completions")

    try:
        import time
        import json as _json
        import requests  # type: ignore
    except Exception:
        typer.echo("Error: 'requests' is required. Install with: uv pip install requests", err=True)
        raise typer.Exit(code=1)

    payload = {
        "model": "default",
        "prompt": "ping",
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    start_time = time.time()
    backoff_seconds = 2.0
    max_backoff_seconds = 15.0
    last_error_message: Optional[str] = None

    typer.echo(f"Checking endpoint status at: {probe_url}")
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            typer.echo("Status: unhealthy (timed out waiting for readiness).", err=True)
            if last_error_message:
                typer.echo(f"Last error: {last_error_message}", err=True)
            raise typer.Exit(code=1)
        try:
            if is_vllm:
                response = requests.get(probe_url, timeout=10)
            else:
                response = requests.post(probe_url, headers=headers, data=_json.dumps(payload), timeout=10)
            if 200 <= response.status_code < 300:
                typer.echo(f"Status: healthy (backend={backend}, url={resolved_server_url})")
                _print_test_request_command(backend, resolved_server_url)
                raise typer.Exit(code=0)
            last_error_message = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error_message = str(exc)
        time.sleep(backoff_seconds)
        backoff_seconds = min(max_backoff_seconds, backoff_seconds * 1.5)


@app.command()
def logs(
    backend: str = typer.Option(
        BACKEND_LLAMA,
        help="Backend logs to view: llamacpp or vllm",
    ),
    follow: bool = typer.Option(True, help="Follow logs stream"),
) -> None:
    """Show logs for a deployed launchpad backend."""
    _ensure_modal_cli_available()
    _ensure_modal_authenticated()
    _print_banner()

    backend = _ensure_launchpad_backend(backend)
    command = ["modal", "app", "logs"]
    if follow:
        command.extend(_modal_logs_follow_args())
    command.append(_backend_app_name(backend))

    code = _run_command(command)
    if code != 0:
        raise typer.Exit(code=code)
    raise typer.Exit(code=0)


@app.command()
def stop(
    backend: str = typer.Option(
        BACKEND_LLAMA,
        help="Backend app to stop: llamacpp or vllm",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    """Stop a deployed launchpad backend app in Modal."""
    _ensure_modal_cli_available()
    _ensure_modal_authenticated()
    _print_banner()

    backend = _ensure_launchpad_backend(backend)
    app_name = _backend_app_name(backend)
    if not yes:
        confirmed = typer.confirm(f"Stop deployed app '{app_name}'?")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=1)

    code = _run_command(["modal", "app", "stop", app_name])
    if code != 0:
        raise typer.Exit(code=code)
    typer.echo(f"Stopped app: {app_name}")
    raise typer.Exit(code=0)


def main() -> None:  # console script entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()



#!/usr/bin/env python3
"""Weekly live canary: deploy, chat, and stop the smallest model on each provider.

Three regressions this year passed the whole test suite and broke real
deployments -- every Prime and Vast llama.cpp deploy for days after 623ba8b,
and two vLLM defaults that only one provider exposed. Nothing short of a real
endpoint answering a real request catches that class, so this drives the
same headless CLI a user runs:

    llm-launchpad deploy --do-warmup ...   # warmup fails unless chat answers
    llm-launchpad stop --yes ...           # always, even after a failure

Spend is bounded three ways: the smallest GGUF on the cheapest GPU, a hard
wall-clock limit per provider, and a 30-minute idle shutdown on the rental
itself as the backstop if this process dies before ``stop`` runs.

Pass ``--live`` to authorize billable resources. Providers whose credentials
are missing are reported as skipped rather than failed, so the scheduled job
works with whichever providers the repository has secrets for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MODEL_REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_QUANT = "Q4_K_M"
# A backstop, not the plan: `stop` normally ends the rental within minutes.
IDLE_SHUTDOWN = "30m"
PROVIDERS = ("modal", "prime", "vast")
# The report is uploaded as a CI artifact, so nothing credential-shaped leaves
# this process: provider keys from the environment, bearer tokens, and the
# endpoint key the deploy summary prints.
_SECRET_ENV = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "PRIME_API_KEY", "VAST_API_KEY", "HF_TOKEN")
_API_KEY_LINE = re.compile(r"(api[ _-]?key\s*[:=]\s*)\S+", re.IGNORECASE)


def _redact(lines: list[str]) -> list[str]:
    from llm_launchpad.core.prime_live import redact_live_value

    secrets = tuple(value for name in _SECRET_ENV if (value := os.getenv(name, "").strip()))
    return [_API_KEY_LINE.sub(r"\1[redacted]", redact_live_value(line, secrets)) for line in lines]


@dataclass
class ProviderResult:
    provider: str
    status: str = "skipped"  # passed | failed | skipped
    detail: str = ""
    app_name: str = ""
    deploy_seconds: float | None = None
    stopped: bool | None = None
    output_tail: list[str] = field(default_factory=list)


def _cli() -> list[str]:
    """The installed CLI, or this checkout's when run from source."""
    found = shutil.which("llm-launchpad")
    return [found] if found else [sys.executable, "-m", "llm_launchpad.cli.main"]


def _credentials_present(provider: str) -> str | None:
    """Return why a provider cannot run, or ``None`` when it can."""
    if provider == "modal":
        has_env = os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")
        has_file = (Path.home() / ".modal.toml").exists()
        return None if has_env or has_file else "no Modal token"
    if provider == "prime":
        has_env = os.getenv("PRIME_API_KEY")
        has_file = (Path.home() / ".prime" / "config.json").exists()
        return None if has_env or has_file else "no Prime API key"
    from llm_launchpad.core.vast_auth import resolve_vast_credentials

    try:
        return None if resolve_vast_credentials().api_key else "no Vast.ai API key"
    except ValueError as exc:  # unreadable key file
        return f"Vast.ai key unreadable: {exc}"


def _deploy_args(provider: str, app_name: str, max_hourly_cost: float) -> list[str]:
    args = [
        "deploy",
        "--provider", provider,
        "--backend", "llamacpp",
        "--repo-id", MODEL_REPO,
        "--quant", MODEL_QUANT,
        "--app-name", app_name,
        "--instance-name", app_name,
        "--server-args", "--ctx-size 4096",
        "--vision", "off",
        "--do-warmup",
        "--summary-logs",
        "--no-tail-logs",
        "--timeout", "1500",
    ]
    if provider == "modal":
        args += ["--gpu-type", "T4", "--gpu-count", "1"]
    else:
        args += ["--idle-shutdown", IDLE_SHUTDOWN]
    if provider == "prime":
        args += ["--gpu-count", "1", "--no-prime-disk"]
    if provider == "vast":
        args += ["--max-hourly-cost", f"{max_hourly_cost:g}", "--vast-disk-gb", "40"]
    return args


def _run(command: list[str], timeout: int) -> tuple[int, list[str]]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, _redact(output.splitlines()[-40:])
    lines = (completed.stdout + completed.stderr).splitlines()
    return completed.returncode, _redact(lines[-40:])


def run_provider(provider: str, *, run_id: str, max_hourly_cost: float, max_minutes: int) -> ProviderResult:
    result = ProviderResult(provider=provider)
    missing = _credentials_present(provider)
    if missing:
        result.detail = missing
        return result
    result.app_name = f"llp-canary-{provider}-{run_id}"
    started = time.monotonic()
    try:
        code, tail = _run(
            [*_cli(), *_deploy_args(provider, result.app_name, max_hourly_cost)],
            timeout=max_minutes * 60,
        )
        result.deploy_seconds = round(time.monotonic() - started, 1)
        result.output_tail = tail
        if code == 0:
            result.status = "passed"
        else:
            result.status = "failed"
            result.detail = "deploy timed out" if code == 124 else f"deploy exited {code}"
    finally:
        # Always stop, including after a failed or timed-out deploy: a
        # half-provisioned rental is exactly what this must not leave behind.
        stop_code, stop_tail = _run(
            [*_cli(), "stop", "--provider", provider, "--backend", "llamacpp",
             "--app-name", result.app_name, "--yes"],
            timeout=600,
        )
        result.stopped = stop_code == 0
        if not result.stopped:
            result.output_tail += ["--- stop ---", *stop_tail]
            if result.status == "passed":
                result.status = "failed"
                result.detail = f"stop exited {stop_code}; check for a leftover rental"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Authorize billable deployments.")
    parser.add_argument("--provider", action="append", choices=PROVIDERS, help="Repeat to choose; default all.")
    parser.add_argument("--max-hourly-cost", type=float, default=0.25, help="Vast hourly ceiling, disk included.")
    parser.add_argument("--max-minutes", type=int, default=35, help="Wall-clock limit per provider deploy.")
    parser.add_argument("--report", type=Path, default=Path("canary-report.json"))
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to authorize billable deployments.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    results = [
        run_provider(provider, run_id=run_id, max_hourly_cost=args.max_hourly_cost, max_minutes=args.max_minutes)
        for provider in (args.provider or PROVIDERS)
    ]
    report = {
        "run_id": run_id,
        "model": f"{MODEL_REPO}:{MODEL_QUANT}",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(result) for result in results],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for result in results:
        line = f"{result.provider:6} {result.status:7} {result.detail}"
        print(line.rstrip(), flush=True)
    if all(result.status == "skipped" for result in results):
        print("No provider had credentials; nothing was tested.", file=sys.stderr)
        return 2
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

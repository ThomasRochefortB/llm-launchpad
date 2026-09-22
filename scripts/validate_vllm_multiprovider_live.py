#!/usr/bin/env python3
"""Opt-in Advanced-deploy vLLM certification, driven through the real TUI.

One model, one form, three providers: this rents (or allocates) real compute,
serves the model, proves the endpoint answers, proves OpenCode can reach it,
and then destroys what it created. Nothing here runs without ``--live``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from unittest.mock import patch
import uuid

import requests
from textual.widgets import Button, Input, Select, Switch

from llm_launchpad.core.connection_store import remove_connection
from llm_launchpad.core.opencode import provider_id_for_app
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    ComputeOffer,
    DeploymentConfig,
    VastOffer,
    VastOfferQuery,
)
from llm_launchpad.tui.screens.deploy import (
    PrimeOffersLoaded,
    VastOffersLoaded,
    VllmDeployScreen,
)
from scripts.validate_prime_live import LiveTuiApp
from scripts.validate_vast_tui_live import settle


# Weights this stage has to move onto the rental, for cost ranking only.
WEIGHTS_GB = 56.0


class GuardedApp(LiveTuiApp):
    """Run the real deploy worker only for the configuration this run approved."""

    def __init__(self, provider: ComputeProvider, name: str, model: str, rehearse: bool) -> None:
        super().__init__(BackendType.VLLM)
        self.authorized_provider = provider
        self.authorized_name = name
        self.authorized_model = model
        self.rehearse = rehearse

    def begin_deploy(self, config: DeploymentConfig) -> None:
        if (
            config.provider != self.authorized_provider
            or config.app_name != self.authorized_name
            or (config.model_name or "") != self.authorized_model
        ):
            raise ValueError("The live form built an unauthorized deployment.")
        # A fallback would spend on hardware this run never approved.
        config.fallback_configs = ()
        if self.rehearse:
            self.deployed_config = config
            return
        super().begin_deploy(config)


def _shot(app: GuardedApp, shots: Path, name: str) -> None:
    shots.mkdir(parents=True, exist_ok=True)
    app.save_screenshot(filename=name + ".svg", path=str(shots))
    print(f"  shot {name}", flush=True)


def prime_offer(
    offer_id: str | None, *, gpu_type: str, min_vram_gb: float, max_hourly_cost: float
) -> ComputeOffer:
    """Resolve the approved Prime offer, or pick the cheapest that qualifies.

    Prime offer ids churn between listings, so naming one and then renting it
    minutes later is a race the harness loses more often than not.
    """
    from llm_launchpad.core.prime_backend import PrimeBackend

    offers = PrimeBackend().list_offers()
    if offer_id:
        for offer in offers:
            if offer.id == offer_id:
                return offer
        raise ValueError(f"Prime offer {offer_id} is no longer available.")
    wanted = gpu_type.strip().casefold()
    candidates = [
        offer
        for offer in offers
        if (offer.price_per_hour or 0) > 0
        and (offer.price_per_hour or 0) <= max_hourly_cost
        and (offer.gpu_memory_gb or 0) * (offer.gpu_count or 1) >= min_vram_gb
        and (not wanted or offer.gpu_type.casefold() == wanted)
    ]
    if not candidates:
        raise ValueError(
            f"No Prime offer under ${max_hourly_cost:.2f}/hr carries {min_vram_gb:.0f} GB."
        )
    return min(candidates, key=lambda offer: offer.price_per_hour or 0)


def vast_offer(
    offer_id: str | None, *, min_vram_gb: float, max_hourly_cost: float, disk_gb: int
) -> VastOffer:
    """Resolve the approved rental, or pick the cheapest live one that fits.

    Vast ask ids are a live market: naming one and renting it minutes later
    loses the race often enough that the first attempt of this run died on
    ``no_such_ask``.
    """
    from llm_launchpad.core.vast_deployment import VastDeploymentBackend

    api = VastDeploymentBackend().api
    if offer_id:
        return api.get_offer(offer_id, VastOfferQuery(gpu_count=None))
    from llm_launchpad.core.vast_runtime import load_vast_runtime

    runtime = load_vast_runtime(BackendType.VLLM)
    query = VastOfferQuery(gpu_count=1, disk_gb=disk_gb, limit=400)
    candidates = [
        offer
        for offer in api.list_offers(query)
        if (offer.gpu_memory_gb or 0) >= min_vram_gb
        and 0 < (offer.costs.total_per_hour_usd or 0) <= max_hourly_cost
        # The deploy refuses a host below the image's own floor, so a picker
        # that ignores it just rents nothing, slowly.
        and (offer.cuda_max_good or 0) >= runtime.min_cuda_version
        and (offer.compute_capability or 0) >= runtime.min_compute_capability
    ]
    if not candidates:
        raise ValueError(
            f"No Vast rental under ${max_hourly_cost:.2f}/hr carries {min_vram_gb:.0f} GB."
        )

    def estimated_cost(offer: VastOffer) -> float:
        """Price the whole stage: a slow link is billed while it downloads."""
        hourly = offer.costs.total_per_hour_usd or 0.0
        mbps = offer.inet_down_mbps or 200.0
        download_hours = (WEIGHTS_GB * 8_000) / max(mbps, 50.0) / 3_600
        transfer = WEIGHTS_GB * (offer.costs.download_per_gb_usd or 0.0)
        return hourly * (download_hours + 0.25) + transfer

    return min(candidates, key=estimated_cost)


async def fill_form(
    pilot: Any,
    screen: VllmDeployScreen,
    args: argparse.Namespace,
    name: str,
    offer_id: str | None,
) -> None:
    """Drive the Advanced vLLM form the way a person would."""
    await settle(pilot, lambda: bool(screen.query("#model-name")), "Model field")
    screen.query_one("#model-name", Input).value = args.model
    screen.query_one("#max-model-len", Input).value = str(args.max_model_len)
    await asyncio.sleep(3)
    await pilot.pause()
    screen.query_one("#provider-vllm", Select).value = args.provider
    await pilot.pause()

    if args.provider == "modal":
        await settle(
            pilot,
            lambda: bool(screen.query_one("#gpu-type-vllm", Select)._options),
            "Modal GPU catalog",
            seconds=60,
        )
        screen.query_one("#gpu-type-vllm", Select).value = args.gpu_type
        screen.query_one("#gpu-count-vllm", Input).value = str(args.gpu_count)
        screen.query_one("#n-gpu", Input).value = str(args.gpu_count)
    elif args.provider == "prime":
        await settle(
            pilot,
            lambda: screen._selected_prime_offer_id == offer_id,
            "Approved Prime offer",
            seconds=60,
        )
    else:
        await settle(
            pilot,
            lambda: screen._selected_vast_offer_id == offer_id,
            "Approved Vast rental",
            seconds=60,
        )

    screen.query_one("#toggle-advanced-vllm", Button).press()
    await pilot.pause()
    screen.query_one("#app-name-vllm", Input).value = name
    screen.query_one("#warmup-vllm", Switch).value = True
    screen.query_one("#show-debug-logs-vllm", Switch).value = bool(args.show_debug_logs)
    if args.tool_call_parser:
        screen.query_one("#tool-call-parser", Input).value = args.tool_call_parser
    if args.reasoning_parser:
        screen.query_one("#reasoning-parser", Input).value = args.reasoning_parser
    await asyncio.sleep(2)
    await pilot.pause()


def stored_connection(app_name: str) -> dict[str, Any]:
    """Fall back to the connection summary the deploy wrote to disk."""
    from llm_launchpad.core.connection_store import load_connection_entries

    entry = load_connection_entries().get(app_name) or {}
    return {
        "base_url": entry.get("base_url") or "",
        "api_key": entry.get("endpoint_api_key") or entry.get("api_key") or "",
        "model_id": entry.get("model_id") or "",
    }


def _chat_url(base_url: str) -> str:
    """Build the completions URL whether or not the base already ends in /v1."""
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root += "/v1"
    return root + "/chat/completions"


def probe_endpoint(base_url: str, api_key: str, model_id: str) -> dict[str, Any]:
    """Prove the endpoint rejects strangers and then actually generates tokens."""
    checks: dict[str, Any] = {}
    url = _chat_url(base_url)
    with requests.Session() as session:
        session.trust_env = False
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with the word hello."}],
            "max_tokens": 64,
        }
        rejected = []
        for headers in ({}, {"Authorization": "Bearer deliberately-invalid-live-key"}):
            with session.post(url, json=payload, headers=headers, timeout=60) as response:
                rejected.append(response.status_code)
        checks["unauthorized_statuses"] = rejected
        if any(code not in {401, 403} for code in rejected):
            raise RuntimeError(
                f"Endpoint answered {rejected} to a missing and to an incorrect bearer token; "
                "expected 401/403 from both."
            )

        headers = {"Authorization": "Bearer " + api_key}
        started = time.monotonic()
        with session.post(url, json=payload, headers=headers, timeout=300) as response:
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage") or {}
        if not int(usage.get("completion_tokens") or 0):
            raise RuntimeError("The endpoint returned no completion tokens.")
        checks["chat"] = {"seconds": time.monotonic() - started, "usage": usage}

        started = time.monotonic()
        chunks = 0
        with session.post(
            url,
            json={**payload, "stream": True, "max_tokens": 256,
                  "messages": [{"role": "user", "content": "Count from 1 to 40."}]},
            headers=headers,
            timeout=300,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunks += 1
        if chunks < 2:
            raise RuntimeError("The endpoint streamed nothing.")
        checks["stream"] = {"chunks": chunks, "seconds": time.monotonic() - started}
    return checks


def _isolated_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(root / "opencode.json")
    env["XDG_CONFIG_HOME"] = str(root / "xdg-config")
    env["XDG_DATA_HOME"] = str(root / "xdg-data")
    env["XDG_CACHE_HOME"] = str(root / "xdg-cache")
    env["XDG_STATE_HOME"] = str(root / "xdg-state")
    env["NO_COLOR"] = "1"
    return env


def opencode_round_trip(
    config: DeploymentConfig, provider: ComputeProvider, root: Path, timeout: int = 900
) -> dict[str, Any]:
    """Sync the deployment into an isolated OpenCode and make it process tokens."""
    executable = shutil.which("opencode")
    if not executable:
        raise RuntimeError("OpenCode is not installed.")
    root.mkdir(parents=True, exist_ok=True)
    app_name = str(config.app_name or "")
    model_id = str(config.served_model_name or config.model_name or "")
    provider_id = provider_id_for_app(app_name)
    reference = f"{provider_id}/{model_id}"

    launcher = (
        "from pathlib import Path\n"
        "import os\n"
        "from llm_launchpad.core import opencode\n"
        "root = Path(os.environ['LLM_LAUNCHPAD_LIVE_OPENCODE_ROOT'])\n"
        "opencode.OPENCODE_CONFIG_PATH = root / 'opencode.json'\n"
        "opencode.OPENCODE_JSONC_CONFIG_PATH = root / 'opencode.jsonc'\n"
        "opencode.OPENCODE_REGISTRY_PATH = root / 'opencode_registry.json'\n"
        "from llm_launchpad.cli.main import main\n"
        "main()\n"
    )
    env = _isolated_env(root)
    env["LLM_LAUNCHPAD_LIVE_OPENCODE_ROOT"] = str(root)
    sync = subprocess.run(
        [sys.executable, "-c", launcher, "opencode", "sync",
         "--provider", provider.value, "--app-name", app_name],
        cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, check=False, timeout=300,
    )
    if sync.returncode != 0:
        raise RuntimeError("opencode sync failed: " + (sync.stderr or sync.stdout)[-400:])

    listed = subprocess.run(
        [executable, "models", provider_id], cwd=root, env=_isolated_env(root),
        capture_output=True, text=True, check=False, timeout=300,
    )
    listing = "\n".join(part for part in (listed.stdout, listed.stderr) if part).strip()
    if reference not in listing:
        raise AssertionError(f"OpenCode did not list {reference}.")

    sentinel = f"LAUNCHPAD-{uuid.uuid4().hex[:8].upper()}"
    run = subprocess.run(
        [executable, "run", "--pure", "--model", reference,
         f"Reply with exactly this identifier and nothing else: {sentinel}"],
        cwd=root, env=_isolated_env(root),
        capture_output=True, text=True, check=False, timeout=timeout,
    )
    combined = "\n".join(part for part in (run.stdout, run.stderr) if part).strip()
    if run.returncode != 0:
        raise RuntimeError("opencode run failed: " + combined[-600:])
    if sentinel not in combined:
        raise AssertionError(
            "OpenCode returned no sentinel; tail: " + combined[-600:]
        )
    return {
        "model_reference": reference,
        "sentinel": sentinel,
        "run_exit_code": run.returncode,
        "output_tail": combined.splitlines()[-8:],
    }


def destroy(provider: ComputeProvider, name: str) -> dict[str, Any]:
    """Stop whatever this run started, and say whether it is really gone."""
    result = subprocess.run(
        [sys.executable, "-m", "llm_launchpad.cli.main", "stop",
         "--provider", provider.value, "--backend", "vllm", "--app-name", name, "--yes"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False, timeout=900,
    )
    remove_connection(name)
    return {
        "exit_code": result.returncode,
        "tail": (result.stdout + result.stderr).strip().splitlines()[-8:],
    }


async def validate(args: argparse.Namespace) -> int:
    provider = ComputeProvider(args.provider)
    name = f"llp-{args.provider}-vllm-q38-" + uuid.uuid4().hex[:8]
    shots = args.shots or args.report.parent / (args.report.stem + "-shots")
    app = GuardedApp(provider, name, args.model, args.rehearse)
    report: dict[str, Any] = {
        "provider": provider.value, "model": args.model, "name": name,
        "max_model_len": args.max_model_len, "checks": {},
        "success": False, "cleanup_confirmed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")

    approved_prime = (
        prime_offer(
            args.offer_id,
            gpu_type=args.prime_gpu_type,
            min_vram_gb=args.min_vram_gb,
            max_hourly_cost=args.max_hourly_cost,
        )
        if provider == ComputeProvider.PRIME
        else None
    )
    approved_vast = (
        vast_offer(
            args.offer_id,
            min_vram_gb=args.min_vram_gb,
            max_hourly_cost=args.max_hourly_cost,
            disk_gb=args.vast_disk_gb,
        )
        if provider == ComputeProvider.VAST
        else None
    )
    if approved_prime is not None:
        report["offer"] = asdict(approved_prime)
        if (approved_prime.price_per_hour or 0) > args.max_hourly_cost:
            raise ValueError("Prime offer exceeds the approved hourly cost.")
    if approved_vast is not None:
        report["offer"] = asdict(approved_vast)
        if (approved_vast.costs.total_per_hour_usd or 0) > args.max_hourly_cost:
            raise ValueError("Vast offer exceeds the approved hourly cost.")

    def live_prime(screen: VllmDeployScreen) -> None:
        screen.post_message(PrimeOffersLoaded([approved_prime]))

    def live_vast(screen: VllmDeployScreen) -> None:
        screen.post_message(VastOffersLoaded([approved_vast]))

    patches = []
    if approved_prime is not None:
        patches.append(patch.object(VllmDeployScreen, "_run_fetch_prime_offers", live_prime))
    if approved_vast is not None:
        patches.append(patch.object(VllmDeployScreen, "_run_fetch_vast_offers", live_vast))

    started = time.monotonic()
    save()
    print(f"LIVE {provider.value}: {args.model} as {name}", flush=True)
    try:
        for entered in patches:
            entered.__enter__()
        async with app.run_test(size=(140, 45)) as pilot:
            try:
                async with asyncio.timeout(args.max_minutes * 60):
                    await pilot.pause()
                    if not isinstance(app.screen, VllmDeployScreen):
                        app.push_screen(VllmDeployScreen())
                    await settle(
                        pilot, lambda: isinstance(app.screen, VllmDeployScreen), "Deploy form", seconds=30
                    )
                    screen = app.screen
                    assert isinstance(screen, VllmDeployScreen)
                    approved_id = (
                        approved_prime.id if approved_prime is not None
                        else (approved_vast.id if approved_vast is not None else None)
                    )
                    await fill_form(pilot, screen, args, name, approved_id)
                    _shot(app, shots, f"{provider.value}-01-form")
                    report["estimated_vram"] = str(
                        screen.query_one("#vllm-vram-status").content  # type: ignore[attr-defined]
                    )
                    save()

                    screen.query_one("#deploy-vllm-btn", Button).press()
                    await settle(pilot, lambda: app.deployed_config is not None, "Deploy worker", seconds=30)
                    config = app.deployed_config
                    assert config is not None
                    report["config"] = {
                        "gpu_type": config.gpu_type, "gpu_count": config.gpu_count,
                        "n_gpu": config.n_gpu, "max_context_tokens": config.max_context_tokens,
                        "required_vram_gb": config.required_vram_gb,
                        "tool_call_parser": config.tool_call_parser,
                        "reasoning_parser": config.reasoning_parser,
                        "served_model_name": config.served_model_name,
                    }
                    save()
                    if args.rehearse:
                        report["rehearsal"] = True
                        report["success"] = True
                        _shot(app, shots, f"{provider.value}-02-rehearsal")
                        return 0

                    last = 0.0
                    while app.capture_monitor is None or app.capture_monitor.last_done is None:
                        await asyncio.sleep(1)
                        if time.monotonic() - last > 60:
                            last = time.monotonic()
                            print(f"  deploying ({time.monotonic() - started:.0f}s)", flush=True)
                            _shot(app, shots, f"{provider.value}-02-deploy-{int(last - started)}s")
                    done = app.capture_monitor.last_done
                    _shot(app, shots, f"{provider.value}-03-outcome")
                    # "See root cause above" is only useful with the above.
                    viewer = app.capture_monitor.log_viewer.log_widget
                    log_lines = list(getattr(viewer, "_lines", []) or [])
                    (shots / f"{provider.value}-deploy.log").write_text(
                        "\n".join(str(line) for line in log_lines) + "\n"
                    )
                    report["checks"]["deploy_and_warmup_seconds"] = time.monotonic() - started
                    if not done.success:
                        raise RuntimeError(done.detail or "Deployment failed.")

                    # The connection summary is published after the operation
                    # completes, so it is not on screen the instant Deploy
                    # reports success.
                    payload: dict[str, Any] = {}
                    for _ in range(60):
                        payload = dict(app.capture_monitor._connection_payload or {})
                        if payload.get("base_url"):
                            break
                        await asyncio.sleep(1)
                    if not payload.get("base_url"):
                        payload = stored_connection(name)
                    base_url = str(payload.get("base_url") or "")
                    api_key = str(payload.get("api_key") or "")
                    model_id = str(payload.get("model_id") or "")
                    if not base_url or not model_id:
                        raise RuntimeError("The deployment produced no connection details.")
                    report["checks"]["endpoint"] = {"base_url": base_url, "model_id": model_id}
                    save()
                    report["checks"].update(
                        await asyncio.to_thread(probe_endpoint, base_url, api_key, model_id)
                    )
                    save()
                    report["checks"]["opencode"] = await asyncio.to_thread(
                        opencode_round_trip, config, provider,
                        args.report.parent / (args.report.stem + "-opencode"),
                    )
                    report["success"] = True
            finally:
                if not args.rehearse and "cleanup" not in report:
                    report["cleanup"] = await asyncio.to_thread(destroy, provider, name)
    except Exception as exc:
        detail = str(exc)
        key = (app.deployed_config.endpoint_api_key if app.deployed_config else None) or ""
        if key:
            detail = detail.replace(key, "[redacted]")
        report["error"] = detail
        report["notifications"] = app.live_notifications[-10:]
        print("FAILED: " + detail, flush=True)
    finally:
        for entered in reversed(patches):
            entered.__exit__(None, None, None)
        if not args.rehearse:
            try:
                if "cleanup" not in report:
                    report["cleanup"] = await asyncio.to_thread(destroy, provider, name)
                report["cleanup_confirmed"] = report["cleanup"]["exit_code"] == 0
            except Exception as exc:
                report["cleanup_error"] = str(exc)
        else:
            report["cleanup_confirmed"] = True
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        print(
            f"Report {args.report}; success={report['success']} "
            f"cleanup={report['cleanup_confirmed']}",
            flush=True,
        )
    return 0 if report["success"] and report["cleanup_confirmed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--rehearse", action="store_true")
    parser.add_argument("--provider", choices=["modal", "prime", "vast"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-type", default="A100-80GB")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--offer-id")
    parser.add_argument("--prime-gpu-type", default="A100_80GB")
    parser.add_argument("--vast-disk-gb", type=int, default=100)
    parser.add_argument("--show-debug-logs", action="store_true")
    parser.add_argument("--min-vram-gb", type=float, default=78.0)
    parser.add_argument("--tool-call-parser", default="")
    parser.add_argument("--reasoning-parser", default="")
    parser.add_argument("--max-hourly-cost", type=float, default=3.0)
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shots", type=Path)
    args = parser.parse_args()
    if not args.live and not args.rehearse:
        parser.error("--live authorizes paid compute and its destruction.")
    if not 1 <= args.max_minutes <= 180:
        parser.error("Use a deadline of 1-180 minutes.")
    return asyncio.run(validate(args))


if __name__ == "__main__":
    raise SystemExit(main())

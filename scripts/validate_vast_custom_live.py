#!/usr/bin/env python3
"""Opt-in Custom Deploy certification, including vision and tensor parallelism."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch
import uuid

import requests
from textual.widgets import Button, Input, Select, Switch

from llm_launchpad.core.connection_store import remove_connection
from llm_launchpad.core.vast_deployment import VastDeploymentBackend
from llm_launchpad.core.vast_runtime import GPU_INVENTORY_COMMAND, parse_gpu_inventory, verify_streaming
from llm_launchpad.core.vast_ssh import VastSsh
from llm_launchpad.core.vision_probe import assistant_text, image_probe_payload
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig, EndpointInfo, VastOfferQuery, VastProviderOptions
from llm_launchpad.tui.screens.deploy import LlamaCppDeployScreen, VastOffersLoaded, VllmDeployScreen
from scripts.validate_prime_live import LiveTuiApp
from scripts.validate_vast_live import account_credit, budget_estimate, long_stream, probe_chat
from scripts.validate_vast_tui_live import settle


class CustomVastApp(LiveTuiApp):
    """Run the real worker only for the exact rental approved by this test."""

    def __init__(self, backend: BackendType, name: str, offer_id: str, model: str, rehearse: bool) -> None:
        super().__init__(backend)
        self.authorized_name = name
        self.authorized_offer_id = offer_id
        self.authorized_model = model
        self.rehearse = rehearse

    def begin_deploy(self, config: DeploymentConfig) -> None:
        options = config.provider_options
        print(json.dumps({
            "model": config.model_name or config.repo_id,
            "required_vram_gb": config.required_vram_gb,
            "gpu_count": config.gpu_count, "n_gpu": config.n_gpu,
        }), flush=True)
        if (
            config.provider != ComputeProvider.VAST
            or config.app_name != self.authorized_name
            or not isinstance(options, VastProviderOptions)
            or options.offer_id != self.authorized_offer_id
            or (config.model_name or config.repo_id) != self.authorized_model
        ):
            raise ValueError("The live form selected an unauthorized deployment.")
        config.fallback_configs = ()
        if self.rehearse:
            self.deployed_config = config
            return
        super().begin_deploy(config)


def probe_vision(
    session: requests.Session, endpoint: EndpointInfo, config: DeploymentConfig,
) -> dict[str, Any]:
    """Check image inference using each engine's own vision configuration."""
    if config.vision is None or not config.vision.enabled:
        raise ValueError("The form did not enable image input.")
    if not endpoint.web_url or not endpoint.endpoint_api_key or not endpoint.served_model_name:
        raise ValueError("Vision verification requires a complete serving endpoint.")
    checks: dict[str, Any] = {"vision_verification": config.vision.verification.value}
    if config.backend == BackendType.LLAMACPP:
        if config.vision.projector is None:
            raise ValueError("llama.cpp vision requires a staged projector.")
        checks["projector"] = asdict(config.vision.projector)
    with session.post(
        endpoint.web_url + "/v1/chat/completions",
        headers={"Authorization": "Bearer " + endpoint.endpoint_api_key},
        json=image_probe_payload(endpoint.served_model_name), timeout=90,
    ) as response:
        response.raise_for_status()
        answer = assistant_text(response.json()["choices"][0]["message"])
    if not answer.strip():
        raise RuntimeError("Image request returned no text.")
    checks["image_answer"] = answer
    return checks


async def validate(args: argparse.Namespace) -> int:
    backend = VastDeploymentBackend()
    offer = backend.api.get_offer(args.offer_id, VastOfferQuery(gpu_count=None))
    before = account_credit(backend.api)
    estimate = budget_estimate(offer, args, before)
    kind = BackendType(args.backend)
    name = f"llp-vast-{kind.value}-custom-" + uuid.uuid4().hex[:10]
    app = CustomVastApp(kind, name, offer.id, args.model, args.rehearse)
    report = {
        "name": name, "backend": kind.value, "vision": args.vision,
        "offer": asdict(offer), "credit_before": before,
        "budget_usd": args.budget_usd, "estimated_cost_with_headroom_usd": estimate,
        "checks": {}, "success": False, "cleanup_confirmed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    def live_offers(screen: LlamaCppDeployScreen | VllmDeployScreen) -> None:
        # Inject the actual approved quote; production re-quotes before rent.
        screen.post_message(VastOffersLoaded([offer]))

    started = time.monotonic()
    save()
    print(f"CUSTOM {name}: {offer.gpu_count}x {offer.gpu_type}, budget ${estimate:.3f}", flush=True)
    instance_id = None
    try:
        screen_type = VllmDeployScreen if kind == BackendType.VLLM else LlamaCppDeployScreen
        with patch.object(screen_type, "_run_fetch_vast_offers", live_offers):
            async with app.run_test(size=(120, 48)) as pilot:
                try:
                    async with asyncio.timeout(args.max_minutes * 60):
                        await pilot.pause()
                        if not isinstance(app.screen, screen_type):
                            app.push_screen(screen_type())
                        await settle(pilot, lambda: isinstance(app.screen, screen_type), "Custom Deploy")
                        screen = app.screen
                        assert isinstance(screen, (LlamaCppDeployScreen, VllmDeployScreen))
                        suffix = "vllm" if kind == BackendType.VLLM else "llama"
                        field = "#model-name" if kind == BackendType.VLLM else "#repo-id"
                        await settle(pilot, lambda: bool(screen.query(field)), "Model fields")
                        screen.query_one(field, Input).value = args.model
                        screen.query_one("#vision-mode", Select).value = "on" if args.vision else "off"
                        screen.query_one(f"#provider-{suffix}", Select).value = "vast"
                        await settle(
                            pilot, lambda: screen._selected_vast_offer_id == offer.id,
                            "Approved rental", seconds=30,
                        )
                        screen.query_one(f"#toggle-advanced-{suffix}", Button).press()
                        await pilot.pause()
                        screen.query_one(f"#app-name-{suffix}", Input).value = name
                        if kind == BackendType.VLLM:
                            screen.query_one("#fast-boot", Switch).value = True
                        else:
                            screen.query_one("#quant", Input).value = args.quant
                            screen.query_one("#server-args", Input).value = "--ctx-size 4096 --parallel 1 --n-gpu-layers all --jinja"
                        if args.projector_file:
                            screen.query_one("#projector-file", Input).value = args.projector_file
                        await pilot.pause()
                        app.save_screenshot(filename=args.report.stem + "-form.svg", path=str(args.report.parent))
                        deploy_button = "#deploy-vllm-btn" if kind == BackendType.VLLM else "#deploy-btn"
                        screen.query_one(deploy_button, Button).press()
                        await settle(pilot, lambda: app.deployed_config is not None, "Deployment worker")
                        if args.rehearse:
                            report["rehearsal"] = True
                            report["success"] = True
                            return 0
                        last_progress = 0.0
                        while app.capture_monitor is None or app.capture_monitor.last_done is None:
                            await asyncio.sleep(1)
                            record = backend.state.load(name)
                            if record and record.instance_id:
                                instance_id = record.instance_id
                                report["instance_id"] = instance_id
                                save()
                            if time.monotonic() - last_progress > 30:
                                print(f"Waiting for Custom Deploy ({time.monotonic() - started:.0f}s)", flush=True)
                                last_progress = time.monotonic()
                        done = app.capture_monitor.last_done
                        if not done.success:
                            raise RuntimeError(done.detail or "Custom Deploy failed.")
                        config = app.deployed_config
                        assert config is not None and instance_id is not None
                        checks = report["checks"]
                        checks["deploy_and_warmup_seconds"] = time.monotonic() - started
                        checks["form_gpu_count"] = config.gpu_count
                        checks["tensor_parallel_size"] = config.n_gpu
                        endpoint = await asyncio.to_thread(backend.connect, instance_id)
                        assert endpoint.web_url and endpoint.endpoint_api_key and endpoint.served_model_name
                        instance = backend.api.get_instance(instance_id)
                        assert instance is not None
                        ssh = VastSsh(backend.state.directory(name))
                        devices = parse_gpu_inventory(ssh.run(instance, GPU_INVENTORY_COMMAND))
                        checks["devices_after_load"] = [asdict(device) for device in devices]
                        if len(devices) != offer.gpu_count or any(d.memory_used_mib < 512 for d in devices):
                            raise RuntimeError("Not every rented GPU holds model weights.")
                        with requests.Session() as session:
                            session.trust_env = False
                            if args.vision:
                                checks.update(await asyncio.to_thread(probe_vision, session, endpoint, config))
                            else:
                                await asyncio.to_thread(probe_chat, session, endpoint, checks)
                                checks["long_stream"] = await asyncio.to_thread(long_stream, session, endpoint, args.stream_seconds)
                        verify_streaming(endpoint.web_url, endpoint.endpoint_api_key, endpoint.served_model_name)
                        checks["post_probe_chat"] = True
                        report["success"] = True
                finally:
                    await asyncio.to_thread(backend.destroy, name=name)
                    remove_connection(name)
    except Exception as exc:
        detail = str(exc)
        if app.deployed_config and app.deployed_config.endpoint_api_key:
            detail = detail.replace(app.deployed_config.endpoint_api_key, "[redacted]")
        report["error"] = detail
        report["notifications"] = app.live_notifications[-10:]
        print("FAILED: " + detail, flush=True)
    finally:
        try:
            await asyncio.to_thread(backend.destroy, name=name)
            report["cleanup_confirmed"] = backend.state.load(name) is None and (
                not instance_id or backend.api.get_instance(instance_id) is None
            )
        except Exception as exc:
            report["cleanup_error"] = str(exc)
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            report["credit_after"] = account_credit(backend.api)
            report["credit_delta_usd"] = before - report["credit_after"]
        except Exception:
            report["credit_after"] = None
        save()
        print(f"Report {args.report}; cleanup confirmed: {report['cleanup_confirmed']}", flush=True)
    return 0 if report["success"] and report["cleanup_confirmed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--rehearse", action="store_true")
    parser.add_argument("--offer-id", required=True)
    parser.add_argument("--backend", choices=["llamacpp", "vllm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", default="Q8_0")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--projector-file")
    parser.add_argument("--max-hourly-cost", type=float, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--max-minutes", type=int, default=20)
    parser.add_argument("--transfer-gb", type=float, default=30)
    parser.add_argument("--stream-seconds", type=int, default=60)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.live and not args.rehearse:
        parser.error("--live authorizes a paid rental and its destruction.")
    if not 1 <= args.max_minutes <= 60 or not all(
        math.isfinite(value) and value > 0 for value in
        (args.max_hourly_cost, args.budget_usd, args.transfer_gb, args.stream_seconds)
    ):
        parser.error("Use positive finite budgets and a deadline of 1–60 minutes.")
    return asyncio.run(validate(args))


if __name__ == "__main__":
    raise SystemExit(main())

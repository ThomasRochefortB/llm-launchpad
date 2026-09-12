#!/usr/bin/env python3
"""Opt-in Fast Deploy validation on one Vast host with real GGUF metadata."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any, Callable
from unittest.mock import patch
import uuid

from textual.widgets import Button, Input, OptionList

from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.connection_store import remove_connection
from llm_launchpad.core.hf_models import fetch_gguf_quant_metadata
from llm_launchpad.core.modal_gpu import fetch_modal_gpu_catalog
from llm_launchpad.core.quick_deploy import QuickDeployCatalogInfo, QuickDeployModel, list_quick_deploy_recipes
from llm_launchpad.core.quick_deploy_refresh import _profiles_for_quant
from llm_launchpad.core.vast_backend import VastBackend
from llm_launchpad.core.vast_deployment import VastDeploymentBackend
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig, VastOfferQuery
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen
from scripts.validate_prime_live import LiveFastDeployScreen, LiveTuiApp
from scripts.validate_vast_live import account_credit


class VastLiveTuiApp(LiveTuiApp):
    def begin_deploy(self, config: DeploymentConfig) -> None:
        if config.provider != ComputeProvider.VAST:
            raise ValueError("This live test only authorizes Vast rentals.")
        config.fallback_configs = ()
        super().begin_deploy(config)


async def settle(pilot: Any, ready: Callable[[], bool], what: str, seconds: float = 10.0) -> None:
    """Let the real TUI reach a state instead of assuming one frame is enough."""

    for _ in range(int(seconds * 10)):
        if ready():
            return
        await pilot.pause()
        await asyncio.sleep(0.1)
    raise RuntimeError(f"{what} did not settle within {seconds:.0f}s.")


async def validate(args: argparse.Namespace) -> int:
    backend = VastDeploymentBackend()
    offer = backend.api.get_offer(args.offer_id, VastOfferQuery())
    price = offer.costs.total_per_hour_usd
    down, up = offer.costs.download_per_gb_usd, offer.costs.upload_per_gb_usd
    before = account_credit(backend.api)
    if price is None or price > 0.2 or down is None or up is None or price / 4 + 20 * down + up > 1 or before < 1:
        raise ValueError("TUI live test requires <$0.20/hour and <$1 with transfer headroom.")
    modal = fetch_modal_gpu_catalog()
    metadata = fetch_gguf_quant_metadata("Qwen/Qwen3-0.6B-GGUF", inspect_serving=True)
    if not metadata.context_length:
        raise ValueError("Live GGUF metadata did not establish full context.")
    profiles = tuple(_profiles_for_quant(
        repo_id="Qwen/Qwen3-0.6B-GGUF", display_name="Qwen3 0.6B (live validation)",
        slug_hint="vast-live", context_tokens=metadata.context_length, quant="Q8_0", metadata=metadata,
        modal_gpu_catalog=modal, llamacpp_runtime_id="llama.cpp-b10689-cuda12",
    ))
    model = QuickDeployModel(
        id="vast-live-qwen3", display_name="Qwen3 0.6B (live validation)",
        recipes=list_quick_deploy_recipes(profiles), profiles=profiles, max_context_tokens=metadata.context_length,
    )
    snapshot = replace(aggregate_compute_availability(modal_catalog=modal), vast_offers=(offer,), vast_configured=True)
    name = "llp-vast-llamacpp-tui-" + uuid.uuid4().hex[:10]
    report = {
        "name": name, "offer": asdict(offer), "full_context_tokens": metadata.context_length,
        "credit_before": before, "success": False, "cleanup_confirmed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    app = VastLiveTuiApp(BackendType.LLAMACPP, fast_snapshot=snapshot, fast_provider=ComputeProvider.VAST)
    started = time.monotonic()
    try:
        # Only inject the small validation catalog and the actual offer snapshot;
        # selection, confirmation, deployment worker, and certification are real.
        # The confirmation screen resolves its profile from the shared catalog,
        # so the validation profile has to be visible there too.
        with patch("llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models", return_value=(model,)), patch(
            "llm_launchpad.tui.screens.fast_deploy.get_quick_deploy_catalog_info",
            return_value=QuickDeployCatalogInfo("Live validation model", "", "", is_live=True),
        ), patch("llm_launchpad.core.quick_deploy.list_quick_deploy_profiles", return_value=profiles):
            async with app.run_test(size=(120, 42)) as pilot:
                try:
                    async with asyncio.timeout(900):
                        await pilot.pause()
                        await asyncio.sleep(0.2)
                        # Textual dispatches on_mount up the MRO, so TuiApp's
                        # own handler lands the main menu on top of the
                        # harness screen. Re-push Fast Deploy before driving it.
                        if not isinstance(app.screen, FastDeployScreen):
                            app.push_screen(LiveFastDeployScreen(snapshot))
                        await settle(
                            pilot,
                            lambda: isinstance(app.screen, FastDeployScreen)
                            and bool(app.screen.query("#fast-deploy-model-search")),
                            "Fast Deploy",
                        )
                        screen = app.screen
                        assert isinstance(screen, FastDeployScreen)
                        screen._open_model(model.id)
                        await settle(pilot, lambda: screen._phase == "infra", "Placement list")
                        row = next((row for row in screen._infra_rows.values() if row.plan.quote.provider == ComputeProvider.VAST), None)
                        if row is None:
                            raise RuntimeError(f"No Vast placement fits full context; phase={screen._phase}, rows={len(screen._infra_rows)}")
                        report["vast_price_per_hour"] = row.plan.quote.price_per_hour_usd
                        report["tiers"] = [{"tier": tier.key, "provider": tier.plan.quote.provider.value} for tier in screen._tiers_for(tuple(screen._infra_rows.values()))]
                        key = next(key for key, candidate in screen._infra_rows.items() if candidate is row)
                        choices = screen.query_one(OptionList)
                        choices.highlighted = choices.get_option_index(key)
                        app.save_screenshot(filename=args.report.stem + "-offers.svg", path=str(args.report.parent))
                        screen._choose(key)
                        await settle(
                            pilot,
                            lambda: isinstance(app.screen, QuickDeployScreen)
                            and bool(app.screen.query("#quick-deploy-btn")),
                            "Confirmation form",
                        )
                        confirm = app.screen
                        assert isinstance(confirm, QuickDeployScreen)
                        confirm.query_one("#quick-app-name", Input).value = name
                        confirm.query_one("#quick-instance-name", Input).value = name.removeprefix("llp-vast-llamacpp-")
                        app.save_screenshot(filename=args.report.stem + "-confirm.svg", path=str(args.report.parent))
                        if args.rehearse:
                            report["success"] = True
                            report["rehearsal"] = True
                            print(f"Rehearsal reached the confirmation form for {name}; no rental created.", flush=True)
                            return 0
                        print(f"TUI deploying {name}: {metadata.context_length} tokens, ${price:.4f}/hr", flush=True)
                        confirm.query_one("#quick-deploy-btn", Button).press()
                        await pilot.pause()
                        while app.capture_monitor is None or app.capture_monitor.last_done is None:
                            await asyncio.sleep(1)
                        done = app.capture_monitor.last_done
                        report["success"] = done.success
                        report["operation"] = done.operation.value
                        report["detail"] = done.detail
                        config = app.deployed_config
                        report["certification"] = config.placement_assessment.certification.value if config and config.placement_assessment else None
                        print(f"TUI {done.operation.value}: success={done.success}, detail={done.detail}", flush=True)
                finally:
                    record = backend.state.load(name)
                    if record:
                        report["instance_id"] = record.instance_id
                    await asyncio.to_thread(backend.destroy, name=name)
                    remove_connection(name)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # The target is the unique name owned by this run, even if the UI failed.
        record = backend.state.load(name)
        instance_id = record.instance_id if record else report.get("instance_id")
        report["instance_id"] = instance_id
        try:
            await asyncio.to_thread(backend.destroy, name=name)
            report["cleanup_confirmed"] = backend.state.load(name) is None and (not instance_id or backend.api.get_instance(instance_id) is None)
        except Exception as exc:
            report["cleanup_error"] = str(exc)
        report["elapsed_seconds"] = time.monotonic() - started
        report["credit_after"] = account_credit(VastBackend())
        report["credit_delta_usd"] = before - report["credit_after"]
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report {args.report}; cleanup confirmed: {report['cleanup_confirmed']}", flush=True)
    return 0 if report["success"] and report["cleanup_confirmed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--rehearse", action="store_true", help="Drive the UI up to confirmation without renting.")
    parser.add_argument("--offer-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.live and not args.rehearse:
        parser.error("--live authorizes one rental, up to $1 including transfer headroom, and destruction after testing.")
    return asyncio.run(validate(args))


if __name__ == "__main__":
    raise SystemExit(main())

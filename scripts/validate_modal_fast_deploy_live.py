#!/usr/bin/env python3
"""Run an opt-in, TUI-driven Modal Fast Deploy live validation."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import traceback
from typing import Any
from urllib.parse import urlsplit

import requests
from textual.widgets import Button, Input, OptionList, Select, Switch

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.modal_auth import get_modal_auth_status
from llm_launchpad.core.modal_gpu import fetch_modal_gpu_catalog
from llm_launchpad.core.naming import build_deployment_name
from llm_launchpad.core.llamacpp_planner import (
    KV_CACHE_TYPES,
    assess_memory_placement,
    memory_for_cache_type,
    with_cache_type,
)
from llm_launchpad.core.quick_deploy import (
    get_quick_deploy_profile,
    list_quick_deploy_models,
)
from llm_launchpad.core.warmup import endpoint_root_url, extract_effective_context
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, ServingObjective
from llm_launchpad.protocol.models import (
    ComputeAvailabilitySnapshot,
    DeploymentConfig,
    StorageSnapshot,
)
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen
from llm_launchpad.tui.workers import OperationDone, StorageLoaded
from scripts.validate_prime_live import (
    LiveFastDeployScreen,
    _llamacpp_management_evidence,
)


# The planner-generated catalog renames profiles when tuning changes, so the
# default is resolved against the live catalog rather than pinned by hand.
DEFAULT_PROFILE_ID = "qwen3-8-27b-xhigh-q2xl-cheap-rtx-pro-6000"
STOPPED_STATES = {"stopped", "stopping", "terminated", "archived"}
# A long prompt is what actually exercises the placement the planner certified.
# Short probes pass even when the KV cache was never budgeted for full context.
LONG_PROMPT_CONTEXT_FRACTION = 0.25
MAX_LONG_PROMPT_TOKENS = 32768
# llama.cpp tokenizes far denser than this for English prose; four characters
# per token is the conservative direction because it undershoots the prompt.
CHARS_PER_TOKEN = 4
# Depths as a fraction of the probe prompt. Quantized caches degrade oldest
# entries first, so the shallow depth is the one that discriminates.
RECALL_DEPTHS = (0.05, 0.25, 0.5, 0.75, 0.95)
# A probe far below the advertised window barely exercises the cache at all:
# 16K against a 262K model tests about 6% of the range where quantization
# damage concentrates. Default to a deep probe and let the caller trade it
# down when prefill time matters more than confidence.
DEFAULT_RECALL_CONTEXT_TOKENS = 131072



class CaptureMonitorScreen(MonitorScreen):
    """Monitor that preserves the final operation result."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.last_done: OperationDone | None = None

    def on_operation_done(self, message: OperationDone) -> None:
        self.last_done = message
        super().on_operation_done(message)


class ModalFastDeployApp(TuiApp):
    """Real Modal TUI workflow isolated from the user's OpenCode config."""

    CSS_PATH = Path(__file__).resolve().parents[1] / "llm_launchpad" / "tui" / "theme.tcss"

    def __init__(
        self,
        snapshot: ComputeAvailabilitySnapshot,
        cache_type: str | None = None,
    ) -> None:
        super().__init__(mouse_enabled=False)
        self.snapshot = snapshot
        self.cache_type = cache_type
        self.selected_gpu_memory_gb: float | None = None
        self.deployed_config: DeploymentConfig | None = None
        self.capture_monitor: CaptureMonitorScreen | None = None
        self.live_notifications: list[tuple[str, str]] = []

    def on_mount(self) -> None:
        self.push_screen(LiveFastDeployScreen(self.snapshot))

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        _ = force
        poster = getattr(receiver, "post_message", None)
        if callable(poster):
            poster(StorageLoaded(StorageSnapshot(llamacpp_models=[], vllm_models=[])))

    def begin_deploy(self, config: DeploymentConfig) -> None:
        if self.cache_type is not None:
            if self.selected_gpu_memory_gb is None:
                raise RuntimeError("No per-GPU memory was captured for re-planning.")
            _apply_cache_type(
                config,
                self.cache_type,
                gpu_memory_gb=self.selected_gpu_memory_gb,
            )
        if not config.app_name:
            config.app_name = build_deployment_name(
                config.provider,
                config.backend,
                config.instance_name,
            )
        self.deployed_config = config
        monitor = CaptureMonitorScreen(
            title="Deploy",
            deploy_backend=config.backend,
            summarize_backend_logs=True,
            show_debug_logs=config.show_debug_logs,
        )
        self.capture_monitor = monitor
        self.push_screen(monitor)
        self.run_worker(
            lambda: self._run_deploy(config, monitor),
            name="deploy-worker",
            thread=True,
        )

    def begin_capture_stop(self, config: DeploymentConfig) -> None:
        monitor = CaptureMonitorScreen(title="Stop")
        self.capture_monitor = monitor
        self.push_screen(monitor)

        def start_stop() -> None:
            self.run_worker(
                lambda: self._run_stop(
                    config.backend,
                    config.app_name or "",
                    config.app_name or "",
                    monitor,
                    provider=ComputeProvider.MODAL,
                ),
                name="stop-worker",
                thread=True,
            )

        self.call_after_refresh(start_stop)

    def _sync_opencode(self, **_: object) -> None:
        """Keep the live TUI from changing the user's OpenCode config."""

    def notify(
        self,
        message: object,
        *,
        severity: str = "information",
        **kwargs: object,
    ) -> None:
        self.live_notifications.append((str(message), severity))
        super().notify(str(message), severity=severity, **kwargs)


def _apply_cache_type(
    config: DeploymentConfig,
    cache_type: str,
    *,
    gpu_memory_gb: float,
) -> None:
    """Re-plan one deployment at a different KV precision, in place.

    Placement, context, and parallelism are held constant so a comparison
    across cache types measures the cache and nothing else. The KV term is
    rescaled from the estimate the catalog already carries, so this needs no
    Hugging Face round trip per arm.
    """

    tuning = config.runtime_tuning
    assessment = config.placement_assessment
    if tuning is None or config.serving_requirements is None:
        raise RuntimeError("Config carries no planner serving contract to re-plan.")
    if assessment is None:
        raise RuntimeError("Config carries no placement assessment to re-plan.")
    retuned = with_cache_type(tuning, cache_type)
    memory = memory_for_cache_type(
        assessment.memory,
        from_cache_type=tuning.cache_type_k,
        to_cache_type=cache_type,
    )
    replanned = assess_memory_placement(
        memory,
        model_id=config.repo_id or "",
        revision=None,
        quant=config.quant,
        runtime_id=config.llamacpp_runtime_id,
        requirements=config.serving_requirements,
        tuning=retuned,
        gpu_type=config.gpu_type or "",
        gpu_count=config.gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        price_per_hour_usd=config.serving_requirements.max_hourly_cost_usd,
    )
    if not replanned.fits or not replanned.gpu_resident:
        raise RuntimeError(
            f"{cache_type} does not fit this placement: {replanned.rejection_reason}"
        )
    config.runtime_tuning = retuned
    config.placement_assessment = replanned
    config.required_vram_gb = replanned.memory.total_gb


async def _navigate_and_submit(
    pilot: Any,
    app: ModalFastDeployApp,
    *,
    run_id: str,
    profile_id: str,
    objective: str,
    gpu_type: str | None,
) -> dict[str, Any]:
    if not isinstance(app.screen, FastDeployScreen):
        app.push_screen(LiveFastDeployScreen(app.snapshot))
    for _ in range(50):
        await pilot.pause()
        if isinstance(app.screen, FastDeployScreen):
            break
        await asyncio.sleep(0.1)
    else:
        stack = [type(candidate).__name__ for candidate in app.screen_stack]
        raise RuntimeError(
            f"Fast Deploy model picker did not mount; current={type(app.screen).__name__}, "
            f"stack={stack}, notifications={app.live_notifications[-3:]}"
        )

    screen = app.screen
    assert isinstance(screen, FastDeployScreen)
    model = next(
        (
            candidate
            for candidate in list_quick_deploy_models()
            if any(profile.id == profile_id for profile in candidate.profiles)
        ),
        None,
    )
    if model is None:
        raise RuntimeError(f"{profile_id} is missing from the Fast Deploy catalog.")
    option_list = screen.query_one("#fast-deploy-list", OptionList)
    model_index = next(
        (
            index
            for index in range(option_list.option_count)
            if option_list.get_option_at_index(index).id == model.id
        ),
        None,
    )
    if model_index is None:
        raise RuntimeError(f"{profile_id} is missing from the rendered model picker.")
    option_list.highlighted = model_index
    await pilot.press("enter")

    for _ in range(50):
        await pilot.pause()
        if screen._phase == "infra" and screen._infra_rows:
            break
        await asyncio.sleep(0.1)
    else:
        raise RuntimeError("Modal infrastructure choices did not load.")
    candidates = [
        row
        for row in screen._infra_rows.values()
        if row.plan.quote.provider == ComputeProvider.MODAL
        and row.profile.id == profile_id
    ]
    if not candidates:
        raise RuntimeError(
            f"No live Modal placement holds the full context of {profile_id}."
        )
    if gpu_type is None:
        # The first row is the screen's own recommendation.
        selected = candidates[0]
    else:
        wanted = gpu_type.strip().casefold()
        selected = next(
            (row for row in candidates if row.plan.quote.gpu_type.casefold() == wanted),
            None,
        )
        if selected is None:
            offered = sorted(
                f"{row.plan.quote.gpu_type}x{row.plan.quote.gpu_count}"
                for row in candidates
            )
            raise RuntimeError(
                f"No {gpu_type} placement is offered for {profile_id}; "
                f"available: {', '.join(offered)}."
            )
    app.selected_gpu_memory_gb = selected.plan.quote.gpu_memory_gb
    option_list = screen.query_one("#fast-deploy-list", OptionList)
    option_list.highlighted = next(
        index
        for index in range(option_list.option_count)
        if option_list.get_option_at_index(index).id == selected.plan.quote.id
    )
    await pilot.press("enter")

    for _ in range(50):
        await pilot.pause()
        if isinstance(app.screen, QuickDeployScreen):
            break
        await asyncio.sleep(0.1)
    else:
        raise RuntimeError("Fast Deploy confirmation screen did not mount.")
    confirm = app.screen
    assert isinstance(confirm, QuickDeployScreen)
    confirm.query_one("#toggle-advanced-quick", Button).press()
    await pilot.pause()
    confirm.query_one("#quick-instance-name", Input).value = f"{run_id}-modal"
    # A certified plan always warms up, so the screen only renders the opt-out
    # toggle for legacy plans that carry no serving requirements.
    warmup_toggles = confirm.query("#quick-warmup")
    if warmup_toggles:
        warmup_toggles.first(Switch).value = True
    elif confirm.plan.recipe.serving_requirements is None:
        raise RuntimeError("Legacy plan is missing its warmup toggle.")
    objective_selects = confirm.query("#quick-objective")
    if objective_selects:
        objective_selects.first(Select).value = objective
        await pilot.pause()
    elif objective != ServingObjective.GENERAL_PURPOSE.value:
        raise RuntimeError(
            f"Plan {selected.plan.quote.id} offers no objective selector; "
            f"cannot honor --objective {objective}."
        )
    confirm.query_one("#quick-deploy-btn", Button).press()

    for _ in range(30):
        await pilot.pause()
        if app.deployed_config is not None:
            assessment = confirm.plan.assessment
            return {
                "objective": confirm._objective.value,
                "model_id": model.id,
                "profile_id": selected.profile.id,
                "plan_id": selected.plan.quote.id,
                "gpu_type": selected.plan.quote.gpu_type,
                "gpu_count": selected.plan.quote.gpu_count,
                "gpu_memory_gb": selected.plan.quote.gpu_memory_gb,
                "price_per_hour_usd": selected.plan.quote.price_per_hour_usd,
                "assessment": (
                    None
                    if assessment is None
                    else {
                        "fingerprint": assessment.fingerprint,
                        "fits": assessment.fits,
                        "gpu_resident": assessment.gpu_resident,
                        "certification": assessment.certification.value,
                        "memory": asdict(assessment.memory),
                        "tuning": asdict(assessment.tuning),
                    }
                ),
            }
        await asyncio.sleep(0.1)
    detail = app.live_notifications[-1][0] if app.live_notifications else "no notification"
    raise RuntimeError(f"Fast Deploy confirmation did not submit: {detail}")


async def _wait_operation(
    app: ModalFastDeployApp,
    *,
    timeout_seconds: int,
    label: str,
) -> list[str]:
    started = time.monotonic()
    last_reported = 0
    while time.monotonic() - started < timeout_seconds:
        monitor = app.capture_monitor
        lines = (
            list(monitor.log_viewer.log_widget.lines)
            if monitor is not None and monitor.is_mounted
            else []
        )
        if len(lines) > last_reported:
            for line in lines[last_reported:]:
                if line.strip():
                    print(f"[{label}] {line}", flush=True)
            last_reported = len(lines)
        if monitor is not None and monitor.last_done is not None:
            if not monitor.last_done.success:
                raise RuntimeError(monitor.last_done.detail or f"Modal {label} failed.")
            return lines
        await asyncio.sleep(2)
    raise TimeoutError(f"Modal {label} exceeded {timeout_seconds} seconds.")


def _long_prompt(context_tokens: int, sentinel: str) -> tuple[str, int]:
    """Build a filler prompt that occupies a real share of the context window."""

    target_tokens = min(
        MAX_LONG_PROMPT_TOKENS,
        max(512, int(context_tokens * LONG_PROMPT_CONTEXT_FRACTION)),
    )
    filler_sentence = (
        "Launchpad certifies that this endpoint keeps the whole model and its "
        "key-value cache resident in GPU memory at the advertised context. "
    )
    repetitions = max(1, (target_tokens * CHARS_PER_TOKEN) // len(filler_sentence))
    body = filler_sentence * repetitions
    prompt = (
        "Read the following notice, then reply with exactly the identifier on "
        f"the final line and nothing else.\n\n{body}\n\nIdentifier: {sentinel}"
    )
    return prompt, target_tokens


def _probe_endpoint(
    config: DeploymentConfig,
    endpoint: str,
    *,
    context_tokens: int,
) -> dict[str, Any]:
    if urlsplit(endpoint).scheme != "https":
        raise AssertionError("Modal Fast Deploy did not return an HTTPS endpoint.")
    base = endpoint_root_url(endpoint)
    models = requests.get(f"{base}/v1/models", timeout=300)
    if models.status_code != 200:
        raise AssertionError(f"Modal models endpoint returned HTTP {models.status_code}.")

    props = requests.get(f"{base}/props", timeout=60)
    if props.status_code != 200:
        raise AssertionError(f"llama.cpp /props returned HTTP {props.status_code}.")
    effective_context = extract_effective_context(props.json())
    if effective_context is None:
        raise AssertionError("llama.cpp /props did not report an effective context size.")
    if effective_context < context_tokens:
        raise AssertionError(
            f"Endpoint advertises {context_tokens:,} tokens but serves only "
            f"{effective_context:,}."
        )

    sentinel = f"MODAL-FASTDEPLOY-{int(time.time())}"
    prompt, prompt_tokens = _long_prompt(context_tokens, sentinel)
    started = time.monotonic()
    completion = requests.post(
        f"{base}/v1/chat/completions",
        json={
            "model": config.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            # A reasoning model spends this budget thinking before emitting any
            # content, so a small cap truncates mid-reasoning and looks exactly
            # like the endpoint failing to answer.
            "max_tokens": 512,
            "reasoning_effort": "low",
            "temperature": 0,
        },
        timeout=900,
    )
    elapsed = time.monotonic() - started
    if completion.status_code != 200:
        raise AssertionError(
            f"Modal chat endpoint returned HTTP {completion.status_code}: "
            f"{completion.text[-500:]}"
        )
    payload = completion.json()
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    answer = message.get("content") or message.get("reasoning_content") or ""
    if sentinel not in answer:
        raise AssertionError(
            "Modal chat endpoint did not return the requested sentinel "
            f"(finish_reason={choice.get('finish_reason')!r}): {answer.strip()[:200]!r}"
        )
    usage = payload.get("usage") or {}
    output_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "endpoint_scheme": "https",
        "models_status": models.status_code,
        "chat_status": completion.status_code,
        "chat_sentinel": sentinel,
        "advertised_context_tokens": context_tokens,
        "effective_context_tokens": effective_context,
        "long_prompt_target_tokens": prompt_tokens,
        "long_prompt_usage": usage,
        "long_prompt_seconds": round(elapsed, 3),
        # End-to-end: this divides output tokens by wall time that is dominated
        # by prefilling the long prompt, so it is far below the decode rate.
        # Per-phase throughput comes from the calibration curve, not from here.
        "long_prompt_end_to_end_output_tokens_per_second": (
            round(output_tokens / elapsed, 2) if output_tokens and elapsed > 0 else None
        ),
        "long_prompt_end_to_end_total_tokens_per_second": (
            round(int(usage.get("total_tokens") or 0) / elapsed, 2)
            if usage.get("total_tokens") and elapsed > 0
            else None
        ),
    }



def _recall_probe(base: str, model: str, context_tokens: int) -> dict[str, Any]:
    """Ask for a fact buried at several depths in a long prompt.

    Cache quantization loses fidelity on stored entries, so recall of an early
    fact is the cheapest signal that a precision is too lossy to ship.
    """

    filler = (
        "The deployment pipeline records throughput, latency, and error rate "
        "for every certified serving configuration. "
    )
    total_chars = context_tokens * CHARS_PER_TOKEN
    results: list[dict[str, Any]] = []
    for depth in RECALL_DEPTHS:
        secret = f"{int(depth * 100):02d}-{int(time.time()) % 100000}"
        needle = f"The certification code for this run is {secret}. "
        prefix_chars = int(total_chars * depth)
        body = (
            filler * max(1, prefix_chars // len(filler))
            + needle
            + filler * max(1, (total_chars - prefix_chars) // len(filler))
        )
        try:
            response = requests.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"{body}\n\nWhat is the certification code mentioned "
                                "above? Reply with only the code."
                            ),
                        }
                    ],
                    # Reasoning models spend this budget thinking before they
                    # emit any content, so a small cap returns a null answer and
                    # would look identical to a recall failure.
                    "max_tokens": 512,
                    "reasoning_effort": "low",
                    "temperature": 0,
                },
                timeout=900,
            )
            choice = (response.json().get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content") or message.get("reasoning_content") or ""
            results.append(
                {
                    "depth": depth,
                    "found": secret in text,
                    "finish_reason": choice.get("finish_reason"),
                    "reply": text.strip()[:160],
                }
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a failed recall
            results.append({"depth": depth, "found": False, "error": str(exc)[:200]})
    return {
        "context_tokens": context_tokens,
        "recalled": sum(1 for row in results if row.get("found")),
        "attempted": len(results),
        "detail": results,
    }


def _certification_evidence(
    entry: dict[str, Any],
    config: DeploymentConfig,
    context_tokens: int,
) -> dict[str, Any]:
    """Assert the full-context, GPU-only contract the planner promised."""

    requirements = config.serving_requirements
    if requirements is None:
        raise AssertionError("Fast Deploy produced no serving requirements.")
    if requirements.context_tokens != context_tokens:
        raise AssertionError(
            f"Fast Deploy requested {requirements.context_tokens:,} tokens instead of "
            f"the catalog's full {context_tokens:,}."
        )
    if not requirements.gpu_only:
        raise AssertionError("Fast Deploy did not require GPU-only placement.")

    attestation = entry.get("runtime_attestation")
    if not isinstance(attestation, dict):
        raise AssertionError("Published endpoint carries no runtime attestation.")
    effective = int(attestation.get("effective_context_tokens") or 0)
    if effective < context_tokens:
        raise AssertionError(
            f"Attested context {effective:,} is below the requested {context_tokens:,}."
        )
    if not attestation.get("gpu_resident"):
        raise AssertionError("Attestation reports the model is not fully GPU-resident.")
    gpu_layers = int(attestation.get("gpu_layers") or 0)
    total_layers = int(attestation.get("total_layers") or 0)
    if total_layers and gpu_layers < total_layers:
        raise AssertionError(
            f"Attestation offloaded layers to CPU: {gpu_layers}/{total_layers} on GPU."
        )
    performance = attestation.get("performance") or []
    if not performance:
        raise AssertionError("Calibration recorded no performance measurements.")
    tuning = config.runtime_tuning
    return {
        "requested_context_tokens": requirements.context_tokens,
        "objective": requirements.objective.value,
        "cache_type_k": tuning.cache_type_k if tuning else None,
        "cache_type_v": tuning.cache_type_v if tuning else None,
        "planned_memory": (
            asdict(config.placement_assessment.memory)
            if config.placement_assessment is not None
            else None
        ),
        "attestation": attestation,
    }


def _matching_live_apps(app_name: str) -> list[dict[str, str]]:
    rows = ModalBackend.list_apps()
    if rows is None:
        raise RuntimeError("Could not list Modal apps during cleanup verification.")
    return [
        {"name": row.name, "state": row.state}
        for row in rows
        if (row.name or "") == app_name
        and (row.state or "").strip().casefold() not in STOPPED_STATES
    ]


async def _wait_no_live_apps(
    app_name: str,
    *,
    timeout_seconds: int = 180,
) -> list[dict[str, str]]:
    started = time.monotonic()
    remaining: list[dict[str, str]] = []
    while time.monotonic() - started < timeout_seconds:
        remaining = await asyncio.to_thread(_matching_live_apps, app_name)
        if not remaining:
            return []
        await asyncio.sleep(2)
    raise TimeoutError(f"Modal app remained live after stop: {remaining}")


async def run(
    report_path: Path,
    *,
    profile_id: str,
    objective: str,
    gpu_type: str | None,
    cache_type: str | None,
    recall_probe: bool,
    recall_context_tokens: int = DEFAULT_RECALL_CONTEXT_TOKENS,
) -> int:
    auth = get_modal_auth_status()
    if not auth.authenticated:
        raise RuntimeError("Modal authentication is required for live validation.")
    profile = get_quick_deploy_profile(profile_id)
    context_tokens = profile.max_context_tokens
    snapshot = replace(
        aggregate_compute_availability(
            modal_catalog=fetch_modal_gpu_catalog(force_refresh=True)
        ),
        providers=(ComputeProvider.MODAL,),
    )
    run_id = f"fastdeploy-{int(time.time())}"
    report: dict[str, Any] = {
        "provider": ComputeProvider.MODAL.value,
        "profile": auth.profile,
        "profile_id": profile_id,
        "objective": objective,
        "cache_type": cache_type,
        "context_tokens": context_tokens,
        "run_id": run_id,
        "started_at_epoch": time.time(),
        "success": False,
        "evidence": {},
        "cleanup": {},
    }
    app = ModalFastDeployApp(snapshot, cache_type=cache_type)
    config: DeploymentConfig | None = None
    stopped = False
    error: str | None = None
    try:
        async with app.run_test(size=(150, 48)) as pilot:
            report["evidence"]["fast_deploy_selection"] = await _navigate_and_submit(
                pilot,
                app,
                run_id=run_id,
                profile_id=profile_id,
                objective=objective,
                gpu_type=gpu_type,
            )
            config = app.deployed_config
            if config is None:
                raise RuntimeError("TUI did not preserve the Modal deployment config.")
            lines = await _wait_operation(
                app,
                timeout_seconds=1800,
                label="deploy",
            )
            entry = app._deploy_connection_cache.get(config.app_name or "", {})
            endpoint = str(entry.get("base_url") or "")
            if not endpoint:
                raise RuntimeError("TUI did not preserve the Modal endpoint URL.")
            if config.repo_id != profile.repo_id or config.quant != profile.quant:
                raise AssertionError("Fast Deploy changed the curated model or quantization.")
            if config.backend != BackendType.LLAMACPP:
                raise AssertionError("Fast Deploy did not select llama.cpp.")
            report["evidence"]["config"] = {
                key: value
                for key, value in asdict(config).items()
                if key not in {"endpoint_api_key", "provider_options"}
            }
            report["evidence"]["certification"] = _certification_evidence(
                entry,
                config,
                context_tokens,
            )
            report["evidence"]["endpoint"] = await asyncio.to_thread(
                _probe_endpoint,
                config,
                endpoint,
                context_tokens=context_tokens,
            )
            if recall_probe:
                report["evidence"]["recall"] = await asyncio.to_thread(
                    _recall_probe,
                    endpoint_root_url(endpoint),
                    config.served_model_name or "",
                    min(recall_context_tokens, context_tokens),
                )
            report["evidence"]["management"] = await asyncio.to_thread(
                _llamacpp_management_evidence,
                config,
                provider=ComputeProvider.MODAL,
            )
            report["evidence"]["tui_milestones"] = lines[-100:]
            app.begin_capture_stop(config)
            await _wait_operation(app, timeout_seconds=300, label="stop")
            stopped = True
        remaining = await _wait_no_live_apps(config.app_name or "")
        report["cleanup"] = {"matching_live_apps": remaining}
        report["success"] = True
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["error"] = error
        traceback.print_exc()
        return 1
    finally:
        if config is not None and config.app_name and not stopped:
            try:
                events = app._orchestrator.stop_app(
                    config.backend,
                    app_name=config.app_name,
                    provider=ComputeProvider.MODAL,
                )
                list(events)
                stopped = True
            except Exception as cleanup_exc:
                report["cleanup"]["fallback_stop_error"] = str(cleanup_exc)
        if config is not None and config.app_name:
            try:
                report["cleanup"]["matching_live_apps"] = await _wait_no_live_apps(
                    config.app_name,
                )
            except Exception as cleanup_exc:
                report["cleanup"]["verification_error"] = str(cleanup_exc)
        report["finished_at_epoch"] = time.time()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Report: {report_path}", flush=True)
        if error:
            print(f"Failed: {error}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that this creates a billable Modal deployment.",
    )
    parser.add_argument(
        "--profile-id",
        default=DEFAULT_PROFILE_ID,
        help=(
            "Fast Deploy catalog profile to validate. Use a 131K or 262K context "
            "entry to exercise long-context placement."
        ),
    )
    parser.add_argument(
        "--gpu-type",
        default=None,
        help=(
            "Pin the placement to one GPU type (for example RTX-PRO-6000). "
            "Defaults to the screen's own recommendation."
        ),
    )
    parser.add_argument(
        "--recall-probe",
        action="store_true",
        help="Probe long-context recall before stopping (adds a few requests).",
    )
    parser.add_argument(
        "--recall-context",
        type=int,
        default=DEFAULT_RECALL_CONTEXT_TOKENS,
        help=(
            "Prompt size for the recall probe, capped at the model's window. "
            "A small value is cheap but barely exercises the cache."
        ),
    )
    parser.add_argument(
        "--cache-type",
        default=None,
        choices=list(KV_CACHE_TYPES),
        help=(
            "Override the KV cache precision before deploying. Holding the "
            "placement fixed and varying only this compares cache formats."
        ),
    )
    parser.add_argument(
        "--objective",
        default=ServingObjective.GENERAL_PURPOSE.value,
        choices=[item.value for item in ServingObjective],
        help="Serving objective to select under advanced options.",
    )
    parser.add_argument(
        "--report",
        default=f"/tmp/llm-launchpad-modal-fastdeploy-{int(time.time())}.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_live:
        print("Refusing to create billable resources without --confirm-live.")
        return 2
    return asyncio.run(
        run(
            Path(args.report),
            profile_id=args.profile_id,
            objective=args.objective,
            gpu_type=args.gpu_type,
            cache_type=args.cache_type,
            recall_probe=args.recall_probe,
            recall_context_tokens=args.recall_context,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

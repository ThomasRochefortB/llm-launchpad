#!/usr/bin/env python3
"""Opt-in Vast smoke validation with a deadline, cost headroom, and cleanup."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import shlex
import time
from typing import Any
import uuid

import requests

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.vast_backend import VastBackend
from llm_launchpad.core.vast_deployment import VastDeploymentBackend
from llm_launchpad.core.vast_runtime import VAST_RUNTIME_DIR, endpoint_healthy, verify_streaming
from llm_launchpad.core.vast_ssh import VastSsh
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, VisionMode
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent, StateChangeEvent
from llm_launchpad.protocol.models import DeploymentConfig, EndpointInfo, VastOfferQuery, VastProviderOptions


def account_credit(api: VastBackend) -> float:
    """Read credit without recording account credentials or personal fields."""
    row = api._request("GET", "/users/current/")
    return float(row["credit"]) + float(row["balance"])


def expired(signum: int, frame: Any) -> None:
    del signum, frame
    raise TimeoutError("Live validation reached its wall-clock deadline.")


def probe_chat(session: requests.Session, endpoint: EndpointInfo, report: dict[str, Any]) -> None:
    url = endpoint.web_url or ""
    payload = {
        "model": endpoint.served_model_name,
        "messages": [{"role": "user", "content": "Reply with the word hello."}],
        "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
    }
    rejected = []
    for headers in ({}, {"Authorization": "Bearer deliberately-invalid-live-test-key"}):
        with session.post(url + "/v1/chat/completions", json=payload, headers=headers, timeout=30) as response:
            rejected.append(response.status_code)
    report["unauthorized_statuses"] = rejected
    if any(code not in {401, 403} for code in rejected):
        raise RuntimeError("Endpoint accepted a missing or incorrect bearer token.")
    headers = {"Authorization": "Bearer " + (endpoint.endpoint_api_key or "")}
    started = time.monotonic()
    with session.post(url + "/v1/chat/completions", json=payload, headers=headers, timeout=60) as response:
        response.raise_for_status()
        data = response.json()
    if not data.get("choices"):
        raise RuntimeError("No chat completion returned.")
    report["chat"] = {"seconds": time.monotonic() - started, "usage": data.get("usage")}
    payload.update({
        "messages": [{"role": "user", "content": "Call get_weather for Paris."}],
        "max_tokens": 256,
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    })
    with session.post(url + "/v1/chat/completions", json=payload, headers=headers, timeout=90) as response:
        response.raise_for_status()
        data = response.json()
    calls = data["choices"][0]["message"].get("tool_calls", [])
    if not calls or calls[0]["function"]["name"] != "get_weather":
        raise RuntimeError("Structured tool call was not returned.")
    arguments = json.loads(calls[0]["function"]["arguments"])
    if "paris" not in str(arguments.get("city", "")).lower():
        raise RuntimeError("Tool arguments did not match the request.")
    report["tool_call"] = {"name": "get_weather", "arguments_valid": True}


def cached_model_files(listing: str) -> list[list[str]]:
    """Path and byte size of each cached model file, ordered for comparison."""
    return sorted(line.split()[:2] for line in listing.splitlines() if line.strip())


def long_stream(session: requests.Session, endpoint: EndpointInfo, seconds: int) -> dict[str, Any]:
    """Keep one real generation stream open, then test client cancellation."""
    started = time.monotonic()
    chunks = 0
    with session.post(
        (endpoint.web_url or "") + "/v1/chat/completions",
        headers={"Authorization": "Bearer " + (endpoint.endpoint_api_key or "")},
        json={
            "model": endpoint.served_model_name,
            "messages": [{"role": "user", "content": "Write a very long story about a voyage through space."}],
            "max_tokens": 131072, "stream": True, "ignore_eos": True,
            # A constrained repeating response keeps this transport stress
            # test producing text instead of sampling invisible special tokens
            # after an ordinary story has ended.
            "grammar": 'root ::= ("hello world ")+',
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=(5, 60), stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            if value == b"[DONE]":
                break
            data = json.loads(value)
            if data.get("choices"):
                chunks += 1
            if time.monotonic() - started >= seconds:
                return {"seconds": time.monotonic() - started, "chunks": chunks, "client_cancelled": True}
    raise RuntimeError(f"Long stream ended early after {time.monotonic() - started:.1f} seconds.")


def run(args: argparse.Namespace) -> int:
    api = VastBackend()
    backend = VastDeploymentBackend(api)
    offer = api.get_offer(args.offer_id, VastOfferQuery(disk_gb=100))
    costs = offer.costs
    rates = (costs.total_per_hour_usd, costs.download_per_gb_usd, costs.upload_per_gb_usd)
    if any(value is None or not math.isfinite(value) for value in rates):
        raise ValueError("Live validation requires known hourly and transfer prices.")
    hourly, download, upload = (float(value) for value in rates if value is not None)
    # This small model is <1 GB; allow 20 GB inbound for image extraction,
    # model/bootstrap downloads and retries, plus 1 GB outbound headroom.
    estimate = hourly * args.max_minutes / 60 + download * 20 + upload
    before = account_credit(api)
    if hourly > args.max_hourly_cost or estimate > args.budget_usd or before < args.budget_usd:
        raise ValueError("Offer or available credit does not meet the live budget guard.")
    name = "llp-vast-llamacpp-live-" + uuid.uuid4().hex[:10]
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(), "name": name,
        "offer": asdict(offer), "budget_usd": args.budget_usd,
        "estimated_cost_with_headroom_usd": estimate, "credit_before": before,
        "max_minutes": args.max_minutes, "checks": {}, "success": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    config = DeploymentConfig(
        backend=BackendType.LLAMACPP, provider=ComputeProvider.VAST,
        app_name=name, instance_name=name.removeprefix("llp-vast-llamacpp-"),
        repo_id="Qwen/Qwen3-0.6B-GGUF", quant="Q8_0", served_model_name="vast-live-qwen3",
        gpu_type=offer.gpu_type, gpu_count=1, do_deploy=True, vision_mode=VisionMode.OFF,
        server_args=shlex.join(["--ctx-size", "4096", "--parallel", "1", "--context-shift", "--n-gpu-layers", "all", "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}']),
        provider_options=VastProviderOptions(offer.id, 100, args.max_hourly_cost, offer.machine_id),
    )
    save()
    print(f"LIVE {name}: offer {offer.id}, ${hourly:.4f}/hr, headroom estimate ${estimate:.3f}", flush=True)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_minutes * 60)
    instance_id = None
    started = time.monotonic()
    try:
        endpoint = None
        for event in Orchestrator(vast_backend=backend).deploy(config):
            if isinstance(event, (StateChangeEvent, LogEvent)):
                print(getattr(event, "detail", "") or getattr(event, "line", ""), flush=True)
            if isinstance(event, OperationCompleteEvent):
                if not event.success:
                    raise RuntimeError(event.detail or "Deployment failed.")
                if isinstance(event.data, EndpointInfo):
                    endpoint = event.data
        if endpoint is None:
            raise RuntimeError("Deployment did not publish an endpoint.")
        instance_id = endpoint.app_id
        report["instance_id"] = instance_id
        checks = report["checks"]
        checks["cold_start_seconds"] = time.monotonic() - started
        save()
        print(f"READY in {checks['cold_start_seconds']:.1f}s; testing auth and tools", flush=True)
        with requests.Session() as session:
            session.trust_env = False
            probe_chat(session, endpoint, checks)
            checks["streaming"] = "passed by deployment verifier"
            save()
            print(f"Testing one continuous {args.stream_seconds}s generation stream", flush=True)
            checks["long_stream"] = long_stream(session, endpoint, args.stream_seconds)
        verify_streaming(endpoint.web_url or "", endpoint.endpoint_api_key or "", endpoint.served_model_name or "")
        checks["post_cancel_chat"] = True
        record = backend.state.load(name)
        assert record is not None
        instance = api.get_instance(instance_id)
        assert instance is not None
        ssh = VastSsh(backend.state.directory(name))
        logs = backend.logs(instance_id)
        checks["logs"] = {"lines": len(logs), "gpu_offload_reported": any("offloaded" in line for line in logs)}
        ssh.disconnect(instance)
        checks["disconnected_not_published"] = next(row for row in backend.list_deployments() if row.name == name).web_url is None
        connected = VastDeploymentBackend().connect(instance_id)
        checks["reconnect_same_url"] = connected.web_url == endpoint.web_url
        if not checks["disconnected_not_published"] or not checks["reconnect_same_url"]:
            raise RuntimeError("Tunnel lifecycle checks failed.")
        # Hugging Face downloads land as content-addressed blobs with no
        # .gguf suffix, so the cache is every file the runtime kept, by size.
        cache_listing = f"find {VAST_RUNTIME_DIR}/models -type f -printf '%p %s %T@\\n'"
        before_cache = ssh.run(instance, cache_listing)
        ssh.run(instance, "pkill -x llama-server")
        warm_start = time.monotonic()
        ssh.run(instance, f"nohup sh {VAST_RUNTIME_DIR}/runtime.sh > {VAST_RUNTIME_DIR}/server.log 2>&1 < /dev/null &")
        while not endpoint_healthy(endpoint.web_url or "", endpoint.endpoint_api_key or ""):
            if time.monotonic() - warm_start > 180:
                raise RuntimeError("Warm cache restart timed out.")
            time.sleep(2)
        verify_streaming(endpoint.web_url or "", endpoint.endpoint_api_key or "", endpoint.served_model_name or "")
        after_cache = ssh.run(instance, cache_listing)
        checks["warm_restart"] = {
            "seconds": time.monotonic() - warm_start,
            "cold_start_seconds": checks["cold_start_seconds"],
            "files_before": cached_model_files(before_cache),
            "files_after": cached_model_files(after_cache),
            # Reuse means the same weights are still on disk, not that
            # llama.cpp left their timestamps alone when it reopened them.
            "cache_unchanged": bool(before_cache) and cached_model_files(before_cache) == cached_model_files(after_cache),
            "timestamps_unchanged": before_cache == after_cache,
        }
        if not checks["warm_restart"]["cache_unchanged"]:
            raise RuntimeError("Warm restart did not preserve the model cache.")
        report["success"] = True
    except Exception as exc:
        report["error"] = str(exc).replace(config.endpoint_api_key or "<no-key>", "[redacted]")
        print("FAILED: " + report["error"], flush=True)
        if instance_id:
            try:
                report["diagnostic_log_tail"] = backend.logs(instance_id)[-30:]
            except Exception:
                pass
    finally:
        signal.alarm(0)
        print("Destroying this test's rental and verifying absence", flush=True)
        try:
            record = backend.state.load(name)
            if record is not None:
                instance_id = record.instance_id or instance_id
                report["instance_id"] = instance_id
                backend.destroy(name=name)
            report["cleanup_confirmed"] = backend.state.load(name) is None and (not instance_id or api.get_instance(instance_id) is None)
        except Exception as exc:
            report["cleanup_error"] = str(exc)
            report["cleanup_confirmed"] = False
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            report["credit_after"] = account_credit(api)
            report["credit_delta_usd"] = before - report["credit_after"]
        except Exception:
            report["credit_after"] = None
        save()
        print(f"Report: {args.report}; cleanup confirmed: {report['cleanup_confirmed']}", flush=True)
    return 0 if report["success"] and report["cleanup_confirmed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize a billable test and destruction of its rental.")
    parser.add_argument("--offer-id", required=True)
    parser.add_argument("--max-hourly-cost", required=True, type=float)
    parser.add_argument("--budget-usd", required=True, type=float)
    parser.add_argument("--max-minutes", type=int, default=20)
    parser.add_argument("--stream-seconds", type=int, default=310)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to explicitly authorize a paid rental.")
    if not all(math.isfinite(value) and value > 0 for value in (args.max_hourly_cost, args.budget_usd)):
        parser.error("Price and budget must be finite and positive.")
    if not 1 <= args.max_minutes <= 60 or not 1 <= args.stream_seconds < args.max_minutes * 60:
        parser.error("Use 1–60 minutes and a stream duration shorter than the deadline.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

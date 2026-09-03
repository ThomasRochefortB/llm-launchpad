#!/usr/bin/env python3
"""Run the live Fast Deploy validation once per KV cache precision and compare.

Quantizing the KV cache halves its memory but adds dequantization work to every
attention step, so whether it is faster is a property of the placement rather
than of the format. Each arm reuses the full certified deploy path with the
placement pinned, so the only variable is cache precision.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Any

from llm_launchpad.core.llamacpp_planner import KV_CACHE_TYPES
from llm_launchpad.core.quick_deploy import get_quick_deploy_profile
from scripts.validate_modal_fast_deploy_live import (
    DEFAULT_PROFILE_ID,
    DEFAULT_RECALL_CONTEXT_TOKENS,
    run as validate_run,
)


def _await_profile(profile_id: str, *, attempts: int = 5) -> None:
    """Block until the catalog offers the profile again.

    A refresh that cannot reach Hugging Face drops models, so a profile can go
    missing between two arms of the same comparison. Waiting is far better than
    failing an arm and losing the paired measurement.
    """

    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            get_quick_deploy_profile(profile_id)
            return
        except KeyError:
            if attempt >= attempts:
                raise
            print(
                f"  catalog is missing {profile_id}; retrying in {delay:.0f}s "
                f"({attempt}/{attempts - 1})",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("evidence") or {}
    certification = evidence.get("certification") or {}
    attestation = certification.get("attestation") or {}
    points = attestation.get("performance") or []
    memory = certification.get("planned_memory") or {}
    recall = evidence.get("recall") or {}

    def best(key: str, only_single: bool = False) -> float | None:
        values = [
            point[key]
            for point in points
            if point.get(key) is not None
            and (not only_single or point.get("concurrency") == 1)
        ]
        return max(values) if values else None

    return {
        "kv_cache_gb": memory.get("kv_cache_gb"),
        "total_gb": memory.get("total_gb"),
        "single_tps": best("output_tokens_per_second", only_single=True),
        "aggregate_tps": best("aggregate_output_tokens_per_second"),
        "prompt_tps": best("prompt_tokens_per_second"),
        "ttft_s": min(
            (p["time_to_first_token_seconds"] for p in points
             if p.get("time_to_first_token_seconds") is not None),
            default=None,
        ),
        "recall": (
            f"{recall['recalled']}/{recall['attempted']}" if recall else "-"
        ),
        "effective_context": attestation.get("effective_context_tokens"),
    }


def _print_table(summary: dict[str, dict[str, Any]], header: str) -> None:
    if not summary:
        return
    cols = [
        ("kv_cache_gb", "KV GB"),
        ("total_gb", "total GB"),
        ("single_tps", "decode tok/s"),
        ("aggregate_tps", "aggr tok/s"),
        ("prompt_tps", "prefill tok/s"),
        ("ttft_s", "TTFT s"),
        ("recall", "recall"),
        ("effective_context", "context"),
    ]
    print(f"\n{header}")
    print(f"{'cache':>7} " + " ".join(f"{label:>14s}" for _, label in cols))
    print("-" * (8 + 15 * len(cols)))
    for cache_type, row in summary.items():
        cells = []
        for key, _ in cols:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:>14.2f}")
            elif isinstance(value, int):
                cells.append(f"{value:>14,}")
            else:
                cells.append(f"{str(value if value is not None else '-'):>14s}")
        print(f"{cache_type:>7} " + " ".join(cells))

    baseline = summary.get("f16")
    if baseline and baseline.get("single_tps"):
        print("\nrelative to f16:")
        for cache_type, row in summary.items():
            if cache_type == "f16" or not row.get("single_tps"):
                continue
            speed = row["single_tps"] / baseline["single_tps"]
            memory = (
                row["kv_cache_gb"] / baseline["kv_cache_gb"]
                if row.get("kv_cache_gb") and baseline.get("kv_cache_gb")
                else float("nan")
            )
            print(
                f"  {cache_type:>6}: {speed:5.2f}x decode speed, "
                f"{memory:5.2f}x KV memory, recall {row['recall']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement: one billable deploy per cache type.",
    )
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument(
        "--gpu-type",
        default=None,
        help="Pin every arm to one GPU type so only precision varies.",
    )
    parser.add_argument(
        "--cache-types",
        default="f16,q8_0",
        help=f"Comma-separated, from: {', '.join(KV_CACHE_TYPES)}",
    )
    parser.add_argument("--objective", default="general_purpose")
    parser.add_argument(
        "--recall-context",
        type=int,
        default=DEFAULT_RECALL_CONTEXT_TOKENS,
        help="Prompt size for the recall probe, capped at the model's window.",
    )
    parser.add_argument(
        "--out-dir",
        default=f"/tmp/llm-launchpad-kv-bakeoff-{int(time.time())}",
    )
    args = parser.parse_args()
    if not args.confirm_live:
        print("Refusing to create billable resources without --confirm-live.")
        return 2
    cache_types = [item.strip() for item in args.cache_types.split(",") if item.strip()]
    unknown = [item for item in cache_types if item not in KV_CACHE_TYPES]
    if unknown:
        print(f"Unknown cache types: {', '.join(unknown)}")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for cache_type in cache_types:
        report_path = out_dir / f"arm-{cache_type}.json"
        print(f"\n{'=' * 70}\n=== arm: {cache_type}\n{'=' * 70}", flush=True)
        try:
            _await_profile(args.profile_id)
        except KeyError as exc:
            failures[cache_type] = f"profile unavailable: {exc}"
            print(f"!!! arm {cache_type} skipped: {failures[cache_type]}", flush=True)
            continue
        code = asyncio.run(
            validate_run(
                report_path,
                profile_id=args.profile_id,
                objective=args.objective,
                gpu_type=args.gpu_type,
                cache_type=cache_type,
                recall_probe=True,
                recall_context_tokens=args.recall_context,
            )
        )
        report = json.loads(report_path.read_text())
        if code != 0 or not report.get("success"):
            failures[cache_type] = str(report.get("error") or f"exit {code}")
            print(f"!!! arm {cache_type} failed: {failures[cache_type]}", flush=True)
            continue
        summary[cache_type] = _summarize(report)

    combined = out_dir / "summary.json"
    combined.write_text(
        json.dumps({"summary": summary, "failures": failures}, indent=2, sort_keys=True)
        + "\n"
    )
    _print_table(
        summary,
        f"{args.profile_id} on {args.gpu_type or 'recommended placement'}",
    )
    if failures:
        print(f"\nfailed arms: {', '.join(failures)}")
    print(f"\nReports: {out_dir}")
    return 0 if summary and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

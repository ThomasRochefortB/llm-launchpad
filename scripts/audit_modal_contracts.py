#!/usr/bin/env python3
"""Zero-cost read-only audit for Modal GPU contracts and Fast Deploy placements.

Never deploys, rents GPUs, or creates billable resources.

Checks:
1. Placement contract sweep across representative model sizes (5 GB -> 850 GB):
   every emitted Modal placement must use a power-of-two card count within
   `modal_gpu_count_limit`, pass `modal_gpu_supports_fast_deploy`, fit VRAM
   with the promised reserve margin, and agree between catalog & live sizing.
2. Live Modal docs drift check (when `--live-docs` is passed): fetches
   `https://modal.com/docs/guide/gpu` and `https://modal.com/pricing` via HTTP
   GET and checks for new/unmapped GPU types, missing pricing, or changed max
   GPU count documentation vs `llm_launchpad/core/modal_gpu.py`.
3. Optional local catalog audit (`--audit-catalog`): inspects the cached/active
   Fast Deploy catalog and prints a summary table of planned Modal shapes.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from llm_launchpad.core.compute_availability import (
    _placement_gpu_count,
    aggregate_compute_availability,
    assessed_plans_for_compute_profile,
)
from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.llamacpp_planner import (
    compile_server_args,
    device_capacity_bytes,
    estimate_memory,
    gib_to_bytes,
    serving_requirements,
    tuning_for_objective,
)
from llm_launchpad.core.modal_gpu import (
    MODAL_GPU_GUIDE_URL,
    MODAL_GPU_MEMORY_GB,
    MODAL_PRICING_URL,
    ModalGpuSpec,
    _fetch_modal_page_text,
    _parse_modal_gpu_pricing,
    _parse_modal_gpu_types,
    _strip_html_tags,
    modal_gpu_count_limit,
    modal_gpu_count_options,
    modal_gpu_supports_fast_deploy,
)
from llm_launchpad.core.quick_deploy import QuickDeployProfile, list_quick_deploy_profiles
from llm_launchpad.core.quick_deploy_refresh import (
    _FALLBACK_GPU_PRICE_PER_HOUR,
    _select_gpu_shape,
)
from llm_launchpad.protocol.enums import ComputeProvider, ServingObjective

_GPU_COUNT_CLAUSE_RE = re.compile(
    r"(?P<gpus>[A-Za-z0-9,+\-\s]+?)\s+instances\s+support\s+up\s+to\s+(?P<limit>\d+)\s+GPUs",
    re.IGNORECASE,
)
_SWEEP_MODELS: tuple[tuple[str, float, int, int], ...] = (
    ("8B-Q4", 5.0, 32_768, 32),
    ("32B-Q4", 20.0, 131_072, 64),
    ("70B-Q4", 45.0, 131_072, 80),
    ("GLM-5.3-Flash-class", 110.0, 262_144, 45),
    ("DS41-IQ1_M-class", 250.0, 1_048_576, 40),
    ("DS41-Q4_K_XL-class", 500.0, 1_048_576, 40),
    ("Kimi-K3-Q2-class", 850.0, 1_048_576, 93),
)


@dataclass(frozen=True)
class AuditFinding:
    """One contract violation or drift warning detected by the audit."""

    severity: str  # "error" | "warning"
    area: str
    message: str


def parse_documented_gpu_limits(guide_page_text: str) -> dict[str, int]:
    """Extract documented max-GPU-per-container limits from Modal's GPU guide."""
    plain = re.sub(r"\s+", " ", _strip_html_tags(guide_page_text))
    lower = plain.lower()
    start = lower.find("specifying gpu count")
    section = plain[start : start + 1500] if start != -1 else plain
    limits: dict[str, int] = {}
    for match in _GPU_COUNT_CLAUSE_RE.finditer(section):
        limit = int(match.group("limit"))
        raw_list = match.group("gpus")
        for token in re.findall(r"\b(?:[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)\b", raw_list):
            if any(ch.isdigit() for ch in token):
                limits[token.upper()] = limit
    return limits


def audit_modal_docs_text(
    guide_page_text: str,
    pricing_page_text: str,
) -> list[AuditFinding]:
    """Compare Modal's GPU guide and pricing HTML against local constants."""
    findings: list[AuditFinding] = []
    gpu_types = _parse_modal_gpu_types(guide_page_text)
    pricing = _parse_modal_gpu_pricing(pricing_page_text)
    if not gpu_types:
        findings.append(
            AuditFinding("error", "modal-docs", "Parsed 0 GPU types from Modal GPU guide (page format changed?).")
        )
        return findings

    for gpu in gpu_types:
        if gpu not in MODAL_GPU_MEMORY_GB:
            findings.append(
                AuditFinding("error", "modal-docs", f"Documented Modal GPU {gpu!r} is missing from MODAL_GPU_MEMORY_GB.")
            )
        if gpu not in pricing:
            findings.append(
                AuditFinding("warning", "modal-pricing", f"Documented Modal GPU {gpu!r} has no parsed hourly rate.")
            )

    doc_limits = parse_documented_gpu_limits(guide_page_text)
    for gpu, doc_limit in sorted(doc_limits.items()):
        # A100 in docs covers both A100-40GB and A100-80GB.
        targets = ("A100", "A100-40GB", "A100-80GB") if gpu == "A100" else (gpu,)
        for target in targets:
            if target not in MODAL_GPU_MEMORY_GB:
                continue
            local_limit = modal_gpu_count_limit(target)
            if local_limit > doc_limit:
                findings.append(
                    AuditFinding(
                        "error",
                        "modal-limits",
                        f"Local limit for {target} is {local_limit}, exceeding documented Modal limit {doc_limit}.",
                    )
                )
    return findings


def audit_placement_sweep(
    catalog: list[ModalGpuSpec] | None = None,
) -> tuple[list[AuditFinding], list[str]]:
    """Verify every Modal placement emitted across representative model sizes."""
    specs = catalog or [
        ModalGpuSpec(gpu, price_per_hour_usd=_FALLBACK_GPU_PRICE_PER_HOUR.get(gpu, 2.5))
        for gpu in MODAL_GPU_MEMORY_GB
    ]
    snapshot = aggregate_compute_availability(modal_catalog=specs)
    placements_by_gpu = {
        placement.gpu_type: placement
        for config in snapshot.configurations
        for placement in config.placements
        if placement.provider == ComputeProvider.MODAL
    }
    findings: list[AuditFinding] = []
    summary_rows: list[str] = []

    for label, weights_gb, context_tokens, layers in _SWEEP_MODELS:
        metadata = GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": weights_gb},
            architecture="llama",
            context_length=context_tokens,
            block_count=layers,
            embedding_length=4096,
            attention_head_count=32,
            attention_head_count_kv=8,
            attention_key_length=128,
            attention_value_length=128,
        )
        requirements = serving_requirements(context_tokens, objective=ServingObjective.GENERAL_PURPOSE)
        tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)
        memory = estimate_memory(
            metadata,
            weights_gb=weights_gb,
            requirements=requirements,
            tuning=tuning,
            gpu_count=1,
        )
        profile = QuickDeployProfile(
            id=f"audit-{label}",
            display_name=label,
            repo_id="org/audit-GGUF",
            quant="Q4_K_M",
            gpu_type="H100",
            gpu_count=1,
            profile_label="Audit",
            approx_cost_per_hour_usd=4.0,
            max_context_tokens=context_tokens,
            instance_slug_hint="audit",
            summary="Audit profile.",
            server_args=tuple(compile_server_args(requirements, tuning)),
            required_vram_gb=memory.total_gb,
            serving_requirements=requirements,
            runtime_tuning=tuning,
            memory_estimate=memory,
        )
        per_dev_overhead = memory.compute_gb + memory.attention_scratch_gb
        best = _select_gpu_shape(
            profile.quant,
            memory.total_gb,
            specs,
            per_device_overhead_gb=per_dev_overhead,
            layer_count=layers,
        )
        if best is not None:
            summary_rows.append(
                f"  {label:20s} -> {best.gpu_count}x {best.gpu_type:12s} (~${best.cost_per_hour_usd:.2f}/hr)"
            )
            if best.gpu_count not in modal_gpu_count_options(best.gpu_type):
                findings.append(
                    AuditFinding(
                        "error",
                        "placement-sweep",
                        f"{label}: catalog selected unsupported {best.gpu_count}x {best.gpu_type}",
                    )
                )
        else:
            summary_rows.append(f"  {label:20s} -> (no fitting Modal shape)")

        for gpu_type, placement in placements_by_gpu.items():
            if not modal_gpu_supports_fast_deploy(gpu_type):
                continue
            catalog_for_gpu = _select_gpu_shape(
                profile.quant,
                memory.total_gb,
                specs,
                gpu_type=gpu_type,
                per_device_overhead_gb=per_dev_overhead,
                layer_count=layers,
            )
            live_count = _placement_gpu_count(placement, memory.total_gb, memory_estimate=memory)
            expected = catalog_for_gpu.gpu_count if catalog_for_gpu is not None else None
            if live_count != expected:
                findings.append(
                    AuditFinding(
                        "error",
                        "placement-parity",
                        f"{label} on {gpu_type}: catalog={expected} vs live={live_count}",
                    )
                )

        for config in snapshot.configurations:
            plans, _ = assessed_plans_for_compute_profile(config, profile)
            for plan in plans:
                quote = plan.quote
                if quote.gpu_count not in modal_gpu_count_options(quote.gpu_type):
                    findings.append(
                        AuditFinding(
                            "error",
                            "live-placement",
                            f"{label}: live plan emitted unsupported {quote.gpu_count}x {quote.gpu_type}",
                        )
                    )
                if plan.assessment is not None and quote.gpu_memory_gb is not None:
                    cap = device_capacity_bytes(quote.gpu_memory_gb)
                    for req in plan.assessment.memory.per_device_required_gb:
                        if gib_to_bytes(req) > cap:
                            findings.append(
                                AuditFinding(
                                    "error",
                                    "memory-fit",
                                    f"{label} on {quote.gpu_count}x {quote.gpu_type}: {req:.2f} GiB > {quote.gpu_memory_gb:.0f} GB",
                                )
                            )

    return findings, summary_rows


def audit_active_catalog() -> tuple[list[AuditFinding], list[str]]:
    """Audit the currently loaded Fast Deploy catalog profiles (no network calls)."""
    findings: list[AuditFinding] = []
    rows: list[str] = []
    for profile in list_quick_deploy_profiles():
        if profile.gpu_count <= 0:
            continue
        gpu_type = profile.gpu_type.strip().upper()
        options = modal_gpu_count_options(gpu_type)
        rows.append(
            f"  {profile.display_name:32s} {profile.quant:10s} "
            f"{profile.gpu_count}x {gpu_type:12s} ~${profile.approx_cost_per_hour_usd:5.2f}/hr"
        )
        if not modal_gpu_supports_fast_deploy(gpu_type):
            findings.append(
                AuditFinding("error", "catalog", f"{profile.id}: unsupported Fast Deploy GPU {gpu_type}")
            )
        if profile.gpu_count not in options:
            findings.append(
                AuditFinding(
                    "error",
                    "catalog",
                    f"{profile.id}: {profile.gpu_count}x {gpu_type} is not in Modal ladder {options}",
                )
            )
    return findings, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--live-docs",
        action="store_true",
        help="Fetch live Modal GPU guide and pricing pages (read-only HTTP GET, $0).",
    )
    parser.add_argument(
        "--audit-catalog",
        action="store_true",
        help="Also inspect active Fast Deploy catalog profiles.",
    )
    args = parser.parse_args(argv)

    findings: list[AuditFinding] = []
    live_specs: list[ModalGpuSpec] | None = None

    if args.live_docs:
        guide_text = _fetch_modal_page_text(MODAL_GPU_GUIDE_URL, timeout=15.0)
        pricing_text = _fetch_modal_page_text(MODAL_PRICING_URL, timeout=15.0)
        findings.extend(audit_modal_docs_text(guide_text, pricing_text))
        gpu_types = _parse_modal_gpu_types(guide_text)
        pricing = _parse_modal_gpu_pricing(pricing_text)
        live_specs = [ModalGpuSpec(gpu, price_per_hour_usd=pricing.get(gpu)) for gpu in gpu_types]

    sweep_findings, sweep_rows = audit_placement_sweep(live_specs)
    findings.extend(sweep_findings)
    print("Placement sweep (Modal):")
    for row in sweep_rows:
        print(row)

    if args.audit_catalog:
        cat_findings, cat_rows = audit_active_catalog()
        findings.extend(cat_findings)
        print("\nActive Fast Deploy catalog:")
        if cat_rows:
            for row in cat_rows:
                print(row)
        else:
            print("  (empty — run a catalog refresh to populate)")

    errors = [item for item in findings if item.severity == "error"]
    warnings = [item for item in findings if item.severity == "warning"]
    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"  [WARN] {item.area}: {item.message}")
    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"  [ERR]  {item.area}: {item.message}")
        return 1
    print("\nAll Modal contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

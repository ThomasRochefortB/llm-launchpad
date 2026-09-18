"""Runtime evidence parsed from llama.cpp itself, independent of the planner.

The warmup path used to certify GPU residency from the placement assessment:
configured ``gpu_layers == "all"`` plus a predicted fit. That is the plan
confirming itself. Everything here is read off the live runtime instead --
``/props`` for the effective context, server logs for the offload report, and
the fit binary's own arithmetic as a projection, never as proof of residency.
Missing telemetry is unknown, not failure and not success.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ..protocol.enums import EvidenceLevel

# "load_tensors: offloading 63 layers" / "load_tensors: offloaded 63/63 layers
# to GPU". llama.cpp logs the offload report while loading the model; the
# planner's configured flags never appear here.
_OFFLOAD_RE = re.compile(
    r"load_tensors:\s+offload(?:ing|ed)\s+(?P<done>\d+)(?:\s*/\s*(?P<total>\d+))?",
    flags=re.IGNORECASE,
)

# The fit binary states what it will allocate per device, whether or not the
# plan is accepted. A projection sizes the next plan; it does not prove the
# current one is resident.
_FIT_DEVICE_RE = re.compile(
    r"-\s+(?P<device>\S+)\s*\((?P<description>[^)]*)\):\s*"
    r"(?P<total>\d+)\s+total,\s*"
    r"(?P<used>\d+)\s+used,\s*"
    r"(?P<free>\d+)\s+free\s+vs\.\s+target\s+of\s+(?P<target>\d+)"
)


@dataclass(frozen=True)
class LlamacppRuntimeEvidence:
    """What the runtime itself said, with each claim graded by strength."""

    effective_context_tokens: int | None = None
    context_evidence: EvidenceLevel | None = None
    gpu_layers: int | None = None
    total_layers: int | None = None
    offload_evidence: EvidenceLevel | None = None
    fit_projection_mib: tuple[int, ...] = ()
    detail: str = ""

    @property
    def gpu_resident(self) -> bool | None:
        """Whether observed offload covers every layer, or None if unknown."""

        if self.gpu_layers is None or self.total_layers is None:
            return None
        if self.total_layers <= 0:
            return None
        return self.gpu_layers >= self.total_layers

    @property
    def context_verified(self) -> bool:
        return (
            self.effective_context_tokens is not None
            and self.context_evidence is not None
        )


def parse_offload_report(text: str) -> tuple[int | None, int | None]:
    """Return (offloaded_layers, total_layers) from server log output.

    A bare "offloading N layers" names only the offloaded count; the total is
    unknown until an "offloaded N/M" line arrives. Later lines win: the final
    report describes the running server, earlier ones describe attempts.
    """

    done: int | None = None
    total: int | None = None
    for line in (text or "").splitlines():
        match = _OFFLOAD_RE.search(line)
        if match is None:
            continue
        try:
            done = int(match.group("done"))
        except (TypeError, ValueError):
            continue
        raw_total = match.group("total")
        if raw_total is not None:
            try:
                total = int(raw_total)
            except (TypeError, ValueError):
                pass
    return done, total


def parse_fit_projection_mib(text: str) -> tuple[int, ...]:
    """Return per-device projected MiB from fit output, without interpreting it."""

    used: list[int] = []
    for line in (text or "").splitlines():
        match = _FIT_DEVICE_RE.search(line)
        if match is None:
            continue
        try:
            used.append(int(match.group("used")))
        except (TypeError, ValueError):
            continue
    return tuple(used)


def runtime_props_evidence(payload: Any) -> LlamacppRuntimeEvidence:
    """Grade a ``/props`` payload: context is observed, offload is not.

    ``/props`` reports the configured context window. It says nothing about
    which layers sit on GPU, so offload evidence stays absent here and must
    come from the server logs.
    """

    from .warmup import extract_effective_context

    effective = extract_effective_context(payload)
    if effective is None:
        return LlamacppRuntimeEvidence(detail="/props reported no context size")
    return LlamacppRuntimeEvidence(
        effective_context_tokens=effective,
        context_evidence=EvidenceLevel.OBSERVED,
    )


def runtime_log_evidence(text: str) -> LlamacppRuntimeEvidence:
    """Grade server log output: offload lines are one observation each."""

    done, total = parse_offload_report(text)
    if done is None:
        return LlamacppRuntimeEvidence()
    return LlamacppRuntimeEvidence(
        gpu_layers=done,
        total_layers=total,
        offload_evidence=EvidenceLevel.OBSERVED,
        detail=f"runtime reported {done}" + (f"/{total} layers on GPU" if total else " layers offloaded"),
    )


def remembered_attestation_evidence(attestation: Any) -> LlamacppRuntimeEvidence:
    """Grade an earlier certificate for this exact placement as observation.

    A placement fingerprint covers the model, quantization, context, tuning and
    shape, so an earlier run's offload report describes the same thing this run
    just started. It is still an observation -- made then, not now -- which is
    why it is graded here rather than treated as proof of the running process.
    """

    if attestation is None:
        return LlamacppRuntimeEvidence()
    gpu_layers = getattr(attestation, "gpu_layers", None)
    total_layers = getattr(attestation, "total_layers", None)
    if not gpu_layers or not total_layers:
        return LlamacppRuntimeEvidence()
    verified_at = str(getattr(attestation, "verified_at", "") or "")
    when = f" on {verified_at[:10]}" if verified_at else ""
    return LlamacppRuntimeEvidence(
        gpu_layers=int(gpu_layers),
        total_layers=int(total_layers),
        offload_evidence=EvidenceLevel.OBSERVED,
        detail=(
            f"an earlier run of this placement reported {gpu_layers}/{total_layers} "
            f"layers on GPU{when}"
        ),
    )


def combine_runtime_evidence(
    *pieces: LlamacppRuntimeEvidence,
) -> LlamacppRuntimeEvidence:
    """Merge evidence pieces, keeping the strongest claim for each axis."""

    effective: int | None = None
    context_evidence: EvidenceLevel | None = None
    gpu_layers: int | None = None
    total_layers: int | None = None
    offload_evidence: EvidenceLevel | None = None
    projections: list[int] = []
    details: list[str] = []
    for piece in pieces:
        if piece.effective_context_tokens is not None:
            effective = piece.effective_context_tokens
            context_evidence = piece.context_evidence
        if piece.gpu_layers is not None:
            gpu_layers = piece.gpu_layers
            offload_evidence = piece.offload_evidence
        if piece.total_layers is not None:
            total_layers = piece.total_layers
            offload_evidence = piece.offload_evidence or offload_evidence
        projections.extend(piece.fit_projection_mib)
        if piece.detail:
            details.append(piece.detail)
    return LlamacppRuntimeEvidence(
        effective_context_tokens=effective,
        context_evidence=context_evidence,
        gpu_layers=gpu_layers,
        total_layers=total_layers,
        offload_evidence=offload_evidence,
        fit_projection_mib=tuple(projections),
        detail="; ".join(details),
    )

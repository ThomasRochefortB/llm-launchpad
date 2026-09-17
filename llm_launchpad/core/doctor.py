"""Onboarding self-checks for the local Launchpad environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import SETTINGS_DIR
from .diagnostics import LOG_FILE, log_file_path
from .hf_auth import get_huggingface_auth_status
from .artificial_analysis import get_artificial_analysis_auth_status
from .provider_readiness import (
    ProviderReadinessStage,
    check_modal_readiness,
    check_prime_readiness,
    check_vast_readiness,
)


@dataclass(frozen=True)
class DoctorCheck:
    """Result of one doctor probe with a user-facing fix hint on failure."""

    name: str
    ok: bool
    required: bool = True
    detail: str = ""
    hint: str | None = None


def _state_dir_check(settings_dir: Path | None = None) -> DoctorCheck:
    directory = settings_dir or SETTINGS_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".doctor_write_test"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return DoctorCheck(
            name="Local state directory",
            ok=False,
            detail=f"{directory} is not writable ({exc})",
            hint="Check home directory permissions; the TUI cannot persist settings without it",
        )
    return DoctorCheck(name="Local state directory", ok=True, detail=str(directory))


def run_doctor_checks(
    *,
    settings_dir: Path | None = None,
    check_artificial_analysis: bool = True,
) -> tuple[DoctorCheck, ...]:
    """Probe local prerequisites and return results in display order.

    The Artificial Analysis check is optional: it validates a configured key
    against the live API and only ever warns, matching the optional role the
    key plays in deploy recommendations.
    """

    checks: list[DoctorCheck] = []

    checks.append(_state_dir_check(settings_dir))

    # Deploying needs one compute provider, not all of them: that is what the
    # TUI gates on. Each provider reports for itself and only warns, so an
    # unused one never fails the run; the aggregate below is the requirement.
    # Readiness distinguishes installation, stored credentials, verified
    # access, rejected credentials, and unreachable providers.
    def _readiness_check(name: str, readiness: object) -> tuple[bool, str, str | None]:
        stage = getattr(readiness, "stage", None)
        detail = str(getattr(readiness, "detail", "") or "")
        hint = getattr(readiness, "hint", None)
        if stage == ProviderReadinessStage.READY:
            return True, detail or "authenticated", None
        if stage == ProviderReadinessStage.UNREACHABLE:
            return False, f"{detail or 'verification unavailable'} (credentials present; not proven invalid)", hint
        if stage == ProviderReadinessStage.CREDENTIALS_PRESENT:
            return False, detail or "credentials present; not verified", hint
        return False, detail or "not configured", hint

    try:
        modal_readiness = check_modal_readiness(verify=True)
    except Exception as exc:  # pragma: no cover - defensive around auth probing
        checks.append(DoctorCheck(name="Modal auth", ok=False, required=False, detail=str(exc), hint="run: modal setup"))
        modal_available = False
    else:
        ok, detail, hint = _readiness_check("Modal auth", modal_readiness)
        modal_available = modal_readiness.verified
        label = "Modal auth" if modal_readiness.stage != ProviderReadinessStage.NOT_INSTALLED else "Modal CLI"
        checks.append(DoctorCheck(name=label, ok=ok, required=False, detail=detail, hint=hint))

    try:
        prime_readiness = check_prime_readiness(verify=True)
    except Exception as exc:  # pragma: no cover - defensive around auth probing
        checks.append(
            DoctorCheck(name="Prime Intellect auth", ok=False, required=False, detail=str(exc), hint="run: prime login (or set PRIME_API_KEY)")
        )
        prime_available = False
    else:
        ok, detail, hint = _readiness_check("Prime Intellect auth", prime_readiness)
        prime_available = prime_readiness.verified
        # A stored key that could not be verified is still credentials the TUI
        # can attempt to use; only verified access counts as available here so
        # `doctor` does not promise what the network has not confirmed.
        checks.append(DoctorCheck(name="Prime Intellect auth", ok=ok, required=False, detail=detail, hint=hint))

    try:
        vast_readiness = check_vast_readiness(verify=True)
    except Exception as exc:  # pragma: no cover - defensive around auth probing
        checks.append(
            DoctorCheck(name="Vast.ai auth", ok=False, required=False, detail=str(exc), hint="run: llm-launchpad vast-auth login (or set VAST_API_KEY)")
        )
        vast_ok = False
    else:
        ok, detail, hint = _readiness_check("Vast.ai auth", vast_readiness)
        vast_ok = vast_readiness.verified
        checks.append(DoctorCheck(name="Vast.ai auth", ok=ok, required=False, detail=detail, hint=hint))

    configured = [
        name for name, available in (
            ("Modal", modal_available), ("Prime Intellect", prime_available), ("Vast.ai", vast_ok),
        ) if available
    ]
    # Credentials that exist but are unverified still let the user into the TUI
    # for an explicit retry; doctor only promises verified access.
    checks.append(DoctorCheck(
        name="Compute provider",
        ok=bool(configured),
        detail=", ".join(configured) if configured else "none verified",
        hint="authenticate one: modal setup | prime login | llm-launchpad vast-auth login",
    ))

    hf_status = get_huggingface_auth_status()
    if hf_status.authenticated:
        username = f" ({hf_status.username})" if hf_status.username else ""
        checks.append(DoctorCheck(name="Hugging Face auth", ok=True, detail=f"logged in{username}"))
    else:
        checks.append(
            DoctorCheck(
                name="Hugging Face auth",
                ok=False,
                detail=hf_status.error or "no local token",
                hint="run: huggingface-cli login",
            )
        )

    if check_artificial_analysis:
        try:
            aai_status = get_artificial_analysis_auth_status()
        except Exception as exc:
            aai_status_detail = str(exc)
            aai_authenticated = False
        else:
            aai_status_detail = aai_status.error or ""
            aai_authenticated = aai_status.authenticated
        if aai_authenticated:
            checks.append(
                DoctorCheck(
                    name="Artificial Analysis key",
                    ok=True,
                    required=False,
                    detail="validated",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="Artificial Analysis key",
                    ok=False,
                    required=False,
                    detail=aai_status_detail or "no API key configured",
                    hint="optional: run: llm-launchpad aai-auth login",
                )
            )

    active_log = log_file_path() or LOG_FILE
    checks.append(
        DoctorCheck(name="Debug log", ok=True, detail=str(active_log))
    )

    return tuple(checks)


def doctor_exit_code(checks: tuple[DoctorCheck, ...]) -> int:
    """Return 1 when any required check failed, else 0."""

    return 1 if any(not check.ok and check.required for check in checks) else 0

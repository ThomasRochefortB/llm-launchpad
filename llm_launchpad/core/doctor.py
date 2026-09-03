"""Onboarding self-checks for the local Launchpad environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backend import ModalBackend
from .config import SETTINGS_DIR
from .diagnostics import LOG_FILE, log_file_path
from .hf_auth import get_huggingface_auth_status
from .modal_auth import get_modal_auth_status
from .prime_auth import get_prime_auth_status
from .quick_deploy_refresh import get_artificial_analysis_auth_status


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

    if ModalBackend.is_cli_available():
        modal_status = get_modal_auth_status()
        if modal_status.authenticated:
            profile = f" (profile: {modal_status.profile})" if modal_status.profile else ""
            checks.append(
                DoctorCheck(name="Modal auth", ok=True, detail=f"authenticated{profile}")
            )
        else:
            checks.append(
                DoctorCheck(
                    name="Modal auth",
                    ok=False,
                    detail=modal_status.error or "not authenticated",
                    hint="run: modal setup",
                )
            )
    else:
        checks.append(
            DoctorCheck(
                name="Modal CLI",
                ok=False,
                detail="modal executable not found",
                hint="reinstall llm-launchpad, then run: modal setup",
            )
        )

    try:
        prime_status = get_prime_auth_status()
        if prime_status.authenticated:
            checks.append(DoctorCheck(name="Prime Intellect auth", ok=True, detail="API key found"))
        else:
            checks.append(
                DoctorCheck(
                    name="Prime Intellect auth",
                    ok=False,
                    detail=prime_status.error or "no API key configured",
                    hint="run: prime login (or set PRIME_API_KEY)",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive around auth probing
        checks.append(
            DoctorCheck(
                name="Prime Intellect auth",
                ok=False,
                detail=str(exc),
                hint="run: prime login (or set PRIME_API_KEY)",
            )
        )

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

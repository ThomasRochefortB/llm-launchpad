"""Modal authentication helpers."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

from .modal_cli import _CLI_TIMEOUT_SECONDS, resolve_modal_cli_path


@dataclass(frozen=True)
class ModalAuthStatus:
    """Best-effort status for local Modal CLI authentication."""

    authenticated: bool
    profile: str | None = None
    detail: str | None = None
    error: str | None = None


def _modal_cli_path() -> str | None:
    return resolve_modal_cli_path()


def _run_modal_command(*args: str) -> subprocess.CompletedProcess[str] | None:
    modal_cli = _modal_cli_path()
    if modal_cli is None:
        return None
    return subprocess.run(
        [modal_cli, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=_CLI_TIMEOUT_SECONDS,
    )


def _summarize_output(text: str) -> str | None:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _looks_unauthenticated(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return any(
        marker in normalized
        for marker in (
            "modal setup",
            "authenticate",
            "authentication required",
            "not logged in",
            "not authenticated",
            "no token",
            "missing token",
            "token not found",
            "token is required",
        )
    )


def get_modal_profile() -> str | None:
    """Return the current Modal profile name, if available."""
    try:
        result = _run_modal_command("profile", "current")
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    if result is None or result.returncode != 0:
        return None
    profile = (result.stdout or "").strip()
    return profile or None


def get_modal_auth_status() -> ModalAuthStatus:
    """Return a best-effort Modal auth status for the current CLI session."""
    if _modal_cli_path() is None:
        return ModalAuthStatus(
            authenticated=False,
            error="Modal CLI not found (reinstall llm-launchpad, then run: modal setup)",
        )

    profile = get_modal_profile()
    try:
        result = _run_modal_command("token", "info")
    except subprocess.TimeoutExpired:
        return ModalAuthStatus(
            authenticated=False,
            profile=profile,
            error="Timed out checking Modal authentication",
        )
    except Exception as exc:
        return ModalAuthStatus(
            authenticated=False,
            profile=profile,
            error=str(exc),
        )

    if result is None:
        return ModalAuthStatus(
            authenticated=False,
            profile=profile,
            error="Modal CLI not found (reinstall llm-launchpad, then run: modal setup)",
        )

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    combined = "\n".join(part for part in (stdout, stderr) if part).strip()
    if result.returncode == 0:
        detail = _summarize_output(stdout)
        return ModalAuthStatus(authenticated=True, profile=profile, detail=detail)

    if _looks_unauthenticated(combined):
        return ModalAuthStatus(authenticated=False, profile=profile)

    return ModalAuthStatus(
        authenticated=False,
        profile=profile,
        error=_summarize_output(combined) or "Modal auth check failed",
    )

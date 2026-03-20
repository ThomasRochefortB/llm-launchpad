"""Hugging Face authentication helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HuggingFaceAuthStatus:
    """Best-effort status for local Hugging Face Hub authentication."""

    authenticated: bool
    username: str | None = None
    error: str | None = None


def _extract_username(payload: Any) -> str | None:
    if isinstance(payload, str):
        text = payload.strip()
        return text or None

    if not isinstance(payload, dict):
        return None

    for key in ("name", "username", "user"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = _extract_username(value)
            if nested:
                return nested

    auth_payload = payload.get("auth")
    if isinstance(auth_payload, dict):
        for key in ("username", "name"):
            value = auth_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _looks_unauthenticated_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return (
        "localtokennotfounderror" in text
        or "not logged in" in text
        or "no token" in text
        or "token is required" in text
        or "run `hf auth login`" in text
        or "run 'hf auth login'" in text
    )


def _looks_invalid_token_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return (
        "invalid user token" in text
        or "invalid token" in text
        or "unauthorized" in text
        or "401" in text
    )


def _summarize_error(exc: Exception) -> str:
    text = " ".join(str(exc).split()).strip()
    return text or exc.__class__.__name__


def _has_local_token() -> bool:
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        # Fall back to a live whoami() call if token lookup isn't available.
        return True
    return bool((token or "").strip())


def get_huggingface_auth_status() -> HuggingFaceAuthStatus:
    """Return a best-effort Hugging Face Hub auth status for the current user."""
    try:
        from huggingface_hub import HfApi
    except Exception:
        return HuggingFaceAuthStatus(
            authenticated=False,
            error="huggingface_hub is not installed",
        )

    if not _has_local_token():
        return HuggingFaceAuthStatus(authenticated=False)

    try:
        payload = HfApi().whoami()
    except Exception as exc:
        if _looks_unauthenticated_error(exc):
            return HuggingFaceAuthStatus(authenticated=False)
        if _looks_invalid_token_error(exc):
            return HuggingFaceAuthStatus(authenticated=False, error="Invalid Hugging Face token")
        return HuggingFaceAuthStatus(authenticated=False, error=_summarize_error(exc))

    username = _extract_username(payload)
    if username:
        return HuggingFaceAuthStatus(authenticated=True, username=username)
    return HuggingFaceAuthStatus(authenticated=True)

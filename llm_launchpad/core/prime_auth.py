"""Prime Intellect authentication and configuration discovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


PRIME_CONFIG_DIR = Path.home() / ".prime"
PRIME_CONFIG_PATH = PRIME_CONFIG_DIR / "config.json"
DEFAULT_PRIME_BASE_URL = "https://api.primeintellect.ai"


@dataclass(frozen=True)
class PrimeConfig:
    """Resolved Prime API settings without mutating Prime CLI state."""

    api_key: str = ""
    team_id: str | None = None
    user_id: str | None = None
    base_url: str = DEFAULT_PRIME_BASE_URL
    ssh_key_path: str | None = None
    context: str = "production"


@dataclass(frozen=True)
class PrimeAuthStatus:
    """Best-effort Prime authentication status."""

    authenticated: bool
    team_id: str | None = None
    user_id: str | None = None
    error: str | None = None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Prime config must contain an object: {path}")
    return payload


def load_prime_config(config_path: Path | None = None) -> PrimeConfig:
    """Resolve Prime settings with the same environment-first precedence as its CLI."""

    root_path = config_path or PRIME_CONFIG_PATH
    root = _read_json_object(root_path)
    context = (os.getenv("PRIME_CONTEXT") or str(root.get("current_environment", "production"))).strip()
    selected = root
    if context and context.casefold() != "production":
        context_path = root_path.parent / "environments" / f"{context}.json"
        selected = {**root, **_read_json_object(context_path)}

    api_key = (os.getenv("PRIME_API_KEY") or str(selected.get("api_key", ""))).strip()
    team_id = (os.getenv("PRIME_TEAM_ID") or str(selected.get("team_id") or "")).strip() or None
    user_id = (os.getenv("PRIME_USER_ID") or str(selected.get("user_id") or "")).strip() or None
    base_url = (
        os.getenv("PRIME_API_BASE_URL")
        or os.getenv("PRIME_BASE_URL")
        or str(selected.get("base_url") or DEFAULT_PRIME_BASE_URL)
    ).strip().rstrip("/")
    if base_url.endswith("/api/v1"):
        base_url = base_url[: -len("/api/v1")]
    ssh_key_path = (
        os.getenv("PRIME_SSH_KEY_PATH") or str(selected.get("ssh_key_path") or "")
    ).strip() or None
    return PrimeConfig(
        api_key=api_key,
        team_id=team_id,
        user_id=user_id,
        base_url=base_url or DEFAULT_PRIME_BASE_URL,
        ssh_key_path=ssh_key_path,
        context=context or "production",
    )


def get_prime_auth_status(config_path: Path | None = None) -> PrimeAuthStatus:
    """Return whether a Prime API key is available locally."""

    try:
        config = load_prime_config(config_path)
    except Exception as exc:
        return PrimeAuthStatus(authenticated=False, error=str(exc))
    return PrimeAuthStatus(
        authenticated=bool(config.api_key),
        team_id=config.team_id,
        user_id=config.user_id,
    )

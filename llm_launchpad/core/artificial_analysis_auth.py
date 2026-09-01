"""Artificial Analysis API key persistence for llm-launchpad."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import SETTINGS_DIR

AAI_API_KEY_ENV = "ARTIFICIAL_ANALYSIS_API_KEY"
AAI_AUTH_PATH = SETTINGS_DIR / "artificial_analysis_auth.json"


def resolve_artificial_analysis_api_key() -> str:
    """Resolve the AAI key with environment-first precedence."""

    environment_key = (os.getenv(AAI_API_KEY_ENV) or "").strip()
    if environment_key:
        return environment_key
    return load_saved_artificial_analysis_api_key().strip()


def load_saved_artificial_analysis_api_key(path: Path | None = None) -> str:
    """Load the stored AAI key, returning an empty string when absent."""

    root_path = path or AAI_AUTH_PATH
    try:
        raw = json.loads(root_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("api_key", "")).strip()


def save_artificial_analysis_api_key(api_key: str, path: Path | None = None) -> Path:
    """Persist the AAI key with owner-only file permissions."""

    normalized_key = (api_key or "").strip()
    if not normalized_key:
        raise ValueError("An Artificial Analysis API key is required")
    root_path = path or AAI_AUTH_PATH
    root_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = root_path.with_suffix(f"{root_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps({"api_key": normalized_key}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary_path, 0o600)
    temporary_path.replace(root_path)
    return root_path


def clear_artificial_analysis_api_key(path: Path | None = None) -> bool:
    """Remove the stored AAI key. Returns True when a file was removed."""

    root_path = path or AAI_AUTH_PATH
    try:
        root_path.unlink()
    except FileNotFoundError:
        return False
    return True

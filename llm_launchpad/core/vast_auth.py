"""Environment-first Vast credentials without modifying the Vast CLI's state."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile

from .config import SETTINGS_DIR

VAST_AUTH_PATH = SETTINGS_DIR / "vast_auth.json"


@dataclass(frozen=True)
class VastCredentials:
    """Resolved key and its origin; never include the key in repr output."""

    api_key: str = field(default="", repr=False)
    source: str = "none"


def normalize_vast_api_key(value: str) -> str:
    """Validate a key before it is persisted or used as an HTTP header."""
    key = value.strip()
    if not key or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError("A non-empty Vast API key without whitespace is required.")
    return key


def resolve_vast_credentials(
    *, path: Path | None = None, cli_path: Path | None = None,
) -> VastCredentials:
    """Resolve environment, Launchpad file, then Vast's XDG CLI key file."""
    env_key = os.getenv("VAST_API_KEY", "").strip()
    if env_key:
        return VastCredentials(normalize_vast_api_key(env_key), "environment")
    saved_path = path or VAST_AUTH_PATH
    try:
        raw = json.loads(saved_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read the saved Vast key. Run vast-auth login again.") from exc
    else:
        if not isinstance(raw, dict) or not isinstance(raw.get("api_key"), str):
            raise ValueError("Invalid saved Vast key file. Run vast-auth login again.")
        return VastCredentials(normalize_vast_api_key(raw["api_key"]), "stored")
    config_home = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    try:
        key = (cli_path or config_home / "vastai" / "vast_api_key").read_text(encoding="utf-8")
    except FileNotFoundError:
        return VastCredentials()
    except OSError as exc:
        raise ValueError("Cannot read the Vast CLI key file.") from exc
    return VastCredentials(normalize_vast_api_key(key), "Vast CLI")


def save_vast_api_key(api_key: str, path: Path | None = None) -> Path:
    """Atomically store a key with owner-only permissions from file creation."""
    key = normalize_vast_api_key(api_key)
    target = path or VAST_AUTH_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".vast-auth-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"api_key": key}, stream)
            stream.write("\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def clear_vast_api_key(path: Path | None = None) -> bool:
    """Remove only Launchpad's saved key, leaving environment/CLI keys intact."""
    try:
        (path or VAST_AUTH_PATH).unlink()
    except FileNotFoundError:
        return False
    return True

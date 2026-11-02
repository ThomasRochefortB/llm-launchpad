"""Artificial Analysis credentials in the operating system's secret store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import keyring
from keyring.backend import KeyringBackend
from keyring.backends.chainer import ChainerBackend

from .config import SETTINGS_DIR

AAI_API_KEY_ENV = "ARTIFICIAL_ANALYSIS_API_KEY"
# Read only for migration from releases that stored credentials in JSON.
AAI_AUTH_PATH = SETTINGS_DIR / "artificial_analysis_auth.json"
_SERVICE = "llm-launchpad/artificial-analysis"
_ACCOUNT = "api-key"
_SECURE_BACKENDS = {
    "keyring.backends.SecretService",
    "keyring.backends.kwallet",
    "keyring.backends.macOS",
    "keyring.backends.Windows",
}
_STORE_ERROR = (
    "A secure OS credential store is unavailable or locked. Unlock/configure "
    "your system keyring, or use ARTIFICIAL_ANALYSIS_API_KEY for this session."
)


def _credential_store() -> KeyringBackend:
    """Select an OS backend, never a plaintext or null fallback."""
    backend = keyring.get_keyring()
    candidates = (
        backend.backends
        if isinstance(backend, ChainerBackend)
        else [backend]
    )
    for candidate in candidates:
        if type(candidate).__module__ in _SECURE_BACKENDS:
            return candidate
    raise RuntimeError(_STORE_ERROR)


def _remove_legacy_credentials(path: Path) -> None:
    """Remove both the old JSON file and an interrupted writer's temporary file."""
    path.unlink(missing_ok=True)
    path.with_suffix(f"{path.suffix}.tmp").unlink(missing_ok=True)


def resolve_artificial_analysis_api_key() -> str:
    """Resolve the AAI key with environment-first precedence."""
    environment_key = (os.getenv(AAI_API_KEY_ENV) or "").strip()
    if environment_key:
        return environment_key
    return load_saved_artificial_analysis_api_key()


def load_saved_artificial_analysis_api_key(path: Path | None = None) -> str:
    """Read the OS store and migrate an old JSON credential when possible."""
    legacy_path = path or AAI_AUTH_PATH
    try:
        store = _credential_store()
        saved = store.get_password(_SERVICE, _ACCOUNT)
        if saved:
            _remove_legacy_credentials(legacy_path)
            return saved.strip()
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        legacy_key = raw.get("api_key") if isinstance(raw, dict) else None
        if not isinstance(legacy_key, str) or not legacy_key.strip():
            return ""
        save_artificial_analysis_api_key(legacy_key, path=legacy_path)
        return legacy_key.strip()
    except Exception:
        # Optional catalog authentication must not stop the rest of the app.
        # Preserve the legacy file if migration could not complete.
        return ""


def save_artificial_analysis_api_key(api_key: str, path: Path | None = None) -> None:
    """Save to a secure OS store; remove legacy plaintext only after verification."""
    normalized_key = (api_key or "").strip()
    if not normalized_key:
        raise ValueError("An Artificial Analysis API key is required")
    try:
        store = _credential_store()
        store.set_password(_SERVICE, _ACCOUNT, normalized_key)
        if store.get_password(_SERVICE, _ACCOUNT) != normalized_key:
            raise RuntimeError(_STORE_ERROR)
        _remove_legacy_credentials(path or AAI_AUTH_PATH)
    except Exception:
        # Backend errors may include credential values; do not expose them.
        raise RuntimeError(_STORE_ERROR) from None


def clear_artificial_analysis_api_key(path: Path | None = None) -> bool:
    """Remove the OS credential and any legacy plaintext credential."""
    legacy_path = path or AAI_AUTH_PATH
    removed = False
    try:
        store = _credential_store()
        if store.get_password(_SERVICE, _ACCOUNT) is not None:
            store.delete_password(_SERVICE, _ACCOUNT)
            removed = True
    except Exception:
        raise RuntimeError(_STORE_ERROR) from None
    if legacy_path.exists() or legacy_path.with_suffix(f"{legacy_path.suffix}.tmp").exists():
        _remove_legacy_credentials(legacy_path)
        removed = True
    return removed

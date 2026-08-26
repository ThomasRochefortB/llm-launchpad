"""OpenCode config sync helpers for Launchpad-managed deployments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Iterable

from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import DeploymentConfig, EndpointInfo
from .config import SETTINGS_DIR
from .naming import default_llamacpp_served_model_name, default_served_model_name

OPENCODE_CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
OPENCODE_JSONC_CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.jsonc"
OPENCODE_REGISTRY_PATH = SETTINGS_DIR / "opencode_registry.json"
OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"

_PROVIDER_PREFIX = "llm-launchpad-"
_PROVIDER_NAME = "llm-launchpad"
_LEGACY_PROVIDER_NAME_PREFIX = "llm-launchpad: "
_MANAGED_NPM = "@ai-sdk/openai-compatible"
_REMOVABLE_STATES = {"stopped", "stopping", "terminated", "archived"}
_SYNC_LOCK = threading.Lock()


@dataclass(frozen=True)
class OpenCodeConnection:
    """Canonical provider/model descriptor for one Launchpad deployment."""

    app_name: str
    instance_name: str
    provider_id: str
    provider_name: str
    base_url: str
    model_id: str
    display_name: str
    backend: BackendType
    provider: ComputeProvider = ComputeProvider.MODAL
    api_key: str | None = None


@dataclass
class OpenCodeSyncResult:
    """Best-effort sync summary for CLI/TUI reporting."""

    detected: bool
    changed: bool = False
    config_path: Path = OPENCODE_CONFIG_PATH
    registry_path: Path = OPENCODE_REGISTRY_PATH
    messages: list[str] = field(default_factory=list)
    created_provider_ids: list[str] = field(default_factory=list)
    updated_provider_ids: list[str] = field(default_factory=list)
    removed_provider_ids: list[str] = field(default_factory=list)
    dropped_registry_app_names: list[str] = field(default_factory=list)


def is_opencode_installed() -> bool:
    """Return True when OpenCode is discoverable via CLI or an existing config file."""
    return bool(shutil.which("opencode")) or resolve_opencode_config_path().exists()


def resolve_opencode_config_path() -> Path:
    """Resolve the config path OpenCode/Launchpad should read and write."""
    configured = os.environ.get("OPENCODE_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser()
    if OPENCODE_CONFIG_PATH.exists():
        return OPENCODE_CONFIG_PATH
    if OPENCODE_JSONC_CONFIG_PATH.exists():
        return OPENCODE_JSONC_CONFIG_PATH
    return OPENCODE_CONFIG_PATH


def _strip_repo_owner(value: str) -> str:
    """Return the repo/model segment for Hugging Face-style owner/model labels."""
    text = value.strip()
    if text.count("/") != 1:
        return text
    owner, model = text.split("/", 1)
    if not owner or not model:
        return text
    return model.strip() or text


def _format_opencode_display_name(source: str, quant: str = "") -> str:
    """Build the user-facing OpenCode label without the HF namespace prefix."""
    base = _strip_repo_owner(source) or "Model"
    quant_text = quant.strip()
    return f"{base} ({quant_text})" if quant_text else base


def build_openai_connection_payload(
    config: DeploymentConfig,
    server_url: str,
) -> dict[str, str]:
    """Return a shared OpenAI-compatible connection summary payload."""
    base_root = server_url.rstrip("/")
    base_url = base_root if base_root.endswith("/v1") else f"{base_root}/v1"

    if config.backend == BackendType.VLLM:
        model_id = (config.served_model_name or default_served_model_name(config.model_name)).strip()
        display_name = _format_opencode_display_name(config.served_model_name or config.model_name or model_id)
    else:
        model_id = (
            (config.served_model_name or "").strip()
            or default_llamacpp_served_model_name(config.repo_id, config.quant)
        )
        display_name = _format_opencode_display_name(config.repo_id or "llama.cpp GGUF", config.quant or "")

    return {
        "base_url": base_url,
        "model_id": model_id,
        "display_name": display_name,
    }


def build_connection_from_config(
    config: DeploymentConfig,
    server_url: str,
) -> OpenCodeConnection | None:
    """Build an OpenCode descriptor from deployment config + endpoint URL."""
    app_name = (config.app_name or "").strip()
    if not app_name:
        return None

    payload = build_openai_connection_payload(config, server_url)
    instance_name = (config.instance_name or app_name).strip() or app_name
    return OpenCodeConnection(
        app_name=app_name,
        instance_name=instance_name,
        provider_id=provider_id_for_app(app_name),
        provider_name=_PROVIDER_NAME,
        base_url=payload["base_url"],
        model_id=payload["model_id"],
        display_name=payload["display_name"],
        backend=config.backend,
        provider=config.provider,
        api_key=config.endpoint_api_key,
    )


def build_connection_from_endpoint(
    row: EndpointInfo,
    *,
    username: str = "",
    server_url: str | None = None,
) -> OpenCodeConnection | None:
    """Build an OpenCode descriptor from a Modal app row."""
    app_name = (row.name or "").strip()
    backend = row.backend
    if not app_name or backend is None:
        return None

    # Prefer Modal's currently reported web URL for the app whenever present.
    # A fallback URL may come from deploy/warmup output and can point at a
    # transient or stale web endpoint after the app is redeployed.
    base_root = (row.web_url or server_url or "").strip().rstrip("/")
    if not base_root and username.strip():
        from .backend import ModalBackend

        base_root = ModalBackend.default_server_url(username.strip(), app_name=app_name).rstrip("/")
    if not base_root:
        return None

    base_url = base_root if base_root.endswith("/v1") else f"{base_root}/v1"
    instance_name = (row.instance_name or app_name).strip() or app_name

    if backend == BackendType.VLLM:
        model_id = (row.served_model_name or default_served_model_name(row.model_name)).strip()
        display_name = (
            (row.display_name or "").strip()
            or (row.model_name or "").strip()
            or (row.served_model_name or "").strip()
            or model_id
        )
        display_name = _format_opencode_display_name(display_name)
    else:
        model_id = (
            (row.served_model_name or "").strip()
            or default_llamacpp_served_model_name(row.repo_id, row.quant)
        )
        display_name = (row.display_name or "").strip()
        if not display_name:
            display_name = _format_opencode_display_name(row.repo_id or "llama.cpp GGUF", row.quant or "")
        else:
            display_name = _format_opencode_display_name(display_name)

    return OpenCodeConnection(
        app_name=app_name,
        instance_name=instance_name,
        provider_id=provider_id_for_app(app_name),
        provider_name=_PROVIDER_NAME,
        base_url=base_url,
        model_id=model_id,
        display_name=display_name,
        backend=backend,
        provider=row.provider,
        api_key=row.endpoint_api_key,
    )


def provider_id_for_app(app_name: str) -> str:
    """Return the deterministic OpenCode provider id for a Launchpad app."""
    return f"{_PROVIDER_PREFIX}{app_name.strip()}"


def visible_launchpad_rows(rows: Iterable[EndpointInfo]) -> list[EndpointInfo]:
    """Return the deduped Launchpad app rows used for cleanup decisions."""
    deduped: list[EndpointInfo] = []
    key_to_index: dict[tuple[str, str, str], int] = {}

    for row in rows:
        if row.backend is None:
            continue
        backend_key = row.backend.value
        instance_key = (row.instance_name or "").strip()
        name_key = (row.name or "").strip()
        key = (backend_key, instance_key, name_key)
        existing_index = key_to_index.get(key)
        if existing_index is None:
            key_to_index[key] = len(deduped)
            deduped.append(row)
            continue

        existing = deduped[existing_index]
        if _is_removable_modal_state(existing.state) and not _is_removable_modal_state(row.state):
            deduped[existing_index] = row

    return deduped


def resolve_connection_for_app(
    app_name: str,
    *,
    rows: Iterable[EndpointInfo] | None = None,
    username: str = "",
    fallback_config: DeploymentConfig | None = None,
    fallback_server_url: str | None = None,
) -> OpenCodeConnection | None:
    """Resolve the best available OpenCode connection descriptor for an app."""
    target_app_name = (app_name or "").strip()
    if not target_app_name:
        return None

    for row in visible_launchpad_rows(rows or []):
        if (row.name or "").strip() != target_app_name:
            continue
        resolved = build_connection_from_endpoint(
            row,
            username=username,
            server_url=fallback_server_url,
        )
        if resolved is not None:
            return resolved

    if fallback_config is not None:
        if fallback_server_url:
            return build_connection_from_config(fallback_config, fallback_server_url)
    return None


def resolve_connections_for_rows(
    rows: Iterable[EndpointInfo] | None,
    *,
    username: str = "",
) -> list[OpenCodeConnection]:
    """Resolve OpenCode descriptors for all visible Launchpad rows."""
    resolved_by_app: dict[str, OpenCodeConnection] = {}
    for row in visible_launchpad_rows(rows or []):
        connection = build_connection_from_endpoint(row, username=username)
        if connection is None:
            continue
        resolved_by_app[connection.app_name] = connection
    return list(resolved_by_app.values())


def sync_opencode_config(
    *,
    target: OpenCodeConnection | None = None,
    targets: Iterable[OpenCodeConnection] | None = None,
    current_rows: Iterable[EndpointInfo] | None = None,
    remove_app_names: Iterable[str] | None = None,
    dry_run: bool = False,
) -> OpenCodeSyncResult:
    """Upsert the target provider and prune stale Launchpad-managed providers."""
    result = OpenCodeSyncResult(
        detected=False,
        config_path=resolve_opencode_config_path(),
        registry_path=OPENCODE_REGISTRY_PATH,
    )
    if not is_opencode_installed():
        result.messages.append("OpenCode not detected; skipping OpenCode sync.")
        return result

    result.detected = True
    with _SYNC_LOCK:
        config = _load_opencode_config(result.config_path)
        registry, bootstrapped = _load_registry(result.registry_path, config)
        config_changed = False
        registry_changed = False
        if bootstrapped and registry:
            registry_changed = True

        provider_map = config.get("provider")
        if not isinstance(provider_map, dict):
            provider_map = {}

        for provider_id, provider_value in provider_map.items():
            if not str(provider_id or "").strip().startswith(_PROVIDER_PREFIX):
                continue
            if not isinstance(provider_value, dict):
                continue
            if str(provider_value.get("npm", "") or "").strip() != _MANAGED_NPM:
                continue
            provider_name = str(provider_value.get("name", "") or "").strip()
            if not provider_name.startswith(_LEGACY_PROVIDER_NAME_PREFIX):
                continue
            next_provider = dict(provider_value)
            next_provider["name"] = _PROVIDER_NAME
            provider_map[provider_id] = next_provider
            config_changed = True

        desired_targets: list[OpenCodeConnection] = []
        if target is not None:
            desired_targets.append(target)
        for candidate in targets or []:
            if candidate is None:
                continue
            desired_targets.append(candidate)
        deduped_targets: dict[str, OpenCodeConnection] = {}
        for connection in desired_targets:
            deduped_targets[connection.app_name] = connection

        if deduped_targets:
            if not isinstance(config.get("provider"), dict):
                config["provider"] = provider_map
                config_changed = True
            if "$schema" not in config:
                config["$schema"] = OPENCODE_SCHEMA_URL
                config_changed = True
            for connection in deduped_targets.values():
                desired_provider = _provider_payload(connection)
                existing_provider = provider_map.get(connection.provider_id)
                if existing_provider != desired_provider:
                    provider_map[connection.provider_id] = desired_provider
                    registry[connection.app_name] = _registry_entry_for_connection(connection)
                    config_changed = True
                    registry_changed = True
                    if existing_provider is None:
                        result.created_provider_ids.append(connection.provider_id)
                        result.messages.append(
                            _prefixed_message(
                                dry_run,
                                f"upsert provider {connection.provider_id} ({connection.model_id} -> {connection.base_url})",
                            )
                        )
                    else:
                        result.updated_provider_ids.append(connection.provider_id)
                        result.messages.append(
                            _prefixed_message(
                                dry_run,
                                f"update provider {connection.provider_id} ({connection.model_id} -> {connection.base_url})",
                            )
                        )
                else:
                    next_entry = _registry_entry_for_connection(connection)
                    if registry.get(connection.app_name) != next_entry:
                        registry[connection.app_name] = next_entry
                        registry_changed = True

        removed_app_names = {
            str(app_name or "").strip()
            for app_name in (remove_app_names or [])
            if str(app_name or "").strip()
        }
        protected_app_names = {
            connection.app_name
            for connection in deduped_targets.values()
            if connection.app_name not in removed_app_names
        }

        for app_name in sorted(removed_app_names):
            entry = registry.get(app_name)
            provider_id = str(entry.get("provider_id", "") or "") if isinstance(entry, dict) else ""
            if provider_id and provider_id in provider_map:
                provider_map.pop(provider_id, None)
                result.removed_provider_ids.append(provider_id)
                result.messages.append(_prefixed_message(dry_run, f"remove provider {provider_id}"))
                config_changed = True
            if app_name in registry:
                registry.pop(app_name, None)
                registry_changed = True

        for app_name in sorted(list(registry.keys())):
            entry = registry.get(app_name)
            if not isinstance(entry, dict):
                registry.pop(app_name, None)
                result.dropped_registry_app_names.append(app_name)
                registry_changed = True
                continue

            provider_id = str(entry.get("provider_id", "") or "").strip()
            if not provider_id or provider_id not in provider_map:
                registry.pop(app_name, None)
                result.dropped_registry_app_names.append(app_name)
                registry_changed = True

        if current_rows is not None:
            visible_rows = visible_launchpad_rows(current_rows)
            visible_by_name: dict[str, EndpointInfo] = {}
            for row in visible_rows:
                row_name = str(row.name or "").strip()
                if row_name:
                    visible_by_name[row_name] = row

            for app_name in sorted(list(registry.keys())):
                entry = registry.get(app_name)
                if not isinstance(entry, dict):
                    continue

                provider_id = str(entry.get("provider_id", "") or "").strip()
                if not provider_id:
                    continue

                row = visible_by_name.get(app_name)
                if row is None and app_name not in protected_app_names:
                    provider_map.pop(provider_id, None)
                    registry.pop(app_name, None)
                    result.removed_provider_ids.append(provider_id)
                    result.messages.append(
                        _prefixed_message(dry_run, f"remove stale provider {provider_id} (missing in Modal app list)")
                    )
                    config_changed = True
                    registry_changed = True
                    continue

                if row is not None and _is_removable_modal_state(row.state):
                    provider_map.pop(provider_id, None)
                    registry.pop(app_name, None)
                    result.removed_provider_ids.append(provider_id)
                    result.messages.append(
                        _prefixed_message(
                            dry_run,
                            f"remove stale provider {provider_id} (state={row.state.strip().lower() or 'unknown'})",
                        )
                    )
                    config_changed = True
                    registry_changed = True

        if not result.messages:
            result.messages.append(_prefixed_message(dry_run, "no OpenCode changes"))

        result.changed = config_changed or registry_changed
        if dry_run:
            return result

        if config_changed:
            _write_opencode_config(result.config_path, config)
        if registry_changed:
            _write_registry(result.registry_path, registry)

    return result


def _prefixed_message(dry_run: bool, message: str) -> str:
    prefix = "OpenCode dry run:" if dry_run else "OpenCode:"
    return f"{prefix} {message}"


def _provider_payload(connection: OpenCodeConnection) -> dict[str, Any]:
    options: dict[str, str] = {"baseURL": connection.base_url}
    if connection.api_key:
        options["apiKey"] = connection.api_key
    return {
        "npm": _MANAGED_NPM,
        "name": connection.provider_name,
        "options": options,
        "models": {
            connection.model_id: {
                "name": connection.display_name,
            }
        },
    }


def _registry_entry_for_connection(connection: OpenCodeConnection) -> dict[str, Any]:
    return {
        "provider_id": connection.provider_id,
        "app_name": connection.app_name,
        "instance_name": connection.instance_name,
        "backend": connection.backend.value,
        "provider": connection.provider.value,
        "base_url": connection.base_url,
        "model_id": connection.model_id,
        "display_name": connection.display_name,
    }


def _load_opencode_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    payload = _parse_jsonc(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("OpenCode config must contain a JSON object at the top level.")
    return payload


def _write_opencode_config(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == serialized:
            return
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(current, encoding="utf-8")
        try:
            os.chmod(backup_path, 0o600)
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _load_registry(
    path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8") or "{}")
            entries = payload.get("entries")
            if isinstance(entries, dict):
                normalized: dict[str, dict[str, Any]] = {}
                for key, value in entries.items():
                    app_name = str(key or "").strip()
                    if app_name and isinstance(value, dict):
                        normalized[app_name] = value
                return normalized, False
        except Exception:
            pass

    return _bootstrap_registry_from_config(config), True


def _write_registry(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": entries}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _bootstrap_registry_from_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    provider_map = config.get("provider")
    if not isinstance(provider_map, dict):
        return {}

    adopted: dict[str, dict[str, Any]] = {}
    for provider_id, value in provider_map.items():
        key = str(provider_id or "").strip()
        if not key.startswith(_PROVIDER_PREFIX) or not isinstance(value, dict):
            continue
        if str(value.get("npm", "") or "").strip() != _MANAGED_NPM:
            continue
        app_name = key.removeprefix(_PROVIDER_PREFIX).strip()
        if not app_name:
            continue
        provider_name = str(value.get("name", "") or "").strip()
        if provider_name == _PROVIDER_NAME:
            instance_name = app_name
        elif provider_name.startswith(_LEGACY_PROVIDER_NAME_PREFIX):
            instance_name = provider_name.removeprefix(_LEGACY_PROVIDER_NAME_PREFIX).strip() or app_name
        else:
            continue
        options = value.get("options")
        models = value.get("models")
        base_url = ""
        model_id = ""
        display_name = ""
        if isinstance(options, dict):
            base_url = str(options.get("baseURL", "") or "").strip()
        if isinstance(models, dict):
            for model_key, model_value in models.items():
                model_id = str(model_key or "").strip()
                if isinstance(model_value, dict):
                    display_name = str(model_value.get("name", "") or "").strip()
                break
        adopted[app_name] = {
            "provider_id": key,
            "app_name": app_name,
            "instance_name": instance_name,
            "backend": "",
            "base_url": base_url,
            "model_id": model_id,
            "display_name": display_name,
        }
    return adopted


def _parse_jsonc(text: str) -> Any:
    without_comments = _strip_json_comments(text)
    without_trailing_commas = _strip_trailing_commas(without_comments)
    return json.loads(without_trailing_commas)


def _strip_json_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if ch in "\r\n":
                in_line_comment = False
                result.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            if ch in "\r\n":
                result.append(ch)
            i += 1
            continue

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _strip_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in "}]":
                i += 1
                continue

        result.append(ch)
        i += 1

    return "".join(result)


def _is_removable_modal_state(state: str) -> bool:
    return (state or "").strip().lower() in _REMOVABLE_STATES

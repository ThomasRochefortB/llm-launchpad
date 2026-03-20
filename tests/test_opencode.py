from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_launchpad.core import opencode
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo


def _connection(
    *,
    app_name: str = "vllm-qwen3",
    instance_name: str = "qwen3",
    base_url: str = "https://alice--vllm-qwen3-serve.modal.run/v1",
    model_id: str = "Qwen3-4B",
    display_name: str = "Qwen3-4B",
    backend: BackendType = BackendType.VLLM,
) -> opencode.OpenCodeConnection:
    return opencode.OpenCodeConnection(
        app_name=app_name,
        instance_name=instance_name,
        provider_id=opencode.provider_id_for_app(app_name),
        provider_name=f"llm-launchpad: {instance_name}",
        base_url=base_url,
        model_id=model_id,
        display_name=display_name,
        backend=backend,
    )


def _provider_payload(connection: opencode.OpenCodeConnection) -> dict[str, object]:
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": connection.provider_name,
        "options": {"baseURL": connection.base_url},
        "models": {connection.model_id: {"name": connection.display_name}},
    }


def _endpoint(
    *,
    name: str,
    state: str,
    backend: BackendType = BackendType.VLLM,
    instance_name: str | None = None,
) -> EndpointInfo:
    return EndpointInfo(
        name=name,
        state=state,
        backend=backend,
        instance_name=instance_name or name.removeprefix(f"{backend.value}-"),
    )


class OpenCodeSyncTests(unittest.TestCase):
    def _patched_paths(self, tmp: str):
        return (
            patch.object(opencode, "OPENCODE_CONFIG_PATH", Path(tmp) / "opencode.json"),
            patch.object(opencode, "OPENCODE_REGISTRY_PATH", Path(tmp) / "opencode_registry.json"),
        )

    def test_sync_skips_when_opencode_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            config_path.write_text('{"provider":{"keep":{"name":"Keep"}}}\n', encoding="utf-8")
            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1]:
                with patch("llm_launchpad.core.opencode.shutil.which", return_value=None):
                    result = opencode.sync_opencode_config(target=_connection())

            self.assertFalse(result.detected)
            self.assertEqual(config_path.read_text(encoding="utf-8"), '{"provider":{"keep":{"name":"Keep"}}}\n')
            self.assertFalse((Path(tmp) / "opencode_registry.json").exists())

    def test_sync_creates_config_and_registry_on_first_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            target = _connection()
            with patch.object(opencode, "OPENCODE_CONFIG_PATH", config_path), patch.object(
                opencode, "OPENCODE_REGISTRY_PATH", registry_path
            ), patch("llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"):
                result = opencode.sync_opencode_config(target=target)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertTrue(result.detected)
            self.assertTrue(result.changed)
            self.assertEqual(payload["$schema"], opencode.OPENCODE_SCHEMA_URL)
            self.assertEqual(payload["provider"][target.provider_id], _provider_payload(target))
            self.assertEqual(registry["entries"][target.app_name]["provider_id"], target.provider_id)

    def test_sync_updates_existing_provider_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            first = _connection()
            second = _connection(
                base_url="https://alice--vllm-qwen3-serve-abc.modal.run/v1",
                model_id="Qwen3-8B",
                display_name="Qwen3-8B",
            )
            with patch.object(opencode, "OPENCODE_CONFIG_PATH", config_path), patch.object(
                opencode, "OPENCODE_REGISTRY_PATH", registry_path
            ), patch("llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"):
                opencode.sync_opencode_config(target=first)
                result = opencode.sync_opencode_config(target=second)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(list(payload["provider"].keys()), [first.provider_id])
            self.assertEqual(payload["provider"][first.provider_id]["options"]["baseURL"], second.base_url)
            self.assertEqual(list(payload["provider"][first.provider_id]["models"].keys()), [second.model_id])
            self.assertTrue((config_path.with_suffix(".json.bak")).exists())
            self.assertEqual(result.updated_provider_ids, [first.provider_id])

    def test_sync_prunes_missing_and_stopped_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            stopped = _connection(app_name="vllm-stopped", instance_name="stopped")
            missing = _connection(app_name="vllm-missing", instance_name="missing")
            config_payload = {
                "$schema": opencode.OPENCODE_SCHEMA_URL,
                "provider": {
                    stopped.provider_id: _provider_payload(stopped),
                    missing.provider_id: _provider_payload(missing),
                    "custom-provider": {"name": "keep"},
                },
            }
            registry_payload = {
                "entries": {
                    stopped.app_name: {"provider_id": stopped.provider_id},
                    missing.app_name: {"provider_id": missing.provider_id},
                }
            }
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")
            rows = [_endpoint(name=stopped.app_name, state="stopped")]

            with patch.object(opencode, "OPENCODE_CONFIG_PATH", config_path), patch.object(
                opencode, "OPENCODE_REGISTRY_PATH", registry_path
            ), patch("llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"):
                result = opencode.sync_opencode_config(current_rows=rows)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertTrue(result.changed)
            self.assertEqual(set(result.removed_provider_ids), {stopped.provider_id, missing.provider_id})
            self.assertEqual(payload["provider"], {"custom-provider": {"name": "keep"}})
            self.assertEqual(registry["entries"], {})

    def test_sync_retains_failed_and_deploying_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            failed = _connection(app_name="vllm-failed", instance_name="failed")
            deploying = _connection(app_name="vllm-deploying", instance_name="deploying")
            config_path.write_text(
                json.dumps(
                    {
                        "$schema": opencode.OPENCODE_SCHEMA_URL,
                        "provider": {
                            failed.provider_id: _provider_payload(failed),
                            deploying.provider_id: _provider_payload(deploying),
                        },
                    }
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            failed.app_name: {"provider_id": failed.provider_id},
                            deploying.app_name: {"provider_id": deploying.provider_id},
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                _endpoint(name=failed.app_name, state="failed"),
                _endpoint(name=deploying.app_name, state="deploying"),
            ]

            with patch.object(opencode, "OPENCODE_CONFIG_PATH", config_path), patch.object(
                opencode, "OPENCODE_REGISTRY_PATH", registry_path
            ), patch("llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"):
                result = opencode.sync_opencode_config(current_rows=rows)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertFalse(result.removed_provider_ids)
            self.assertIn(failed.provider_id, payload["provider"])
            self.assertIn(deploying.provider_id, payload["provider"])

    def test_sync_self_heals_registry_when_provider_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            ghost = _connection(app_name="vllm-ghost", instance_name="ghost")
            config_path.write_text(json.dumps({"provider": {}}), encoding="utf-8")
            registry_path.write_text(
                json.dumps({"entries": {ghost.app_name: {"provider_id": ghost.provider_id}}}),
                encoding="utf-8",
            )

            with patch.object(opencode, "OPENCODE_CONFIG_PATH", config_path), patch.object(
                opencode, "OPENCODE_REGISTRY_PATH", registry_path
            ), patch("llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"):
                result = opencode.sync_opencode_config()

            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(result.dropped_registry_app_names, [ghost.app_name])
            self.assertEqual(registry["entries"], {})

    def test_sync_reads_jsonc_and_writes_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            config_path.write_text(
                """
                {
                  // existing config
                  "provider": {
                    "custom": {
                      "name": "Keep me",
                    },
                  },
                }
                """,
                encoding="utf-8",
            )
            target = _connection()

            with patch.object(opencode, "OPENCODE_CONFIG_PATH", config_path), patch.object(
                opencode, "OPENCODE_REGISTRY_PATH", registry_path
            ), patch("llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"):
                opencode.sync_opencode_config(target=target)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            backup = config_path.with_suffix(".json.bak")
            self.assertTrue(backup.exists())
            self.assertIn("// existing config", backup.read_text(encoding="utf-8"))
            self.assertIn("custom", payload["provider"])
            self.assertIn(target.provider_id, payload["provider"])


if __name__ == "__main__":
    unittest.main()

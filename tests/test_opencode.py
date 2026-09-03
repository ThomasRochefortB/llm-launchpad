from __future__ import annotations

import json
import tempfile
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_launchpad.core import opencode
from llm_launchpad.protocol.models import DeploymentConfig
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, ReasoningCapabilities


def _reasoning(
    *,
    efforts: tuple[str, ...] = ("low", "medium", "xhigh"),
    default_effort: str = "xhigh",
    interleaved_field: str | None = None,
) -> ReasoningCapabilities:
    return ReasoningCapabilities(
        profile_id="hf-test-profile",
        canonical_model_id="acme/Future-Reasoner",
        model_revision="a" * 40,
        efforts=efforts,
        default_effort=default_effort,
        source_repo="acme/Future-Reasoner",
        source_revision="b" * 40,
        source_path="chat_template.jinja",
        request_option_path="chat_template_kwargs.reasoning_effort",
        enable_thinking=True,
        interleaved_field=interleaved_field,
    )


def _connection(
    *,
    app_name: str = "vllm-qwen3",
    instance_name: str = "qwen3",
    base_url: str = "https://alice--vllm-qwen3-serve.modal.run/v1",
    model_id: str = "Qwen3-4B",
    display_name: str = "Qwen3-4B",
    backend: BackendType = BackendType.VLLM,
    provider: ComputeProvider = ComputeProvider.MODAL,
    context_limit: int | None = None,
    output_limit: int | None = None,
    reasoning: ReasoningCapabilities | None = None,
) -> opencode.OpenCodeConnection:
    return opencode.OpenCodeConnection(
        app_name=app_name,
        instance_name=instance_name,
        provider_id=opencode.provider_id_for_app(app_name),
        provider_name="llm-launchpad",
        base_url=base_url,
        model_id=model_id,
        display_name=display_name,
        backend=backend,
        provider=provider,
        context_limit=context_limit,
        output_limit=output_limit,
        reasoning=reasoning,
    )


def _provider_payload(connection: opencode.OpenCodeConnection) -> dict[str, object]:
    model: dict[str, object] = {"name": connection.display_name}
    if connection.context_limit is not None and connection.output_limit is not None:
        model["limit"] = {
            "context": connection.context_limit,
            "output": connection.output_limit,
        }
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": connection.provider_name,
        "options": {"baseURL": connection.base_url},
        "models": {connection.model_id: model},
    }


def _endpoint(
    *,
    name: str,
    state: str,
    backend: BackendType = BackendType.VLLM,
    instance_name: str | None = None,
    provider: ComputeProvider = ComputeProvider.MODAL,
) -> EndpointInfo:
    return EndpointInfo(
        name=name,
        state=state,
        backend=backend,
        instance_name=instance_name or name.removeprefix(f"{backend.value}-"),
        provider=provider,
    )


class OpenCodeSyncTests(unittest.TestCase):
    def _patched_paths(self, tmp: str):
        return (
            patch.object(opencode, "OPENCODE_CONFIG_PATH", Path(tmp) / "opencode.json"),
            patch.object(opencode, "OPENCODE_JSONC_CONFIG_PATH", Path(tmp) / "opencode.jsonc"),
            patch.object(opencode, "OPENCODE_REGISTRY_PATH", Path(tmp) / "opencode_registry.json"),
        )

    def test_sync_skips_when_opencode_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2]:
                with patch("llm_launchpad.core.opencode.shutil.which", return_value=None):
                    result = opencode.sync_opencode_config(target=_connection())

            self.assertFalse(result.detected)
            self.assertFalse(config_path.exists())
            self.assertFalse((Path(tmp) / "opencode_registry.json").exists())

    def test_sync_creates_config_and_registry_on_first_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            target = _connection()
            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
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
            first = _connection()
            second = _connection(
                base_url="https://alice--vllm-qwen3-serve-abc.modal.run/v1",
                model_id="Qwen3-8B",
                display_name="Qwen3-8B",
            )
            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                opencode.sync_opencode_config(target=first)
                result = opencode.sync_opencode_config(target=second)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(list(payload["provider"].keys()), [first.provider_id])
            self.assertEqual(payload["provider"][first.provider_id]["options"]["baseURL"], second.base_url)
            self.assertEqual(list(payload["provider"][first.provider_id]["models"].keys()), [second.model_id])
            self.assertTrue((config_path.with_suffix(".json.bak")).exists())
            self.assertEqual(result.updated_provider_ids, [first.provider_id])

    def test_sync_upserts_multiple_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            vllm_target = _connection()
            llamacpp_target = _connection(
                app_name="llamacpp-phi",
                instance_name="phi",
                base_url="https://alice--llamacpp-phi-serve.modal.run/v1",
                model_id="Phi-3-GGUF-Q4_K_M",
                display_name="unsloth/Phi-3-GGUF (Q4_K_M)",
                backend=BackendType.LLAMACPP,
            )
            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                result = opencode.sync_opencode_config(targets=[vllm_target, llamacpp_target])

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertTrue(result.changed)
            self.assertEqual(
                set(payload["provider"].keys()),
                {vllm_target.provider_id, llamacpp_target.provider_id},
            )
            self.assertEqual(
                set(registry["entries"].keys()),
                {vllm_target.app_name, llamacpp_target.app_name},
            )

    def test_build_openai_connection_payload_strips_hf_owner_from_llamacpp_display_name(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-glm",
            instance_name="glm",
            repo_id="unsloth/GLM-4.7-Flash-GGUF",
            quant="Q4_K_M",
        )

        payload = opencode.build_openai_connection_payload(config, "https://example.com")

        self.assertEqual(payload["display_name"], "GLM-4.7-Flash-GGUF (Q4_K_M)")

    def test_build_connection_from_config_advertises_effective_llamacpp_limits(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-qwen",
            instance_name="qwen",
            repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
            quant="UD-Q2_K_XL",
            server_args="--ctx-size 131072 --parallel 1",
            max_context_tokens=262144,
        )

        connection = opencode.build_connection_from_config(
            config,
            "https://example.com",
        )

        self.assertIsNotNone(connection)
        assert connection is not None
        self.assertEqual(connection.context_limit, 131072)
        self.assertEqual(connection.output_limit, 32768)
        self.assertEqual(
            opencode._provider_payload(connection)["models"][connection.model_id]["limit"],
            {"context": 131072, "output": 32768},
        )

    def test_build_connection_uses_smaller_runtime_window_and_output_reserve(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-kimi",
            repo_id="unsloth/Kimi-K2.5-GGUF",
            quant="UD-Q4_K_XL",
            server_args="--ctx-size=98304",
            max_context_tokens=262144,
        )

        connection = opencode.build_connection_from_config(
            config,
            "https://example.com/v1",
        )

        self.assertIsNotNone(connection)
        assert connection is not None
        self.assertEqual(connection.context_limit, 98304)
        self.assertEqual(connection.output_limit, 24576)

    def test_sync_persists_model_limits_in_opencode_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            target = _connection(context_limit=262144, output_limit=32768)
            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                opencode.sync_opencode_config(target=target)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            model = payload["provider"][target.provider_id]["models"][target.model_id]
            self.assertEqual(
                model["limit"],
                {"context": 262144, "output": 32768},
            )

    def test_sync_persists_discovered_reasoning_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            target = _connection(
                app_name="llamacpp-qwen38",
                backend=BackendType.LLAMACPP,
                model_id="Qwen3.8-27B-GGUF-UD-Q4_K_XL",
                display_name="Qwen3.8-27B-GGUF (UD-Q4_K_XL)",
                reasoning=_reasoning(),
            )
            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                opencode.sync_opencode_config(target=target)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            model = payload["provider"][target.provider_id]["models"][target.model_id]
            self.assertTrue(model["reasoning"])
            self.assertEqual(
                list(model["variants"]),
                ["default", "low", "medium", "xhigh"],
            )
            self.assertEqual(
                model["variants"]["medium"],
                {
                    "chat_template_kwargs": {
                        "reasoning_effort": "medium",
                        "enable_thinking": True,
                    }
                },
            )
            self.assertNotIn("high", model["variants"])
            self.assertEqual(
                registry["entries"][target.app_name]["reasoning"]["source_revision"],
                "b" * 40,
            )

    def test_build_connection_from_config_uses_selection_discovery(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            app_name="vllm-future-reasoner",
            model_name="acme/Future-Reasoner",
            reasoning=_reasoning(
                efforts=("high", "max"),
                default_effort="max",
            ),
        )

        connection = opencode.build_connection_from_config(config, "https://example.com")

        self.assertIsNotNone(connection)
        assert connection is not None
        self.assertIsNotNone(connection.reasoning)
        assert connection.reasoning is not None
        self.assertEqual(connection.reasoning.efforts, ("high", "max"))
        model = opencode._provider_payload(connection)["models"][connection.model_id]
        self.assertEqual(list(model["variants"]), ["default", "high", "max"])

    def test_endpoint_uses_source_discovered_interleaved_reasoning(self) -> None:
        row = EndpointInfo(
            name="llamacpp-future-reasoner",
            state="running",
            backend=BackendType.LLAMACPP,
            web_url="https://example.com",
            served_model_name="Future-Reasoner-GGUF-UD-Q2_K_XL",
            reasoning=_reasoning(
                efforts=("low", "high", "max"),
                default_effort="low",
                interleaved_field="reasoning_content",
            ),
        )

        connection = opencode.build_connection_from_endpoint(row)

        self.assertIsNotNone(connection)
        assert connection is not None
        model = opencode._provider_payload(connection)["models"][connection.model_id]
        self.assertEqual(model["interleaved"], {"field": "reasoning_content"})
        self.assertEqual(
            list(model["variants"]),
            ["default", "low", "high", "max"],
        )

    def test_unknown_model_does_not_advertise_reasoning(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            app_name="vllm-unknown",
            model_name="example/Unknown-7B",
        )

        connection = opencode.build_connection_from_config(config, "https://example.com")

        self.assertIsNotNone(connection)
        assert connection is not None
        model = opencode._provider_payload(connection)["models"][connection.model_id]
        self.assertNotIn("reasoning", model)
        self.assertNotIn("variants", model)

    def test_build_connection_from_endpoint_strips_hf_owner_from_vllm_display_name(self) -> None:
        row = EndpointInfo(
            name="vllm-glm",
            state="running",
            backend=BackendType.VLLM,
            model_name="zai-org/GLM-5",
        )

        connection = opencode.build_connection_from_endpoint(
            row,
            server_url="https://example.com",
        )

        self.assertIsNotNone(connection)
        assert connection is not None
        self.assertEqual(connection.display_name, "GLM-5")

    def test_resolve_connection_for_app_prefers_modal_row_web_url_over_fallback_url(self) -> None:
        row = EndpointInfo(
            name="llamacpp-glm",
            state="running",
            backend=BackendType.LLAMACPP,
            instance_name="glm",
            web_url="https://alice--llamacpp-glm-serve-live.modal.run",
            repo_id="unsloth/GLM-4.7-Flash-GGUF",
            quant="Q4_K_M",
        )
        fallback_config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-glm",
            instance_name="glm",
            repo_id="unsloth/GLM-4.7-Flash-GGUF",
            quant="Q4_K_M",
            reasoning=_reasoning(
                efforts=("low", "high"),
                default_effort="high",
            ),
        )

        resolved = opencode.resolve_connection_for_app(
            "llamacpp-glm",
            rows=[row],
            fallback_config=fallback_config,
            fallback_server_url="https://alice--llamacpp-glm-serve-stale.modal.run",
        )

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(
            resolved.base_url,
            "https://alice--llamacpp-glm-serve-live.modal.run/v1",
        )
        self.assertIsNotNone(resolved.reasoning)
        assert resolved.reasoning is not None
        self.assertEqual(resolved.reasoning.efforts, ("low", "high"))

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

            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                result = opencode.sync_opencode_config(current_rows=rows)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertTrue(result.changed)
            self.assertEqual(set(result.removed_provider_ids), {stopped.provider_id, missing.provider_id})
            self.assertEqual(payload["provider"], {"custom-provider": {"name": "keep"}})
            self.assertEqual(registry["entries"], {})
            self.assertTrue(
                any("Modal deployment list" in line for line in result.messages)
            )

    def test_sync_does_not_prune_other_compute_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            registry_path = Path(tmp) / "opencode_registry.json"
            modal_missing = _connection(app_name="vllm-missing", instance_name="missing")
            prime_live = _connection(
                app_name="llp-prime-llamacpp-qwen",
                instance_name="qwen",
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.PRIME,
                base_url="https://example.tunnel.pinfra.io/v1",
                model_id="Qwen-GGUF",
                display_name="Qwen GGUF",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "$schema": opencode.OPENCODE_SCHEMA_URL,
                        "provider": {
                            modal_missing.provider_id: _provider_payload(modal_missing),
                            prime_live.provider_id: _provider_payload(prime_live),
                        },
                    }
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            modal_missing.app_name: {
                                "provider_id": modal_missing.provider_id,
                                "provider": "modal",
                            },
                            prime_live.app_name: {
                                "provider_id": prime_live.provider_id,
                                "provider": "prime",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                result = opencode.sync_opencode_config(
                    current_rows=[],
                    prune_providers=(ComputeProvider.MODAL,),
                )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(result.removed_provider_ids, [modal_missing.provider_id])
            self.assertNotIn(modal_missing.provider_id, payload["provider"])
            self.assertIn(prime_live.provider_id, payload["provider"])
            self.assertIn(prime_live.app_name, registry["entries"])
            self.assertTrue(
                any("Modal deployment list" in line for line in result.messages)
            )

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

            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
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

            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                result = opencode.sync_opencode_config()

            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(result.dropped_registry_app_names, [ghost.app_name])
            self.assertEqual(registry["entries"], {})

    def test_sync_reads_jsonc_and_writes_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
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

            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                opencode.sync_opencode_config(target=target)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            backup = config_path.with_suffix(".json.bak")
            self.assertTrue(backup.exists())
            self.assertIn("// existing config", backup.read_text(encoding="utf-8"))
            self.assertIn("custom", payload["provider"])
            self.assertIn(target.provider_id, payload["provider"])

    def test_sync_uses_existing_jsonc_config_when_cli_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonc_path = Path(tmp) / "opencode.jsonc"
            jsonc_path.write_text('{"provider":{"custom":{"name":"Keep"}}}\n', encoding="utf-8")
            target = _connection()

            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value=None
            ):
                result = opencode.sync_opencode_config(target=target)

            payload = json.loads(jsonc_path.read_text(encoding="utf-8"))
            self.assertTrue(result.detected)
            self.assertEqual(result.config_path, jsonc_path)
            self.assertIn("custom", payload["provider"])
            self.assertIn(target.provider_id, payload["provider"])

    def test_bootstrap_registry_accepts_legacy_and_flat_launchpad_provider_names(self) -> None:
        payload = {
            "provider": {
                "llm-launchpad-vllm-qwen3": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "llm-launchpad",
                    "options": {"baseURL": "https://example.com/v1"},
                    "models": {"Qwen3-4B": {"name": "Qwen3-4B"}},
                },
                "llm-launchpad-llamacpp-phi": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "llm-launchpad: phi",
                    "options": {"baseURL": "https://phi.example.com/v1"},
                    "models": {"Phi-3": {"name": "Phi-3"}},
                },
            }
        }

        adopted = opencode._bootstrap_registry_from_config(payload)

        self.assertEqual(adopted["vllm-qwen3"]["instance_name"], "vllm-qwen3")
        self.assertEqual(adopted["vllm-qwen3"]["provider"], "modal")
        self.assertEqual(adopted["llamacpp-phi"]["instance_name"], "phi")
        self.assertEqual(adopted["llamacpp-phi"]["provider"], "modal")

    def test_sync_migrates_legacy_launchpad_provider_names_to_flat_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "opencode.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": {
                            "llm-launchpad-vllm-qwen3": {
                                "npm": "@ai-sdk/openai-compatible",
                                "name": "llm-launchpad: qwen3",
                                "options": {"baseURL": "https://example.com/v1"},
                                "models": {"Qwen3-4B": {"name": "Qwen3-4B"}},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self._patched_paths(tmp)[0], self._patched_paths(tmp)[1], self._patched_paths(tmp)[2], patch(
                "llm_launchpad.core.opencode.shutil.which", return_value="/usr/bin/opencode"
            ):
                result = opencode.sync_opencode_config()

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertTrue(result.changed)
            self.assertEqual(payload["provider"]["llm-launchpad-vllm-qwen3"]["name"], "llm-launchpad")



class AtomicConfigWriteTests(unittest.TestCase):
    """A partially written config stops OpenCode from starting at all."""

    def test_a_shorter_config_leaves_no_trailing_bytes(self) -> None:
        from llm_launchpad.core.opencode import _write_opencode_config

        with TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            path.write_text(
                json.dumps({"provider": {f"p{i}": {} for i in range(40)}}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            long_size = path.stat().st_size

            _write_opencode_config(path, {"provider": {}})

            written = path.read_text(encoding="utf-8")
            self.assertLess(len(written), long_size)
            # The failure this guards against parses as valid JSON followed by
            # the tail of the previous, longer file.
            self.assertEqual(json.loads(written), {"provider": {}})

    def test_no_temporary_file_is_left_behind(self) -> None:
        from llm_launchpad.core.opencode import _write_opencode_config

        with TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            _write_opencode_config(path, {"provider": {}})

            self.assertEqual(
                sorted(child.name for child in Path(directory).iterdir()),
                ["opencode.json"],
            )

    def test_the_replacement_is_a_rename_not_a_truncating_write(self) -> None:
        from llm_launchpad.core import opencode as opencode_module

        with TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            path.write_text('{"provider": {"old": {}}}\n', encoding="utf-8")
            observed: list[Path] = []
            original = Path.replace

            def record(self: Path, target: Any) -> Any:
                observed.append(Path(target))
                return original(self, target)

            with patch.object(Path, "replace", record):
                opencode_module._write_opencode_config(path, {"provider": {}})

            self.assertIn(path, observed)


if __name__ == "__main__":
    unittest.main()

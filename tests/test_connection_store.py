from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_launchpad.core import connection_store
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo, ReasoningCapabilities


class ConnectionStoreReasoningMigrationTests(unittest.TestCase):
    def test_legacy_llamacpp_row_is_repaired_from_revision_pinned_storage(self) -> None:
        revision = "c8b5954a88c2775c546b92593eda40ea041d3176"
        capabilities = ReasoningCapabilities(
            profile_id="hf-test",
            canonical_model_id="unsloth/Qwen3.8-Flash-Next-GGUF",
            model_revision=revision,
            efforts=("low", "medium", "high", "xhigh"),
            default_effort="xhigh",
            source_repo="unsloth/Qwen3.8-Flash-Next-GGUF",
            source_revision=revision,
            source_path="gguf.chat_template",
            request_option_path="chat_template_kwargs.reasoning_effort",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            connections_path = root / "connections.json"
            storage_path = root / "storage.json"
            connections_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            "llamacpp-qwen-flash": {
                                "backend": "llamacpp",
                                "base_url": "https://example.com/v1",
                                "model_id": "Qwen3.8-Flash-Next-GGUF-UD-Q2_K_XL",
                                "display_name": (
                                    "Qwen3.8-Flash-Next-GGUF (UD-Q2_K_XL)"
                                ),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            storage_path.write_text(
                json.dumps(
                    {
                        "snapshot": {
                            "llamacpp_models": [
                                {
                                    "backend": "llamacpp",
                                    "model_id": (
                                        "unsloth/Qwen3.8-Flash-Next-GGUF"
                                    ),
                                    "revision": revision,
                                    "quant": "Q2_K_XL",
                                }
                            ],
                            "vllm_models": [],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                connection_store,
                "discover_reasoning_capabilities",
                return_value=capabilities,
            ) as discover:
                rows = connection_store.rows_from_connection_cache(
                    connections_path,
                    storage_path,
                )

            discover.assert_called_once_with(
                BackendType.LLAMACPP,
                "unsloth/Qwen3.8-Flash-Next-GGUF",
                revision,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0].repo_id,
                "unsloth/Qwen3.8-Flash-Next-GGUF",
            )
            self.assertEqual(rows[0].reasoning, capabilities)
            persisted = json.loads(connections_path.read_text(encoding="utf-8"))
            repaired = persisted["entries"]["llamacpp-qwen-flash"]
            self.assertEqual(
                repaired["reasoning"]["efforts"],
                ["low", "medium", "high", "xhigh"],
            )
            self.assertEqual(repaired["reasoning_checked_revision"], revision)

    def test_ambiguous_storage_identity_is_not_guessed(self) -> None:
        entry = {
            "model_id": "shared-GGUF-Q4_K_M",
            "display_name": "shared-GGUF (Q4_K_M)",
        }
        storage_rows = [
            {
                "backend": "llamacpp",
                "model_id": "first/shared-GGUF",
                "revision": "a" * 40,
            },
            {
                "backend": "llamacpp",
                "model_id": "second/shared-GGUF",
                "revision": "b" * 40,
            },
        ]

        selected = connection_store._resolve_storage_model(
            entry,
            BackendType.LLAMACPP,
            storage_rows,
        )

        self.assertIsNone(selected)

    def test_merge_hydrates_repaired_reasoning_into_live_row(self) -> None:
        revision = "d" * 40
        capabilities = ReasoningCapabilities(
            profile_id="hf-test",
            canonical_model_id="acme/Future-Reasoner",
            model_revision=revision,
            efforts=("brief", "deep"),
            default_effort="deep",
            source_repo="acme/Future-Reasoner",
            source_revision=revision,
            source_path="chat_template.jinja",
            request_option_path="chat_template_kwargs.reasoning_effort",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            connections_path = root / "connections.json"
            storage_path = root / "storage.json"
            connections_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            "vllm-future": {
                                "backend": "vllm",
                                "base_url": "https://example.com/v1",
                                "model_id": "Future-Reasoner",
                                "display_name": "Future-Reasoner",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            storage_path.write_text(
                json.dumps(
                    {
                        "snapshot": {
                            "llamacpp_models": [],
                            "vllm_models": [
                                {
                                    "backend": "vllm",
                                    "model_id": "acme/Future-Reasoner",
                                    "revision": revision,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            row = EndpointInfo(
                name="vllm-future",
                backend=BackendType.VLLM,
                web_url="https://live.example.com",
            )

            with patch.object(
                connection_store,
                "discover_reasoning_capabilities",
                return_value=capabilities,
            ):
                connection_store.merge_connections(
                    [row],
                    connections_path,
                    storage_path,
                )

            self.assertEqual(row.model_name, "acme/Future-Reasoner")
            self.assertEqual(row.reasoning, capabilities)


if __name__ == "__main__":
    unittest.main()

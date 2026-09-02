from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm_launchpad.core import reasoning_profiles
from llm_launchpad.core.reasoning_profiles import (
    _ReasoningSource,
    _RepositoryInspection,
    clear_reasoning_discovery_cache,
    discover_reasoning_capabilities,
    discover_selected_model_reasoning,
    reasoning_capabilities_from_dict,
    reasoning_capabilities_to_dict,
    reasoning_request_options,
    reasoning_variants,
)
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import DeploymentConfig


MODEL_SHA = "a" * 40
BASE_SHA = "b" * 40


def _inspection(
    repo_id: str,
    revision: str,
    path: str,
    text: str,
    *,
    quantized_base_models: tuple[str, ...] = (),
) -> _RepositoryInspection:
    sources = (
        _ReasoningSource(
            repo_id=repo_id,
            revision=revision,
            path=path,
            text=text,
        ),
    )
    return _RepositoryInspection(
        repo_id=repo_id,
        revision=revision,
        sources=sources,
        quantized_base_models=quantized_base_models,
    )


def _multi_source_inspection(
    repo_id: str,
    revision: str,
    sources: tuple[tuple[str, str], ...],
) -> _RepositoryInspection:
    return _RepositoryInspection(
        repo_id=repo_id,
        revision=revision,
        sources=tuple(
            _ReasoningSource(repo_id, revision, path, text) for path, text in sources
        ),
    )


class ReasoningProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_reasoning_discovery_cache()

    def tearDown(self) -> None:
        clear_reasoning_discovery_cache()

    def test_discovers_unlisted_selected_repo_from_pinned_template(self) -> None:
        selected = _inspection(
            "acme/Future-Reasoner",
            MODEL_SHA,
            "chat_template.jinja",
            """
            {% set selected_reasoning_effort = reasoning_effort|default('deep') %}
            {% if selected_reasoning_effort not in ('brief', 'balanced', 'deep') %}
              {{ raise_exception('unsupported') }}
            {% endif %}
            {% if enable_thinking %}{% endif %}
            """,
        )

        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ) as inspect_repo:
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                "acme/Future-Reasoner",
                "release-1",
            )

        inspect_repo.assert_called_once_with("acme/Future-Reasoner", "release-1")
        self.assertIsNotNone(capabilities)
        assert capabilities is not None
        self.assertEqual(capabilities.canonical_model_id, "acme/Future-Reasoner")
        self.assertEqual(capabilities.model_revision, MODEL_SHA)
        self.assertEqual(capabilities.efforts, ("brief", "balanced", "deep"))
        self.assertEqual(capabilities.default_effort, "deep")
        self.assertEqual(capabilities.source_path, "chat_template.jinja")
        self.assertTrue(capabilities.enable_thinking)

    def test_parses_ternary_default_without_model_specific_mapping(self) -> None:
        selected = _inspection(
            "new-org/New-Model",
            MODEL_SHA,
            "chat_template.jinja",
            """
            {% set effective_reasoning_effort = 'high'
                if reasoning_effort is defined and reasoning_effort == 'high'
                else 'max' %}
            """,
        )

        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ):
            capabilities = discover_reasoning_capabilities(
                BackendType.LLAMACPP,
                selected.repo_id,
            )

        self.assertIsNotNone(capabilities)
        assert capabilities is not None
        self.assertEqual(capabilities.efforts, ("high", "max"))
        self.assertEqual(capabilities.default_effort, "max")

    def test_quantized_repo_uses_its_declared_base_model_for_complete_evidence(self) -> None:
        selected = _inspection(
            "quantizer/New-Model-GGUF",
            MODEL_SHA,
            "gguf.chat_template",
            """
            {% if enable_thinking %}
              {% if reasoning_effort == 'high' %}high
              {% elif reasoning_effort == 'max' %}max{% endif %}
              {{ message['reasoning_content'] }}
            {% endif %}
            """,
            quantized_base_models=("authors/New-Model",),
        )
        base = _inspection(
            "authors/New-Model",
            BASE_SHA,
            "encoding/reasoning.py",
            """
REASONING_EFFORT_PROMPTS: dict[str, str] = {
    "low": "",
    "high": "careful",
    "max": "exhaustive",
}
DEFAULT_REASONING_EFFORT = "low"
            """,
        )

        def inspect(repo_id: str, revision: str | None) -> _RepositoryInspection:
            self.assertIsNone(revision)
            return selected if repo_id == selected.repo_id else base

        with patch.object(reasoning_profiles, "_inspect_hf_repository", side_effect=inspect):
            capabilities = discover_reasoning_capabilities(
                BackendType.LLAMACPP,
                selected.repo_id,
            )

        self.assertIsNotNone(capabilities)
        assert capabilities is not None
        self.assertEqual(capabilities.canonical_model_id, selected.repo_id)
        self.assertEqual(capabilities.model_revision, MODEL_SHA)
        self.assertEqual(capabilities.efforts, ("low", "high", "max"))
        self.assertEqual(capabilities.default_effort, "low")
        self.assertEqual(capabilities.source_repo, base.repo_id)
        self.assertEqual(capabilities.source_revision, BASE_SHA)
        self.assertTrue(capabilities.enable_thinking)
        self.assertEqual(capabilities.interleaved_field, "reasoning_content")

    def test_unverified_repo_does_not_advertise_reasoning(self) -> None:
        selected = _inspection(
            "acme/Ordinary-Model",
            MODEL_SHA,
            "config.json",
            '{"architectures": ["OrdinaryModel"]}',
        )
        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ):
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )

        self.assertIsNone(capabilities)

    def test_partial_template_uses_readme_documented_default(self) -> None:
        # The chat template lists efforts but no default, while the README
        # documents the default. Discovery must still succeed instead of
        # discarding the README evidence because a code source is present.
        selected = _multi_source_inspection(
            "acme/Partial-Template-Reasoner",
            MODEL_SHA,
            (
                (
                    "chat_template.jinja",
                    "{% if reasoning_effort not in ('low', 'high', 'xhigh') %}"
                    "{% endif %}",
                ),
                (
                    "README.md",
                    "Supported reasoning_effort values are one of 'low', "
                    "'high', or 'xhigh'. 'xhigh' is the default reasoning_effort.",
                ),
            ),
        )
        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ):
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )

        self.assertIsNotNone(capabilities)
        assert capabilities is not None
        self.assertEqual(capabilities.efforts, ("low", "high", "xhigh"))
        self.assertEqual(capabilities.default_effort, "xhigh")

    def test_complete_code_merge_is_not_widened_by_readme(self) -> None:
        # Chat template + tokenizer config form a complete, code-only profile.
        # A more permissive README must not widen the advertised efforts.
        selected = _multi_source_inspection(
            "acme/Code-Complete-Reasoner",
            MODEL_SHA,
            (
                (
                    "chat_template.jinja",
                    "{% if reasoning_effort not in ('low', 'medium') %}{% endif %}",
                ),
                (
                    "tokenizer_config.json",
                    json.dumps({"reasoning_effort": {"default": "medium"}}),
                ),
                (
                    "README.md",
                    "Supported reasoning_effort values are one of 'low', "
                    "'medium', or 'ultra'. 'ultra' is the default.",
                ),
            ),
        )
        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ):
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )

        self.assertIsNotNone(capabilities)
        assert capabilities is not None
        self.assertEqual(capabilities.efforts, ("low", "medium"))
        self.assertEqual(capabilities.default_effort, "medium")

    def test_discovers_structured_reasoning_metadata_from_json(self) -> None:
        selected = _inspection(
            "acme/Structured-Reasoner",
            MODEL_SHA,
            "config.json",
            """
            {
              "reasoning_effort": {
                "supported_values": ["quick", "thorough"],
                "default": "quick"
              }
            }
            """,
        )
        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ):
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )

        self.assertIsNotNone(capabilities)
        assert capabilities is not None
        self.assertEqual(capabilities.efforts, ("quick", "thorough"))
        self.assertEqual(capabilities.default_effort, "quick")

    def test_selection_discovery_cache_reuses_the_pinned_result(self) -> None:
        selected = _inspection(
            "acme/Cached-Reasoner",
            MODEL_SHA,
            "chat_template.jinja",
            """
            {% set reasoning_effort = reasoning_effort|default('high') %}
            {% if reasoning_effort not in ('low', 'high') %}{% endif %}
            """,
        )
        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ) as inspect_repo:
            first = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )
            second = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )

        self.assertIs(first, second)
        inspect_repo.assert_called_once_with(selected.repo_id, None)

    def test_local_model_path_is_not_sent_to_hugging_face(self) -> None:
        with patch.object(reasoning_profiles, "_inspect_hf_repository") as inspect_repo:
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                "/models/local-checkpoint",
            )

        self.assertIsNone(capabilities)
        inspect_repo.assert_not_called()

    def test_selected_config_uses_backend_specific_repo_and_revision(self) -> None:
        vllm = DeploymentConfig(
            backend=BackendType.VLLM,
            model_name="acme/Vllm-Model",
            model_revision="vllm-revision",
        )
        llamacpp = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="acme/Llama-GGUF",
            revision="gguf-revision",
        )
        with patch.object(
            reasoning_profiles,
            "discover_reasoning_capabilities",
            return_value=None,
        ) as discover:
            discover_selected_model_reasoning(vllm)
            discover_selected_model_reasoning(llamacpp)

        self.assertEqual(
            discover.call_args_list[0].args,
            (BackendType.VLLM, "acme/Vllm-Model", "vllm-revision"),
        )
        self.assertEqual(
            discover.call_args_list[1].args,
            (BackendType.LLAMACPP, "acme/Llama-GGUF", "gguf-revision"),
        )

    def test_variants_use_each_discovered_effort(self) -> None:
        selected = _inspection(
            "acme/Future-Reasoner",
            MODEL_SHA,
            "chat_template.jinja",
            """
            {% set resolved_reasoning_effort = reasoning_effort|default('deep') %}
            {% if resolved_reasoning_effort not in ('brief', 'deep') %}{% endif %}
            """,
        )
        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ):
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )
        assert capabilities is not None

        variants = reasoning_variants(capabilities)

        self.assertEqual(list(variants), ["default", "brief", "deep"])
        self.assertEqual(
            variants["default"],
            {"chat_template_kwargs": {"reasoning_effort": "deep"}},
        )
        with self.assertRaisesRegex(ValueError, "Unsupported reasoning effort"):
            reasoning_request_options(capabilities, "invented")

    def test_capability_snapshot_round_trips_and_requires_model_revision(self) -> None:
        selected = _inspection(
            "acme/Future-Reasoner",
            MODEL_SHA,
            "chat_template.jinja",
            """
            {% set reasoning_effort = reasoning_effort|default('high') %}
            {% if reasoning_effort not in ('low', 'high') %}{% endif %}
            """,
        )
        with patch.object(
            reasoning_profiles,
            "_inspect_hf_repository",
            return_value=selected,
        ):
            capabilities = discover_reasoning_capabilities(
                BackendType.VLLM,
                selected.repo_id,
            )
        assert capabilities is not None
        payload = reasoning_capabilities_to_dict(capabilities)

        self.assertEqual(reasoning_capabilities_from_dict(payload), capabilities)
        assert payload is not None
        payload.pop("model_revision")
        self.assertIsNone(reasoning_capabilities_from_dict(payload))

    def test_reads_structured_quantized_base_model_metadata(self) -> None:
        info = SimpleNamespace(
            base_models={
                "relation": "quantized",
                "models": [{"id": "authors/Base-Model"}],
            },
            card_data=None,
        )

        self.assertEqual(
            reasoning_profiles._quantized_base_models(info),
            ("authors/Base-Model",),
        )


if __name__ == "__main__":
    unittest.main()

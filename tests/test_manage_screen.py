from __future__ import annotations

import unittest

from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.screens.manage import (
    _available_actions,
    _endpoint_compact_label,
    _endpoint_key,
    _is_stoppable_state,
    _state_label,
)


class ManageScreenHelpersTests(unittest.TestCase):
    def test_state_labels_hide_provider_specific_lifecycle_terms(self) -> None:
        self.assertEqual(_state_label("active"), "Running")
        self.assertEqual(_state_label("deployed"), "Ready")
        self.assertEqual(_state_label("ephemeral"), "Temporary")
        self.assertEqual(_state_label("some_new_state"), "Some New State")

    def test_compact_endpoint_label_explains_duplicate_names(self) -> None:
        row = EndpointInfo(
            instance_name="logbeauty",
            state="active",
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
        )

        self.assertEqual(_endpoint_compact_label(row), "logbeauty [Prime/llama.cpp]")

    def test_is_stoppable_state_includes_active_in_progress_states(self) -> None:
        self.assertTrue(_is_stoppable_state("active"))
        self.assertTrue(_is_stoppable_state("deployed"))
        self.assertTrue(_is_stoppable_state("queued"))
        self.assertTrue(_is_stoppable_state("running"))
        self.assertFalse(_is_stoppable_state("ephemeral"))
        self.assertFalse(_is_stoppable_state("stopped"))
        self.assertFalse(_is_stoppable_state("unknown"))

    def test_endpoint_key_includes_provider_and_exact_resource_id(self) -> None:
        row = EndpointInfo(
            name="vllm-qwen",
            app_id="ap-1",
            state="deployed",
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
        )

        self.assertEqual(_endpoint_key(row), "modal:id:ap-1")

    def test_available_actions_are_state_specific(self) -> None:
        running = EndpointInfo(backend=BackendType.VLLM, state="running")
        failed = EndpointInfo(backend=BackendType.VLLM, state="failed")
        starting = EndpointInfo(backend=BackendType.VLLM, state="starting")

        self.assertEqual(
            _available_actions(running),
            frozenset({"status", "logs", "benchmark", "stop"}),
        )
        self.assertEqual(_available_actions(failed), frozenset({"logs"}))
        self.assertEqual(_available_actions(starting), frozenset({"logs", "stop"}))


if __name__ == "__main__":
    unittest.main()

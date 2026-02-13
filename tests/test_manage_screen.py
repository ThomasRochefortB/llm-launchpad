from __future__ import annotations

import unittest
from types import SimpleNamespace

from llm_launchpad.tui.screens.manage import (
    _build_backend_app_options,
    _is_stoppable_state,
)


class ManageScreenHelpersTests(unittest.TestCase):
    def test_is_stoppable_state_only_allows_running_or_deployed(self) -> None:
        self.assertTrue(_is_stoppable_state("deployed"))
        self.assertTrue(_is_stoppable_state("running"))
        self.assertFalse(_is_stoppable_state("stopped"))
        self.assertFalse(_is_stoppable_state("unknown"))

    def test_build_backend_app_options_maps_to_backend_and_app_name(self) -> None:
        vllm_backend = SimpleNamespace(value="vllm")
        llamacpp_backend = SimpleNamespace(value="llamacpp")
        instances = [
            SimpleNamespace(
                name="vllm-qwen",
                app_id="ap-1",
                state="deployed",
                backend=vllm_backend,
            ),
            SimpleNamespace(
                name="llamacpp-phi",
                app_id="ap-2",
                state="running",
                backend=llamacpp_backend,
            ),
        ]

        options, option_to_target = _build_backend_app_options(instances)
        option_ids = [str(option.id) for option in options]
        self.assertEqual(option_ids, ["app-id:ap-1", "app-id:ap-2"])
        self.assertEqual(option_to_target["app-id:ap-1"], (vllm_backend, "vllm-qwen"))
        self.assertEqual(option_to_target["app-id:ap-2"], (llamacpp_backend, "llamacpp-phi"))


if __name__ == "__main__":
    unittest.main()

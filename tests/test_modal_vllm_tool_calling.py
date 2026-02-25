from __future__ import annotations

import unittest

from llm_launchpad.backends import modal_vllm_app


class ModalVllmToolCallingTests(unittest.TestCase):
    def test_tool_call_flags_do_not_infer_parser_from_model_name(self) -> None:
        flags, parser, enabled = modal_vllm_app.tool_call_flags()
        self.assertEqual(flags, [])
        self.assertIsNone(parser)
        self.assertFalse(enabled)

    def test_tool_call_flags_enable_auto_by_default_when_parser_explicit(self) -> None:
        flags, parser, enabled = modal_vllm_app.tool_call_flags(
            tool_call_parser="hermes",
        )
        self.assertEqual(parser, "hermes")
        self.assertTrue(enabled)
        self.assertEqual(flags, ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"])

    def test_tool_call_flags_can_disable_auto_tool_choice(self) -> None:
        flags, parser, enabled = modal_vllm_app.tool_call_flags(
            tool_call_parser="qwen3_xml",
            enable_auto_tool_choice=False,
        )
        self.assertEqual(parser, "qwen3_xml")
        self.assertFalse(enabled)
        self.assertEqual(flags, ["--tool-call-parser", "qwen3_xml"])

    def test_tool_call_flags_warnable_state_when_auto_enabled_but_no_parser(self) -> None:
        flags, parser, enabled = modal_vllm_app.tool_call_flags(enable_auto_tool_choice=True)
        self.assertEqual(flags, [])
        self.assertIsNone(parser)
        self.assertTrue(enabled)


if __name__ == "__main__":
    unittest.main()

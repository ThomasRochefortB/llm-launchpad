from __future__ import annotations

import unittest

from unittest.mock import MagicMock, patch

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


class ModalVllmServeCommandTests(unittest.TestCase):
    def _serve_command(self, **env: str) -> list[str]:
        from llm_launchpad.backends import modal_vllm_app

        captured: list[list[str]] = []

        def _record(cmd, *args, **kwargs):
            captured.append(list(cmd))
            return MagicMock()

        with (
            patch.dict(modal_vllm_app.os.environ, env, clear=False),
            patch("subprocess.Popen", side_effect=_record),
        ):
            modal_vllm_app.serve.local()
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_serve_caps_the_context_when_max_model_len_is_set(self) -> None:
        cmd = self._serve_command(MAX_MODEL_LEN="32768")
        self.assertIn("--max-model-len", cmd)
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "32768")

    def test_serve_states_the_concurrency_it_was_given(self) -> None:
        cmd = self._serve_command(MAX_NUM_SEQS="256")
        self.assertIn("--max-num-seqs", cmd)
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "256")

    def test_serve_leaves_the_context_to_vllm_by_default(self) -> None:
        cmd = self._serve_command(MAX_MODEL_LEN="0")
        self.assertNotIn("--max-model-len", cmd)


if __name__ == "__main__":
    unittest.main()

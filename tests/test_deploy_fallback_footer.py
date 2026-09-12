from __future__ import annotations

import unittest
from dataclasses import replace

from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig
from llm_launchpad.tui.app import _defers_completion_footer


def _config(**changes: object) -> DeploymentConfig:
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP, provider=ComputeProvider.VAST,
        app_name="llp-test", repo_id="acme/model-GGUF", quant="Q4_K_M",
        gpu_type="RTX 4090", gpu_count=1, do_deploy=True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class CompletionFooterTests(unittest.TestCase):
    """The footer must not offer to return while the session carries on."""

    def test_a_failure_a_fallback_will_answer_defers_the_footer(self) -> None:
        event = OperationCompleteEvent(
            operation=OperationType.DEPLOY, success=False,
            detail="The selected Vast offer is no longer available with these requirements.",
        )
        config = _config(fallback_configs=(_config(provider=ComputeProvider.MODAL),))
        self.assertTrue(_defers_completion_footer(event, config=config, will_run_warmup=False))

    def test_a_failure_with_nothing_left_to_try_shows_the_footer(self) -> None:
        event = OperationCompleteEvent(operation=OperationType.DEPLOY, success=False)
        self.assertFalse(
            _defers_completion_footer(event, config=_config(), will_run_warmup=False)
        )

    def test_a_success_defers_only_while_warmup_follows(self) -> None:
        event = OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)
        self.assertTrue(
            _defers_completion_footer(event, config=_config(), will_run_warmup=True)
        )
        self.assertFalse(
            _defers_completion_footer(event, config=_config(), will_run_warmup=False)
        )

    def test_other_operations_always_show_their_footer(self) -> None:
        config = _config(fallback_configs=(_config(),))
        for operation in (OperationType.WARMUP, OperationType.STOP, OperationType.STATUS):
            with self.subTest(operation=operation):
                event = OperationCompleteEvent(operation=operation, success=False)
                self.assertFalse(
                    _defers_completion_footer(event, config=config, will_run_warmup=True)
                )


if __name__ == "__main__":
    unittest.main()

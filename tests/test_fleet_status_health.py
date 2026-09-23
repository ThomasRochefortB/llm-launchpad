"""A stored health check reaches the fleet line of a fresh Modal row."""

from __future__ import annotations

import unittest

from textual.content import Content

from llm_launchpad.core.runtime_health import record_explicit_health, reset
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.fleet_status import deployment_and_health_line


def _modal_row() -> EndpointInfo:
    return EndpointInfo(
        name="vllm-demo",
        app_id="ap-demo",
        state="deployed",
        backend=BackendType.VLLM,
        provider=ComputeProvider.MODAL,
        web_url="https://demo.modal.run",
    )


class StoredHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        reset()
        self.addCleanup(reset)

    def test_a_status_check_is_remembered_by_the_next_refresh(self) -> None:
        """The lookup imported `...core.runtime_health`, one level above the
        package, and a bare `except` swallowed the ImportError -- so every
        running Modal endpoint read "Health: Not checked", even straight
        after a status check had recorded it healthy."""
        record_explicit_health(_modal_row(), "healthy")

        # A fresh row, as the next fleet refresh would build it.
        line = Content.from_markup(deployment_and_health_line(_modal_row())).plain

        self.assertIn("Health: Healthy", line)

    def test_an_unchecked_row_still_says_so(self) -> None:
        line = Content.from_markup(deployment_and_health_line(_modal_row())).plain
        self.assertIn("Health: Not checked", line)


if __name__ == "__main__":
    unittest.main()

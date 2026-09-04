"""Endpoint URL precedence, resolved once instead of per caller."""

from __future__ import annotations

import unittest

from llm_launchpad.core.deploy_events import (
    EndpointUrlResolver,
    collect_error,
    failure_detail,
    is_deploy_completion,
)
from llm_launchpad.protocol.enums import OperationType
from llm_launchpad.protocol.events import (
    EndpointAvailableEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
)
from llm_launchpad.protocol.models import EndpointInfo

DEPLOYED = "https://user--app-serve.modal.run"
DEV = "https://user--app-serve-dev.modal.run"
INCIDENTAL = "https://user--other.modal.run"


def _resolve(*events: object) -> str:
    resolver = EndpointUrlResolver()
    for event in events:
        resolver.observe(event)
    return resolver.url


class EndpointUrlResolverTests(unittest.TestCase):
    def test_nothing_seen_yields_no_url(self) -> None:
        self.assertEqual(_resolve(), "")
        self.assertEqual(_resolve(LogEvent(line="starting up")), "")

    def test_an_announced_function_beats_an_incidental_log_url(self) -> None:
        self.assertEqual(
            _resolve(
                LogEvent(line=f"see {INCIDENTAL}"),
                LogEvent(line=f"Created web function => {DEPLOYED}"),
            ),
            DEPLOYED,
        )

    def test_the_deployed_url_beats_the_ephemeral_dev_one(self) -> None:
        # Certifying against -dev means probing an endpoint that is not the one
        # being published; readiness never arrives.
        self.assertEqual(
            _resolve(
                LogEvent(line=f"Created web function => {DEV}"),
                LogEvent(line=f"Created web function => {DEPLOYED}"),
            ),
            DEPLOYED,
        )
        self.assertEqual(
            _resolve(
                LogEvent(line=f"Created web function => {DEPLOYED}"),
                LogEvent(line=f"Created web function => {DEV}"),
            ),
            DEPLOYED,
        )

    def test_an_event_carried_url_outranks_anything_scraped(self) -> None:
        self.assertEqual(
            _resolve(
                EndpointAvailableEvent(endpoint=EndpointInfo(web_url=DEPLOYED)),
                LogEvent(line=f"Created web function => {DEV}"),
            ),
            DEPLOYED,
        )

    def test_a_completion_payload_supplies_the_url(self) -> None:
        self.assertEqual(
            _resolve(
                OperationCompleteEvent(
                    operation=OperationType.DEPLOY,
                    success=True,
                    data=EndpointInfo(web_url=DEPLOYED),
                )
            ),
            DEPLOYED,
        )

    def test_a_dev_url_is_still_better_than_none(self) -> None:
        self.assertEqual(_resolve(LogEvent(line=f"Created web function => {DEV}")), DEV)


class FailureDetailTests(unittest.TestCase):
    def test_a_detail_is_used_when_present(self) -> None:
        event = OperationCompleteEvent(
            operation=OperationType.WARMUP, success=False, detail="context mismatch"
        )
        self.assertEqual(failure_detail(event, ["earlier"]), "context mismatch")

    def test_preceding_errors_explain_a_detail_less_failure(self) -> None:
        # Several failure paths complete without a detail; the reason arrived
        # earlier on an ErrorEvent.
        event = OperationCompleteEvent(operation=OperationType.WARMUP, success=False)
        self.assertEqual(
            failure_detail(event, ["Invalid endpoint URL: bad scheme"]),
            "Invalid endpoint URL: bad scheme",
        )

    def test_a_silent_failure_says_so_rather_than_nothing(self) -> None:
        event = OperationCompleteEvent(operation=OperationType.WARMUP, success=False)
        self.assertEqual(failure_detail(event, []), "no reason reported")


class EventHelperTests(unittest.TestCase):
    def test_error_events_are_collected(self) -> None:
        errors: list[str] = []
        self.assertTrue(collect_error(ErrorEvent(message="boom"), errors))
        self.assertFalse(collect_error(LogEvent(line="fine"), errors))
        self.assertEqual(errors, ["boom"])

    def test_only_deploy_completions_count_as_deploy_completions(self) -> None:
        self.assertTrue(
            is_deploy_completion(
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)
            )
        )
        self.assertFalse(
            is_deploy_completion(
                OperationCompleteEvent(operation=OperationType.WARMUP, success=True)
            )
        )
        self.assertFalse(is_deploy_completion(LogEvent(line="x")))


if __name__ == "__main__":
    unittest.main()

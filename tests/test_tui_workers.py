from __future__ import annotations

import unittest

from llm_launchpad.protocol.enums import DeploymentState, OperationType
from llm_launchpad.protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent, StateChangeEvent
from llm_launchpad.protocol.models import StorageSnapshot
from llm_launchpad.tui.workers import (
    LogMessage,
    OperationDone,
    OperationError,
    StorageFailed,
    StorageLoaded,
    StateChanged,
    _dispatch_event,
)


class _Poster:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def post_message(self, message: object) -> None:
        self.messages.append(message)


class TuiWorkersDispatchTests(unittest.TestCase):
    def test_dispatch_event_maps_protocol_events(self) -> None:
        poster = _Poster()
        _dispatch_event(
            poster,
            LogEvent(line="hello", stream="stderr", is_milestone=True),
        )
        _dispatch_event(
            poster,
            StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail="running",
            ),
        )
        _dispatch_event(
            poster,
            OperationCompleteEvent(
                operation=OperationType.DEPLOY,
                success=False,
                exit_code=7,
                detail="failed",
                data={"k": "v"},
            ),
        )
        _dispatch_event(
            poster,
            ErrorEvent(
                message="boom",
                operation=OperationType.DEPLOY,
                recoverable=False,
            ),
        )

        self.assertIsInstance(poster.messages[0], LogMessage)
        self.assertEqual(poster.messages[0].line, "hello")
        self.assertEqual(poster.messages[0].stream, "stderr")
        self.assertTrue(poster.messages[0].is_milestone)

        self.assertIsInstance(poster.messages[1], StateChanged)
        self.assertEqual(poster.messages[1].state, DeploymentState.DEPLOYING)
        self.assertEqual(poster.messages[1].operation, OperationType.DEPLOY)

        self.assertIsInstance(poster.messages[2], OperationDone)
        self.assertFalse(poster.messages[2].success)
        self.assertEqual(poster.messages[2].exit_code, 7)
        self.assertEqual(poster.messages[2].detail, "failed")
        self.assertEqual(poster.messages[2].data, {"k": "v"})

        self.assertIsInstance(poster.messages[3], OperationError)
        self.assertEqual(poster.messages[3].message, "boom")
        self.assertFalse(poster.messages[3].recoverable)

    def test_dispatch_event_is_noop_without_post_message(self) -> None:
        class _NoPoster:
            pass

        _dispatch_event(_NoPoster(), LogEvent(line="ignored"))

    def test_storage_messages_store_payload(self) -> None:
        snapshot = StorageSnapshot(llamacpp_models=[], vllm_models=[])
        loaded = StorageLoaded(snapshot=snapshot)
        failed = StorageFailed(error="boom")
        self.assertIs(loaded.snapshot, snapshot)
        self.assertEqual(failed.error, "boom")


if __name__ == "__main__":
    unittest.main()

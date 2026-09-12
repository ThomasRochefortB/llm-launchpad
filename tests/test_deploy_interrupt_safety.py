"""Quitting mid-deploy must not leave a provider resource billing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.deploy_journal import (
    InFlightDeployment,
    load_in_flight,
    record_in_flight,
)
from llm_launchpad.protocol.enums import (
    BackendType,
    ComputeProvider,
    OperationType,
)
from llm_launchpad.protocol.events import OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig, EndpointInfo
from llm_launchpad.tui.app import AbandonedDeploymentsChecked, TuiApp


def _config() -> DeploymentConfig:
    config = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.MODAL,
    )
    config.app_name = "llamacpp-interrupted"
    config.instance_name = "interrupted"
    config.gpu_type = "RTX-PRO-6000"
    config.gpu_count = 1
    return config


class InFlightJournalTests(unittest.TestCase):
    def test_a_deploy_is_journalled_before_the_provider_is_contacted(self) -> None:
        app = TuiApp()

        app._begin_in_flight(_config())

        recorded = load_in_flight()
        self.assertEqual([row.app_name for row in recorded], ["llamacpp-interrupted"])
        self.assertEqual(app._in_flight_deploy.app_name, "llamacpp-interrupted")
        self.assertGreater(app._in_flight_deploy.started_at_epoch, 0.0)

    def test_a_finished_deploy_clears_its_journal_entry(self) -> None:
        app = TuiApp()
        config = _config()

        app._begin_in_flight(config)
        app._finish_in_flight(config)

        self.assertEqual(load_in_flight(), ())
        self.assertIsNone(app._in_flight_deploy)


class QuitStopsInFlightDeploymentTests(unittest.IsolatedAsyncioTestCase):
    async def test_quitting_stops_the_deployment_the_provider_already_created(
        self,
    ) -> None:
        app = TuiApp()
        app._begin_in_flight(_config())
        stopped: list[tuple[str, ComputeProvider]] = []

        def fake_stop(backend, app_name=None, app_id=None, provider=ComputeProvider.MODAL):
            stopped.append((app_name, provider))
            yield OperationCompleteEvent(operation=OperationType.STOP, success=True)

        with patch.object(app._orchestrator, "stop_app", fake_stop), patch.object(
            app, "exit"
        ), patch("llm_launchpad.tui.app.ModalBackend.terminate_all"), patch.object(
            app, "notify"
        ):
            await app.action_quit()

        self.assertEqual(stopped, [("llamacpp-interrupted", ComputeProvider.MODAL)])
        # The journal is only cleared once the stop is confirmed.
        self.assertEqual(load_in_flight(), ())

    async def test_a_failed_stop_keeps_the_entry_for_the_next_session(self) -> None:
        app = TuiApp()
        app._begin_in_flight(_config())

        def failing_stop(backend, app_name=None, app_id=None, provider=ComputeProvider.MODAL):
            yield OperationCompleteEvent(
                operation=OperationType.STOP, success=False, detail="provider unreachable"
            )

        with patch.object(app._orchestrator, "stop_app", failing_stop), patch.object(
            app, "exit"
        ), patch("llm_launchpad.tui.app.ModalBackend.terminate_all"), patch.object(
            app, "notify"
        ):
            await app.action_quit()

        # Still recorded, so the next launch can warn about it.
        self.assertEqual(
            [row.app_name for row in load_in_flight()], ["llamacpp-interrupted"]
        )

    async def test_quitting_without_a_deploy_stops_nothing(self) -> None:
        app = TuiApp()
        calls: list[str] = []

        def fake_stop(*args, **kwargs):
            calls.append("stop")
            yield OperationCompleteEvent(operation=OperationType.STOP, success=True)

        with patch.object(app._orchestrator, "stop_app", fake_stop), patch.object(
            app, "exit"
        ), patch("llm_launchpad.tui.app.ModalBackend.terminate_all"):
            await app.action_quit()

        self.assertEqual(calls, [])


class AbandonedDeploymentRecoveryTests(unittest.TestCase):
    @staticmethod
    def _entry(app_name: str, **changes: object) -> InFlightDeployment:
        base = {
            "app_name": app_name,
            "provider": ComputeProvider.MODAL.value,
            "backend": BackendType.LLAMACPP.value,
            "price_per_hour_usd": 3.03,
            "started_at_epoch": 1.0,
        }
        base.update(changes)
        return InFlightDeployment(**base)  # type: ignore[arg-type]

    def _check(self, app: TuiApp, rows: list[EndpointInfo], enumerated: tuple) -> list:
        """Run the verification pass and return what it still reports."""
        posted: list = []
        with patch.object(app, "_visible_rows_and_prune_scope", return_value=(rows, enumerated)), patch.object(
            app, "post_message", lambda message: posted.append(message)
        ):
            app._run_check_abandoned_deployments()
        return posted

    def test_an_entry_the_provider_still_lists_is_reported(self) -> None:
        record_in_flight(self._entry("llamacpp-abandoned"))
        app = TuiApp()
        rows = [EndpointInfo(name="llamacpp-abandoned", provider=ComputeProvider.MODAL)]

        posted = self._check(app, rows, (ComputeProvider.MODAL,))

        self.assertEqual(len(posted), 1)
        self.assertEqual(len(load_in_flight()), 1)
        messages: list[str] = []
        with patch.object(app, "notify", lambda message, **kwargs: messages.append(str(message))):
            app.on_abandoned_deployments_checked(posted[0])
        self.assertEqual(len(messages), 1)
        self.assertIn("llamacpp-abandoned", messages[0])
        self.assertIn("may still be running", messages[0])
        # A months-old entry must not claim a five-figure bill.
        self.assertNotIn("$", messages[0])

    def test_an_entry_the_provider_says_is_gone_is_resolved_silently(self) -> None:
        """The journal records an interrupted deploy, not a live resource.

        Warning about one the provider positively reports as absent sends the
        user hunting for a rental that does not exist, on every launch.
        """
        record_in_flight(self._entry("llamacpp-finished"))
        app = TuiApp()

        posted = self._check(app, [], (ComputeProvider.MODAL,))

        self.assertEqual(posted, [])
        self.assertEqual(load_in_flight(), ())

    def test_a_provider_that_could_not_be_listed_keeps_its_entry(self) -> None:
        # Absence of evidence is not confirmation: an unreachable provider must
        # not resolve an entry.
        record_in_flight(self._entry("llamacpp-unknown"))
        app = TuiApp()

        posted = self._check(app, [], ())

        self.assertEqual(len(posted), 1)
        self.assertEqual(len(load_in_flight()), 1)

    def test_a_failed_listing_keeps_every_entry(self) -> None:
        record_in_flight(self._entry("llamacpp-unknown"))
        app = TuiApp()
        posted: list = []
        with patch.object(
            app, "_visible_rows_and_prune_scope", side_effect=RuntimeError("provider down")
        ), patch.object(app, "post_message", lambda message: posted.append(message)):
            app._run_check_abandoned_deployments()

        self.assertEqual(len(posted), 1)
        self.assertEqual(len(load_in_flight()), 1)

    def test_many_unresolved_entries_do_not_bury_the_screen(self) -> None:
        entries = tuple(self._entry(f"llamacpp-{index}") for index in range(5))
        app = TuiApp()
        messages: list[str] = []
        with patch.object(app, "notify", lambda message, **kwargs: messages.append(str(message))):
            app.on_abandoned_deployments_checked(AbandonedDeploymentsChecked(entries))

        self.assertEqual(len(messages), app._ABANDONED_WARNING_LIMIT + 1)
        self.assertIn("2 more unresolved deployments", messages[-1])

    def test_nothing_is_reported_when_the_journal_is_empty(self) -> None:
        app = TuiApp()
        messages: list[str] = []

        with patch.object(app, "notify", lambda message, **kwargs: messages.append(str(message))):
            app._warn_about_abandoned_deployments()

        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()

"""Concurrent deployment ownership, retained monitors, and cancellation."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
import io
import unittest
from unittest.mock import Mock, patch

from textual.screen import Screen

from llm_launchpad.core.deploy_journal import clear_in_flight, load_in_flight, record_in_flight
from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent, ResourceAllocatedEvent
from llm_launchpad.protocol.models import DeploymentConfig, EndpointInfo
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.deployment_jobs import DeploymentJob
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.operations import OperationsScreen


def config(name: str, provider: ComputeProvider = ComputeProvider.MODAL) -> DeploymentConfig:
    return DeploymentConfig(
        backend=BackendType.LLAMACPP, provider=provider, app_name=name,
        do_deploy=True, do_warmup=False,
    )


class DeploymentTrackingTests(unittest.TestCase):
    def test_closing_one_modal_stream_terminates_only_its_client(self) -> None:
        process = Mock(stdout=io.StringIO("Starting deployment\n"))
        process.poll.return_value = None
        with patch("llm_launchpad.core.backend.subprocess.Popen", return_value=process), patch.object(
            ModalBackend, "_resolve_command", return_value=["modal", "deploy"],
        ), patch.object(ModalBackend, "terminate_all") as terminate_all:
            stream = ModalBackend.run_streaming(["modal", "deploy"])
            next(stream)
            stream.close()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        self.assertTrue(process.stdout.closed)
        terminate_all.assert_not_called()

    def test_finishing_one_deployment_preserves_another_and_its_journal(self) -> None:
        app = TuiApp()
        first, second = config("first"), config("second")
        app._begin_in_flight(first)
        app._begin_in_flight(second)
        app._finish_in_flight(first)
        self.assertEqual(list(app._in_flight_deploys), [app._deployment_key(second)])
        self.assertEqual([entry.app_name for entry in load_in_flight()], ["second"])

    def test_identical_names_on_different_providers_are_independent(self) -> None:
        app = TuiApp()
        first, second = config("shared"), config("shared", ComputeProvider.PRIME)
        app._begin_in_flight(first)
        app._begin_in_flight(second)
        self.assertEqual(len(load_in_flight()), 2)
        app._finish_in_flight(first)
        self.assertEqual([entry.provider for entry in load_in_flight()], ["prime"])

    def test_fallback_cannot_overwrite_another_jobs_target(self) -> None:
        app = TuiApp()
        other = config("occupied")
        app._begin_in_flight(other)
        original = load_in_flight()[0]
        initial = config("first")
        initial.fallback_configs = (config("occupied"),)
        job = DeploymentJob("job", initial, MonitorScreen())
        app.deployment_jobs[job.id] = job
        with patch.object(app._orchestrator, "deploy", return_value=iter([
            OperationCompleteEvent(operation=OperationType.DEPLOY, success=False),
        ])) as deploy:
            app._run_deployment_job(job)
        self.assertEqual(deploy.call_count, 1)
        # The skipped fallback never began, so the owner's entry survives
        # untouched; this job's own entry resolves like any plain failure.
        self.assertIn(original, load_in_flight())
        self.assertIn("Failed", job.outcome)

    def test_concurrent_journal_writers_do_not_lose_entries(self) -> None:
        app = TuiApp()
        app._begin_in_flight(config("seed"))
        seed = load_in_flight()[0]
        entries = [replace(seed, app_name=f"job-{index}") for index in range(30)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(record_in_flight, entries))
            list(pool.map(clear_in_flight, [entry.app_name for entry in entries[:15]]))
        self.assertEqual(
            {entry.app_name for entry in load_in_flight()},
            {"seed", *(entry.app_name for entry in entries[15:])},
        )

    def test_cancel_during_provisioning_captures_resource_id_and_skips_fallback(self) -> None:
        app = TuiApp()
        initial = config("cancel-me", ComputeProvider.PRIME)
        initial.fallback_configs = (config("fallback"),)
        monitor = MonitorScreen()
        job = DeploymentJob("job", initial, monitor)
        app.deployment_jobs[job.id] = job

        def deploy(_config):
            app.cancel_deployment(job.id)
            yield ResourceAllocatedEvent(app_id="pod-123")
            self.fail("Cancelled deployment must not consume further provider events")

        with patch.object(app._orchestrator, "deploy", side_effect=deploy) as deploying, patch.object(
            app._orchestrator, "stop_app", return_value=iter([
                OperationCompleteEvent(operation=OperationType.STOP, success=True),
            ]),
        ) as stopping:
            app._run_deployment_job(job)
        self.assertEqual(deploying.call_count, 1)
        self.assertEqual(stopping.call_args.kwargs["app_id"], "pod-123")
        self.assertEqual(stopping.call_args.kwargs["provider"], ComputeProvider.PRIME)
        self.assertEqual(load_in_flight(), ())
        self.assertTrue(job.finished.is_set())
        self.assertIn("resource stopped", job.outcome)

    def test_failed_cancellation_keeps_recovery_record(self) -> None:
        app = TuiApp()
        job = DeploymentJob("job", config("keep-record"), MonitorScreen())
        app.deployment_jobs[job.id] = job
        def deploy(_config):
            app.cancel_deployment(job.id)
            yield LogEvent(line="Provisioning")

        with patch.object(app._orchestrator, "deploy", side_effect=deploy), patch.object(
            app._orchestrator, "stop_app", return_value=iter(()),
        ):
            app._run_deployment_job(job)
        self.assertEqual([entry.app_name for entry in load_in_flight()], ["keep-record"])
        self.assertIn("cleanup failed", job.outcome)

    def test_cancel_during_warmup_stops_the_allocated_pod(self) -> None:
        app = TuiApp()
        current = config("warming", ComputeProvider.PRIME)
        current.do_warmup = True
        job = DeploymentJob("job", current, MonitorScreen())
        app.deployment_jobs[job.id] = job

        def warmup(*args, **kwargs):
            app.cancel_deployment(job.id)
            yield LogEvent(line="Loading weights")

        with patch.object(app._orchestrator, "deploy", return_value=iter([
            OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=EndpointInfo(
                name="warming", app_id="pod-warming", provider=ComputeProvider.PRIME,
                web_url="https://example.invalid",
            )),
        ])), patch.object(app._orchestrator, "warmup", side_effect=warmup), patch.object(
            app._orchestrator, "stop_app", return_value=iter([
                OperationCompleteEvent(operation=OperationType.STOP, success=True),
            ]),
        ) as stop:
            app._run_deployment_job(job)
        self.assertEqual(stop.call_args.kwargs["app_id"], "pod-warming")
        self.assertEqual(load_in_flight(), ())
        self.assertIn("resource stopped", job.outcome)

    def test_cancel_before_worker_starts_does_not_stop_existing_endpoint(self) -> None:
        app = TuiApp()
        job = DeploymentJob("job", config("existing"), MonitorScreen())
        app.deployment_jobs[job.id] = job
        app.cancel_deployment(job.id)
        with patch.object(app._orchestrator, "deploy") as deploy, patch.object(
            app._orchestrator, "stop_app",
        ) as stop:
            app._run_deployment_job(job)
        deploy.assert_not_called()
        stop.assert_not_called()
        self.assertEqual(load_in_flight(), ())
        self.assertTrue(job.finished.is_set())

    def test_worker_exception_retains_recovery_record_and_finishes_job(self) -> None:
        app = TuiApp()
        job = DeploymentJob("job", config("crashed"), MonitorScreen())
        app.deployment_jobs[job.id] = job
        with patch.object(app, "_run_deploy_inner", side_effect=RuntimeError("provider disconnected")):
            app._run_deployment_job(job)
        self.assertTrue(job.finished.is_set())
        self.assertIn("Failed", job.outcome)
        self.assertEqual([entry.app_name for entry in load_in_flight()], ["crashed"])

    def test_fallback_cleanup_failure_is_not_cleared_by_previous_attempt(self) -> None:
        app = TuiApp()
        initial = config("shared")
        initial.fallback_configs = (config("shared"),)
        job = DeploymentJob("job", initial, MonitorScreen())
        app.deployment_jobs[job.id] = job
        attempts = []

        def deploy(current):
            attempts.append(current)
            if len(attempts) == 2:
                app.cancel_deployment(job.id)
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=False)

        with patch.object(app._orchestrator, "deploy", side_effect=deploy), patch.object(
            app._orchestrator, "stop_app", return_value=iter(()),
        ):
            app._run_deployment_job(job)
        self.assertEqual(len(attempts), 2)
        self.assertEqual([entry.app_name for entry in load_in_flight()], ["shared"])


class JobsApp(TuiApp):
    def on_mount(self) -> None:
        self.push_screen(Screen())

    def _deploy_endpoint_url(self, config: DeploymentConfig, observed_url: str | None) -> str | None:
        return observed_url


async def wait_until(predicate) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


def _inline_spawn(app: JobsApp) -> object:
    """Run durable jobs in-process so parent mocks apply (hermetic tests)."""
    from llm_launchpad.core.job_runner import run_job

    def _spawn(persistent_id: str) -> None:
        import threading

        store = app._get_job_store()
        thread = threading.Thread(
            target=run_job, args=(persistent_id, store), daemon=True
        )
        thread.start()

    return _spawn


class DeploymentJobInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_jobs_reopen_logs_and_cancel_only_selected_resource(self) -> None:
        app = JobsApp()
        releases = {"first": Event(), "second": Event()}
        stopped = []

        def deploy(current):
            yield LogEvent(line=f"Started {current.app_name}")
            releases[current.app_name].wait(timeout=10)
            yield LogEvent(line=f"Returned {current.app_name}")
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)

        def stop(_backend, **kwargs):
            stopped.append(kwargs.get("app_name"))
            yield OperationCompleteEvent(operation=OperationType.STOP, success=True)

        with patch.object(app._orchestrator, "deploy", side_effect=deploy), patch.object(
            app._orchestrator, "stop_app", side_effect=stop,
        ), patch(
            "llm_launchpad.core.job_runner._orchestrator", return_value=app._orchestrator
        ), patch.object(app, "_spawn_job_worker", side_effect=_inline_spawn(app)):
            async with app.run_test(size=(80, 24)) as pilot:
                try:
                    app.begin_deploy(config("first"))
                    await pilot.pause()
                    first = next(iter(app.deployment_jobs.values()))
                    await pilot.press("escape")
                    app.begin_deploy(config("second"))
                    await pilot.pause()
                    second = list(app.deployment_jobs.values())[1]
                    app.begin_deploy(config("second"))
                    self.assertEqual(len(app.deployment_jobs), 2)
                    await pilot.press("escape")
                    # Durable jobs are tracked in the store across sessions, not the legacy journal dict.
                    store = app._get_job_store()
                    self.assertEqual(len(store.list_jobs(include_terminal=False)), 2)
                    releases["first"].set()
                    await wait_until(first.finished.is_set)
                    self.assertFalse(second.finished.is_set())
                    await pilot.press("ctrl+o")
                    self.assertIsInstance(app.screen, OperationsScreen)
                    await pilot.press("enter")
                    self.assertIs(app.screen, first.monitor)
                    await pilot.pause()
                    self.assertTrue(any("Returned first" in line for line in first.monitor._raw_log_lines))
                    await pilot.press("escape", "ctrl+o")
                    await pilot.press("down", "x")
                    await pilot.click("#keep-running")
                    self.assertFalse(second.cancel_requested.is_set())
                    await pilot.press("x")
                    await pilot.click("#cancel-deployment")
                    self.assertTrue(second.cancel_requested.is_set())
                    releases["second"].set()
                    await wait_until(second.finished.is_set)
                    # Only the cancelled deployment is torn down, and it *is*
                    # torn down: a Modal app is addressed by its name, so the
                    # worker stops the app the deploy may already have
                    # published. It used to skip this because Modal emits no
                    # allocation event, which the TUI path never relied on.
                    self.assertEqual(stopped, ["second"])
                    app.reopen_deployment(second.id)
                    await pilot.pause()
                    self.assertIs(app.screen, second.monitor)
                    self.assertTrue(second.monitor._done)
                finally:
                    for release in releases.values():
                        release.set()

    async def test_quit_detaches_from_persistent_jobs(self) -> None:
        """Closing the TUI leaves background workers running; cancel is separate."""
        app = JobsApp()
        release = Event()
        entered = set()

        def deploy(current):
            entered.add(current.app_name)
            release.wait(timeout=10)
            yield LogEvent(line="Provider returned")
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)

        with patch.object(app._orchestrator, "deploy", side_effect=deploy), patch(
            "llm_launchpad.core.job_runner._orchestrator", return_value=app._orchestrator
        ), patch.object(app, "_spawn_job_worker", side_effect=_inline_spawn(app)), patch.object(
            app, "exit"
        ), patch("llm_launchpad.tui.app.ModalBackend.terminate_all"):
            async with app.run_test() as pilot:
                try:
                    app.begin_deploy(config("first"))
                    await pilot.pause()
                    await wait_until(lambda: len(entered) == 1)
                    await app.action_quit()
                    # Detached, not cancelled: the worker still owns the job.
                    jobs = list(app.deployment_jobs.values())
                    self.assertEqual(len(jobs), 1)
                    self.assertFalse(jobs[0].cancel_requested.is_set())
                    store = app._get_job_store()
                    records = store.list_jobs(include_terminal=False)
                    self.assertEqual(len(records), 1)
                    self.assertFalse(records[0].terminal)
                finally:
                    release.set()


class OperationsEmptyStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_operations_offers_a_direct_deploy_action(self) -> None:
        from textual.widgets import Button, OptionList, Static

        from llm_launchpad.tui.screens.operations import OperationsScreen

        app = TuiApp()
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(OperationsScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, OperationsScreen)
            deploy = screen.query_one("#operations-deploy-btn", Button)
            self.assertTrue(deploy.display)
            self.assertIn(
                "No deployments yet",
                str(screen.query_one("#operations-empty", Static).content),
            )
            options = screen.query_one("#deployment-jobs", OptionList)
            self.assertEqual(options.option_count, 1)
            self.assertTrue(options.get_option_at_index(0).disabled)

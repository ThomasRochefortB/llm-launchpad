from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_launchpad.core.job_runner import run_job
from llm_launchpad.core.job_store import JobStatus, JobStore
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent, ResourceAllocatedEvent
from llm_launchpad.protocol.models import DeploymentConfig


def _config(name: str = "durable-app", *, do_warmup: bool = False) -> DeploymentConfig:
    return DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.MODAL,
        app_name=name,
        do_deploy=True,
        do_warmup=do_warmup,
    )


class DurableJobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.tmp.name) / "jobs.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_create_and_run_job_to_success(self) -> None:
        record = self.store.create_job(_config())

        def deploy(_config):
            yield LogEvent(line="hello")
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)

        with patch("llm_launchpad.core.job_runner._orchestrator") as factory:
            factory.return_value.deploy.side_effect = deploy
            code = run_job(record.id, self.store)
        self.assertEqual(code, 0)
        updated = self.store.get_job(record.id)
        assert updated is not None
        self.assertEqual(updated.status, JobStatus.SUCCEEDED)
        events = self.store.get_events(record.id)
        self.assertTrue(any(type(event.event).__name__ == "LogEvent" for event in events))

    def test_warmup_runs_after_deploy_and_publishes_url(self) -> None:
        from llm_launchpad.protocol.models import EndpointInfo

        record = self.store.create_job(_config(do_warmup=True))
        endpoint = EndpointInfo(name="durable-app", app_id="ap-1", web_url="https://example.test/ap-1")

        def deploy(_config):
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=endpoint)

        def warmup(*args, **kwargs):
            yield OperationCompleteEvent(operation=OperationType.WARMUP, success=True, data={"url": endpoint.web_url})

        with patch("llm_launchpad.core.job_runner._orchestrator") as factory:
            factory.return_value.deploy.side_effect = deploy
            factory.return_value.warmup.side_effect = warmup
            code = run_job(record.id, self.store)
        self.assertEqual(code, 0)
        self.assertEqual(factory.return_value.warmup.call_count, 1)
        updated = self.store.get_job(record.id)
        assert updated is not None
        self.assertEqual(updated.status, JobStatus.SUCCEEDED)
        result = self.store.get_result(record.id)
        assert isinstance(result, dict)
        self.assertEqual(result["url"], "https://example.test/ap-1")

    def test_warmup_failure_falls_back_to_next_placement(self) -> None:
        from llm_launchpad.protocol.models import EndpointInfo

        first = _config("fallback-app", do_warmup=True)
        second = _config("fallback-app-2", do_warmup=False)
        first.fallback_configs = (second,)
        endpoint = EndpointInfo(name="fallback-app", app_id="ap-1", web_url="https://example.test/ap-1")

        def deploy(config):
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=endpoint)

        calls = {"warmup": 0}

        def warmup(*args, **kwargs):
            calls["warmup"] += 1
            if calls["warmup"] == 1:
                yield OperationCompleteEvent(operation=OperationType.WARMUP, success=False, detail="not ready")
            else:
                yield OperationCompleteEvent(operation=OperationType.WARMUP, success=True)

        record = self.store.create_job(first)
        with patch("llm_launchpad.core.job_runner._orchestrator") as factory:
            factory.return_value.deploy.side_effect = deploy
            factory.return_value.warmup.side_effect = warmup
            factory.return_value.stop_app.return_value = iter([
                OperationCompleteEvent(operation=OperationType.STOP, success=True)
            ])
            run_job(record.id, self.store)
        # First placement warmed (and failed), second has no warmup: one warmup call.
        self.assertEqual(calls["warmup"], 1)
        updated = self.store.get_job(record.id)
        assert updated is not None
        self.assertEqual(updated.status, JobStatus.SUCCEEDED)

    def test_duplicate_active_target_is_refused(self) -> None:
        self.store.create_job(_config("same"))
        with self.assertRaisesRegex(ValueError, "already has an active deployment"):
            self.store.create_job(_config("same"))

    def test_cancel_during_allocation_cleans_up(self) -> None:
        record = self.store.create_job(_config())

        def deploy(_config):
            self.store.request_cancel(record.id)
            yield ResourceAllocatedEvent(app_id="app-123")
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)

        stopped: list[dict] = []

        def stop_app(backend, app_name=None, app_id=None, provider=ComputeProvider.MODAL):
            stopped.append({"app_id": app_id})
            yield OperationCompleteEvent(operation=OperationType.STOP, success=True)

        with patch("llm_launchpad.core.job_runner._orchestrator") as factory:
            factory.return_value.deploy.side_effect = deploy
            factory.return_value.stop_app.side_effect = stop_app
            run_job(record.id, self.store)
        updated = self.store.get_job(record.id)
        assert updated is not None
        self.assertEqual(updated.status, JobStatus.CANCELLED)
        self.assertEqual(stopped, [{"app_id": "app-123"}])

    def test_failed_cleanup_stays_visible(self) -> None:
        record = self.store.create_job(_config())

        def deploy(_config):
            self.store.request_cancel(record.id)
            yield ResourceAllocatedEvent(app_id="app-123")
            yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)

        def stop_app(*args, **kwargs):
            yield OperationCompleteEvent(operation=OperationType.STOP, success=False, detail="provider unreachable")

        with patch("llm_launchpad.core.job_runner._orchestrator") as factory:
            factory.return_value.deploy.side_effect = deploy
            factory.return_value.stop_app.side_effect = stop_app
            run_job(record.id, self.store)
        updated = self.store.get_job(record.id)
        assert updated is not None
        self.assertEqual(updated.status, JobStatus.CANCELLED)
        self.assertIsNotNone(updated.cleanup_error)
        # Still listed across restarts.
        self.assertEqual(len(self.store.list_jobs()), 1)

    def test_worker_death_is_interrupted_not_retried(self) -> None:
        record = self.store.create_job(_config())
        claimed = self.store.claim_job(record.id, pid=999999)
        self.assertTrue(claimed)
        # Fake a stale heartbeat from a dead pid.
        import time

        with self.store._connect() as conn:
            conn.execute("UPDATE jobs SET heartbeat=? WHERE id=?;", (time.time() - 1000, record.id))
            conn.commit()
        interrupted = self.store.reconcile_workers()
        self.assertEqual(len(interrupted), 1)
        updated = self.store.get_job(record.id)
        assert updated is not None
        self.assertEqual(updated.status, JobStatus.INTERRUPTED)

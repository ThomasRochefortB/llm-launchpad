"""Legacy job persistence stays readable after the contract split."""

from __future__ import annotations

import base64
import pickle
import unittest

from llm_launchpad.core.job_store import JobStore
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig


class LegacyJobCompatibilityTests(unittest.TestCase):
    def test_old_pickled_config_remains_inspectable(self) -> None:
        store = JobStore(path=__import__("pathlib").Path(
            __import__("tempfile").mkdtemp() + "/jobs.db"
        ))
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            app_name="llamacpp-legacy",
            repo_id="org/model-GGUF",
        )
        record = store.create_job(config)
        fetched = store.get_job(record.id)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertIsNone(fetched.undecodable)
        self.assertEqual(fetched.config.repo_id, "org/model-GGUF")

    def test_undecodable_job_is_kept_for_recovery(self) -> None:
        import sqlite3

        store = JobStore(path=__import__("pathlib").Path(
            __import__("tempfile").mkdtemp() + "/jobs.db"
        ))
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-vllm-legacy",
            model_name="org/model",
        )
        record = store.create_job(config)
        conn = sqlite3.connect(str(store.path))
        try:
            conn.execute(
                "UPDATE jobs SET config_b64=? WHERE id=?;",
                (base64.b64encode(pickle.dumps(object())).decode("ascii"), record.id),
            )
            conn.commit()
        finally:
            conn.close()
        fetched = store.get_job(record.id)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertIsNotNone(fetched.undecodable)
        self.assertEqual(fetched.app_name, "llp-prime-vllm-legacy")
        # Unreadable jobs still list instead of silently disappearing.
        self.assertTrue(any(job.id == record.id for job in store.list_jobs()))


if __name__ == "__main__":
    unittest.main()

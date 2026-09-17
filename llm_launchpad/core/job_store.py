"""Durable deployment jobs that survive closing the TUI.

Execution lives in detached worker processes, not UI threads. This store is
the source of truth both workers and interfaces share: configuration,
lifecycle state, ordered events, allocated resource references, cancellation
requests, and worker heartbeats. SQLite with WAL gives cross-process
transactions without a service.
"""

from __future__ import annotations

import base64
import os
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.events import BaseEvent, EndpointAvailableEvent, OperationCompleteEvent, ResourceAllocatedEvent
from ..protocol.models import DeploymentConfig, EndpointInfo
from .config import SETTINGS_DIR
from .diagnostics import log_exception

def _default_job_db_path() -> Path:
    override = os.getenv("LLM_LAUNCHPAD_JOB_DB", "").strip()
    if override:
        return Path(override)
    return SETTINGS_DIR / "deployment_jobs.db"


JOB_DB_PATH = _default_job_db_path()

# Heartbeat freshness: workers update every few seconds, even inside long
# provider calls (via a heartbeat thread). Past this, a live pid is trusted
# but flagged; a dead pid is interrupted.
HEARTBEAT_TIMEOUT_SECONDS = 120.0
_SCHEMA_VERSION = 1


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = frozenset({
    JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED,
})
ACTIVE_STATUSES = frozenset({JobStatus.PENDING, JobStatus.RUNNING})


def _encode(obj: object) -> str:
    return base64.b64encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _decode(payload: str) -> object:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


@dataclass
class JobRecord:
    """One durable deployment job with its execution state."""

    id: str
    status: str
    config: DeploymentConfig
    created_at: float = 0.0
    updated_at: float = 0.0
    worker_pid: int | None = None
    heartbeat: float = 0.0
    cancel_requested: bool = False
    outcome: str = "Pending"
    app_name: str = ""
    provider: str = ""
    backend: str = ""
    resource_app_id: str | None = None
    cleanup_error: str | None = None
    event_count: int = 0

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass
class StoredEvent:
    """One ordered protocol event retained for a job."""

    seq: int
    timestamp: float
    type: str
    event: BaseEvent = field(repr=False)


def _deployment_key(provider: str, backend: str, app_name: str) -> tuple[str, str, str]:
    return (provider, backend, app_name)


class JobStore:
    """SQLite-backed durable jobs, safe across threads and processes."""

    def __init__(self, path: Path | None = None) -> None:
        if path is not None:
            self.path = path
        else:
            override = os.getenv("LLM_LAUNCHPAD_JOB_DB", "").strip()
            self.path = Path(override) if override else JOB_DB_PATH
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        config_b64 TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        worker_pid INTEGER,
                        heartbeat REAL DEFAULT 0,
                        cancel_requested INTEGER DEFAULT 0,
                        outcome TEXT DEFAULT '',
                        app_name TEXT DEFAULT '',
                        provider TEXT DEFAULT '',
                        backend TEXT DEFAULT '',
                        resource_app_id TEXT,
                        result_b64 TEXT,
                        cleanup_error TEXT
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        job_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        ts REAL NOT NULL,
                        type TEXT NOT NULL,
                        payload_b64 TEXT NOT NULL,
                        PRIMARY KEY (job_id, seq)
                    );
                    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                    CREATE INDEX IF NOT EXISTS idx_jobs_target ON jobs(provider, backend, app_name, status);
                    CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, seq);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def create_job(self, config: DeploymentConfig) -> JobRecord:
        """Persist a job before any provider work; raises on duplicate active target."""
        app_name = (config.app_name or "").strip()
        provider = config.provider.value if isinstance(config.provider, ComputeProvider) else str(config.provider)
        backend = config.backend.value if isinstance(config.backend, BackendType) else str(config.backend)
        if not app_name:
            raise ValueError("Deployment config requires app_name for durable jobs.")
        job_id = f"job-{uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                duplicate = conn.execute(
                    "SELECT id FROM jobs WHERE provider=? AND backend=? AND app_name=? AND status IN ('pending','running') LIMIT 1;",
                    (provider, backend, app_name),
                ).fetchone()
                if duplicate is not None:
                    conn.execute("ROLLBACK;")
                    raise ValueError(
                        f"{app_name} already has an active deployment ({duplicate['id']}); reopen it instead of launching a duplicate."
                    )
                conn.execute(
                    "INSERT INTO jobs (id, config_b64, status, created_at, updated_at, worker_pid, heartbeat, cancel_requested, outcome, app_name, provider, backend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",
                    (job_id, _encode(config), JobStatus.PENDING, now, now, None, 0.0, 0, "Pending", app_name, provider, backend),
                )
                conn.commit()
            finally:
                conn.close()
        record = self.get_job(job_id)
        assert record is not None
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM jobs WHERE id=?;", (job_id,)).fetchone()
                if row is None:
                    return None
                count = conn.execute("SELECT COUNT(*) AS n FROM events WHERE job_id=?;", (job_id,)).fetchone()
            finally:
                conn.close()
        try:
            config = _decode(row["config_b64"])
        except Exception:
            log_exception(f"Could not decode job config for {job_id}")
            return None
        if not isinstance(config, DeploymentConfig):
            return None
        return JobRecord(
            id=row["id"],
            status=row["status"],
            config=config,
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
            worker_pid=row["worker_pid"],
            heartbeat=row["heartbeat"] or 0.0,
            cancel_requested=bool(row["cancel_requested"]),
            outcome=row["outcome"] or "",
            app_name=row["app_name"] or "",
            provider=row["provider"] or "",
            backend=row["backend"] or "",
            resource_app_id=row["resource_app_id"],
            cleanup_error=row["cleanup_error"],
            event_count=int((count["n"] if count else 0) or 0),
        )

    def list_jobs(self, *, include_terminal: bool = True, limit: int = 200) -> list[JobRecord]:
        with self._lock:
            conn = self._connect()
            try:
                if include_terminal:
                    rows = conn.execute(
                        "SELECT id FROM jobs ORDER BY updated_at DESC LIMIT ?;", (limit,)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id FROM jobs WHERE status IN ('pending','running') ORDER BY updated_at DESC LIMIT ?;",
                        (limit,),
                    ).fetchall()
                ids = [row["id"] for row in rows]
            finally:
                conn.close()
        records = []
        for job_id in ids:
            record = self.get_job(job_id)
            if record is not None:
                records.append(record)
        return records

    def claim_job(self, job_id: str, pid: int) -> bool:
        """Atomically move PENDING->RUNNING for one worker; False if taken/cancelled."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                row = conn.execute("SELECT status, cancel_requested FROM jobs WHERE id=?;", (job_id,)).fetchone()
                if row is None or row["status"] != JobStatus.PENDING:
                    conn.execute("ROLLBACK;")
                    return False
                if row["cancel_requested"]:
                    conn.execute(
                        "UPDATE jobs SET status=?, outcome=?, updated_at=? WHERE id=?;",
                        (JobStatus.CANCELLED, "Cancelled before starting.", now, job_id),
                    )
                    conn.commit()
                    return False
                conn.execute(
                    "UPDATE jobs SET status=?, worker_pid=?, heartbeat=?, updated_at=? WHERE id=?;",
                    (JobStatus.RUNNING, pid, now, now, job_id),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def heartbeat_job(self, job_id: str, pid: int) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE jobs SET heartbeat=?, updated_at=? WHERE id=? AND worker_pid=?;",
                    (now, now, job_id, pid),
                )
                conn.commit()
            finally:
                conn.close()

    def request_cancel(self, job_id: str) -> bool:
        """Persist a cooperative cancellation request; workers observe it between provider calls."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT status FROM jobs WHERE id=?;", (job_id,)).fetchone()
                if row is None or row["status"] not in ACTIVE_STATUSES:
                    return False
                conn.execute(
                    "UPDATE jobs SET cancel_requested=1, outcome=?, updated_at=? WHERE id=?;",
                    ("Cancelling; waiting for provider call to return.", time.time(), job_id),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT cancel_requested FROM jobs WHERE id=?;", (job_id,)).fetchone()
            finally:
                conn.close()
        return bool(row["cancel_requested"]) if row is not None else False

    def update_config(self, job_id: str, config: DeploymentConfig) -> None:
        """Point a job at its current fallback placement."""
        app_name = (config.app_name or "").strip()
        provider = config.provider.value if isinstance(config.provider, ComputeProvider) else str(config.provider)
        backend = config.backend.value if isinstance(config.backend, BackendType) else str(config.backend)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE jobs SET config_b64=?, app_name=?, provider=?, backend=?, updated_at=? WHERE id=?;",
                    (_encode(config), app_name, provider, backend, time.time(), job_id),
                )
                conn.commit()
            finally:
                conn.close()

    def set_resource(self, job_id: str, resource_app_id: str | None) -> None:
        if not resource_app_id:
            return
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE jobs SET resource_app_id=?, updated_at=? WHERE id=? AND (resource_app_id IS NULL OR resource_app_id='');",
                    (resource_app_id, time.time(), job_id),
                )
                conn.commit()
            finally:
                conn.close()

    def append_event(self, job_id: str, event: BaseEvent) -> int:
        """Append one event; returns its sequence number."""
        event_type = type(event).__name__
        now = time.time()
        resource_id: str | None = None
        if isinstance(event, ResourceAllocatedEvent) and event.app_id:
            resource_id = event.app_id
        elif isinstance(event, EndpointAvailableEvent):
            try:
                if event.endpoint.app_id:
                    resource_id = event.endpoint.app_id
            except Exception:
                pass
        elif (
            isinstance(event, OperationCompleteEvent)
            and isinstance(event.data, EndpointInfo)
            and event.data.app_id
        ):
            resource_id = event.data.app_id
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM events WHERE job_id=?;", (job_id,)).fetchone()
                seq = int(row["m"] or 0) + 1
                conn.execute(
                    "INSERT INTO events (job_id, seq, ts, type, payload_b64) VALUES (?,?,?,?,?);",
                    (job_id, seq, now, event_type, _encode(event)),
                )
                if resource_id:
                    conn.execute(
                        "UPDATE jobs SET resource_app_id=?, updated_at=? WHERE id=? AND (resource_app_id IS NULL OR resource_app_id='');",
                        (resource_id, now, job_id),
                    )
                else:
                    conn.execute("UPDATE jobs SET updated_at=? WHERE id=?;", (now, job_id))
                conn.commit()
                return seq
            finally:
                conn.close()

    def get_events(self, job_id: str, *, after_seq: int = 0, limit: int = 5000) -> list[StoredEvent]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT seq, ts, type, payload_b64 FROM events WHERE job_id=? AND seq>? ORDER BY seq ASC LIMIT ?;",
                    (job_id, after_seq, limit),
                ).fetchall()
            finally:
                conn.close()
        stored: list[StoredEvent] = []
        for row in rows:
            try:
                event = _decode(row["payload_b64"])
            except Exception:
                continue
            if not isinstance(event, BaseEvent):
                continue
            stored.append(StoredEvent(seq=row["seq"], timestamp=row["ts"], type=row["type"], event=event))
        return stored

    def finish_job(
        self,
        job_id: str,
        status: str,
        outcome: str,
        *,
        result: object = None,
        cleanup_error: str | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE jobs SET status=?, outcome=?, result_b64=?, cleanup_error=?, updated_at=? WHERE id=?;",
                    (status, outcome, _encode(result) if result is not None else None, cleanup_error, time.time(), job_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_result(self, job_id: str) -> object:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT result_b64 FROM jobs WHERE id=?;", (job_id,)).fetchone()
            finally:
                conn.close()
        if row is None or not row["result_b64"]:
            return None
        try:
            return _decode(row["result_b64"])
        except Exception:
            return None

    @staticmethod
    def pid_alive(pid: int | None) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
        return True

    _STALLED_OUTCOME = (
        "Worker has not reported in; it may be wedged. Its resource may be billing."
    )

    def note_outcome(self, job_id: str, outcome: str) -> None:
        """Change what a job says about itself without ending it."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE jobs SET outcome=? WHERE id=? AND status IN ('pending','running');",
                    (outcome, job_id),
                )
                conn.commit()
            finally:
                conn.close()

    def reconcile_workers(self) -> list[JobRecord]:
        """Mark jobs whose workers died; never auto-restart allocation.

        A dead pid is interrupted. A live pid is trusted -- only the worker may
        end its own job -- but a live worker that has stopped heartbeating is
        flagged, because a wedged provider call bills exactly like a working
        one and nothing else on screen would say so.
        """
        interrupted: list[JobRecord] = []
        for record in self.list_jobs(include_terminal=False):
            if record.status not in ACTIVE_STATUSES:
                continue
            alive = self.pid_alive(record.worker_pid)
            stale = (time.time() - (record.heartbeat or 0.0)) > HEARTBEAT_TIMEOUT_SECONDS
            if record.worker_pid and not alive:
                self.finish_job(
                    record.id,
                    JobStatus.INTERRUPTED,
                    "Worker interrupted; check provider for a billing resource, then clean up from Jobs or Manage.",
                )
                updated = self.get_job(record.id)
                if updated is not None:
                    interrupted.append(updated)
            elif record.worker_pid and stale and record.outcome != self._STALLED_OUTCOME:
                self.note_outcome(record.id, self._STALLED_OUTCOME)
            # A job with no pid was never claimed (e.g. spawn failed); it stays
            # pending for inspection rather than being called stalled.
        return interrupted

    def import_journal_entries(self) -> list[JobRecord]:
        """Adopt legacy journal entries as interrupted recovery records."""
        try:
            from .deploy_journal import clear_in_flight, load_in_flight
        except Exception:
            return []
        imported: list[JobRecord] = []
        for entry in load_in_flight():
            try:
                provider = ComputeProvider(entry.provider)
            except ValueError:
                continue
            try:
                backend = BackendType(entry.backend)
            except ValueError:
                continue
            config = DeploymentConfig(
                backend=backend,
                provider=provider,
                app_name=entry.app_name,
                instance_name=entry.instance_name,
                gpu_type=entry.gpu_type,
                gpu_count=entry.gpu_count,
                do_deploy=True,
                do_warmup=False,
            )
            try:
                record = self.create_job(config)
            except ValueError:
                continue
            except Exception:
                log_exception("Could not import journal entry as a job")
                continue
            self.finish_job(
                record.id,
                JobStatus.INTERRUPTED,
                f"Interrupted before resolving ({entry.app_name}); imported from a previous session. Check Manage for billing resources.",
            )
            try:
                clear_in_flight(entry.app_name, provider=entry.provider, backend=entry.backend)
            except Exception:
                pass
            updated = self.get_job(record.id)
            if updated is not None:
                imported.append(updated)
        return imported

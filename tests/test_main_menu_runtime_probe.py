"""Cover the fleet health probe and runtime annotation (main menu)."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from contextlib import contextmanager

from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.screens.main_menu import (
    _annotate_runtime_statuses,
    _probe_row_runtime_status,
    _runtime_bucket,
    _runtime_bucket_from_modal_state,
)


def _row(**overrides: object) -> EndpointInfo:
    kwargs: dict[str, object] = {
        "name": "vllm-qwen",
        "app_id": "ap-1",
        "state": "running",
        "backend": BackendType.VLLM,
        "provider": ComputeProvider.MODAL,
        "web_url": "https://example.modal.run",
        "endpoint_api_key": "secret",
    }
    kwargs.update(overrides)
    return EndpointInfo(**kwargs)  # type: ignore[arg-type]


@contextmanager
def _fake_requests(fake: object):  # type: ignore[no-untyped-def]
    """Swap the real requests module; the probe imports it function-locally."""
    import sys

    with patch.dict(sys.modules, {"requests": fake}):
        yield fake


def _never_called(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    raise AssertionError("no HTTP probe expected for this row")


def _response(status_code: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(status_code=status_code)


class ProbeRowRuntimeStatusTests(unittest.TestCase):
    def test_non_healthy_modal_state_short_circuits(self) -> None:
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            status, detail = _probe_row_runtime_status(_row(state="deploying"), "alice")
        self.assertEqual((status, detail), ("in_progress", None))

    def test_unknown_backend_is_not_probed(self) -> None:
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            status, detail = _probe_row_runtime_status(_row(backend=None), "alice")
        self.assertEqual((status, detail), ("in_progress", "unknown backend"))

    def test_missing_url_is_not_probed(self) -> None:
        row = _row(web_url="", name="")
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            status, detail = _probe_row_runtime_status(row, "")
        self.assertEqual((status, detail), ("in_progress", "missing URL"))

    def test_healthy_probe_marks_row_healthy(self) -> None:
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(200))
        with _fake_requests(fake):
            status, detail = _probe_row_runtime_status(_row(), "alice")
        self.assertEqual((status, detail), ("healthy", None))

    def test_explicit_url_auth_failure_is_an_error(self) -> None:
        seen: dict[str, object] = {}

        def _get(url: str, **kwargs: object) -> types.SimpleNamespace:
            seen["url"] = url
            seen["headers"] = kwargs.get("headers")
            return _response(401)

        fake = types.SimpleNamespace(get=_get)
        with _fake_requests(fake):
            status, detail = _probe_row_runtime_status(_row(), "alice")
        self.assertEqual(seen["url"], "https://example.modal.run/health")
        self.assertEqual(seen["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual((status, detail), ("error", "HTTP 401"))

    def test_derived_url_auth_failure_stays_in_progress(self) -> None:
        # A derived URL 401 usually means the slug guess was wrong, not that
        # the deployment is broken — it must not flip the row to error.
        row = _row(web_url="", name="vllm-qwen")
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(403))
        with _fake_requests(fake):
            status, detail = _probe_row_runtime_status(row, "alice")
        self.assertEqual((status, detail), ("in_progress", "HTTP 403"))

    def test_server_error_stays_in_progress(self) -> None:
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(503))
        with _fake_requests(fake):
            status, detail = _probe_row_runtime_status(_row(), "alice")
        self.assertEqual((status, detail), ("in_progress", "HTTP 503"))

    def test_probe_exception_stays_in_progress(self) -> None:
        def _raise(*args: object, **kwargs: object) -> types.SimpleNamespace:
            raise ConnectionError("tunnel reset")

        fake = types.SimpleNamespace(get=_raise)
        with _fake_requests(fake):
            status, detail = _probe_row_runtime_status(_row(), "alice")
        self.assertEqual((status, detail), ("in_progress", "tunnel reset"))

    def test_probe_without_api_key_sends_no_auth_header(self) -> None:
        seen: dict[str, object] = {}

        def _get(url: str, **kwargs: object) -> types.SimpleNamespace:
            seen["headers"] = kwargs.get("headers")
            return _response(200)

        fake = types.SimpleNamespace(get=_get)
        with _fake_requests(fake):
            _probe_row_runtime_status(_row(endpoint_api_key=""), "alice")
        self.assertIsNone(seen["headers"])


class AnnotateRuntimeStatusesTests(unittest.TestCase):
    def test_non_healthy_rows_annotated_without_network(self) -> None:
        rows = [_row(state="failed"), _row(state="stopped")]
        probed: list[str] = []
        with _fake_requests(types.SimpleNamespace(get=lambda *a, **k: probed.append("x") or _response(200))):
            _annotate_runtime_statuses(rows, "alice")
        self.assertEqual(probed, [])
        # "stopped" maps to the in_progress runtime bucket: a stopped Modal
        # app has no probe target, so it shares the pending bucket.
        self.assertEqual(
            [(r.runtime_status, r.runtime_status_detail) for r in rows],
            [("error", None), ("in_progress", None)],
        )

    def test_running_rows_are_probed_concurrently(self) -> None:
        rows = [_row(name="a"), _row(name="b", web_url="https://b.example.run")]
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(200))
        with _fake_requests(fake):
            _annotate_runtime_statuses(rows, "alice")
        self.assertTrue(all(r.runtime_status == "healthy" for r in rows))
        self.assertTrue(all(r.runtime_status_detail is None for r in rows))

    def test_probe_failures_do_not_raise(self) -> None:
        rows = [_row()]
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(500))
        with _fake_requests(fake):
            _annotate_runtime_statuses(rows, "alice")
        self.assertEqual(rows[0].runtime_status, "in_progress")
        self.assertEqual(rows[0].runtime_status_detail, "HTTP 500")

    def test_runtime_bucket_prefers_stored_status(self) -> None:
        self.assertEqual(_runtime_bucket(_row(runtime_status="error")), "error")
        self.assertEqual(
            _runtime_bucket_from_modal_state("deploying"), "in_progress"
        )


if __name__ == "__main__":
    unittest.main()

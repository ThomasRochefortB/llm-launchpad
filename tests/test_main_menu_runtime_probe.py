"""Cover the fleet health probe and runtime annotation (main menu)."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from contextlib import contextmanager

from llm_launchpad.core.runtime_health import reset as reset_runtime_health
from llm_launchpad.core.serving_metrics import default_tracker, usage_key
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, ServingStats
from llm_launchpad.tui.screens.main_menu import (
    _annotate_runtime_statuses,
    _probe_row_runtime_status,
    _runtime_bucket,
    _runtime_bucket_from_modal_state,
)

_VLLM_METRICS = """vllm:prompt_tokens_total{model_name="qwen"} 1200.0
vllm:generation_tokens_total{model_name="qwen"} 3400.0
vllm:num_requests_running{model_name="qwen"} 2.0
"""


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


def _response(status_code: int, text: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(status_code=status_code, text=text)


class ProbeRowRuntimeStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        default_tracker().reset()
        reset_runtime_health()

    def test_non_healthy_modal_state_short_circuits(self) -> None:
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            result = _probe_row_runtime_status(_row(state="deploying"), "alice")
        self.assertEqual((result.status, result.detail), ("in_progress", None))

    def test_unknown_backend_is_not_probed(self) -> None:
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            result = _probe_row_runtime_status(
                _row(backend=None, provider=ComputeProvider.PRIME), "alice"
            )
        self.assertEqual((result.status, result.detail), ("in_progress", "unknown backend"))

    def test_missing_url_is_not_probed(self) -> None:
        row = _row(web_url="", name="", provider=ComputeProvider.PRIME)
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            result = _probe_row_runtime_status(row, "")
        self.assertEqual((result.status, result.detail), ("in_progress", "missing URL"))

    def test_modal_background_never_touches_the_runtime(self) -> None:
        # The core guarantee: a background refresh sends no request to a
        # Modal inference endpoint, not even the /health fallback.
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            result = _probe_row_runtime_status(_row(), "alice")
        self.assertEqual((result.status, result.detail), ("unchecked", "health not checked"))

    def test_modal_explicit_check_may_start_the_gpu(self) -> None:
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(200))
        with _fake_requests(fake):
            result = _probe_row_runtime_status(_row(), "alice", explicit=True)
        self.assertEqual((result.status, result.detail), ("healthy", None))

    def test_healthy_probe_marks_row_healthy(self) -> None:
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(200))
        with _fake_requests(fake):
            result = _probe_row_runtime_status(
                _row(provider=ComputeProvider.PRIME), "alice"
            )
        self.assertEqual((result.status, result.detail), ("healthy", None))

    def test_explicit_url_auth_failure_is_an_error(self) -> None:
        seen: dict[str, object] = {}

        def _get(url: str, **kwargs: object) -> types.SimpleNamespace:
            seen["url"] = url
            seen["headers"] = kwargs.get("headers")
            return _response(401)

        fake = types.SimpleNamespace(get=_get)
        with _fake_requests(fake):
            result = _probe_row_runtime_status(
                _row(provider=ComputeProvider.PRIME), "alice"
            )
        self.assertEqual(seen["url"], "https://example.modal.run/health")
        self.assertEqual(seen["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual((result.status, result.detail), ("error", "HTTP 401"))

    def test_modal_explicit_auth_failure_is_an_error(self) -> None:
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(401))
        with _fake_requests(fake):
            result = _probe_row_runtime_status(_row(), "alice", explicit=True)
        self.assertEqual((result.status, result.detail), ("error", "HTTP 401"))

    def test_derived_url_auth_failure_stays_in_progress(self) -> None:
        # A derived URL 401 usually means the slug guess was wrong, not that
        # the deployment is broken — it must not flip the row to error.
        # Derived URLs only exist for Modal, so this exercises the explicit
        # path that is allowed to wake the container.
        row = _row(web_url="", name="vllm-qwen", provider=ComputeProvider.MODAL)
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(403))
        with _fake_requests(fake):
            result = _probe_row_runtime_status(row, "alice", explicit=True)
        self.assertEqual((result.status, result.detail), ("in_progress", "HTTP 403"))

    def test_server_error_stays_in_progress(self) -> None:
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(503))
        with _fake_requests(fake):
            result = _probe_row_runtime_status(
                _row(provider=ComputeProvider.PRIME), "alice"
            )
        self.assertEqual((result.status, result.detail), ("in_progress", "HTTP 503"))

    def test_probe_exception_stays_in_progress(self) -> None:
        def _raise(*args: object, **kwargs: object) -> types.SimpleNamespace:
            raise ConnectionError("tunnel reset")

        fake = types.SimpleNamespace(get=_raise)
        with _fake_requests(fake):
            result = _probe_row_runtime_status(
                _row(provider=ComputeProvider.PRIME), "alice"
            )
        self.assertEqual((result.status, result.detail), ("in_progress", "tunnel reset"))

    def test_probe_without_api_key_sends_no_auth_header(self) -> None:
        seen: dict[str, object] = {}

        def _get(url: str, **kwargs: object) -> types.SimpleNamespace:
            seen["headers"] = kwargs.get("headers")
            return _response(200)

        fake = types.SimpleNamespace(get=_get)
        with _fake_requests(fake):
            _probe_row_runtime_status(
                _row(endpoint_api_key="", provider=ComputeProvider.PRIME), "alice"
            )
        self.assertIsNone(seen["headers"])


class AnnotateRuntimeStatusesTests(unittest.TestCase):
    def setUp(self) -> None:
        default_tracker().reset()
        reset_runtime_health()

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

    def test_modal_background_leaves_deployed_rows_unchecked(self) -> None:
        rows = [_row(name="a"), _row(name="b", web_url="https://b.example.run")]
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            _annotate_runtime_statuses(rows, "alice")
        self.assertTrue(all(r.runtime_status == "unchecked" for r in rows))
        self.assertTrue(
            all(r.runtime_status_detail == "health not checked" for r in rows)
        )
        self.assertTrue(all(r.runtime_checked_at is None for r in rows))

    def test_mixed_fleet_probes_prime_but_not_modal(self) -> None:
        modal = _row(name="modal-a", provider=ComputeProvider.MODAL)
        prime = _row(
            name="prime-a",
            provider=ComputeProvider.PRIME,
            web_url="https://prime.example.run",
        )
        seen: list[str] = []

        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            seen.append(url)
            return _response(200)

        with _fake_requests(types.SimpleNamespace(get=_get)):
            _annotate_runtime_statuses([modal, prime], "alice")
        self.assertEqual(modal.runtime_status, "unchecked")
        self.assertEqual(prime.runtime_status, "healthy")
        self.assertTrue(all("prime.example.run" in url for url in seen))
        self.assertTrue(all("modal" not in url for url in seen))

    def test_running_rows_are_probed_concurrently(self) -> None:
        rows = [
            _row(name="a", provider=ComputeProvider.PRIME),
            _row(
                name="b",
                web_url="https://b.example.run",
                provider=ComputeProvider.PRIME,
            ),
        ]
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(200))
        with _fake_requests(fake):
            _annotate_runtime_statuses(rows, "alice")
        self.assertTrue(all(r.runtime_status == "healthy" for r in rows))
        self.assertTrue(all(r.runtime_status_detail is None for r in rows))

    def test_probe_failures_do_not_raise(self) -> None:
        rows = [_row(provider=ComputeProvider.PRIME)]
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(500))
        with _fake_requests(fake):
            _annotate_runtime_statuses(rows, "alice")
        self.assertEqual(rows[0].runtime_status, "in_progress")
        self.assertEqual(rows[0].runtime_status_detail, "HTTP 500")

    def test_explicit_modal_probe_records_health_for_later_passive_passes(self) -> None:
        from llm_launchpad.core.runtime_health import get_health

        row = _row()
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(200))
        with _fake_requests(fake):
            _annotate_runtime_statuses([row], "alice", explicit=True)
        self.assertEqual(row.runtime_status, "healthy")
        self.assertIsNotNone(row.runtime_checked_at)
        self.assertIsNotNone(get_health(row))

        # A later background pass re-attaches the verdict without a request.
        fresh = _row()
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            _annotate_runtime_statuses([fresh], "alice")
        self.assertEqual(fresh.runtime_status, "healthy")
        self.assertEqual(fresh.runtime_checked_at, row.runtime_checked_at)


class ServingMetricsProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        default_tracker().reset()
        reset_runtime_health()

    def test_metrics_answer_health_and_traffic_in_one_request(self) -> None:
        urls: list[str] = []

        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            urls.append(url)
            return _response(200, _VLLM_METRICS)

        with _fake_requests(types.SimpleNamespace(get=_get)):
            result = _probe_row_runtime_status(
                _row(provider=ComputeProvider.PRIME), "alice"
            )

        self.assertEqual(urls, ["https://example.modal.run/metrics"])
        self.assertEqual(result.status, "healthy")
        assert result.serving is not None
        self.assertEqual(result.serving.total_tokens, 4600.0)
        self.assertEqual(result.serving.stats.requests_running, 2.0)

    def test_modal_metrics_require_an_explicit_probe(self) -> None:
        urls: list[str] = []

        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            urls.append(url)
            return _response(200, _VLLM_METRICS)

        with _fake_requests(types.SimpleNamespace(get=_get)):
            background = _probe_row_runtime_status(_row(), "alice")
        self.assertEqual(background.status, "unchecked")
        self.assertEqual(urls, [])

        with _fake_requests(types.SimpleNamespace(get=_get)):
            explicit = _probe_row_runtime_status(_row(), "alice", explicit=True)
        self.assertEqual(explicit.status, "healthy")
        self.assertEqual(urls, ["https://example.modal.run/metrics"])

    def test_blocked_modal_metrics_never_fall_back_to_health(self) -> None:
        # Regression guard for the core guarantee: a passive pass that cannot
        # read /metrics must not spend a /health request instead.
        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            raise AssertionError(f"no request expected, got {url}")

        with _fake_requests(types.SimpleNamespace(get=_get)):
            result = _probe_row_runtime_status(_row(), "alice")
        self.assertEqual(result.status, "unchecked")

    def test_runtime_without_metrics_falls_back_to_health_once(self) -> None:
        urls: list[str] = []

        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            urls.append(url)
            # llama.cpp started without --metrics answers 501 here.
            return _response(501) if url.endswith("/metrics") else _response(200)

        row = _row(backend=BackendType.LLAMACPP, provider=ComputeProvider.PRIME)
        with _fake_requests(types.SimpleNamespace(get=_get)):
            first = _probe_row_runtime_status(row, "alice")
            second = _probe_row_runtime_status(row, "alice")

        self.assertEqual(first.status, "healthy")
        self.assertIsNone(first.serving)
        self.assertEqual(second.status, "healthy")
        # The verdict is remembered, so the second refresh spends one request
        # rather than re-learning that this runtime serves no metrics.
        self.assertEqual(
            urls,
            [
                "https://example.modal.run/metrics",
                "https://example.modal.run/health",
                "https://example.modal.run/health",
            ],
        )

    def test_a_metrics_transport_failure_is_not_a_verdict(self) -> None:
        urls: list[str] = []

        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            urls.append(url)
            if url.endswith("/metrics") and len(urls) == 1:
                raise ConnectionError("reset")
            return _response(200, _VLLM_METRICS if url.endswith("/metrics") else "")

        row = _row(provider=ComputeProvider.PRIME)
        with _fake_requests(types.SimpleNamespace(get=_get)):
            _probe_row_runtime_status(row, "alice")
            second = _probe_row_runtime_status(row, "alice")

        # A dropped connection says nothing about whether metrics are served,
        # so the endpoint keeps its place in the queue.
        self.assertEqual(urls[:2], ["https://example.modal.run/metrics", "https://example.modal.run/health"])
        self.assertEqual(urls[2], "https://example.modal.run/metrics")
        assert second.serving is not None

    def test_a_non_metrics_page_does_not_become_a_reading(self) -> None:
        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            if url.endswith("/metrics"):
                return _response(200, "<html>starting up</html>")
            return _response(200)

        with _fake_requests(types.SimpleNamespace(get=_get)):
            result = _probe_row_runtime_status(
                _row(provider=ComputeProvider.PRIME), "alice"
            )

        self.assertEqual(result.status, "healthy")
        self.assertIsNone(result.serving)

    def test_stopped_rows_keep_their_banked_total_without_a_request(self) -> None:
        row = _row(state="stopped")
        default_tracker().record(
            usage_key(row),
            ServingStats(captured_at=1.0, prompt_tokens=100.0, generation_tokens=400.0),
        )
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            _annotate_runtime_statuses([row], "alice")

        assert row.serving is not None
        self.assertEqual(row.serving.total_tokens, 500.0)
        # The container is gone, so there are no live gauges to show with it.
        self.assertIsNone(row.serving.tokens_per_second)
        self.assertFalse(row.serving.stats.has_readings)

    def test_probed_rows_carry_their_snapshot_onto_the_row(self) -> None:
        fake = types.SimpleNamespace(get=lambda *a, **k: _response(200, _VLLM_METRICS))
        rows = [_row(provider=ComputeProvider.PRIME)]
        with _fake_requests(fake):
            _annotate_runtime_statuses(rows, "alice")
        assert rows[0].serving is not None
        self.assertEqual(rows[0].serving.total_tokens, 4600.0)


class RuntimeBucketTests(unittest.TestCase):
    def setUp(self) -> None:
        default_tracker().reset()
        reset_runtime_health()

    def test_runtime_bucket_prefers_stored_status(self) -> None:
        self.assertEqual(_runtime_bucket(_row(runtime_status="error")), "error")
        self.assertEqual(
            _runtime_bucket_from_modal_state("deploying"), "in_progress"
        )

    def test_deployed_modal_without_observation_is_not_healthy(self) -> None:
        # A deployed app row is not liveness evidence: without an explicit
        # check the bucket must not claim the container is warm.
        self.assertEqual(_runtime_bucket(_row(state="deployed")), "unchecked")
        self.assertEqual(_runtime_bucket(_row(state="running")), "unchecked")

    def test_deployed_prime_without_observation_keeps_live_semantics(self) -> None:
        self.assertEqual(
            _runtime_bucket(_row(state="deployed", provider=ComputeProvider.PRIME)),
            "healthy",
        )

    def test_explicit_modal_verdict_beats_provider_state(self) -> None:
        self.assertEqual(
            _runtime_bucket(_row(state="deployed", runtime_status="healthy")),
            "healthy",
        )


if __name__ == "__main__":
    unittest.main()

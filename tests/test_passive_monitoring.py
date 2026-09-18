"""No-request guarantee: background fleet refreshes never wake Modal GPUs.

Leaving Home or Manage open indefinitely must send no requests to Modal
inference endpoints. Explicit checks (status, warmup) may wake the container
and their verdict is remembered with its age.
"""

from __future__ import annotations

import sys
import time
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from llm_launchpad.core.runtime_health import (
    get_health,
    record_explicit_health,
    reset as reset_health,
)
from llm_launchpad.core.serving_metrics import default_tracker
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, ServingStats
from llm_launchpad.tui.fleet_status import (
    annotate_serving_stats,
    attach_cached_serving_stats,
    can_report_traffic,
    fetch_serving_snapshot,
    is_passive_traffic_row,
    is_passively_monitored,
)


def _row(**overrides: object) -> EndpointInfo:
    fields: dict[str, object] = {
        "name": "vllm-qwen",
        "app_id": "ap-1",
        "state": "deployed",
        "backend": BackendType.VLLM,
        "provider": ComputeProvider.MODAL,
        "web_url": "https://example.modal.run",
        "endpoint_api_key": "secret",
    }
    fields.update(overrides)
    return EndpointInfo(**fields)  # type: ignore[arg-type]


@contextmanager
def _fake_requests(fake: object):  # type: ignore[no-untyped-def]
    with patch.dict(sys.modules, {"requests": fake}):
        yield fake


def _never_called(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    raise AssertionError("background refresh must not contact a Modal runtime")


def _response(status_code: int, text: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(status_code=status_code, text=text)


class PassivePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        default_tracker().reset()
        reset_health()

    def test_modal_is_passively_monitored_prime_is_not(self) -> None:
        self.assertTrue(is_passively_monitored(_row()))
        self.assertFalse(
            is_passively_monitored(_row(provider=ComputeProvider.PRIME))
        )
        self.assertFalse(
            is_passively_monitored(_row(provider=ComputeProvider.VAST))
        )

    def test_background_never_reports_modal_traffic(self) -> None:
        self.assertFalse(can_report_traffic(_row()))
        self.assertTrue(can_report_traffic(_row(), explicit=True))
        self.assertTrue(
            can_report_traffic(_row(provider=ComputeProvider.PRIME))
        )

    def test_fetch_snapshot_returns_before_any_http_for_modal(self) -> None:
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            self.assertIsNone(fetch_serving_snapshot(_row(), "alice"))
            # Explicit checks are the only path allowed to wake the container.
            fake = types.SimpleNamespace(
                get=lambda *a, **k: _response(
                    200, "vllm:generation_tokens_total 10\n"
                )
            )
        with _fake_requests(fake):
            snapshot = fetch_serving_snapshot(_row(), "alice", explicit=True)
        self.assertIsNotNone(snapshot)

    def test_annotate_attaches_banked_totals_without_network(self) -> None:
        row = _row()
        default_tracker().record(
            usage_key_row(row),
            ServingStats(captured_at=1.0, prompt_tokens=100.0, generation_tokens=400.0),
        )
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            annotate_serving_stats([row], "alice")
        assert row.serving is not None
        self.assertEqual(row.serving.total_tokens, 500.0)
        self.assertIsNone(row.serving.tokens_per_second)
        self.assertTrue(is_passive_traffic_row(row))

    def test_mixed_fleet_probes_prime_but_skips_modal(self) -> None:
        modal = _row(name="modal-a")
        prime = _row(
            name="prime-a",
            provider=ComputeProvider.PRIME,
            web_url="https://prime.example.run",
        )
        seen: list[str] = []

        def _get(url: str, **_kwargs: object) -> types.SimpleNamespace:
            seen.append(url)
            return _response(200, "vllm:generation_tokens_total 5\n")

        with _fake_requests(types.SimpleNamespace(get=_get)):
            annotate_serving_stats([modal, prime], "alice")
        self.assertTrue(all("prime.example.run" in url for url in seen))
        self.assertTrue(seen)
        assert prime.serving is not None
        self.assertGreater(prime.serving.total_tokens, 0)

    def test_skipping_a_probe_never_marks_metrics_unsupported(self) -> None:
        from llm_launchpad.core.serving_metrics import usage_key

        row = _row()
        key = usage_key(row)
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            annotate_serving_stats([row], "alice")
        # Still worth asking on an explicit check: the passive skip left no
        # "this runtime serves no metrics" verdict behind.
        self.assertTrue(default_tracker().metrics_supported(key))

    def test_attach_cached_never_touches_the_network(self) -> None:
        rows = [_row(name="a"), _row(name="b")]
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            attach_cached_serving_stats(rows)


def usage_key_row(row: EndpointInfo) -> str:
    from llm_launchpad.core.serving_metrics import usage_key

    return usage_key(row)


class ExplicitHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        default_tracker().reset()
        reset_health()

    def test_explicit_verdict_is_remembered_with_its_age(self) -> None:
        row = _row()
        before = time.time()
        record_explicit_health(row, "healthy", None)
        stored = get_health(row)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.status, "healthy")
        self.assertGreaterEqual(stored.checked_at_epoch, before)
        self.assertEqual(row.runtime_status, "healthy")
        self.assertIsNotNone(row.runtime_checked_at)

    def test_explicit_failure_is_remembered_too(self) -> None:
        row = _row()
        record_explicit_health(row, "error", "HTTP 500")
        stored = get_health(row)
        assert stored is not None
        self.assertEqual((stored.status, stored.detail), ("error", "HTTP 500"))

    def test_passive_refresh_reattaches_without_network(self) -> None:
        from llm_launchpad.tui.screens.main_menu import _annotate_runtime_statuses

        row = _row()
        record_explicit_health(row, "healthy", None)
        checked_at = row.runtime_checked_at
        fresh = _row()
        with _fake_requests(types.SimpleNamespace(get=_never_called)):
            _annotate_runtime_statuses([fresh], "alice")
        self.assertEqual(fresh.runtime_status, "healthy")
        self.assertEqual(fresh.runtime_checked_at, checked_at)


if __name__ == "__main__":
    unittest.main()

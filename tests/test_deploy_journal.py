"""An interrupted deployment keeps billing, so it has to leave a trace."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_launchpad.core.deploy_journal import (
    InFlightDeployment,
    clear_in_flight,
    load_in_flight,
    record_in_flight,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider


def _entry(app_name: str = "llamacpp-app", **overrides: object) -> InFlightDeployment:
    fields: dict[str, object] = {
        "app_name": app_name,
        "provider": ComputeProvider.MODAL.value,
        "backend": BackendType.LLAMACPP.value,
        "gpu_type": "RTX-PRO-6000",
        "gpu_count": 1,
        "price_per_hour_usd": 3.03,
        "started_at_epoch": 1_000_000.0,
    }
    fields.update(overrides)
    return InFlightDeployment(**fields)  # type: ignore[arg-type]


class DeployJournalTests(unittest.TestCase):
    def test_a_recorded_deployment_survives_into_the_next_session(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "in_flight.json"
            record_in_flight(_entry(), path)

            # Nothing cleared it, which is what a SIGKILL looks like.
            restored = load_in_flight(path)

            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].app_name, "llamacpp-app")
            self.assertEqual(restored[0].compute_provider, ComputeProvider.MODAL)
            self.assertEqual(restored[0].backend_type, BackendType.LLAMACPP)

    def test_a_resolved_deployment_leaves_nothing_to_recover(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "in_flight.json"
            record_in_flight(_entry(), path)
            clear_in_flight("llamacpp-app", path)

            self.assertEqual(load_in_flight(path), ())

    def test_clearing_one_deployment_keeps_the_others(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "in_flight.json"
            record_in_flight(_entry("first"), path)
            record_in_flight(_entry("second"), path)

            clear_in_flight("first", path)

            self.assertEqual([row.app_name for row in load_in_flight(path)], ["second"])

    def test_recording_the_same_app_twice_does_not_duplicate_it(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "in_flight.json"
            record_in_flight(_entry(gpu_type="L4"), path)
            record_in_flight(_entry(gpu_type="B200"), path)

            rows = load_in_flight(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].gpu_type, "B200")

    def test_a_missing_or_unreadable_journal_recovers_nothing(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.json"
            self.assertEqual(load_in_flight(missing), ())

            corrupt = Path(directory) / "corrupt.json"
            corrupt.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_in_flight(corrupt), ())

    def test_an_entry_from_a_future_schema_is_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "in_flight.json"
            path.write_text('{"schema_version": 999, "entries": [{}]}', encoding="utf-8")

            self.assertEqual(load_in_flight(path), ())

    def test_exposure_grows_with_elapsed_time(self) -> None:
        entry = _entry()

        # Half an hour on a $3.03/hr placement.
        self.assertAlmostEqual(
            entry.exposure_usd(now=1_000_000.0 + 1800), 1.515, places=3
        )
        self.assertEqual(entry.exposure_usd(now=1_000_000.0), 0.0)

    def test_a_stale_entry_reports_no_exposure(self) -> None:
        entry = _entry()

        # Multiplying an hourly rate by a forgotten timestamp yields a number
        # that alarms without informing, so nothing is claimed past a day.
        self.assertIsNone(entry.exposure_usd(now=1_000_000.0 + 60 * 60 * 25))
        self.assertIsNotNone(entry.exposure_usd(now=1_000_000.0 + 60 * 60 * 23))

    def test_exposure_is_unknown_without_a_price(self) -> None:
        self.assertIsNone(_entry(price_per_hour_usd=None).exposure_usd())


if __name__ == "__main__":
    unittest.main()

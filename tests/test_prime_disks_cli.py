"""The command that removes what a stopped Prime deployment leaves billing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from llm_launchpad.cli.main import app
from llm_launchpad.core.prime_disks import RetainedPrimeDisk


def _disk(disk_id: str = "disk-kept", **overrides: object) -> RetainedPrimeDisk:
    fields: dict[str, object] = {
        "id": disk_id, "name": "llp-cache", "size_gb": 100,
        "location": "eu-north1", "status": "READY", "managed": True,
    }
    fields.update(overrides)
    return RetainedPrimeDisk(**fields)  # type: ignore[arg-type]


class PrimeDiskCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        backend = patch("llm_launchpad.cli.prime._backend")
        self.backend = backend.start().return_value
        self.addCleanup(backend.stop)

    def test_list_names_every_billing_disk_and_how_to_remove_one(self) -> None:
        with patch(
            "llm_launchpad.core.prime_disks.list_retained_prime_disks",
            return_value=[_disk(), _disk("disk-foreign", managed=False)],
        ):
            result = self.runner.invoke(app, ["prime-disks", "list"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("disk-kept", result.output)
        self.assertIn("100 GB", result.output)
        self.assertIn("disk-foreign", result.output)
        # A listing with no next step is what let these go unnoticed.
        self.assertIn("prime-disks delete", result.output)

    def test_list_says_plainly_when_nothing_is_billing(self) -> None:
        with patch(
            "llm_launchpad.core.prime_disks.list_retained_prime_disks", return_value=[]
        ):
            result = self.runner.invoke(app, ["prime-disks", "list"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("nothing is billing", result.output)

    def test_delete_confirms_first_and_names_what_it_destroys(self) -> None:
        with patch(
            "llm_launchpad.core.prime_disks.list_retained_prime_disks", return_value=[_disk()]
        ), patch(
            "llm_launchpad.core.prime_disks.delete_retained_prime_disk",
            return_value="Terminated Prime disk disk-kept.",
        ) as delete:
            result = self.runner.invoke(app, ["prime-disks", "delete", "disk-kept"], input="n\n")

        # Declining leaves the disk alone; the weights on it are not cheap to
        # re-download, so the default answer is no.
        delete.assert_not_called()
        self.assertIn("cached weights", result.output)
        self.assertIn("Left alone", result.output)

    def test_delete_removes_the_disk_once_confirmed(self) -> None:
        with patch(
            "llm_launchpad.core.prime_disks.list_retained_prime_disks", return_value=[_disk()]
        ), patch(
            "llm_launchpad.core.prime_disks.delete_retained_prime_disk",
            return_value="Terminated Prime disk disk-kept.",
        ) as delete:
            result = self.runner.invoke(app, ["prime-disks", "delete", "disk-kept", "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        delete.assert_called_once()
        self.assertIn("Terminated Prime disk disk-kept", result.output)

    def test_an_unknown_disk_is_refused_without_calling_delete(self) -> None:
        with patch(
            "llm_launchpad.core.prime_disks.list_retained_prime_disks", return_value=[_disk()]
        ), patch("llm_launchpad.core.prime_disks.delete_retained_prime_disk") as delete:
            result = self.runner.invoke(app, ["prime-disks", "delete", "nope", "--yes"])

        self.assertEqual(result.exit_code, 1)
        delete.assert_not_called()
        self.assertIn("no Prime disk nope", result.output)

    def test_a_still_attached_disk_reports_why_rather_than_a_traceback(self) -> None:
        with patch(
            "llm_launchpad.core.prime_disks.list_retained_prime_disks", return_value=[_disk()]
        ), patch(
            "llm_launchpad.core.prime_disks.delete_retained_prime_disk",
            side_effect=RuntimeError("Prime disk disk-kept is still attached to a pod."),
        ):
            result = self.runner.invoke(app, ["prime-disks", "delete", "disk-kept", "--yes"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("still attached", result.output)

    def test_an_unreachable_account_fails_the_listing_cleanly(self) -> None:
        with patch(
            "llm_launchpad.core.prime_disks.list_retained_prime_disks",
            side_effect=RuntimeError("Prime API unreachable"),
        ):
            result = self.runner.invoke(app, ["prime-disks", "list"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Prime API unreachable", result.output)

from __future__ import annotations

import unittest

from llm_launchpad.core.storage_resources import (
    format_stop_preview,
    prime_storage_resources,
    stop_effect_for_provider,
    storage_scope_label,
    vast_storage_resources,
)
from llm_launchpad.core.prime_disks import RetainedPrimeDisk
from llm_launchpad.protocol.enums import ComputeProvider


class StorageLifecycleTests(unittest.TestCase):
    def test_modal_stop_keeps_cache_without_billing(self) -> None:
        effect = stop_effect_for_provider(ComputeProvider.MODAL)
        self.assertFalse(effect.destructive)
        self.assertFalse(effect.remains_billable)
        self.assertIn("remains", effect.storage_consequence)

    def test_prime_stop_keeps_disk_billable(self) -> None:
        effect = stop_effect_for_provider(ComputeProvider.PRIME)
        self.assertTrue(effect.remains_billable)
        self.assertFalse(effect.destructive)
        preview = format_stop_preview(ComputeProvider.PRIME, "qwen3")
        self.assertIn("qwen3", preview)
        self.assertIn("keeps billing", preview.lower() + effect.storage_consequence.lower())

    def test_vast_stop_destroys_disk(self) -> None:
        effect = stop_effect_for_provider(ComputeProvider.VAST)
        self.assertTrue(effect.destructive)
        self.assertFalse(effect.remains_billable)
        preview = format_stop_preview(ComputeProvider.VAST, "qwen3")
        self.assertIn("deleted", preview.lower() + effect.storage_consequence.lower())

    def test_prime_resources_sort_attached_first(self) -> None:
        disks = [
            RetainedPrimeDisk(id="disk-b", size_gb=100, managed=True),
            RetainedPrimeDisk(id="disk-a", size_gb=100, managed=True),
        ]
        resources = prime_storage_resources(disks, attached_disk_id="disk-a")
        self.assertEqual(resources[0].resource_id, "disk-a")
        self.assertEqual(resources[0].attached_to, "this deployment")
        self.assertTrue(resources[0].billable_after_stop)

    def test_vast_resources_state_no_survival(self) -> None:
        resources = vast_storage_resources(disk_gb=100, instance_label="vast-123")
        self.assertEqual(len(resources), 1)
        self.assertFalse(resources[0].survives_stop)
        self.assertFalse(resources[0].deletable)

    def test_storage_scope_labels_name_provider(self) -> None:
        self.assertIn("Modal", storage_scope_label(ComputeProvider.MODAL))
        self.assertIn("Prime", storage_scope_label(ComputeProvider.PRIME))
        self.assertIn("Vast", storage_scope_label(ComputeProvider.VAST))

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_launchpad.core.prime_backend import PrimeApiError, PrimeDiskOffer, parse_prime_offer
from llm_launchpad.core.prime_disks import (
    StoredPrimeDisk,
    delete_retained_prime_disk,
    list_retained_prime_disks,
    cache_disk_size_gb,
    disk_matches_gpu_offer,
    load_stored_prime_disks,
    matching_disk_offer,
    remember_prime_disk,
    resolve_prime_offer_and_disk,
    wait_for_prime_disk_ready,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    MemoryEstimate,
    PlacementAssessment,
    PrimeProviderOptions,
    RuntimeTuning,
)


def _offer_payload() -> dict[str, object]:
    return {
        "cloudId": "n3-H100x1",
        "gpuType": "H100_80GB",
        "socket": "PCIe",
        "provider": "hyperstack",
        "region": "canada",
        "dataCenter": "CANADA-1",
        "country": "CA",
        "gpuCount": 1,
        "gpuMemory": 80,
        "disk": {"defaultCount": 100},
        "vcpu": {"defaultCount": 16},
        "memory": {"defaultCount": 180},
        "stockStatus": "Available",
        "security": "secure_cloud",
        "prices": {"onDemand": 1.9, "isVariable": False},
        "images": ["ubuntu_22_cuda_12"],
        "isSpot": False,
    }


class _DiskBackend:
    def __init__(self) -> None:
        self.offer = parse_prime_offer(_offer_payload())
        self.disk_offer = PrimeDiskOffer(
            cloud_id="n3-H100x1",
            provider_name="hyperstack",
            data_center="CANADA-1",
            country="CA",
            region="canada",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=40,
            maximum_size_gb=500,
            raw={},
        )
        self.created: list[tuple[int, str]] = []
        self.deleted: list[str] = []
        self.disk_statuses = ["UNATTACHED"]
        self.list_offer_disk_ids: list[object] = []

    def list_offers(self, **kwargs: object) -> list[object]:
        self.list_offer_disk_ids.append(kwargs.get("disk_id"))
        return [self.offer]

    def list_disk_offers(self) -> list[PrimeDiskOffer]:
        return [self.disk_offer]

    def create_disk(
        self,
        _offer: PrimeDiskOffer,
        *,
        size_gb: int,
        name: str,
    ) -> dict[str, str]:
        self.created.append((size_gb, name))
        return {"id": "disk-new"}

    def get_disk(self, disk_id: str) -> dict[str, str]:
        status = self.disk_statuses.pop(0) if len(self.disk_statuses) > 1 else self.disk_statuses[0]
        return {"id": disk_id, "status": status}

    def delete_disk(self, disk_id: str) -> None:
        self.deleted.append(disk_id)


class PrimeDiskHelperTests(unittest.TestCase):
    def test_wait_for_prime_disk_ready_polls_until_unattached(self) -> None:
        backend = _DiskBackend()
        backend.disk_statuses = ["PROVISIONING", "UNATTACHED"]

        with patch("llm_launchpad.core.prime_disks.time.sleep") as sleep:
            disk = wait_for_prime_disk_ready(backend, "disk-new")

        self.assertEqual(disk["status"], "UNATTACHED")
        sleep.assert_called_once_with(3)

    def test_cache_disk_size_respects_offer_bounds(self) -> None:
        offer = PrimeDiskOffer(
            cloud_id="cloud",
            provider_name="hyperstack",
            data_center="CANADA-1",
            country="CA",
            region="canada",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=200,
            maximum_size_gb=500,
            raw={},
        )
        self.assertEqual(cache_disk_size_gb(offer), 200)

    def test_cache_disk_is_sized_for_the_model_not_the_default(self) -> None:
        """A model larger than the fixed default must still fit its disk."""
        offer = PrimeDiskOffer(
            cloud_id="cloud",
            provider_name="hyperstack",
            data_center="CANADA-1",
            country="CA",
            region="canada",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=20,
            maximum_size_gb=1000,
            raw={},
        )
        memory = MemoryEstimate(
            weights_gb=109.0, kv_cache_gb=1.0, compute_gb=1.0,
            speculative_gb=0.0, reserve_gb=1.0, total_gb=112.0,
        )
        assessment = PlacementAssessment(
            fingerprint="fp", memory=memory, tuning=RuntimeTuning(), performance=(),
        )
        config = DeploymentConfig(placement_assessment=assessment)

        size = cache_disk_size_gb(offer, config)

        self.assertGreaterEqual(size, 109)
        self.assertLess(size, 500)

    def test_cache_disk_floor_holds_without_a_planner_estimate(self) -> None:
        offer = PrimeDiskOffer(
            cloud_id="cloud", provider_name="hyperstack", data_center="CANADA-1",
            country="CA", region="canada", stock_status="Available",
            price_per_gb_hour=0.0001, minimum_size_gb=20, maximum_size_gb=1000,
            raw={},
        )
        self.assertEqual(cache_disk_size_gb(offer, DeploymentConfig()), 100)

    def test_location_matching_ignores_separators_and_case(self) -> None:
        gpu = parse_prime_offer(_offer_payload())
        disk = PrimeDiskOffer(
            cloud_id="n3-H100x1",
            provider_name="HyperStack",
            data_center="canada_1",
            country="CA",
            region="canada",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=20,
            maximum_size_gb=100,
            raw={},
        )
        self.assertTrue(disk_matches_gpu_offer(disk, gpu))

    def test_an_undersized_remembered_disk_is_skipped_not_retried(self) -> None:
        backend = _DiskBackend()
        memory = MemoryEstimate(
            weights_gb=109.0, kv_cache_gb=1.0, compute_gb=1.0,
            speculative_gb=0.0, reserve_gb=1.0, total_gb=112.0,
        )
        assessment = PlacementAssessment(
            fingerprint="fp", memory=memory, tuning=RuntimeTuning(), performance=(),
        )
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            gpu_type="H100_80GB",
            gpu_count=1,
            model_name="Qwen/Qwen3-4B",
            placement_assessment=assessment,
            provider_options=PrimeProviderOptions(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            remember_prime_disk(StoredPrimeDisk(id="disk-small", size_gb=100), path)
            _offer, disk_id, _messages = resolve_prime_offer_and_disk(
                backend, config, required_image="ubuntu_22_cuda_12", path=path,
            )
        # Too small to hold the weights, so it is skipped and a fitted disk
        # is created instead of attaching a disk that cannot help.
        self.assertNotEqual(disk_id, "disk-small")
        self.assertEqual(disk_id, "disk-new")

    def test_a_failed_remembered_disk_is_forgotten(self) -> None:
        class _FailedDiskBackend(_DiskBackend):
            def get_disk(self, disk_id: str) -> dict[str, str]:
                if disk_id == "disk-bad":
                    return {"id": disk_id, "status": "ERROR"}
                return super().get_disk(disk_id)

        backend = _FailedDiskBackend()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            gpu_type="H100_80GB",
            gpu_count=1,
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            remember_prime_disk(StoredPrimeDisk(id="disk-bad", size_gb=500), path)
            _offer, disk_id, _messages = resolve_prime_offer_and_disk(
                backend, config, required_image="ubuntu_22_cuda_12", path=path,
            )
            remaining = load_stored_prime_disks(path)
        self.assertNotIn("disk-bad", [row.id for row in remaining])
        self.assertEqual(disk_id, "disk-new")

    def test_matching_disk_offer_requires_provider_and_datacenter(self) -> None:
        gpu = parse_prime_offer(_offer_payload())
        other = PrimeDiskOffer(
            cloud_id="other",
            provider_name="runpod",
            data_center="US-1",
            country="US",
            region="united_states",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=20,
            maximum_size_gb=100,
            raw={},
        )
        match = PrimeDiskOffer(
            cloud_id="n3-H100x1",
            provider_name="hyperstack",
            data_center="CANADA-1",
            country="CA",
            region="canada",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=20,
            maximum_size_gb=100,
            raw={},
        )
        self.assertIsNone(matching_disk_offer([other], gpu))
        self.assertIs(matching_disk_offer([other, match], gpu), match)
        self.assertTrue(disk_matches_gpu_offer(match, gpu))

    def test_matching_disk_offer_skips_human_readable_out_of_stock_status(self) -> None:
        gpu = parse_prime_offer(_offer_payload())
        unavailable = PrimeDiskOffer(
            cloud_id="n3-H100x1",
            provider_name="hyperstack",
            data_center="CANADA-1",
            country="CA",
            region="canada",
            stock_status="Out of Stock",
            price_per_gb_hour=0.0001,
            minimum_size_gb=20,
            maximum_size_gb=100,
            raw={},
        )
        available = PrimeDiskOffer(
            cloud_id="n3-H100x1",
            provider_name="hyperstack",
            data_center="CANADA-1",
            country="CA",
            region="canada",
            stock_status="Available",
            price_per_gb_hour=0.0002,
            minimum_size_gb=20,
            maximum_size_gb=100,
            raw={},
        )

        self.assertIs(matching_disk_offer([unavailable, available], gpu), available)

    def test_resolve_creates_disk_and_remembers_it(self) -> None:
        backend = _DiskBackend()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            gpu_type="H100_80GB",
            gpu_count=1,
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            offer, disk_id, messages = resolve_prime_offer_and_disk(
                backend,
                config,
                required_image="ubuntu_22_cuda_12",
                path=path,
            )
            stored = load_stored_prime_disks(path)

        self.assertEqual(offer.gpu_type, "H100_80GB")
        self.assertEqual(disk_id, "disk-new")
        # No planner estimate on this config, so the 100 GB floor holds.
        self.assertEqual(backend.created, [(100, "llp-cache")])
        self.assertEqual(backend.list_offer_disk_ids, [None, "disk-new"])
        self.assertEqual([row.id for row in stored], ["disk-new"])
        self.assertTrue(any("Created Prime cache disk" in line for line in messages))

    def test_resolve_reuses_remembered_disk(self) -> None:
        backend = _DiskBackend()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            gpu_type="H100_80GB",
            gpu_count=1,
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            remember_prime_disk(
                StoredPrimeDisk(
                    id="disk-existing",
                    provider_name="hyperstack",
                    data_center="CANADA-1",
                    cloud_id="n3-H100x1",
                ),
                path,
            )
            _offer, disk_id, messages = resolve_prime_offer_and_disk(
                backend,
                config,
                required_image="ubuntu_22_cuda_12",
                path=path,
            )

        self.assertEqual(disk_id, "disk-existing")
        self.assertEqual(backend.created, [])
        self.assertIn("disk-existing", backend.list_offer_disk_ids)
        self.assertTrue(any("Reusing Prime cache disk" in line for line in messages))


class _DiskAccountBackend:
    """A Prime account whose disks outlive whatever attached them."""

    def __init__(self, rows: list[dict[str, object]], *, attached: set[str] | None = None) -> None:
        self.rows = rows
        self.attached = attached or set()
        self.deleted: list[str] = []

    def list_disks(self) -> list[dict[str, object]]:
        return list(self.rows)

    def delete_disk(self, disk_id: str) -> None:
        if disk_id in self.attached:
            raise PrimeApiError(
                "Prime API HTTP 400: Disk is attached to resources. "
                "Terminate them before terminating disk.",
                status_code=400,
            )
        self.deleted.append(disk_id)
        self.rows = [row for row in self.rows if row.get("id") != disk_id]


class RetainedDiskTests(unittest.TestCase):
    """Seeing and removing what a stopped deployment leaves billing."""

    # Shaped like a real Prime row: size and status at the top level, the
    # placement nested inside "info", and the rate as priceHr.
    ROWS: list[dict[str, object]] = [
        {"id": "disk-kept", "name": "llp-cache", "size": 100, "status": "ACTIVE",
         "priceHr": 0.0111, "info": {"dataCenterId": "eu-north1", "country": "NO"}},
        {"id": "disk-foreign", "name": "someone-elses", "size": 40,
         "info": {"dataCenterId": "us-east-1"}},
    ]

    def test_lists_account_disks_and_flags_the_ones_launchpad_made(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            remember_prime_disk(StoredPrimeDisk(id="disk-kept", size_gb=100), path)
            disks = list_retained_prime_disks(_DiskAccountBackend(list(self.ROWS)), path)

        self.assertEqual([d.id for d in disks], ["disk-kept", "disk-foreign"])
        self.assertTrue(disks[0].managed)
        # A disk Launchpad never made still costs money, so it is still shown.
        self.assertFalse(disks[1].managed)
        described = disks[0].describe()
        self.assertIn("100 GB", described)
        # Placement is nested under "info"; reading only the top level left
        # this blank for every disk on the account.
        self.assertIn("eu-north1", described)
        self.assertEqual(disks[0].location, "eu-north1")
        # A disk with no rate shown reads as free, which is the opposite of
        # what this listing exists to say.
        self.assertIn("$0.0111/hr", described)

    def test_a_disk_the_account_no_longer_has_is_forgotten_not_listed(self) -> None:
        """Stale local state is not spend, and must not read as spend."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            remember_prime_disk(StoredPrimeDisk(id="disk-gone", size_gb=100), path)
            disks = list_retained_prime_disks(_DiskAccountBackend([]), path)
            remaining = [disk.id for disk in load_stored_prime_disks(path)]

        self.assertEqual(disks, [])
        self.assertEqual(remaining, [])

    def test_deleting_a_disk_removes_it_and_forgets_it(self) -> None:
        backend = _DiskAccountBackend(list(self.ROWS))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            remember_prime_disk(StoredPrimeDisk(id="disk-kept", size_gb=100), path)
            message = delete_retained_prime_disk(backend, "disk-kept", path)
            remaining = [disk.id for disk in load_stored_prime_disks(path)]

        self.assertEqual(backend.deleted, ["disk-kept"])
        self.assertIn("disk-kept", message)
        self.assertEqual(remaining, [])

    def test_a_still_attached_disk_says_to_retry_not_that_it_is_impossible(self) -> None:
        """Prime refuses until the pod's termination finishes propagating.

        That is a wait, not a dead end -- a live cleanup hit exactly this and
        succeeded seconds later -- so the message has to send the reader back
        rather than leaving them thinking the disk cannot be removed.
        """
        backend = _DiskAccountBackend(list(self.ROWS), attached={"disk-kept"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disks.json"
            remember_prime_disk(StoredPrimeDisk(id="disk-kept", size_gb=100), path)
            with self.assertRaises(RuntimeError) as caught:
                delete_retained_prime_disk(backend, "disk-kept", path)
            # A refused delete must not forget the disk: it is still billing.
            remaining = [disk.id for disk in load_stored_prime_disks(path)]

        self.assertIn("still attached", str(caught.exception))
        self.assertIn("retry", str(caught.exception))
        self.assertEqual(remaining, ["disk-kept"])

    def test_an_empty_disk_id_is_refused_before_any_request(self) -> None:
        backend = _DiskAccountBackend(list(self.ROWS))
        with self.assertRaises(ValueError):
            delete_retained_prime_disk(backend, "  ")
        self.assertEqual(backend.deleted, [])

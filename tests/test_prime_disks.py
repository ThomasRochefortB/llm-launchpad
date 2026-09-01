from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_launchpad.core.prime_backend import PrimeDiskOffer, parse_prime_offer
from llm_launchpad.core.prime_disks import (
    StoredPrimeDisk,
    cache_disk_size_gb,
    disk_matches_gpu_offer,
    load_stored_prime_disks,
    matching_disk_offer,
    remember_prime_disk,
    resolve_prime_offer_and_disk,
    wait_for_prime_disk_ready,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig, PrimeProviderOptions


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

from __future__ import annotations

import pytest

from llm_launchpad.core.prime_backend import select_prime_offer
from llm_launchpad.protocol.models import ComputeOffer


def _offer() -> ComputeOffer:
    return ComputeOffer(
        id="offer-1",
        cloud_id="cloud-1",
        provider_name="provider",
        gpu_type="H100_80GB",
        gpu_count=1,
        gpu_memory_gb=80,
        region="canada",
        security="secure_cloud",
        price_per_hour=2.0,
        stock_status="available",
        images=("ubuntu_22_cuda_12",),
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"gpu_type": "A100_80GB"},
        {"gpu_count": 2},
        {"region": "usa"},
    ],
)
def test_explicit_prime_offer_respects_requested_filters(filters: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="does not match"):
        select_prime_offer([_offer()], offer_id="offer-1", **filters)

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import types
import unittest
from unittest.mock import patch

from llm_launchpad.core import modal_gpu


_DOC_SNIPPET = """
## Specifying GPU type

Modal supports the following values for this parameter:

* `T4`
* `L4`
* `A10`
* `A100`
* `A100-40GB`
* `A100-80GB`
* `L40S`
* `RTX-PRO-6000`
* `H100`/`H100!`
* `H200`
* `B200`
* `B200+`

For instance, to run a Function with eight H100s:

@app.function(gpu="H100:8")
def run_llama_405b_fp8():
    ...

## Specifying GPU count
"""

_PRICING_SNIPPET = """
### Compute costs

GPU Tasks

Nvidia B200

$0.001736 / sec

Nvidia H200

$0.001261 / sec

Nvidia H100

$0.001097 / sec

Nvidia RTX PRO 6000

$0.000842 / sec

Nvidia A100, 80 GB

$0.000694 / sec

Nvidia A100, 40 GB

$0.000583 / sec

Nvidia L40S

$0.000542 / sec

Nvidia A10

$0.000306 / sec

Nvidia L4

$0.000222 / sec

Nvidia T4

$0.000164 / sec

CPU
"""


class ModalGpuTypesTests(unittest.TestCase):
    def setUp(self) -> None:
        modal_gpu._reset_modal_gpu_catalog_cache()

    def tearDown(self) -> None:
        modal_gpu._reset_modal_gpu_catalog_cache()

    def test_parse_modal_gpu_types_from_docs_snippet(self) -> None:
        parsed = modal_gpu._parse_modal_gpu_types(_DOC_SNIPPET)
        self.assertEqual(
            parsed,
            [
                "T4",
                "L4",
                "A10",
                "A100",
                "A100-40GB",
                "A100-80GB",
                "L40S",
                "RTX-PRO-6000",
                "H100",
                "H100!",
                "H200",
                "B200",
                "B200+",
            ],
        )

    def test_fetch_modal_gpu_types_reads_docs_response(self) -> None:
        class FakeResponse:
            status_code = 200
            text = _DOC_SNIPPET

        def fake_get(url: str, timeout: float, headers: dict[str, str]):
            self.assertEqual(url, modal_gpu.MODAL_GPU_GUIDE_URL)
            self.assertEqual(timeout, 7.0)
            self.assertIn("User-Agent", headers)
            return FakeResponse()

        fake_requests = types.SimpleNamespace(get=fake_get)
        with patch.dict("sys.modules", {"requests": fake_requests}):
            values = modal_gpu.fetch_modal_gpu_types(timeout=7.0)
        self.assertIn("H100", values)
        self.assertIn("A100-80GB", values)
        self.assertIn("RTX-PRO-6000", values)
        self.assertIn("B200+", values)
        self.assertNotIn("H100:8", values)

    def test_parse_modal_gpu_pricing_from_pricing_snippet(self) -> None:
        pricing = modal_gpu._parse_modal_gpu_pricing(_PRICING_SNIPPET)
        self.assertAlmostEqual(pricing["B200"], 6.2496)
        self.assertAlmostEqual(pricing["H200"], 4.5396)
        self.assertAlmostEqual(pricing["H100"], 3.9492)
        self.assertAlmostEqual(pricing["RTX-PRO-6000"], 3.0312)
        self.assertAlmostEqual(pricing["A100-80GB"], 2.4984)
        self.assertAlmostEqual(pricing["A100-40GB"], 2.0988)
        self.assertAlmostEqual(pricing["L40S"], 1.9512)
        self.assertAlmostEqual(pricing["A10"], 1.1016)
        self.assertAlmostEqual(pricing["L4"], 0.7992)
        self.assertAlmostEqual(pricing["T4"], 0.5904)
        self.assertAlmostEqual(pricing["A100"], pricing["A100-40GB"])
        self.assertAlmostEqual(pricing["H100!"], pricing["H100"])
        self.assertAlmostEqual(pricing["B200+"], pricing["B200"])

    def test_fetch_modal_gpu_catalog_merges_docs_and_pricing(self) -> None:
        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.status_code = 200
                self.text = text

        def fake_get(url: str, timeout: float, headers: dict[str, str]):
            self.assertEqual(timeout, 7.0)
            self.assertIn("User-Agent", headers)
            if url == modal_gpu.MODAL_GPU_GUIDE_URL:
                return FakeResponse(_DOC_SNIPPET)
            if url == modal_gpu.MODAL_PRICING_URL:
                return FakeResponse(_PRICING_SNIPPET)
            self.fail(f"Unexpected URL: {url}")

        fake_requests = types.SimpleNamespace(get=fake_get)
        with patch.dict("sys.modules", {"requests": fake_requests}):
            catalog = modal_gpu.fetch_modal_gpu_catalog(timeout=7.0)

        pricing_by_gpu = {entry.value: entry.price_per_hour_usd for entry in catalog}
        self.assertAlmostEqual(pricing_by_gpu["B200"] or 0.0, 6.2496)
        self.assertAlmostEqual(pricing_by_gpu["B200+"] or 0.0, 6.2496)
        self.assertAlmostEqual(pricing_by_gpu["H100"] or 0.0, 3.9492)
        self.assertAlmostEqual(pricing_by_gpu["H100!"] or 0.0, 3.9492)
        self.assertAlmostEqual(pricing_by_gpu["A100"] or 0.0, 2.0988)
        self.assertAlmostEqual(pricing_by_gpu["A100-80GB"] or 0.0, 2.4984)
        self.assertAlmostEqual(pricing_by_gpu["RTX-PRO-6000"] or 0.0, 3.0312)

    def test_fetch_modal_gpu_catalog_keeps_types_when_pricing_fetch_fails(self) -> None:
        class FakeResponse:
            def __init__(self, status_code: int, text: str) -> None:
                self.status_code = status_code
                self.text = text

        def fake_get(url: str, timeout: float, headers: dict[str, str]):
            self.assertEqual(timeout, 7.0)
            self.assertIn("User-Agent", headers)
            if url == modal_gpu.MODAL_GPU_GUIDE_URL:
                return FakeResponse(200, _DOC_SNIPPET)
            if url == modal_gpu.MODAL_PRICING_URL:
                return FakeResponse(503, "")
            self.fail(f"Unexpected URL: {url}")

        fake_requests = types.SimpleNamespace(get=fake_get)
        with patch.dict("sys.modules", {"requests": fake_requests}):
            catalog = modal_gpu.fetch_modal_gpu_catalog(timeout=7.0)

        self.assertEqual(catalog[0].value, "T4")
        self.assertIsNone(catalog[0].price_per_hour_usd)

    def test_fetch_modal_gpu_catalog_reuses_cached_result(self) -> None:
        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.status_code = 200
                self.text = text

        requested_urls: list[str] = []

        def fake_get(url: str, **_kwargs: object) -> FakeResponse:
            requested_urls.append(url)
            if url == modal_gpu.MODAL_GPU_GUIDE_URL:
                return FakeResponse(_DOC_SNIPPET)
            return FakeResponse(_PRICING_SNIPPET)

        fake_requests = types.SimpleNamespace(get=fake_get)
        with patch.dict("sys.modules", {"requests": fake_requests}):
            first = modal_gpu.fetch_modal_gpu_catalog()
            second = modal_gpu.fetch_modal_gpu_catalog()

        self.assertEqual(first, second)
        self.assertCountEqual(
            requested_urls,
            [modal_gpu.MODAL_GPU_GUIDE_URL, modal_gpu.MODAL_PRICING_URL],
        )

    def test_fetch_modal_gpu_catalog_shares_inflight_refresh(self) -> None:
        started = threading.Event()
        release = threading.Event()
        call_count = 0

        def fake_fetch(**_kwargs: object) -> list[modal_gpu.ModalGpuSpec]:
            nonlocal call_count
            call_count += 1
            started.set()
            release.wait(timeout=2.0)
            return [modal_gpu.ModalGpuSpec("H100", 3.95)]

        with patch.object(modal_gpu, "_fetch_modal_gpu_catalog_uncached", fake_fetch):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(
                    modal_gpu.fetch_modal_gpu_catalog,
                    force_refresh=True,
                )
                self.assertTrue(started.wait(timeout=1.0))
                second = executor.submit(
                    modal_gpu.fetch_modal_gpu_catalog,
                    force_refresh=True,
                )
                time.sleep(0.05)
                release.set()
                self.assertEqual(first.result(), second.result())

        self.assertEqual(call_count, 1)

    def test_fetch_modal_gpu_catalog_uses_stale_result_after_refresh_error(self) -> None:
        expected = [modal_gpu.ModalGpuSpec("H100", 3.95)]
        with patch.object(
            modal_gpu,
            "_fetch_modal_gpu_catalog_uncached",
            return_value=expected,
        ):
            self.assertEqual(modal_gpu.fetch_modal_gpu_catalog(), expected)

        with patch.object(
            modal_gpu,
            "_fetch_modal_gpu_catalog_uncached",
            side_effect=RuntimeError("temporary outage"),
        ):
            actual = modal_gpu.fetch_modal_gpu_catalog(force_refresh=True)

        self.assertEqual(actual, expected)

    def test_fetch_modal_gpu_types_raises_on_http_error(self) -> None:
        class FakeResponse:
            status_code = 404
            text = ""

        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: FakeResponse())
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                modal_gpu.fetch_modal_gpu_types()


if __name__ == "__main__":
    unittest.main()

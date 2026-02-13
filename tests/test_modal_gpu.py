from __future__ import annotations

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
* `H100`/`H100!`
* `H200`
* `B200`

For instance, to run a Function with eight H100s:

@app.function(gpu="H100:8")
def run_llama_405b_fp8():
    ...

## Specifying GPU count
"""


class ModalGpuTypesTests(unittest.TestCase):
    def test_parse_modal_gpu_types_from_docs_snippet(self) -> None:
        parsed = modal_gpu._parse_modal_gpu_types(_DOC_SNIPPET)
        self.assertEqual(
            parsed,
            ["T4", "L4", "A10", "A100", "A100-40GB", "A100-80GB", "L40S", "H100", "H100!", "H200", "B200"],
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
        self.assertNotIn("H100:8", values)

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

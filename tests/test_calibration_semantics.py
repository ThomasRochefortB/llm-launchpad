"""Calibration measures usage-counted tokens, never transport chunks."""

from __future__ import annotations

import json
import types
import unittest

from llm_launchpad.core import warmup


def _streaming_response(chunks: list[dict]) -> types.SimpleNamespace:
    lines = [
        "data: " + json.dumps(chunk) for chunk in chunks
    ] + ["data: [DONE]"]

    def _iter_lines(*, decode_unicode: bool = True):  # noqa: ANN001, ANN002, ANN202
        _ = decode_unicode
        yield from lines

    return types.SimpleNamespace(status_code=200, iter_lines=_iter_lines)


def _usage_chunk(completion_tokens: int, text: str = "hello") -> dict:
    return {
        "choices": [{"text": text}],
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": 512},
    }


class StreamingCalibrationTests(unittest.TestCase):
    def test_usage_counts_decide_not_chunk_counts(self) -> None:
        response = _streaming_response(
            [_usage_chunk(0, "a"), _usage_chunk(0, "b"), _usage_chunk(64, "c")]
        )
        requests = types.SimpleNamespace(post=lambda *_, **__: response)

        result = warmup._streaming_calibration_request(
            requests,
            endpoint="https://example/v1/completions",
            model="default",
            headers={},
            prompt_tokens=512,
            output_tokens=128,
            timeout=10.0,
        )

        self.assertEqual(result["completion_tokens"], 64.0)

    def test_a_response_without_usage_is_unmeasurable(self) -> None:
        response = _streaming_response([{"choices": [{"text": "hello"}]}])
        requests = types.SimpleNamespace(post=lambda *_, **__: response)

        with self.assertRaisesRegex(RuntimeError, "no usable completion-token"):
            warmup._streaming_calibration_request(
                requests,
                endpoint="https://example/v1/completions",
                model="default",
                headers={},
                prompt_tokens=512,
                output_tokens=128,
                timeout=10.0,
            )

    def test_distinct_workers_get_distinct_prompts(self) -> None:
        seen: list[str] = []

        def _post(*_, **kwargs):  # noqa: ANN002, ANN202
            payload = json.loads(kwargs["data"])
            seen.append(payload["prompt"])
            return _streaming_response([_usage_chunk(32)])

        requests = types.SimpleNamespace(post=_post)
        points = warmup._calibrate_endpoint(
            requests,
            server_url="https://example.modal.run",
            model="default",
            headers={},
            parallel_slots=4,
            price_per_hour_usd=2.0,
            budget_seconds=30.0,
        )

        multi = next(point for point in points if point.concurrency == 2)
        self.assertEqual(len(set(seen)), len(seen))
        self.assertEqual(multi.sample_count, 2)
        self.assertIsNone(multi.p95_latency_seconds)
        self.assertEqual(multi.actual_output_tokens, 64)
        self.assertEqual(multi.completion_reason, "calibration-curve")

    def test_uncounted_points_fail_acceptance(self) -> None:
        from llm_launchpad.protocol.enums import ServingObjective
        from llm_launchpad.protocol.models import PerformancePoint

        accepted, _ = warmup._calibration_is_acceptable(
            (
                PerformancePoint(
                    prompt_tokens=512,
                    output_tokens=128,
                    concurrency=1,
                    output_tokens_per_second=20.0,
                    error_rate=0.0,
                    measured=True,
                ),
            ),
            ServingObjective.GENERAL_PURPOSE,
        )

        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()

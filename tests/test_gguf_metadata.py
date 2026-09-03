from __future__ import annotations

import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core import gguf_metadata
from llm_launchpad.core.gguf_metadata import (
    GgufMtpStatus,
    fetch_gguf_mtp_capability,
    fetch_gguf_serving_metadata,
    parse_gguf_mtp_metadata,
    parse_gguf_serving_metadata,
    select_target_gguf_file,
)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_metadata(rows: list[tuple[str, int, object]]) -> bytes:
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", 3, 0, len(rows)))
    for key, value_type, value in rows:
        payload.extend(_gguf_string(key))
        payload.extend(struct.pack("<I", value_type))
        if value_type == 8:
            payload.extend(_gguf_string(str(value)))
        elif value_type == 4:
            payload.extend(struct.pack("<I", int(value)))
        elif value_type == 9:
            strings = list(value)  # type: ignore[arg-type]
            payload.extend(struct.pack("<IQ", 8, len(strings)))
            for item in strings:
                payload.extend(_gguf_string(str(item)))
        else:
            raise AssertionError(f"Unsupported test value type: {value_type}")
    return bytes(payload)


class GgufMetadataTests(unittest.TestCase):
    def test_serving_parser_reads_architecture_shape(self) -> None:
        payload = _gguf_metadata(
            [
                ("general.architecture", 8, "qwen35"),
                ("qwen35.context_length", 4, 262_144),
                ("qwen35.block_count", 4, 64),
                ("qwen35.embedding_length", 4, 5120),
                ("qwen35.attention.head_count", 4, 40),
                ("qwen35.attention.head_count_kv", 4, 8),
                ("qwen35.attention.key_length", 4, 128),
                ("qwen35.attention.value_length", 4, 128),
            ]
        )

        metadata = parse_gguf_serving_metadata(payload, "model.gguf")

        self.assertEqual(metadata.architecture, "qwen35")
        self.assertEqual(metadata.context_length, 262_144)
        self.assertEqual(metadata.block_count, 64)
        self.assertEqual(metadata.embedding_length, 5120)
        self.assertEqual(metadata.attention_head_count, 40)
        self.assertEqual(metadata.attention_head_count_kv, 8)
        self.assertEqual(metadata.attention_key_length, 128)
        self.assertEqual(metadata.attention_value_length, 128)
        self.assertEqual(metadata.source_file, "model.gguf")

    def test_serving_fetcher_grows_bounded_ranges(self) -> None:
        payload = _gguf_metadata(
            [
                ("tokenizer.ggml.tokens", 9, ["x" * 40]),
                ("general.architecture", 8, "llama"),
                ("llama.context_length", 4, 131_072),
                ("llama.block_count", 4, 32),
            ]
        )
        ranges: list[tuple[int, int]] = []

        def fake_range(
            _repo_id: str,
            _filename: str,
            *,
            revision: str | None,
            start: int,
            end: int,
        ) -> bytes:
            self.assertEqual(revision, "rev")
            ranges.append((start, end))
            return payload[start : end + 1]

        with patch(
            "llm_launchpad.core.gguf_metadata._fetch_hf_file_range",
            side_effect=fake_range,
        ):
            metadata = fetch_gguf_serving_metadata(
                "org/model",
                [SimpleNamespace(rfilename="Q4_K_M/model.gguf")],
                revision="rev",
                quant="Q4_K_M",
                chunk_bytes=48,
                max_bytes=512,
            )

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.context_length, 131_072)  # type: ignore[union-attr]
        self.assertGreater(len(ranges), 1)

    def test_parser_detects_embedded_nextn_without_tensor_data(self) -> None:
        payload = _gguf_metadata(
            [
                ("general.architecture", 8, "qwen35"),
                ("qwen35.nextn_predict_layers", 4, 1),
            ]
        )

        capability = parse_gguf_mtp_metadata(payload, "model.gguf")

        self.assertEqual(capability.status, GgufMtpStatus.SUPPORTED)
        self.assertEqual(capability.nextn_predict_layers, 1)
        self.assertEqual(capability.source_file, "model.gguf")

    def test_parser_confirms_absence_after_complete_metadata_table(self) -> None:
        payload = _gguf_metadata(
            [
                ("general.architecture", 8, "deepseek4"),
                ("tokenizer.ggml.tokens", 9, ["one", "two"]),
            ]
        )

        capability = parse_gguf_mtp_metadata(payload)

        self.assertEqual(capability.status, GgufMtpStatus.UNSUPPORTED)
        self.assertEqual(capability.nextn_predict_layers, 0)

    def test_fetcher_grows_ranges_until_metadata_is_complete(self) -> None:
        payload = _gguf_metadata(
            [
                ("general.architecture", 8, "glm-dsa"),
                ("tokenizer.ggml.tokens", 9, ["x" * 30]),
                ("glm-dsa.nextn_predict_layers", 4, 1),
            ]
        )
        ranges: list[tuple[int, int]] = []

        def fake_range(
            _repo_id: str,
            _filename: str,
            *,
            revision: str | None,
            start: int,
            end: int,
        ) -> bytes:
            self.assertEqual(revision, "rev")
            ranges.append((start, end))
            return payload[start : end + 1]

        siblings = [SimpleNamespace(rfilename="Q4_K_M/model-00001-of-00002.gguf")]
        with patch(
            "llm_launchpad.core.gguf_metadata._fetch_hf_file_range",
            side_effect=fake_range,
        ):
            capability = fetch_gguf_mtp_capability(
                "org/model",
                siblings,
                revision="rev",
                quant="Q4_K_M",
                chunk_bytes=48,
                max_bytes=512,
            )

        self.assertEqual(capability.status, GgufMtpStatus.SUPPORTED)
        self.assertGreater(len(ranges), 1)

    def test_truncated_or_malformed_metadata_is_unknown(self) -> None:
        siblings = [SimpleNamespace(rfilename="model.gguf")]
        with patch(
            "llm_launchpad.core.gguf_metadata._fetch_hf_file_range",
            return_value=b"GGUF\x03",
        ):
            truncated = fetch_gguf_mtp_capability(
                "org/model", siblings, chunk_bytes=5, max_bytes=10
            )
        with patch(
            "llm_launchpad.core.gguf_metadata._fetch_hf_file_range",
            return_value=b"NOPE" * 8,
        ):
            malformed = fetch_gguf_mtp_capability("org/model", siblings)

        self.assertEqual(truncated.status, GgufMtpStatus.UNKNOWN)
        self.assertEqual(malformed.status, GgufMtpStatus.UNKNOWN)

    def test_file_selection_excludes_auxiliary_artifacts_and_prefers_quant(self) -> None:
        siblings = [
            SimpleNamespace(rfilename="MTP/mtp-model-Q4_K_M.gguf"),
            SimpleNamespace(rfilename="model-mmproj.gguf"),
            SimpleNamespace(rfilename="imatrix.dat"),
            SimpleNamespace(rfilename="DSpark/model-Q4_K_M.gguf"),
            SimpleNamespace(rfilename="Q2_K/model-Q2_K-00001-of-00002.gguf"),
            SimpleNamespace(rfilename="Q4_K_M/model-Q4_K_M-00002-of-00002.gguf"),
            SimpleNamespace(rfilename="Q4_K_M/model-Q4_K_M-00001-of-00002.gguf"),
        ]

        selected = select_target_gguf_file(siblings, quant="Q4_K_M")

        self.assertEqual(selected, "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf")

    def test_http_reader_rejects_servers_that_ignore_range(self) -> None:
        response = SimpleNamespace(
            status_code=200,
            headers={"Content-Length": str(10**9)},
            raise_for_status=lambda: None,
            close=lambda: None,
            iter_content=lambda chunk_size: iter((b"large response",)),
        )
        with patch("requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "ignored"):
                gguf_metadata._fetch_hf_file_range(
                    "org/model",
                    "model.gguf",
                    revision=None,
                    start=0,
                    end=1023,
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from llm_launchpad.core.prime_frpc import (
    ensure_cached_frpc,
    frpc_archive_url,
    normalize_frpc_arch,
)


def _frpc_archive(version: str, arch: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=f"frp_{version}_linux_{arch}/frpc")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class PrimeFrpcTests(unittest.TestCase):
    def test_normalize_frpc_arch(self) -> None:
        self.assertEqual(normalize_frpc_arch("x86_64"), "amd64")
        self.assertEqual(normalize_frpc_arch("aarch64"), "arm64")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            normalize_frpc_arch("ppc64le")

    def test_ensure_cached_frpc_downloads_once_and_reuses_binary(self) -> None:
        version = "0.66.0"
        arch = "amd64"
        binary_payload = b"#!/bin/sh\necho frpc\n"
        archive = _frpc_archive(version, arch, binary_payload)
        digest = hashlib.sha256(archive).hexdigest()
        fetches: list[str] = []

        def fetch(url: str) -> bytes:
            fetches.append(url)
            return archive

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "frpc"
            with patch(
                "llm_launchpad.core.prime_frpc.frpc_release_meta",
                return_value=(version, {"amd64": digest, "arm64": "a" * 64}),
            ):
                first = ensure_cached_frpc(arch, cache_dir=cache_dir, fetch=fetch)
                second = ensure_cached_frpc(arch, cache_dir=cache_dir, fetch=fetch)
            self.assertEqual(first.read_bytes(), binary_payload)
            self.assertEqual(second, first)
        self.assertEqual(fetches, [frpc_archive_url(version, arch)])

    def test_ensure_cached_frpc_rejects_checksum_mismatch(self) -> None:
        archive = _frpc_archive("0.66.0", "amd64", b"frpc")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch(
                    "llm_launchpad.core.prime_frpc.frpc_release_meta",
                    return_value=("0.66.0", {"amd64": "b" * 64, "arm64": "a" * 64}),
                ),
                self.assertRaisesRegex(RuntimeError, "checksum"),
            ):
                ensure_cached_frpc(
                    "amd64",
                    cache_dir=Path(tmp) / "bad",
                    fetch=lambda _url: archive,
                )

    def test_ensure_cached_frpc_replaces_corrupted_cached_binary(self) -> None:
        version = "0.66.0"
        arch = "amd64"
        binary_payload = b"#!/bin/sh\necho frpc\n"
        archive = _frpc_archive(version, arch, binary_payload)
        digest = hashlib.sha256(archive).hexdigest()
        fetches: list[str] = []

        def fetch(url: str) -> bytes:
            fetches.append(url)
            return archive

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "frpc"
            cache_dir.mkdir(parents=True)
            cached_binary = cache_dir / "frpc"
            cached_binary.write_bytes(b"corrupted")
            with patch(
                "llm_launchpad.core.prime_frpc.frpc_release_meta",
                return_value=(version, {"amd64": digest, "arm64": "a" * 64}),
            ):
                resolved = ensure_cached_frpc(
                    arch,
                    cache_dir=cache_dir,
                    fetch=fetch,
                )

            self.assertEqual(resolved.read_bytes(), binary_payload)
        self.assertEqual(fetches, [frpc_archive_url(version, arch)])

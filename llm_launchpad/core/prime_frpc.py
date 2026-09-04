"""Client-side Prime Tunnel client cache.

GPU pods download from GitHub slowly and unreliably. Launchpad caches the
pinned frpc build locally, verifies its checksum, and copies it over SSH.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tarfile
from collections.abc import Callable

import requests

from .config import SETTINGS_DIR
from .prime_backend import prime_networking_runtime

PRIME_FRPC_CACHE_DIR = SETTINGS_DIR / "prime" / "frpc"
_FRPC_FETCH = Callable[[str], bytes]


def normalize_frpc_arch(machine: str) -> str:
    """Map ``uname -m`` output to the frp release architecture name."""

    normalized = machine.strip().casefold()
    if normalized in {"x86_64", "amd64"}:
        return "amd64"
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    raise ValueError(f"Unsupported Prime Tunnel architecture: {machine or 'unknown'}")


def frpc_release_meta() -> tuple[str, dict[str, str]]:
    """Return the pinned frpc version and per-arch SHA-256 checksums."""

    networking = prime_networking_runtime()
    version = str(networking.get("frpc_version") or "").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Prime Tunnel frpc version is not configured correctly.")
    checksums = {
        "amd64": str(networking.get("frpc_linux_amd64_sha256") or "").strip(),
        "arm64": str(networking.get("frpc_linux_arm64_sha256") or "").strip(),
    }
    for checksum in checksums.values():
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Prime Tunnel frpc checksum is not configured correctly.")
    return version, checksums


def frpc_archive_url(version: str, arch: str) -> str:
    """Return the GitHub release URL for one pinned frpc archive."""

    return (
        f"https://github.com/fatedier/frp/releases/download/v{version}/"
        f"frp_{version}_linux_{arch}.tar.gz"
    )


def _default_fetch(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def ensure_cached_frpc(
    arch: str,
    *,
    cache_dir: Path | None = None,
    fetch: _FRPC_FETCH | None = None,
) -> Path:
    """Return a local frpc binary, downloading and verifying it once."""

    resolved_arch = normalize_frpc_arch(arch)
    version, checksums = frpc_release_meta()
    expected_sha = checksums[resolved_arch]
    root = cache_dir or (PRIME_FRPC_CACHE_DIR / version / resolved_arch)
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    binary = root / "frpc"
    archive = root / "frp.tar.gz"
    archive_sha = _sha256_file(archive) if archive.is_file() else ""
    if archive_sha != expected_sha:
        payload = (fetch or _default_fetch)(frpc_archive_url(version, resolved_arch))
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_sha:
            raise RuntimeError(
                "Prime Tunnel client download failed checksum verification."
            )
        tmp = archive.with_name(archive.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(archive)

    member_name = f"frp_{version}_linux_{resolved_arch}/frpc"
    with tarfile.open(archive, "r:gz") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError as exc:
            raise RuntimeError("Prime Tunnel archive is missing the frpc binary.") from exc
        if (
            not member.isfile()
            or member.size <= 0
            or member.name.startswith("/")
            or ".." in Path(member.name).parts
        ):
            raise RuntimeError("Prime Tunnel archive member is not a safe frpc binary.")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError("Could not extract the Prime Tunnel client.")
        binary_payload = extracted.read()
        if len(binary_payload) != member.size:
            raise RuntimeError("Prime Tunnel archive contains a truncated frpc binary.")

    try:
        cached_payload = binary.read_bytes() if binary.is_file() else None
    except OSError:
        cached_payload = None
    if cached_payload != binary_payload:
        temporary_binary = binary.with_name(f"{binary.name}.tmp")
        temporary_binary.write_bytes(binary_payload)
        os.chmod(temporary_binary, 0o755)
        temporary_binary.replace(binary)
    else:
        os.chmod(binary, 0o755)
    return binary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

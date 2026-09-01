#!/usr/bin/env python3
"""Generate the llama.cpp architecture manifest for one pinned image build."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import requests

from llm_launchpad.core.runtime_support import (
    DEFAULT_LLAMACPP_IMAGE_REF,
    extract_llamacpp_architectures,
)


DEFAULT_SOURCE_REVISION = "57291f2644af8c9df0dd8d44395881c5bdcf0ecd"
DEFAULT_IMAGE_DIGEST = "sha256:e52c610406cd18714902d1ca3bffadebca4a2a8370faaba8a5be5cc5d5203921"
DEFAULT_RUNTIME_BUILD = "b10689"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "llm_launchpad"
    / "data"
    / "llamacpp_runtime_support.json"
)


def build_manifest(
    source: str,
    *,
    source_revision: str,
    image_ref: str,
    image_digest: str,
    runtime_build: str,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build a deterministic support manifest from llama.cpp source."""

    architectures = extract_llamacpp_architectures(source)
    if not architectures:
        raise ValueError("No llama.cpp architectures were found in llama-arch.cpp")
    source_url = (
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/"
        f"{source_revision}/src/llama-arch.cpp"
    )
    return {
        "schema_version": 1,
        "runtime_id": f"llama.cpp-{runtime_build}-cuda12",
        "runtime_build": runtime_build,
        "image_ref": image_ref,
        "image_digest": image_digest,
        "compatible_image_refs": [
            image_ref.replace(":server-cuda-", ":server-cuda12-"),
        ],
        "source_revision": source_revision,
        "source_url": source_url,
        "generated_at": generated_at or _utc_now_iso(),
        "architectures": architectures,
    }


def fetch_architecture_source(source_revision: str, timeout: float = 20.0) -> str:
    """Fetch llama-arch.cpp at the source revision embedded in the image."""

    url = (
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/"
        f"{source_revision}/src/llama-arch.cpp"
    )
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def write_manifest(payload: dict[str, object], output: Path) -> None:
    """Write a stable, reviewable JSON manifest."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--image-ref", default=DEFAULT_LLAMACPP_IMAGE_REF)
    parser.add_argument("--image-digest", default=DEFAULT_IMAGE_DIGEST)
    parser.add_argument("--runtime-build", default=DEFAULT_RUNTIME_BUILD)
    args = parser.parse_args(argv)

    source = fetch_architecture_source(args.source_revision)
    payload = build_manifest(
        source,
        source_revision=args.source_revision,
        image_ref=args.image_ref,
        image_digest=args.image_digest,
        runtime_build=args.runtime_build,
    )
    write_manifest(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

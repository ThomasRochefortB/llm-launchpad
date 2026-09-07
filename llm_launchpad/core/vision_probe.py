"""Small, bounded image-request verification with no external image hosting."""

from __future__ import annotations

import base64
import json
import shlex
import struct
import zlib

import requests

from ..protocol.enums import VisionVerification
from ..protocol.models import VisionCapabilities
from .shutdown import is_shutting_down


def image_probe_payload(model: str) -> dict:
    """Build a valid 64px red PNG and a native OpenAI image request."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64))
    png += chunk(b"IEND", b"")
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe the image in one short sentence."},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        "max_tokens": 128, "temperature": 0,
    }


def verify_image_request(url: str, model: str | None, api_key: str | None, vision: VisionCapabilities) -> None:
    """Verify transport and nonempty text output, without judging visual accuracy."""
    root = url.rstrip("/").removesuffix("/v1")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    vision.verification = VisionVerification.UNTESTED
    try:
        if is_shutting_down():
            raise RuntimeError("Image verification cancelled.")
        if not model:
            response = requests.get(f"{root}/v1/models", headers=headers, timeout=(10, 30))
            response.raise_for_status()
            model = response.json()["data"][0]["id"]
        response = requests.post(f"{root}/v1/chat/completions", headers=headers,
                                 json=image_probe_payload(model), timeout=(10, 60))
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Image request returned no assistant text.")
        if is_shutting_down():
            raise RuntimeError("Image verification cancelled.")
    except Exception:
        vision.verification = VisionVerification.FAILED
        vision.message = "Image request failed; inspect warmup diagnostics before advertising image input."
        raise
    vision.verification = VisionVerification.PASSED
    vision.message = "Image request verified; visual accuracy has not been benchmarked."


def image_test_command(url: str, model: str) -> str:
    """Return a copyable image probe; credentials remain environment references."""
    endpoint = url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    return (f"curl --fail-with-body {shlex.quote(endpoint)} -H 'Content-Type: application/json' "
            '-H "Authorization: Bearer $LLM_LAUNCHPAD_API_KEY" '
            f"-d {shlex.quote(json.dumps(image_probe_payload(model)))}")

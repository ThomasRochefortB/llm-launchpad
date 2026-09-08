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


VISION_PROBE_FAILED = "vision_probe_failed"


class ImageProbeCancelled(RuntimeError):
    """The probe stopped on shutdown, having learned nothing about the model."""


def is_vision_probe_failure(event: object) -> bool:
    """Report whether a failed warmup was only the image probe.

    The endpoint answered the readiness probe, so it is still worth keeping
    for diagnosis; callers must not treat this like a failed deployment.
    """
    data = getattr(event, "data", None)
    return isinstance(data, dict) and data.get(VISION_PROBE_FAILED) is True


def assistant_text(message: dict) -> str:
    """Extract assistant text from the shapes OpenAI-compatible servers return.

    Thinking models may leave ``content`` empty and put their output in
    ``reasoning_content``, and some servers return content as a list of parts.
    Any of those still proves the server accepted the image.
    """
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
        if parts.strip():
            return parts
    reasoning = message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) and reasoning.strip() else ""


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
        # Generous enough that a thinking model can finish an answer rather
        # than spending the whole budget inside its reasoning block.
        "max_tokens": 512, "temperature": 0,
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
            raise ImageProbeCancelled("Image verification cancelled.")
        if not model:
            response = requests.get(f"{root}/v1/models", headers=headers, timeout=(10, 30))
            response.raise_for_status()
            model = response.json()["data"][0]["id"]
        response = requests.post(f"{root}/v1/chat/completions", headers=headers,
                                 json=image_probe_payload(model), timeout=(10, 60))
        response.raise_for_status()
        if not assistant_text(response.json()["choices"][0]["message"]):
            raise ValueError("Image request returned no assistant text.")
    except ImageProbeCancelled:
        # Nothing was learned, so the previous verification stands.
        vision.message = "Image verification cancelled before it finished."
        raise
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

"""Fetch Modal GPU types from the Modal docs page."""

from __future__ import annotations

import html
import re
from typing import Final

MODAL_GPU_GUIDE_URL: Final[str] = "https://modal.com/docs/guide/gpu"
_REQUEST_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_GPU_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?:[+!])?)"
    r"(?::\d+)?"
    r"(?![A-Za-z0-9_-])"
)
_VALID_GPU_RE: Final[re.Pattern[str]] = re.compile(
    r"(?=.*\d)[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?:[+!])?"
)
_HTML_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


def fetch_modal_gpu_types(
    url: str = MODAL_GPU_GUIDE_URL,
    timeout: float = 10.0,
) -> list[str]:
    """Fetch currently documented Modal GPU type values."""
    from .backend import ModalBackend
    try:
        import requests
    except Exception as exc:
        raise RuntimeError(
            "requests is required to fetch Modal GPU types. Install with: pip install requests"
        ) from exc

    try:
        response = requests.get(url, timeout=timeout, headers=_REQUEST_HEADERS)
    except Exception as exc:
        if ModalBackend.is_shutting_down():
            raise RuntimeError("Shutdown requested") from exc
        raise RuntimeError(f"Failed to fetch Modal GPU docs at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Modal GPU docs returned HTTP {response.status_code} for {url}")

    gpu_types = _parse_modal_gpu_types(response.text)
    if not gpu_types:
        raise RuntimeError(
            "Could not parse GPU types from Modal docs page. The page format may have changed."
        )
    return gpu_types


def _parse_modal_gpu_types(page_text: str) -> list[str]:
    """Parse GPU identifiers from Modal docs HTML or markdown text."""
    content = _strip_html_tags(_slice_gpu_type_section(page_text))
    raw_tokens = _GPU_TOKEN_RE.findall(content)
    if not raw_tokens:
        raw_tokens = _GPU_TOKEN_RE.findall(_strip_html_tags(page_text))

    parsed: list[str] = []
    seen: set[str] = set()
    for raw in raw_tokens:
        token = _normalize_gpu_token(raw)
        if token is None or token in seen:
            continue
        seen.add(token)
        parsed.append(token)
    return parsed


def _slice_gpu_type_section(page_text: str) -> str:
    lower = page_text.lower()
    start = lower.find("specifying gpu type")
    if start == -1:
        start = lower.find("supports the following values for this parameter")
    if start == -1:
        return page_text

    end = lower.find("specifying gpu count", start)
    if end == -1:
        end = lower.find("gpu fallbacks", start)
    if end == -1:
        end = len(page_text)
    return page_text[start:end]


def _normalize_gpu_token(value: str) -> str | None:
    token = value.strip().strip("`'\"").upper()
    if not token:
        return None
    if not _VALID_GPU_RE.fullmatch(token):
        return None
    return token


def _strip_html_tags(value: str) -> str:
    unescaped = html.unescape(value)
    return _HTML_TAG_RE.sub(" ", unescaped)

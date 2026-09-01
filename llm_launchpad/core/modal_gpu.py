"""Fetch Modal GPU types from the Modal docs page."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time

from .shutdown import is_shutting_down

import html
import re
from typing import Final

MODAL_GPU_GUIDE_URL: Final[str] = "https://modal.com/docs/guide/gpu"
MODAL_PRICING_URL: Final[str] = "https://modal.com/pricing"
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
_PRICING_ROW_RE: Final[re.Pattern[str]] = re.compile(
    r"(NVIDIA\s+[A-Z0-9 ,+!-]+?)\s+\$([0-9]+(?:\.[0-9]+)?)\s*/\s*(SEC(?:OND)?|HR|HOUR)\b",
    re.IGNORECASE,
)
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_SECONDS_PER_HOUR: Final[float] = 3600.0
_CATALOG_CACHE_TTL_SECONDS: Final[float] = 600.0


@dataclass(frozen=True)
class ModalGpuSpec:
    """GPU value plus optional Modal pricing metadata."""

    value: str
    price_per_hour_usd: float | None = None


@dataclass
class _CatalogFlight:
    """One shared in-flight catalog request."""

    completed: threading.Event
    result: tuple[ModalGpuSpec, ...] | None = None
    error: BaseException | None = None


_CATALOG_CACHE: dict[
    tuple[str, str],
    tuple[float, tuple[ModalGpuSpec, ...]],
] = {}
_CATALOG_FLIGHTS: dict[tuple[str, str], _CatalogFlight] = {}
_CATALOG_LOCK = threading.Lock()


def fetch_modal_gpu_types(
    url: str = MODAL_GPU_GUIDE_URL,
    timeout: float = 10.0,
) -> list[str]:
    """Fetch currently documented Modal GPU type values."""
    page_text = _fetch_modal_page_text(url=url, timeout=timeout)
    gpu_types = _parse_modal_gpu_types(page_text)
    if not gpu_types:
        raise RuntimeError(
            "Could not parse GPU types from Modal docs page. The page format may have changed."
        )
    return gpu_types


def fetch_modal_gpu_catalog(
    gpu_guide_url: str = MODAL_GPU_GUIDE_URL,
    pricing_url: str = MODAL_PRICING_URL,
    timeout: float = 10.0,
    *,
    force_refresh: bool = False,
) -> list[ModalGpuSpec]:
    """Fetch Modal GPU metadata with a shared TTL cache and stale fallback."""

    cache_key = (gpu_guide_url, pricing_url)
    now = time.monotonic()
    with _CATALOG_LOCK:
        cached = _CATALOG_CACHE.get(cache_key)
        if (
            not force_refresh
            and cached is not None
            and now - cached[0] <= _CATALOG_CACHE_TTL_SECONDS
        ):
            return list(cached[1])
        flight = _CATALOG_FLIGHTS.get(cache_key)
        if flight is None:
            flight = _CatalogFlight(completed=threading.Event())
            _CATALOG_FLIGHTS[cache_key] = flight
            is_leader = True
        else:
            is_leader = False

    if not is_leader:
        flight.completed.wait()
        if flight.result is not None:
            return list(flight.result)
        assert flight.error is not None
        raise RuntimeError(str(flight.error)) from flight.error

    stale_catalog = cached[1] if cached is not None else None
    try:
        catalog = tuple(
            _fetch_modal_gpu_catalog_uncached(
                gpu_guide_url=gpu_guide_url,
                pricing_url=pricing_url,
                timeout=timeout,
            )
        )
        if stale_catalog is not None:
            stale_prices = {
                item.value: item.price_per_hour_usd
                for item in stale_catalog
                if item.price_per_hour_usd is not None
            }
            catalog = tuple(
                ModalGpuSpec(
                    value=item.value,
                    price_per_hour_usd=(
                        item.price_per_hour_usd
                        if item.price_per_hour_usd is not None
                        else stale_prices.get(item.value)
                    ),
                )
                for item in catalog
            )
    except BaseException as exc:
        if not isinstance(exc, Exception):
            flight.error = exc
            raise
        if stale_catalog is None or is_shutting_down():
            flight.error = exc
            raise
        catalog = stale_catalog
    finally:
        with _CATALOG_LOCK:
            if flight.error is None and "catalog" in locals():
                _CATALOG_CACHE[cache_key] = (time.monotonic(), catalog)
                flight.result = catalog
            _CATALOG_FLIGHTS.pop(cache_key, None)
            flight.completed.set()

    return list(catalog)


def _fetch_modal_gpu_catalog_uncached(
    *,
    gpu_guide_url: str,
    pricing_url: str,
    timeout: float,
) -> list[ModalGpuSpec]:
    """Fetch and combine the public GPU guide and pricing pages."""

    with ThreadPoolExecutor(max_workers=2) as executor:
        guide_future = executor.submit(
            _fetch_modal_page_text,
            url=gpu_guide_url,
            timeout=timeout,
        )
        pricing_future = executor.submit(
            _fetch_modal_page_text,
            url=pricing_url,
            timeout=timeout,
        )
        guide_page_text = guide_future.result()
        gpu_types = _parse_modal_gpu_types(guide_page_text)
        if not gpu_types:
            raise RuntimeError(
                "Could not parse GPU types from Modal docs page. The page format may have changed."
            )

        pricing_by_gpu: dict[str, float] = {}
        try:
            pricing_page_text = pricing_future.result()
        except Exception:
            if is_shutting_down():
                raise
        else:
            pricing_by_gpu = _parse_modal_gpu_pricing(pricing_page_text)

    return [
        ModalGpuSpec(
            value=gpu_type,
            price_per_hour_usd=pricing_by_gpu.get(gpu_type),
        )
        for gpu_type in gpu_types
    ]


def _reset_modal_gpu_catalog_cache() -> None:
    """Clear process-local catalog state for tests and explicit refresh tooling."""
    with _CATALOG_LOCK:
        _CATALOG_CACHE.clear()


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


def _parse_modal_gpu_pricing(page_text: str) -> dict[str, float]:
    """Parse base hourly Modal GPU pricing from the pricing page."""
    content = _normalize_whitespace(_strip_html_tags(_slice_gpu_pricing_section(page_text)))
    pricing: dict[str, float] = {}
    for raw_name, raw_price, raw_unit in _PRICING_ROW_RE.findall(content):
        token = _normalize_pricing_gpu_token(raw_name)
        if token is None:
            continue
        hourly_price = float(raw_price)
        if raw_unit.strip().upper().startswith("SEC"):
            hourly_price *= _SECONDS_PER_HOUR
        pricing[token] = hourly_price
    return _with_pricing_aliases(pricing)


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


def _slice_gpu_pricing_section(page_text: str) -> str:
    lower = page_text.lower()
    start = lower.find("gpu tasks")
    if start == -1:
        return page_text

    end_candidates = [
        lower.find("cpu", start),
        lower.find("pricing plans", start),
        lower.find("modal sandbox", start),
    ]
    valid_end_candidates = [idx for idx in end_candidates if idx != -1]
    end = min(valid_end_candidates) if valid_end_candidates else len(page_text)
    return page_text[start:end]


def _normalize_gpu_token(value: str) -> str | None:
    token = value.strip().strip("`'\"").upper()
    if not token:
        return None
    if not _VALID_GPU_RE.fullmatch(token):
        return None
    return token


def _normalize_pricing_gpu_token(value: str) -> str | None:
    normalized = _normalize_whitespace(_strip_html_tags(value)).upper()
    if normalized.startswith("NVIDIA "):
        normalized = normalized[len("NVIDIA ") :].strip()

    a100_match = re.fullmatch(r"([A-Z0-9]+),\s*(\d+)\s*GB", normalized)
    if a100_match is not None:
        return f"{a100_match.group(1)}-{a100_match.group(2)}GB"

    collapsed = normalized.replace(",", "").replace(" ", "-")
    collapsed = re.sub(r"-{2,}", "-", collapsed)
    if not collapsed:
        return None
    if not _VALID_GPU_RE.fullmatch(collapsed):
        return None
    return collapsed


def _with_pricing_aliases(pricing: dict[str, float]) -> dict[str, float]:
    aliased = dict(pricing)
    alias_map = {
        "A100-40GB": ("A100",),
        "H100": ("H100!",),
        "B200": ("B200+",),
    }
    for source, aliases in alias_map.items():
        price = aliased.get(source)
        if price is None:
            continue
        for alias in aliases:
            aliased.setdefault(alias, price)
    return aliased


def _strip_html_tags(value: str) -> str:
    unescaped = html.unescape(value)
    return _HTML_TAG_RE.sub(" ", unescaped)


def _normalize_whitespace(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _fetch_modal_page_text(url: str, timeout: float) -> str:

    try:
        import requests
    except Exception as exc:
        raise RuntimeError(
            "requests is required to fetch Modal GPU metadata. Install with: pip install requests"
        ) from exc

    try:
        response = requests.get(url, timeout=timeout, headers=_REQUEST_HEADERS)
    except Exception as exc:
        if is_shutting_down():
            raise RuntimeError("Shutdown requested") from exc
        raise RuntimeError(f"Failed to fetch Modal metadata page at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Modal metadata page returned HTTP {response.status_code} for {url}")
    return response.text

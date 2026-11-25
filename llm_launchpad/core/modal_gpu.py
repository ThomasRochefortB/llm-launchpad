"""Fetch Modal GPU types from the Modal docs page."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
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
    r"(NVIDIA\s+[A-Z0-9 ,+!-]+?)\s+\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(SEC(?:OND)?|HR|HOUR)\b",
    re.IGNORECASE,
)
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_SECONDS_PER_HOUR: Final[float] = 3600.0
_CATALOG_CACHE_TTL_SECONDS: Final[float] = 600.0


MODAL_GPU_MEMORY_GB: dict[str, float] = {
    "T4": 16.0,
    "L4": 24.0,
    "A10": 24.0,
    "A100": 40.0,
    "A100-40GB": 40.0,
    "A100-80GB": 80.0,
    "L40S": 48.0,
    "RTX-PRO-6000": 96.0,
    "H100": 80.0,
    "H100!": 80.0,
    "H200": 141.0,
    "B200": 180.0,
    "B200+": 180.0,
    "B300": 288.0,
}


def modal_gpu_supports_fast_deploy(gpu_type: str) -> bool:
    """The pinned runtimes do not guarantee the CUDA 13.1 required for a B300."""
    return gpu_type.upper() not in {"B300", "B200+"}


# Modal's GPU guide documents up to 8 GPUs per container for B300, B200, H200,
# H100, A100, L4, T4, and L40S, and up to 4 for A10. RTX-PRO-6000 is omitted
# from the 8-GPU list and 8-GPU requests stall waiting for scheduling capacity.
_MODAL_FOUR_GPU_MAX: Final[frozenset[str]] = frozenset({"A10", "RTX-PRO-6000"})

# Modal schedules multi-GPU containers on power-of-two slices of 4- and 8-GPU
# hosts; non-power-of-two requests (3, 5, 6, 7) are accepted at deploy time
# but stall indefinitely in the scheduler.
_MODAL_GPU_COUNT_LADDER: Final[tuple[int, ...]] = (1, 2, 4, 8)


def modal_gpu_count_limit(gpu_type: str) -> int:
    """Maximum cards per Modal container for a known GPU type."""
    return 4 if gpu_type.upper() in _MODAL_FOUR_GPU_MAX else 8


def modal_gpu_count_options(gpu_type: str) -> tuple[int, ...]:
    """Supported card counts per Modal container for a known GPU type."""
    limit = modal_gpu_count_limit(gpu_type)
    return tuple(count for count in _MODAL_GPU_COUNT_LADDER if count <= limit)


@dataclass(frozen=True)
class ModalGpuSpec:
    """GPU value plus optional Modal pricing metadata."""

    value: str
    price_per_hour_usd: float | None = None
    stale_since: float | None = field(default=None, compare=False)


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
    cache_timestamp: float | None = None
    try:
        catalog = tuple(
            _fetch_modal_gpu_catalog_uncached(
                gpu_guide_url=gpu_guide_url,
                pricing_url=pricing_url,
                timeout=timeout,
            )
        )
        cache_timestamp = time.monotonic()
        if cached is not None:
            stale_prices = {
                item.value: replace(item, stale_since=item.stale_since or cached[0])
                for item in cached[1]
                if item.price_per_hour_usd is not None
            }
            catalog = tuple(
                stale_prices.get(item.value, item) if item.price_per_hour_usd is None else item
                for item in catalog
            )
    except BaseException as exc:
        if not isinstance(exc, Exception):
            flight.error = exc
            raise
        if stale_catalog is None or is_shutting_down():
            flight.error = exc
            raise
        assert cached is not None
        catalog = tuple(replace(item, stale_since=item.stale_since or cached[0]) for item in stale_catalog)
        cache_timestamp = cached[0]
    finally:
        with _CATALOG_LOCK:
            if flight.error is None and "catalog" in locals():
                _CATALOG_CACHE[cache_key] = (
                    cache_timestamp if cache_timestamp is not None else time.monotonic(),
                    catalog,
                )
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
    def _extract(text: str) -> list[str]:
        parsed: list[str] = []
        seen: set[str] = set()
        for raw in _GPU_TOKEN_RE.findall(text):
            token = _normalize_gpu_token(raw)
            if token is None or token in seen:
                continue
            seen.add(token)
            parsed.append(token)
        return parsed

    parsed = _extract(_strip_html_tags(_slice_gpu_type_section(page_text)))
    if not parsed:
        parsed = _extract(_strip_html_tags(page_text))
    return parsed


class _PriceText(HTMLParser):
    """Read accessible price text, ignoring duplicate animated digits."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden_spans = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "span" and (self.hidden_spans or dict(attrs).get("aria-hidden") == "true"):
            self.hidden_spans += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self.hidden_spans:
            self.hidden_spans -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_spans:
            self.parts.append(data)


def _parse_modal_gpu_pricing(page_text: str) -> dict[str, float]:
    """Parse base hourly Modal GPU pricing from the pricing page."""
    parser = _PriceText()
    parser.feed(_slice_gpu_pricing_section(page_text))
    parser.close()
    content = _normalize_whitespace(" ".join(parser.parts))
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

    normalized = {"H100 SXM5": "H100", "H200 SXM": "H200"}.get(normalized, normalized)
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

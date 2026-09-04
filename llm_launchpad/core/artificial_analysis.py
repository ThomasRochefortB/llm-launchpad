"""Client for the Artificial Analysis model-ranking API.

Quick Deploy seeds its catalog from Artificial Analysis: which models are worth
offering, how they rank, and how large they are. That is a self-contained
concern -- an HTTP client, a 24-hour on-disk cache, an auth probe, and the row
parsing that turns their JSON into candidates -- and it used to sit in the
middle of ``quick_deploy_refresh`` alongside catalog generation, so importing
the doctor's auth check dragged the whole catalog builder in with it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, UTC
from pathlib import Path
from threading import Lock
from typing import Any, Literal
import hashlib
import json
import re

import requests

from .artificial_analysis_auth import resolve_artificial_analysis_api_key
from .coerce import clean_string, optional_float
from .config import SETTINGS_DIR


AA_PRO_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models"

AA_FREE_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models/free"

AA_CACHE_PATH = SETTINGS_DIR / "artificial_analysis_models.json"

AA_CACHE_TTL = timedelta(hours=24)

AA_ATTRIBUTION = "Benchmark data sourced from Artificial Analysis: https://artificialanalysis.ai/"

_AA_CACHE_LOCK = Lock()

ModelSizeBucket = Literal["compact", "medium", "large"]

@dataclass(frozen=True)
class AAModelCandidate:
    """Normalized Artificial Analysis Intelligence Index row."""

    aa_model_id: str
    name: str
    slug: str
    creator_name: str
    coding_score: float | None
    intelligence_score: float
    rank: int
    parameter_count_b: float | None = None
    max_context_tokens: int | None = None
    huggingface_url: str | None = None

@dataclass(frozen=True)
class ArtificialAnalysisAuthStatus:
    """Best-effort status for the configured Artificial Analysis API key."""

    authenticated: bool
    tier: str | None = None
    error: str | None = None

@dataclass(frozen=True)
class _AARankings:
    candidates: tuple[AAModelCandidate, ...]
    freshness: Literal["live", "cached"]
    tier: str | None = None

@dataclass(frozen=True)
class _AACacheEntry:
    fetched_at: datetime
    payload: dict[str, Any]
    key_fingerprint: str | None = None

_AA_AUTH_STATUS_BY_KEY: dict[str, ArtificialAnalysisAuthStatus] = {}

def fetch_artificial_analysis_models(
    api_key: str,
    *,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Fetch every AAI language-model page, using Free when Pro is unavailable."""

    normalized_key = api_key.strip()
    if not normalized_key:
        raise ValueError("An Artificial Analysis API key is required")
    try:
        return _fetch_aa_pages(AA_PRO_MODELS_URL, normalized_key, timeout=timeout)
    except _AAHttpError as exc:
        if exc.status_code != 403:
            raise
    return _fetch_aa_pages(AA_FREE_MODELS_URL, normalized_key, timeout=timeout)

def normalize_aa_model_candidates(payload: Any) -> tuple[AAModelCandidate, ...]:
    """Normalize, deduplicate, and rank AAI rows by Intelligence Index."""

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ()

    deduplicated: dict[str, AAModelCandidate] = {}
    for row in rows:
        if not isinstance(row, dict) or _is_explicitly_closed_weight(row):
            continue
        name = clean_string(row.get("name"))
        slug = clean_string(row.get("slug"))
        if not name and not slug:
            continue
        evaluations = row.get("evaluations")
        evaluations = evaluations if isinstance(evaluations, dict) else {}
        coding_score = optional_float(
            evaluations.get("artificial_analysis_coding_index")
        )
        intelligence_score = optional_float(
            evaluations.get("artificial_analysis_intelligence_index")
        )
        if intelligence_score is None:
            continue
        creator = row.get("model_creator")
        creator_name = (
            clean_string(creator.get("name")) if isinstance(creator, dict) else ""
        )
        candidate = AAModelCandidate(
            aa_model_id=clean_string(row.get("id")),
            name=name or slug,
            slug=slug,
            creator_name=creator_name,
            coding_score=coding_score,
            intelligence_score=intelligence_score,
            rank=0,
            parameter_count_b=_extract_parameter_count_b(row),
            max_context_tokens=_extract_context_tokens(row),
            huggingface_url=clean_string(row.get("huggingface_url")) or None,
        )
        key = slug.casefold() or _model_key(name)
        existing = deduplicated.get(key)
        if existing is None or candidate.intelligence_score > existing.intelligence_score:
            deduplicated[key] = candidate

    ranked = sorted(
        deduplicated.values(),
        key=lambda candidate: (-candidate.intelligence_score, candidate.name.casefold()),
    )
    return tuple(replace(candidate, rank=index + 1) for index, candidate in enumerate(ranked))

def _load_aa_rankings(
    *,
    api_key: str,
    cache_path: Path,
    now: datetime | None = None,
) -> _AARankings | None:
    with _AA_CACHE_LOCK:
        return _load_aa_rankings_locked(
            api_key=api_key,
            cache_path=cache_path,
            now=now,
        )

def _load_aa_rankings_locked(
    *,
    api_key: str,
    cache_path: Path,
    now: datetime | None = None,
) -> _AARankings | None:
    current_time = now or datetime.now(UTC)
    key_fingerprint = _api_key_fingerprint(api_key) if api_key else None
    cached = _read_aa_cache(cache_path)
    if cached is not None:
        cache_is_fresh = current_time - cached.fetched_at <= AA_CACHE_TTL
        cache_matches_key = bool(
            key_fingerprint
            and cached.key_fingerprint == key_fingerprint
        )
        if cache_is_fresh and (not api_key or cache_matches_key):
            candidates = normalize_aa_model_candidates(cached.payload)
            if candidates:
                if key_fingerprint:
                    _AA_AUTH_STATUS_BY_KEY[key_fingerprint] = (
                        ArtificialAnalysisAuthStatus(
                            authenticated=True,
                            tier=clean_string(cached.payload.get("tier")) or None,
                        )
                    )
                return _AARankings(
                    candidates=candidates,
                    freshness="cached",
                    tier=clean_string(cached.payload.get("tier")) or None,
                )

    if api_key:
        try:
            payload = fetch_artificial_analysis_models(api_key)
        except Exception as exc:
            if key_fingerprint:
                _AA_AUTH_STATUS_BY_KEY[key_fingerprint] = (
                    ArtificialAnalysisAuthStatus(
                        authenticated=False,
                        error=_artificial_analysis_auth_error(exc),
                    )
                )
            payload = None
        if payload is not None:
            tier = clean_string(payload.get("tier")) or None
            if key_fingerprint:
                _AA_AUTH_STATUS_BY_KEY[key_fingerprint] = (
                    ArtificialAnalysisAuthStatus(
                        authenticated=True,
                        tier=tier,
                    )
                )
            candidates = normalize_aa_model_candidates(payload)
            if candidates:
                _write_aa_cache(
                    cache_path,
                    payload,
                    fetched_at=current_time,
                    api_key=api_key,
                )
                return _AARankings(
                    candidates=candidates,
                    freshness="live",
                    tier=tier,
                )

    if cached is not None:
        candidates = normalize_aa_model_candidates(cached.payload)
        if candidates:
            return _AARankings(
                candidates=candidates,
                freshness="cached",
                tier=clean_string(cached.payload.get("tier")) or None,
            )
    return None

def get_artificial_analysis_auth_status(
    *,
    api_key: str | None = None,
    cache_path: Path | None = None,
) -> ArtificialAnalysisAuthStatus:
    """Validate the configured AAI key through the shared catalog cache path."""

    normalized_key = (
        resolve_artificial_analysis_api_key() if api_key is None else api_key
    ).strip()
    if not normalized_key:
        return ArtificialAnalysisAuthStatus(authenticated=False)
    key_fingerprint = _api_key_fingerprint(normalized_key)
    with _AA_CACHE_LOCK:
        cached_status = _AA_AUTH_STATUS_BY_KEY.get(key_fingerprint)
    if cached_status is not None:
        return cached_status

    _load_aa_rankings(
        api_key=normalized_key,
        cache_path=cache_path or AA_CACHE_PATH,
    )
    with _AA_CACHE_LOCK:
        return _AA_AUTH_STATUS_BY_KEY.get(
            key_fingerprint,
            ArtificialAnalysisAuthStatus(
                authenticated=False,
                error="Artificial Analysis auth check failed",
            ),
        )

class _AAHttpError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Artificial Analysis API returned HTTP {status_code}")
        self.status_code = status_code

def _fetch_aa_pages(url: str, api_key: str, *, timeout: float) -> dict[str, Any]:

    rows: list[Any] = []
    page = 1
    first_payload: dict[str, Any] | None = None
    while page <= 20:
        response = requests.get(
            url,
            headers={"x-api-key": api_key, "Accept": "application/json"},
            params={"page": page},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise _AAHttpError(response.status_code)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Artificial Analysis API returned a non-object payload")
        if first_payload is None:
            first_payload = payload
        page_rows = payload.get("data")
        if not isinstance(page_rows, list):
            raise RuntimeError("Artificial Analysis API returned invalid model data")
        rows.extend(page_rows)
        pagination = payload.get("pagination")
        has_more = (
            bool(pagination.get("has_more"))
            if isinstance(pagination, dict)
            else False
        )
        if not has_more:
            break
        page += 1
    if first_payload is None:
        raise RuntimeError("Artificial Analysis API returned no response")
    combined = dict(first_payload)
    combined["data"] = rows
    combined["pagination"] = {
        "page": 1,
        "page_size": len(rows),
        "total_pages": page,
        "has_more": False,
    }
    return combined

def _read_aa_cache(path: Path) -> _AACacheEntry | None:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        fetched_at_text = clean_string(envelope.get("fetched_at"))
        payload = envelope.get("payload")
        if not fetched_at_text or not isinstance(payload, dict):
            return None
        fetched_at = datetime.fromisoformat(fetched_at_text.replace("Z", "+00:00"))
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        return _AACacheEntry(
            fetched_at=fetched_at.astimezone(UTC),
            payload=payload,
            key_fingerprint=clean_string(envelope.get("key_fingerprint")) or None,
        )
    except Exception:
        return None

def _write_aa_cache(
    path: Path,
    payload: dict[str, Any],
    *,
    fetched_at: datetime,
    api_key: str | None = None,
) -> None:
    envelope = {
        "schema_version": 1,
        "fetched_at": fetched_at.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "payload": payload,
    }
    if api_key:
        envelope["key_fingerprint"] = _api_key_fingerprint(api_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(envelope, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except Exception:
        return

def _api_key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()

def _artificial_analysis_auth_error(exc: Exception) -> str:
    if isinstance(exc, _AAHttpError):
        if exc.status_code == 401:
            return "Invalid Artificial Analysis API key"
        if exc.status_code == 429:
            return "Artificial Analysis API rate limit exceeded"
    detail = " ".join(str(exc).split()).strip()
    return detail or "Artificial Analysis auth check failed"

def _reset_artificial_analysis_auth_status_cache() -> None:
    """Reset in-memory auth state for tests."""

    with _AA_CACHE_LOCK:
        _AA_AUTH_STATUS_BY_KEY.clear()

def _is_explicitly_closed_weight(row: dict[str, Any]) -> bool:
    licensing = row.get("licensing")
    if isinstance(licensing, dict) and licensing.get("is_open_weights") is False:
        return True
    for key, value in _iter_key_values(row):
        normalized_key = key.strip().casefold()
        if normalized_key in {
            "open_weights",
            "open_weight",
            "is_open_weight",
            "is_open_weights",
            "weights_available",
        } and value is False:
            return True
    return False

def _extract_parameter_count_b(row: dict[str, Any]) -> float | None:
    parameters = row.get("parameters")
    if isinstance(parameters, dict):
        parsed = _parameter_value_in_billions(parameters.get("total"))
        if parsed is not None:
            return parsed
    for key in ("parameter_count", "parameter_count_b", "parameters_total"):
        parsed = _parameter_value_in_billions(row.get(key))
        if parsed is not None:
            return parsed
    for value in (
        clean_string(row.get("name")),
        clean_string(row.get("slug")),
    ):
        parsed = _parameter_count_from_name(value)
        if parsed is not None:
            return parsed
    return None

def _parameter_value_in_billions(value: Any) -> float | None:
    parsed = optional_float(value)
    if parsed is not None:
        if parsed <= 0:
            return None
        return parsed / 1_000_000_000.0 if parsed > 1_000_000 else parsed
    return _parameter_count_from_name(clean_string(value))

def _parameter_count_from_name(value: str) -> float | None:
    if not value:
        return None
    mixture = re.search(
        r"(?i)(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*([bt])(?:illion|rillion)?\b",
        value,
    )
    if mixture:
        multiplier = 1_000.0 if mixture.group(3).casefold() == "t" else 1.0
        return float(mixture.group(1)) * float(mixture.group(2)) * multiplier
    compact_moe = re.search(
        r"(?i)(?<![a-z0-9])(\d+(?:\.\d+)?)\s*a\s*\d+(?:\.\d+)?\s*b\b",
        value,
    )
    if compact_moe:
        return float(compact_moe.group(1))
    match = re.search(
        r"(?i)(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([bt])(?:illion|rillion)?\b",
        value,
    )
    if match is None:
        return None
    multiplier = 1_000.0 if match.group(2).casefold() == "t" else 1.0
    return float(match.group(1)) * multiplier

def _extract_context_tokens(row: dict[str, Any]) -> int | None:
    for key in (
        "context_window_tokens",
        "max_context_tokens",
        "context_window",
        "max_context_length",
    ):
        parsed = _parse_token_count(row.get(key))
        if parsed is not None:
            return parsed
    return None

def _parse_token_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    text = clean_string(value).casefold().replace(",", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([km]?)", text)
    if not match:
        return None
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2)]
    tokens = int(round(float(match.group(1)) * multiplier))
    return tokens if tokens > 0 else None

def _size_bucket_for_parameters(parameter_count_b: float | None) -> ModelSizeBucket | None:
    if parameter_count_b is None or parameter_count_b <= 0:
        return None
    if parameter_count_b <= 40:
        return "compact"
    if parameter_count_b <= 150:
        return "medium"
    return "large"

def _model_key(value: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", value.casefold())
    cleaned = re.sub(r"(?i)gguf", "", cleaned)
    return re.sub(r"[^a-z0-9]+", "", cleaned)

def _iter_key_values(node: Any) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            values.append((str(key), value))
            values.extend(_iter_key_values(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(_iter_key_values(value))
    return values

"""Remember which benchmark models do and do not have GGUF weights.

Matching a benchmark row to an Unsloth GGUF repository costs Hugging Face
requests: one canonical probe, then up to four searches. A *failed* match
costs the most and is by far the most common, because the benchmark feed is
mostly API-only models -- and that answer never changes. Re-deriving it every
build spent the bulk of a quota of 1000 API requests per five minutes on
rediscovering that Claude and GPT do not ship weights, which left nothing for
the open models further down the ranking and eventually throttled the build
into publishing a short catalog.

So both answers are written down. A hit is stable for a long time: repository
ids do not move. A miss is stable for much less, because a model released
today may be quantized next week.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
import json
from pathlib import Path
from threading import Lock
from typing import Any

from .diagnostics import log_debug

SCHEMA_VERSION = 1

# A resolved repository id does not move; re-checking it buys nothing but a
# request. A miss is re-checked far sooner, because "no GGUF yet" is a fact
# about a point in time rather than about the model.
HIT_TTL = timedelta(days=30)
MISS_TTL = timedelta(days=3)

# Entries are small and the feed is finite; this only stops unbounded growth
# as models come and go across many months.
MAX_ENTRIES = 4000


@dataclass(frozen=True)
class RepoMatch:
    """A remembered answer, and whether it is still fresh enough to reuse."""

    repo_id: str | None
    checked_at: datetime

    def is_fresh(self, now: datetime) -> bool:
        ttl = HIT_TTL if self.repo_id else MISS_TTL
        return now - self.checked_at < ttl


def _store_path() -> Path:
    from .config import SETTINGS_DIR as current_settings_dir

    return current_settings_dir / "hf_repo_matches.json"


class RepoMatchStore:
    """A load-once, save-once record of match results for one catalog build.

    Loading and saving happen on the build thread, but answers are recorded
    from the resolution workers as they land, so mutation is locked.
    """

    def __init__(self, entries: dict[str, RepoMatch] | None = None) -> None:
        self._entries: dict[str, RepoMatch] = dict(entries or {})
        self._dirty = False
        self._lock = Lock()

    @classmethod
    def load(cls, *, path: Path | None = None) -> RepoMatchStore:
        target = path or _store_path()
        try:
            envelope = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(envelope, dict) or envelope.get("schema_version") != SCHEMA_VERSION:
            return cls()
        raw_entries = envelope.get("entries")
        if not isinstance(raw_entries, dict):
            return cls()
        entries: dict[str, RepoMatch] = {}
        for key, raw in raw_entries.items():
            match = _match_from_dict(raw)
            if match is not None:
                entries[key] = match
        return cls(entries)

    def get(self, model_key: str, *, now: datetime | None = None) -> RepoMatch | None:
        """Return a remembered answer, or None when absent or stale."""

        if not model_key:
            return None
        with self._lock:
            match = self._entries.get(model_key)
        if match is None:
            return None
        return match if match.is_fresh(now or datetime.now(UTC)) else None

    def record(
        self,
        model_key: str,
        repo_id: str | None,
        *,
        now: datetime | None = None,
    ) -> None:
        """Write down an answer that was just paid for."""

        if not model_key:
            return
        entry = RepoMatch(repo_id=repo_id or None, checked_at=now or datetime.now(UTC))
        with self._lock:
            self._entries[model_key] = entry
            self._dirty = True

    def save(self, *, path: Path | None = None, now: datetime | None = None) -> None:
        """Persist, newest first, dropping anything past the cap."""

        with self._lock:
            if not self._dirty:
                return
            snapshot = dict(self._entries)
        current = now or datetime.now(UTC)
        ordered = sorted(
            snapshot.items(), key=lambda item: item[1].checked_at, reverse=True
        )[:MAX_ENTRIES]
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": current.isoformat().replace("+00:00", "Z"),
            "entries": {key: _match_to_dict(match) for key, match in ordered},
        }
        target = path or _store_path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = target.with_suffix(f"{target.suffix}.tmp")
            temporary_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
            temporary_path.replace(target)
            with self._lock:
                self._dirty = False
        except Exception as exc:
            # A cache that cannot be written costs requests, never correctness.
            log_debug(f"Could not persist Hugging Face repo matches: {exc}")


def _match_to_dict(match: RepoMatch) -> dict[str, Any]:
    return {
        "repo_id": match.repo_id,
        "checked_at": match.checked_at.isoformat().replace("+00:00", "Z"),
    }


def _match_from_dict(raw: Any) -> RepoMatch | None:
    if not isinstance(raw, dict):
        return None
    checked_at = raw.get("checked_at")
    if not isinstance(checked_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    repo_id = raw.get("repo_id")
    if repo_id is not None and not isinstance(repo_id, str):
        return None
    return RepoMatch(repo_id=repo_id or None, checked_at=parsed)

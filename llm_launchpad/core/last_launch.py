"""Remember the last model a user deployed so the home screen can offer it again.

Only the choice is stored -- the catalog model and the hardware shape -- never
the quote. Prices and marketplace offers churn between sessions, so a relaunch
re-prices the same shape live instead of replaying a stale plan.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import SETTINGS_DIR

LAST_LAUNCH_PATH = SETTINGS_DIR / "last_launch.json"


@dataclass(frozen=True)
class LastLaunch:
    model_id: str
    display_name: str
    provider: str
    gpu_type: str
    gpu_count: int
    price_per_hour_usd: float | None = None
    launched_at: float = 0.0


def save_last_launch(launch: LastLaunch, path: Path | None = None) -> None:
    target = path or LAST_LAUNCH_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(launch)
    payload["launched_at"] = launch.launched_at or time.time()
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_last_launch(path: Path | None = None) -> LastLaunch | None:
    target = path or LAST_LAUNCH_PATH
    try:
        raw: Any = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    model_id = str(raw.get("model_id") or "").strip()
    display_name = str(raw.get("display_name") or "").strip()
    if not model_id or not display_name:
        return None
    price = raw.get("price_per_hour_usd")
    try:
        gpu_count = max(1, int(raw.get("gpu_count") or 1))
    except (TypeError, ValueError):
        gpu_count = 1
    return LastLaunch(
        model_id=model_id,
        display_name=display_name,
        provider=str(raw.get("provider") or ""),
        gpu_type=str(raw.get("gpu_type") or ""),
        gpu_count=gpu_count,
        price_per_hour_usd=float(price) if isinstance(price, (int, float)) else None,
        launched_at=float(raw.get("launched_at") or 0.0),
    )

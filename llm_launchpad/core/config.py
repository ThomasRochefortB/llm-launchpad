"""Settings persistence and configuration management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..protocol.models import LaunchpadSettings

SETTINGS_DIR = Path.home() / ".llm_launchpad"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


class ConfigStore:
    """Load and save user settings from ``~/.llm_launchpad/settings.json``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or SETTINGS_PATH

    def load(self) -> LaunchpadSettings:
        """Load settings, returning defaults when missing or corrupt."""
        if self.path.exists():
            try:
                raw: Dict[str, Any] = json.loads(self.path.read_text())
                return LaunchpadSettings.from_dict(raw)
            except Exception:
                return LaunchpadSettings()
        return LaunchpadSettings()

    def save(self, settings: LaunchpadSettings) -> None:
        """Persist settings. Failures are silently ignored."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(settings.to_dict(), indent=2))
        except Exception:
            pass

"""Settings persistence and configuration management."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

from ..protocol.models import LaunchpadSettings

SETTINGS_DIR = Path.home() / ".llm_launchpad"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


@dataclass(frozen=True)
class ConfigLoadResult:
    """Settings load result with non-fatal diagnostics."""

    settings: LaunchpadSettings
    path: Path
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ConfigSaveResult:
    """Settings save result with failure diagnostics."""

    success: bool
    path: Path
    error: str | None = None


class ConfigStore:
    """Load and save user settings from ``~/.llm_launchpad/settings.json``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or SETTINGS_PATH

    def load(self) -> LaunchpadSettings:
        """Load settings, returning defaults when missing or corrupt."""
        return self.load_result().settings

    def load_result(self) -> ConfigLoadResult:
        """Load settings and include diagnostics for corrupt/unreadable files."""
        if self.path.exists():
            try:
                raw: Dict[str, Any] = json.loads(self.path.read_text())
                return ConfigLoadResult(
                    settings=LaunchpadSettings.from_dict(raw),
                    path=self.path,
                )
            except Exception as exc:
                return ConfigLoadResult(
                    settings=LaunchpadSettings(),
                    path=self.path,
                    error=f"Could not load settings from {self.path}: {exc}",
                )
        return ConfigLoadResult(settings=LaunchpadSettings(), path=self.path)

    def save(self, settings: LaunchpadSettings) -> None:
        """Persist settings. Failures are silently ignored."""
        self.save_result(settings)

    def save_result(self, settings: LaunchpadSettings) -> ConfigSaveResult:
        """Persist settings and return diagnostics on failure."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(settings.to_dict(), indent=2))
        except Exception as exc:
            return ConfigSaveResult(
                success=False,
                path=self.path,
                error=f"Could not save settings to {self.path}: {exc}",
            )
        return ConfigSaveResult(success=True, path=self.path)

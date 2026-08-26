"""Single source of truth for resolving the Modal CLI executable."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_CLI_TIMEOUT_SECONDS = 8.0


def resolve_modal_cli_path() -> str | None:
    env_prefix = Path(sys.prefix)
    candidates = [
        env_prefix / "bin" / "modal",
        env_prefix / "Scripts" / "modal.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("modal")

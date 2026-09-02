"""Shared mouse-mode defaults resolved from environment and session type."""

from __future__ import annotations

import os


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def is_ssh_session() -> bool:
    return bool(os.getenv("SSH_CONNECTION") or os.getenv("SSH_TTY"))


def likely_remote_clipboard_supported() -> bool:
    """Best-effort guess for whether remote clipboard writes can reach the local terminal."""
    marker_values = [
        os.getenv("TERM", ""),
        os.getenv("TERM_PROGRAM", ""),
        os.getenv("LC_TERMINAL", ""),
    ]
    normalized_markers = " ".join(value.strip().lower() for value in marker_values if value)

    if any(
        os.getenv(name)
        for name in (
            "ITERM_SESSION_ID",
            "KITTY_WINDOW_ID",
            "KITTY_PUBLIC_KEY",
            "WEZTERM_EXECUTABLE",
            "GHOSTTY_RESOURCES_DIR",
        )
    ):
        return True

    if any(marker in normalized_markers for marker in ("iterm", "wezterm", "kitty", "ghostty", "vscode")):
        return True

    if "apple_terminal" in normalized_markers or "apple terminal" in normalized_markers:
        return False

    # Over SSH, default to terminal-native selection unless we recognize a terminal
    # that is likely to accept clipboard escape sequences from the remote app.
    return False


def default_tui_mouse_enabled() -> bool:
    """Resolve the default TUI mouse mode from session type and env override."""
    default = True
    if is_ssh_session():
        default = likely_remote_clipboard_supported()
    return parse_bool_env("LLM_LAUNCHPAD_TUI_MOUSE", default)

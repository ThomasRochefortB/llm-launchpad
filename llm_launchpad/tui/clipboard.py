"""Best-effort bridge between the TUI and the host operating-system clipboard."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence

_CLIPBOARD_TIMEOUT_SECONDS = 1.0


def _command_available(command: Sequence[str]) -> bool:
    """Return whether the executable for a clipboard command is installed."""
    executable = command[0]
    if os.getenv("SSH_CONNECTION") or os.getenv("SSH_TTY"):
        # The process clipboard belongs to the remote host. In an SSH session
        # OSC 52 (including the tmux/screen passthrough above) is the useful
        # route to the user's local terminal clipboard.
        return False
    if executable in {"wl-copy", "wl-paste"} and not os.getenv("WAYLAND_DISPLAY"):
        return False
    if executable in {"xclip", "xsel"} and not os.getenv("DISPLAY"):
        return False
    return shutil.which(executable) is not None


def _clipboard_commands(*, write: bool) -> list[tuple[str, ...]]:
    """Return clipboard commands in preference order for this host."""
    if sys.platform == "darwin":
        return [("pbcopy",)] if write else [("pbpaste",)]

    if os.name == "nt":
        if write:
            return [("clip",)]
        return [
            (
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::Out.Write((Get-Clipboard -Raw))",
            ),
            ("pwsh", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"),
        ]

    commands: list[tuple[str, ...]] = []
    if write:
        commands.extend(
            [
                ("wl-copy",),
                ("xclip", "-selection", "clipboard"),
                ("xsel", "--clipboard", "--input"),
                ("termux-clipboard-set",),
            ]
        )
    else:
        commands.extend(
            [
                ("wl-paste", "--no-newline"),
                ("xclip", "-selection", "clipboard", "-o"),
                ("xsel", "--clipboard", "--output"),
                ("termux-clipboard-get",),
            ]
        )
    return commands


def write_system_clipboard(text: str) -> bool:
    """Write text to the first available host clipboard provider."""
    for command in _clipboard_commands(write=True):
        if not _command_available(command):
            continue
        try:
            subprocess.run(
                list(command),
                input=text.encode("utf-8"),
                check=True,
                timeout=_CLIPBOARD_TIMEOUT_SECONDS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return True
    return False


def read_system_clipboard() -> str | None:
    """Read text from the first available host clipboard provider.

    ``None`` means no provider was available or a provider failed. An empty
    string is a valid clipboard value and is therefore returned unchanged.
    """
    for command in _clipboard_commands(write=False):
        if not _command_available(command):
            continue
        try:
            result = subprocess.run(
                list(command),
                check=True,
                timeout=_CLIPBOARD_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return result.stdout
    return None

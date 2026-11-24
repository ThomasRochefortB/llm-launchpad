"""Terminal and desktop completion notifications for long-running operations."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

from ..protocol.models import DeploymentConfig

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f;]+")
_MAX_NOTIFICATION_TEXT = 160


def sanitize_notification_text(text: str, *, max_length: int = _MAX_NOTIFICATION_TEXT) -> str:
    """Strip control characters and OSC delimiters so untrusted text stays safe."""
    cleaned = _CONTROL_CHARS_RE.sub(" ", text or "")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_length:
        cleaned = cleaned[: max(1, max_length - 1)].rstrip() + "…"
    return cleaned


def osc_9_sequence(message: str) -> str:
    """Build an OSC 9 terminal notification sequence (Ghostty, iTerm2, Kitty, WezTerm)."""
    safe = sanitize_notification_text(message)
    return f"\x1b]9;{safe}\x07"


def wrap_terminal_passthrough(
    sequence: str,
    *,
    tmux: bool = False,
    screen: bool = False,
) -> str:
    """Wrap an escape sequence for tmux or GNU screen passthrough when needed."""
    if tmux:
        escaped = sequence.replace("\x1b", "\x1b\x1b")
        return f"\x1bPtmux;{escaped}\x1b\\"
    if screen:
        return f"\x1bP{sequence}\x1b\\"
    return sequence


def terminal_notification_sequence(
    title: str,
    body: str,
    *,
    tmux: bool = False,
    screen: bool = False,
) -> str:
    """Build a terminal alert combining BEL with an OSC 9 desktop notification."""
    safe_title = sanitize_notification_text(title, max_length=60)
    safe_body = sanitize_notification_text(body, max_length=140)
    message = f"{safe_title}: {safe_body}" if safe_title and safe_body else (safe_body or safe_title)
    if not message:
        return ""
    osc = wrap_terminal_passthrough(osc_9_sequence(message), tmux=tmux, screen=screen)
    return f"\x07{osc}"


def detect_multiplexer(env: dict[str, str] | None = None) -> tuple[bool, bool]:
    """Return ``(in_tmux, in_screen)`` from process environment variables."""
    source = os.environ if env is None else env
    in_tmux = bool(source.get("TMUX"))
    in_screen = (not in_tmux) and source.get("TERM", "").startswith("screen")
    return in_tmux, in_screen


def format_deployment_notification(
    config: DeploymentConfig,
    *,
    success: bool,
    detail: str = "",
) -> tuple[str, str]:
    """Build a concise ``(title, body)`` pair for a finished deployment."""
    target = (
        (config.served_model_name or "").strip()
        or (config.model_name or "").strip()
        or (config.repo_id or "").strip()
        or (config.instance_name or "").strip()
        or (config.app_name or "").strip()
        or config.backend.display_name
    )
    provider = config.provider.display_name
    if success:
        return "LLM Launchpad", f"{target} is ready on {provider}"
    reason = sanitize_notification_text(detail, max_length=90)
    if reason:
        return "LLM Launchpad", f"{target} failed on {provider}: {reason}"
    return "LLM Launchpad", f"{target} failed on {provider}"


def emit_terminal_notification(
    title: str,
    body: str,
    *,
    writer: object | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    """Write a BEL + OSC 9 sequence to a terminal writer or stdout TTY."""
    in_tmux, in_screen = detect_multiplexer(env)
    sequence = terminal_notification_sequence(title, body, tmux=in_tmux, screen=in_screen)
    if not sequence:
        return False
    if writer is not None:
        write_fn = getattr(writer, "write", None)
        if not callable(write_fn):
            return False
        try:
            write_fn(sequence)
            flush_fn = getattr(writer, "flush", None)
            if callable(flush_fn):
                flush_fn()
            return True
        except Exception:
            return False
    try:
        if not sys.stdout.isatty():
            return False
        sys.stdout.write(sequence)
        sys.stdout.flush()
        return True
    except Exception:
        return False


def send_desktop_notification(title: str, body: str) -> bool:
    """Best-effort native desktop notification (macOS / Linux); never raises."""
    if os.environ.get("LLM_LAUNCHPAD_NO_DESKTOP_NOTIFY", "").strip().lower() in {"1", "true", "yes"}:
        return False
    safe_title = sanitize_notification_text(title, max_length=60)
    safe_body = sanitize_notification_text(body, max_length=140)
    if not safe_body and not safe_title:
        return False
    try:
        if sys.platform == "darwin" and shutil.which("osascript"):
            script = (
                "on run argv\n"
                "  display notification (item 2 of argv) with title (item 1 of argv)\n"
                "end run"
            )
            subprocess.Popen(
                ["osascript", "-e", script, safe_title, safe_body],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DBUS_SESSION_BUS_ADDRESS")):
                return False
            subprocess.Popen(
                ["notify-send", "--app-name=LLM Launchpad", safe_title, safe_body],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    except Exception:
        return False
    return False

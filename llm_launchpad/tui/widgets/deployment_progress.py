"""Persistent deployment progress summary rendered above the monitor logs."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..deployment_progress import DeploymentProgress
from ...core.deploy_log_summary import format_elapsed


def _stage_chip(label: str, state: str) -> str:
    # Status text carries the meaning; color only reinforces it. Monochrome
    # renders every style identically, so done/active/failed stay distinct.
    if state == "done":
        return f"[success]OK {escape(label)}[/]"
    if state == "active":
        return f"[reverse]>> {escape(label)}[/]"
    if state == "failed":
        return f"[error]XX {escape(label)}[/]"
    if state == "skipped":
        return f"[dim]-- {escape(label)}[/]"
    return f"[dim].. {escape(label)}[/]"


class DeploymentProgressWidget(Vertical):
    """Compact lifecycle summary: stages, elapsed time, and current detail."""

    DEFAULT_CSS = """
    DeploymentProgressWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        border: solid $border;
        background: $surface;
    }
    #deploy-progress-title {
        height: auto;
        text-wrap: wrap;
    }
    #deploy-progress-stages {
        height: auto;
        text-wrap: wrap;
        color: $text;
    }
    #deploy-progress-detail {
        height: auto;
        text-wrap: wrap;
        color: $text-muted;
    }
    #deploy-progress-hint {
        height: auto;
        text-wrap: wrap;
        color: $text-muted;
    }
    """

    def __init__(self, progress: DeploymentProgress | None = None) -> None:
        super().__init__()
        self._progress = progress or DeploymentProgress()

    def compose(self) -> ComposeResult:
        yield Static("", id="deploy-progress-title")
        yield Static("", id="deploy-progress-stages")
        yield Static("", id="deploy-progress-detail")
        yield Static("", id="deploy-progress-hint")

    def update_progress(self, progress: DeploymentProgress) -> None:
        """Re-render from the tracker; safe to call before mount."""
        self._progress = progress
        try:
            title = self.query_one("#deploy-progress-title", Static)
            stages = self.query_one("#deploy-progress-stages", Static)
            detail = self.query_one("#deploy-progress-detail", Static)
            hint = self.query_one("#deploy-progress-hint", Static)
        except Exception:
            return
        elapsed = format_elapsed(progress.elapsed_seconds())
        if progress.done:
            status = "[success]Complete[/]" if progress.success else "[error]Failed[/]"
        else:
            status = "[reverse]Running[/]"
        attempt = f"  [dim]attempt {progress.attempt}[/]" if progress.attempt > 1 else ""
        title.update(f"[bold]{escape(progress.title)}[/]  {status}  [dim]{elapsed} elapsed[/]{attempt}")
        if progress.stages:
            chips = " → ".join(
                _stage_chip(label, state)
                for label, state in zip(progress.stages, progress.stage_states, strict=False)
            )
        else:
            chips = ""
        stages.update(chips)
        current = progress.current_detail.strip() or "Working"
        quiet = progress.quiet_explanation()
        if quiet and not progress.done:
            since = format_elapsed(progress.seconds_since_progress())
            detail.update(f"{escape(current)}\n[dim]Last progress {since} ago. {escape(quiet)}[/]")
        elif progress.error and not progress.success:
            detail.update(f"[error]{escape(progress.error)}[/]")
        else:
            detail.update(f"[dim]{escape(current)}[/]")
        if progress.done:
            hint.update("[dim]Press esc, q or enter to return[/]")
        else:
            hint.update("[dim]Esc Back — deployment continues in the background[/]")

    @property
    def progress(self) -> DeploymentProgress:
        return self._progress

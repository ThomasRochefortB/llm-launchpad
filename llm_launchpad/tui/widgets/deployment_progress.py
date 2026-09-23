"""Persistent deployment progress summary rendered above the monitor logs."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..deployment_progress import DeploymentProgress
from ...core.deploy_log_summary import format_elapsed
from ..visual import STATUS_MARKERS


def _stage_chip(label: str, state: str) -> str:
    # The marker shape carries the meaning; color only reinforces it.
    # Monochrome renders every style identically, so done/active/failed stay
    # distinct through their markers.
    marker, style = STATUS_MARKERS.get(state, STATUS_MARKERS["pending"])
    if state == "active":
        return f"[bold {style}]{marker} {escape(label)}[/]"
    if state in ("done", "failed"):
        return f"[{style}]{marker}[/] {escape(label)}"
    return f"[{style}]{marker} {escape(label)}[/]"


# "›" rather than "→": the arrow is East Asian Width "A" and draws two cells in
# a CJK-configured terminal, shifting every chip after it.
_STAGE_SEPARATOR = " [dim]›[/] "


# Detail text that only repeats the title's own status word.
_SAID_BY_TITLE = frozenset({"complete", "completed", "done", "working"})


class DeploymentProgressWidget(Vertical):
    """Compact lifecycle summary: stages, elapsed time, and current detail."""

    DEFAULT_CSS = """
    DeploymentProgressWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        border: round $foreground 20%;
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
            status = "[$success]Complete[/]" if progress.success else "[$error]Failed[/]"
        else:
            status = "[bold $primary]Running[/]"
        attempt = f"  [dim]attempt {progress.attempt}[/]" if progress.attempt > 1 else ""
        title.update(f"[bold]{escape(progress.title)}[/]  {status}  [dim]{elapsed} elapsed[/]{attempt}")
        # A one-stage operation (status, logs) has nothing to chart: its single
        # chip only restated the title beside it.
        # A class rather than `display`: an inline style would override the
        # stylesheet rules that hide these rows on small viewports.
        stages.set_class(len(progress.stages) <= 1, "hidden")
        if progress.stages:
            chips = _STAGE_SEPARATOR.join(
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
            detail.update(f"[$error]{escape(progress.error)}[/]")
        else:
            detail.update(f"[dim]{escape(current)}[/]")
        # "Complete" under a title that already says Complete says nothing.
        detail.set_class(
            progress.done and progress.success and current.casefold() in _SAID_BY_TITLE,
            "hidden",
        )
        # Finished, the footer and the outcome card name the way out; while
        # running, the hint carries what the footer cannot: leaving is safe.
        hint.set_class(progress.done, "hidden")
        if not progress.done:
            hint.update("[dim]Esc Back — deployment continues in the background[/]")

    @property
    def progress(self) -> DeploymentProgress:
        return self._progress

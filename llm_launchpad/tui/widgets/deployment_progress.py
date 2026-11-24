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
    #deploy-progress-stages, #deploy-progress-next {
        height: auto;
        text-wrap: wrap;
        color: $text;
    }
    #deploy-progress-next {
        display: none;
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
        yield Static("", id="deploy-progress-next")
        yield Static("", id="deploy-progress-detail")
        yield Static("", id="deploy-progress-hint")

    def update_progress(self, progress: DeploymentProgress) -> None:
        """Re-render from the tracker; safe to call before mount."""
        self._progress = progress
        try:
            title = self.query_one("#deploy-progress-title", Static)
            stages = self.query_one("#deploy-progress-stages", Static)
            next_stage = self.query_one("#deploy-progress-next", Static)
            detail = self.query_one("#deploy-progress-detail", Static)
            hint = self.query_one("#deploy-progress-hint", Static)
        except Exception:
            return
        elapsed = format_elapsed(progress.elapsed_seconds())
        attempt = f"  [dim]attempt {progress.attempt}[/]" if progress.attempt > 1 else ""
        style = "$success" if progress.success else "$error" if progress.done else "$primary"
        if not progress.done and progress.headline.startswith(("Waiting for", "Cleaning up")):
            style = "$warning"
        time_label = f"{elapsed} elapsed"
        if not progress.done and len(progress.stages) > 1:
            time_label = f"{format_elapsed(progress.stage_elapsed_seconds())} in step"
        title.update(
            f"[dim]{escape(progress.title)}[/]  [bold {style}]{escape(progress.headline)}[/]"
            f"  [dim]{time_label}[/]{attempt}"
        )
        # A one-stage operation (status, logs) has nothing to chart: its single
        # chip only restated the title beside it.
        # A class rather than `display`: an inline style would override the
        # stylesheet rules that hide these rows on small viewports.
        stages.set_class(len(progress.stages) <= 1, "hidden")
        stages.update(_STAGE_SEPARATOR.join(
            _stage_chip(label, state)
            for label, state in zip(progress.stages, progress.stage_states, strict=True)
        ))
        following = progress.next_stage() if not progress.done else ""
        next_stage.set_class(not following, "hidden")
        next_stage.update(f"[dim]Next[/]  {escape(following)}")
        current = progress.current_detail or progress.quiet_explanation()
        detail_markup = escape(current)
        if progress.done:
            detail_markup = f"[$error]{escape(progress.error)}[/]" if not progress.success else ""
        elif progress.percent is not None:
            filled = progress.percent // 10
            bar = f"[$primary]{'━' * filled}[/][dim]{'─' * (10 - filled)}[/]"
            detail_markup = f"{bar}  {progress.percent}%  {detail_markup}".rstrip()
        timeout = progress.warmup_seconds_remaining()
        if timeout is not None and not progress.done:
            limit = f"Warmup timeout in {format_elapsed(timeout)}"
            detail_markup = f"{detail_markup}  ·  {limit}" if detail_markup else limit
        detail.update(detail_markup)
        detail.set_class(not detail_markup, "hidden")
        # Finished, the footer and the outcome card name the way out; while
        # running, the hint carries what the footer cannot: leaving is safe.
        hint.set_class(progress.done, "hidden")
        if not progress.done:
            hint.update("[dim]Continues in background if you leave[/]")

    @property
    def progress(self) -> DeploymentProgress:
        return self._progress

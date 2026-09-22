"""Monitor screen: real-time log streaming and operation status.

Shows the status header + scrolling log output for any running
operation (deploy, warmup, logs, status, stop).
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from rich.markup import escape
from textual import events
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from ...protocol.enums import BackendType, DeploymentState, OperationType
from ..deployment_progress import DeploymentProgress
from ..deploy_log_summary import (
    SUMMARY_SPINNER_FRAMES,
    DeployLogSummarizer,
    beautify_summary_line,
    classify_summary_kind,
    summary_progress_parts,
)
from ..widgets.deployment_progress import DeploymentProgressWidget
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen
from ..widgets.log_viewer import LogViewer, prune_retained_items
from ..widgets.status_header import StatusHeader
from ..workers import (
    ConnectionSummaryReady,
    EndpointAvailable,
    LogMessage,
    OperationDone,
    OperationError,
    ResourceAllocated,
    StateChanged,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text or "")


# The same three keys leave this screen whatever finished on it. Three
# different sentences describing them ("esc or q to return", "enter or esc to
# return home") read as three different behaviours.
_RETURN_HINT = "Press esc, q or enter to return"


def _labelled_rows_markup(rows: list[tuple[str, str]]) -> str:
    """Render label/value rows with the values in one column.

    Hand-counted padding kept drifting: the connection card lined its values up
    and the result card did not, so `Status  Healthy` and `Test command  curl`
    began at different columns in the same style of panel. Measuring the widest
    label keeps every card aligned and every future row aligned with it.
    """
    if not rows:
        return ""
    width = max(len(label) for label, _ in rows) + 2
    return "\n".join(
        f"[dim]{escape(label)}[/dim]{' ' * (width - len(label))}{escape(value)}"
        for label, value in rows
    )


def _connection_card_markup(payload: dict[str, str]) -> str:
    """Render connection fields for the post-deploy card."""
    def _field(key: str) -> str:
        return (payload.get(key) or "").strip() or "(unavailable)"

    return _labelled_rows_markup(
        [
            ("Base URL", _field("base_url")),
            ("Model ID", _field("model_id")),
            ("Display", _field("display_name")),
            ("API key", (payload.get("api_key") or "").strip() or "none"),
        ]
    )


@dataclass
class _SummaryRow:
    text: str
    kind: str


def _connection_copy_text(payload: dict[str, str]) -> str:
    """Plain-text connection block for the clipboard."""
    lines = [
        f"Base URL: {(payload.get('base_url') or '').strip()}",
        f"Model ID: {(payload.get('model_id') or '').strip()}",
        f"Display name: {(payload.get('display_name') or '').strip()}",
    ]
    api_key = (payload.get("api_key") or "").strip()
    if api_key:
        lines.append(f"API key: {api_key}")
    return "\n".join(line for line in lines if not line.endswith(": "))


def _result_card_markup(rows: list[tuple[str, str]]) -> str:
    """Render result-card fields for a finished status check or benchmark."""
    return _labelled_rows_markup(rows)


class MonitorScreen(CopyEnabledScreen):
    """Full-screen operation monitor with streaming logs."""

    AUTO_FOCUS = "#log-output"

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding(
            "y",
            "copy_text",
            "Copy",
            key_display="y",
            show=True,
        ),
        Binding(
            "ctrl+shift+c,super+c,meta+c,cmd+c,command+c",
            "copy_text",
            "Copy",
            show=False,
        ),
        Binding("pageup", "page_up_log", "Page up", show=True, priority=True),
        Binding("pagedown", "page_down_log", "Page down", show=True, priority=True),
        Binding("end", "resume_follow", "Follow", show=True, priority=True),
        Binding("/", "search_logs", "Search", show=True, priority=True),
        Binding("n", "next_search_match", "Next match", show=False),
        Binding("shift+n", "previous_search_match", "Previous match", show=False),
        Binding("v", "toggle_log_view", "Raw/Summary", show=True),
        Binding("ctrl+l", "clear_log", "Clear log", show=True),
        Binding("enter", "submit_or_finish", "Done", show=True, priority=True),
        Binding("u", "copy_base_url", "Copy URL", show=True),
        Binding("k", "copy_api_key", "Copy key", show=False),
    ]

    _RESULT_TITLES = {
        OperationType.STATUS: "Status check complete",
        OperationType.BENCHMARK: "Benchmark complete",
        OperationType.DEPLOY: "Deploy complete",
        OperationType.WARMUP: "Warmup complete",
        OperationType.LOGS: "Logs complete",
        OperationType.STOP: "Stop complete",
    }

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Prioritize result actions once the operation finishes.

        While running, the footer is about following and searching logs; once
        done, Done/Copy URL/Copy take precedence over log navigation, which
        remains reachable by shortcut and help.
        """
        if self._done and action in {
            "page_up_log", "page_down_log", "resume_follow", "clear_log",
        }:
            return False
        if not self._done and action in {"submit_or_finish", "copy_base_url"}:
            # The result card is hidden until completion; advertising Done and
            # Copy URL beforehand promises buttons that do not exist yet.
            if action == "copy_base_url" and self._connection_payload is None:
                return False
            if action == "submit_or_finish":
                return False
        return super().check_action(action, parameters)

    def __init__(
        self,
        title: str = "Operation",
        deploy_backend: BackendType | None = None,
        summarize_backend_logs: bool = False,
        show_debug_logs: bool = True,
    ) -> None:
        super().__init__()
        self._title = title
        self._done = False
        self._deploy_backend = deploy_backend
        self._summarize_backend_logs = summarize_backend_logs
        self._show_debug_logs = show_debug_logs
        self._current_operation: OperationType | None = None
        self._last_summary_state_detail = ""
        self._last_error_message = ""
        self._following = True
        self._unseen_lines = 0
        self._line_count = 0
        self._search_query = ""
        self._search_current = 0
        self._search_total = 0
        self._success = False
        self._connection_payload: dict[str, str] | None = None
        self._status_result: dict[str, str] = {}
        self._result_rows: list[tuple[str, str]] = []
        self._result_title = "Check complete"
        self._failure_rows: list[tuple[str, str]] = []
        self._failure_title = "Operation failed"
        self._progress = DeploymentProgress()
        self._progress.start(None, title)
        self._progress_tick_timer = None
        self._deploy_summarizer = (
            DeployLogSummarizer(deploy_backend)
            if deploy_backend is not None and summarize_backend_logs and not show_debug_logs
            else None
        )
        self._view_mode = "summary" if self._deploy_summarizer is not None else "raw"
        self._raw_log_lines: list[str] = []
        self._summary_log_lines: list[str] = []
        self._summary_items: list[_SummaryRow] = []
        self._spinner_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="monitor-layout"):
            yield StatusHeader(id="monitor-status-header")
            with Horizontal(id="monitor-toolbar"):
                yield Static(self._title_markup(), id="monitor-title")
                yield Static(self._view_status_markup(), id="monitor-view-status")
            yield DeploymentProgressWidget(self._progress)
            yield Input(
                placeholder="Search logs; Enter closes, n/N navigates",
                id="monitor-search",
                classes="hidden",
            )
            # The outcome card mounts on demand via _mount_outcome_card: only
            # one of connection/result/failure ever exists, because each
            # reserves up to 65% of the height and all three mounted at once
            # overflow short terminals even while hidden.
            yield LogViewer(id="monitor-log-viewer")
        yield FittedFooter()

    def _mount_outcome_card(self, card_id: str) -> VerticalScroll | None:
        """Mount the outcome card if absent; return it, or None on failure.

        Mounting must be scheduled, not done inline: message handlers run
        outside the widget-mounted context, so ``mount`` on children raises
        ``MountError``. ``call_after_refresh`` defers until mounting is legal.
        Callers that only need content set can pass it and return; the card
        appears on the next refresh.
        """
        try:
            return self.query_one(f"#{card_id}", VerticalScroll)
        except Exception:
            pass
        try:
            self.call_after_refresh(self._mount_outcome_card_deferred, card_id)
        except Exception:
            return None
        return None

    def _mount_outcome_card_deferred(self, card_id: str) -> None:
        try:
            self.query_one(f"#{card_id}", VerticalScroll)
            return
        except Exception:
            pass
        try:
            viewer = self.query_one("#monitor-log-viewer", LogViewer)
            layout = self.query_one("#monitor-layout", Vertical)
        except Exception:
            return
        card: VerticalScroll | None = None
        if card_id == "connection-card":
            card = VerticalScroll(id="connection-card")
        elif card_id == "result-card":
            card = VerticalScroll(id="result-card")
        elif card_id == "failure-card":
            card = VerticalScroll(id="failure-card")
        else:
            return
        layout.mount(card, before=viewer)
        if card_id == "connection-card":
            card.mount(
                Static("[bold]Connection[/]", id="connection-card-title"),
                Static("", id="connection-card-body"),
            )
            actions = Horizontal(id="connection-card-actions")
            card.mount(actions)
            actions.mount(
                Button("Copy URL", id="copy-url-btn"),
                Button("Copy API key", id="copy-key-btn"),
                Button("Copy all", id="copy-all-btn"),
                Button("Manage endpoint", id="connection-manage-btn"),
                Button("Done", id="connection-done-btn", variant="primary"),
            )
            self._fill_connection_card()
        elif card_id == "result-card":
            card.mount(
                Static("[bold]Result[/]", id="result-card-title"),
                Static("", id="result-card-body"),
            )
            actions = Horizontal(id="result-card-actions")
            card.mount(actions)
            actions.mount(
                Button("Copy result", id="result-copy-btn"),
                Button("Done", id="result-done-btn", variant="primary"),
            )
            self._fill_result_card()
        elif card_id == "failure-card":
            card.mount(
                Static("[bold]Outcome[/]", id="failure-card-title"),
                Static("", id="failure-card-body"),
            )
            actions = Horizontal(id="failure-card-actions")
            card.mount(actions)
            actions.mount(
                Button("Copy error", id="failure-copy-btn"),
                Button("Manage", id="failure-manage-btn"),
                Button("Done", id="failure-done-btn", variant="primary"),
            )
            self._fill_failure_card()

    def _title_markup(self) -> str:
        """Render a compact operation title."""
        return f"[bold]{escape(self._title)}[/]"

    def _view_status_markup(self) -> str:
        """Render live follow and line-count state for the log viewport."""
        line_label = "line" if self._line_count == 1 else "lines"
        if self._following:
            state = "[reverse]FOLLOWING[/]"
        else:
            new_label = "line" if self._unseen_lines == 1 else "lines"
            state = (
                f"[warning]PAUSED[/]  [warning]· {self._unseen_lines} new {new_label}[/]"
            )
        search = ""
        if self._search_query:
            query = self._search_query[:18]
            search = (
                f"  [dim]· /{escape(query)} "
                f"{self._search_current}/{self._search_total}[/dim]"
            )
        return (
            f"{state}  [dim]· {self._line_count} {line_label} · "
            f"{self._view_mode.upper()}[/dim]{search}"
        )

    def refresh_copy_help(self) -> None:
        """Refresh compact monitor chrome."""
        self.query_one("#monitor-title", Static).update(self._title_markup())
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )

    @property
    def log_viewer(self) -> LogViewer:
        return self.query_one("#monitor-log-viewer", LogViewer)

    @property
    def status_header(self) -> StatusHeader:
        return self.query_one("#monitor-status-header", StatusHeader)

    @property
    def _summary_mode_enabled(self) -> bool:
        return self._deploy_summarizer is not None

    # -- Message handlers --

    def on_mount(self) -> None:
        # Every caller knows the backend it is operating on, so name it even
        # when there is no deploy summary to drive. Leaving it unset showed
        # "backend: --" for the whole of logs, status, benchmark and stop.
        if self._deploy_backend is not None:
            self.status_header.update_from_event(backend=self._deploy_backend)
        self._progress_tick_timer = None
        self._refresh_progress()
        if self._summary_mode_enabled:
            self.status_header.update_from_event(
                state=DeploymentState.QUEUED,
                backend=self._deploy_backend,
                operation=OperationType.DEPLOY,
                detail="Preparing deployment",
            )
            self._ensure_progress_operation(OperationType.DEPLOY)
            self._progress.on_state(DeploymentState.QUEUED, "Preparing deployment")
            self._refresh_progress()
            self._append_log_line("Preparing deployment", summary=True, raw=False)
            self._last_summary_state_detail = "Preparing deployment"
            self.set_interval(0.25, self._tick_summary_spinner)

    def _ensure_progress_operation(self, operation: OperationType | None) -> None:
        """Adopt the operation once the event stream names it.

        The screen is built with only a title; the first state/log event names
        the operation. Restarting the tracker then costs <1s of elapsed time
        and buys the correct stage row for deploy vs. status/stop/benchmark.
        """
        if operation is None or operation == self._progress.operation:
            return
        if self._progress.operation is None:
            title = self._progress.title or self._title
            self._progress.start(operation, title)
            self._refresh_progress()

    def _refresh_progress(self) -> None:
        try:
            widget = self.query_one(DeploymentProgressWidget)
        except Exception:
            return
        try:
            widget.update_progress(self._progress)
        except Exception:
            pass
        # Elapsed time must stay fresh without backend events, but a 1s
        # repaint timer keeps the whole screen dirty forever. Only repaint
        # when something visible can change: the elapsed label ticks each
        # minute, and quiet-stage hints appear after a long silence.
        try:
            timer = getattr(self, "_progress_tick_timer", None)
            if self._done or self._progress.done:
                if timer is not None:
                    timer.stop()
                    self._progress_tick_timer = None
            elif timer is None:
                self._progress_tick_timer = self.set_interval(5.0, self._tick_progress)
        except Exception:
            pass

    def _tick_progress(self) -> None:
        """Repaint elapsed/quiet hints while an operation runs."""
        if self._done:
            return
        try:
            widget = self.query_one(DeploymentProgressWidget)
            widget.update_progress(self._progress)
        except Exception:
            pass

    def on_log_message(self, message: LogMessage) -> None:
        prefix = "stderr | " if message.stream == "stderr" else ""
        cleaned = _strip_ansi(message.line)
        self._capture_result_lines(cleaned)
        if message.is_milestone and cleaned.strip():
            self._progress.on_milestone(cleaned.strip())
            self._refresh_progress()
        raw_line = (
            message.line
            if self._show_debug_logs and self._summarize_backend_logs
            else cleaned
        )
        self._append_log_line(
            f"{prefix}{raw_line}" if prefix else raw_line,
            raw=True,
            summary=False,
        )
        if self._summary_mode_enabled:
            assert self._deploy_summarizer is not None
            for line in self._deploy_summarizer.transform(cleaned, self._current_operation):
                stripped = _strip_ansi(line).strip()
                if stripped:
                    self._progress.on_milestone(stripped)
                    self._refresh_progress()
                self._append_log_line(
                    f"{prefix}{line}" if prefix else line,
                    summary=True,
                    raw=False,
                )
            return

    def on_state_changed(self, message: StateChanged) -> None:
        if message.operation is not None:
            self._current_operation = message.operation
            self._ensure_progress_operation(message.operation)
        self.status_header.update_from_event(
            state=message.state,
            operation=message.operation,
            detail=message.detail,
        )
        detail = _strip_ansi(message.detail).strip()
        self._progress.on_state(message.state, detail)
        self._refresh_progress()
        if (
            self._summary_mode_enabled
            and detail
            and detail != self._last_summary_state_detail
        ):
            self._last_summary_state_detail = detail
            assert self._deploy_summarizer is not None
            for line in self._deploy_summarizer.transform_state(
                detail, self._current_operation
            ):
                self._append_log_line(line, summary=True, raw=False)

    def on_resource_allocated(self, message: ResourceAllocated) -> None:
        self._progress.on_resource_allocated(
            f"Resource allocated ({message.app_id})" if message.app_id else "Resource allocated"
        )
        self._refresh_progress()

    def on_endpoint_available(self, message: EndpointAvailable) -> None:
        _ = message.endpoint
        self._progress.on_endpoint_available()
        self._refresh_progress()

    def on_connection_summary_ready(self, message: ConnectionSummaryReady) -> None:
        self._connection_payload = dict(message.payload)
        self._progress.on_connection_ready("Endpoint verified")
        self._refresh_progress()
        if self._success:
            self._show_connection_card()

    # An operation that ended is no longer publishing, deploying or warming up.
    # Without this the context bar kept its last in-progress state forever, so a
    # finished deploy still read "state: publishing · Publishing verified
    # endpoint" while the log below said the operation was complete.
    #
    # Only the operations below leave the deployment somewhere new. Everything
    # else -- logs, status, benchmark, the storage operations -- reads or edits
    # a cache and ends at idle, because nothing is running any more. They are
    # not listed individually: an unmapped operation falling through to None is
    # what left a finished `logs` run still reading "state: running".
    _TERMINAL_STATES = {
        OperationType.DEPLOY: DeploymentState.HEALTHY,
        OperationType.WARMUP: DeploymentState.HEALTHY,
        OperationType.SMOKE_TEST: DeploymentState.HEALTHY,
        OperationType.STOP: DeploymentState.STOPPED,
    }

    def on_operation_done(self, message: OperationDone) -> None:
        self._done = True
        self._success = message.success
        self._ensure_progress_operation(message.operation)
        self._current_operation = message.operation
        detail = _strip_ansi(message.detail or "").strip()
        resource_status = self._progress.resource_status
        if message.success and resource_status == "none":
            resource_status = "allocated"
        if not message.success and resource_status == "none":
            resource_status = "unknown"
        self._progress.on_done(message.success, detail, resource_status=resource_status)
        self._refresh_progress()
        self.refresh_bindings()
        if message.success:
            self.status_header.update_from_event(
                state=self._TERMINAL_STATES.get(
                    message.operation, DeploymentState.IDLE
                )
            )
            # update_from_event keeps the old detail when given an empty one,
            # so clear it directly rather than loosening that for every caller.
            self.status_header.detail = ""
        else:
            self.status_header.report_failure()
        self._append_log_line("")
        if message.success:
            self._append_log_line(f"Operation complete ({message.operation.value}).")
            self._show_result_card(message)
        else:
            self._append_log_line(f"Operation failed (exit code {message.exit_code}).")
            # The operation has already reported the error under its own line,
            # and fail_operation repeats it verbatim as the completion detail.
            # Restating it here printed the same sentence twice, which on a
            # summarized deploy took four of the dozen lines the box has.
            if detail and detail != self._last_error_message:
                self._append_log_line(f"Detail: {detail}")
            if self._summary_mode_enabled:
                self._append_log_line(
                    'Tip: re-run with "Show debug logs" enabled to see full backend logs.'
                )
            # Not "enter to retry": enter pops this screen like esc and q do,
            # and no retry path exists to offer.
            self._append_log_line(_RETURN_HINT)
            self._show_failure_card(message)
            return
        if message.success and self._connection_payload:
            self._append_log_line(_RETURN_HINT)
            self._show_connection_card()
        else:
            self._append_log_line(_RETURN_HINT)

    def _capture_result_lines(self, cleaned_line: str) -> None:
        """Capture structured status-probe output while an operation runs."""
        if self._current_operation != OperationType.STATUS:
            return
        if cleaned_line.startswith("Status: healthy"):
            self._status_result["status"] = cleaned_line.strip()
        elif cleaned_line.startswith("Test command:"):
            self._status_result["test_command"] = (
                cleaned_line.removeprefix("Test command:").strip()
            )

    def _show_result_card(self, message: OperationDone) -> None:
        """Render a structured result card for status checks and benchmarks."""
        rows: list[tuple[str, str]] = []
        if message.operation == OperationType.STATUS:
            status_line = self._status_result.get("status") or ""
            if "healthy" in status_line:
                rows.append(("Status", "Healthy"))
            test_command = self._status_result.get("test_command") or ""
            if test_command:
                rows.append(("Test command", test_command))
        elif message.operation == OperationType.BENCHMARK:
            summary = message.data
            best_concurrency = getattr(summary, "best_concurrency", None)
            best_throughput = getattr(summary, "best_output_token_throughput", None)
            if best_throughput is not None:
                rows.append(
                    (
                        "Best throughput",
                        f"{best_throughput:.2f} tok/s at concurrency {best_concurrency}",
                    )
                )
            run_dir = getattr(summary, "run_dir", "")
            if run_dir:
                rows.append(("Artifacts", run_dir))
        if not rows:
            return
        self._result_rows = rows
        self._result_title = self._RESULT_TITLES.get(message.operation, "Check complete")
        card = self._mount_outcome_card("result-card")
        if card is not None:
            self._fill_result_card()

    def _fill_result_card(self) -> None:
        """Fill a mounted result card from retained rows (deferred mount)."""
        if not self._result_rows:
            return
        try:
            self.query_one("#result-card-title", Static).update(
                f"[bold]{self._result_title}[/]"
            )
            self.query_one("#result-card-body", Static).update(
                _result_card_markup(self._result_rows)
            )
            self.query_one("#result-done-btn", Button).focus()
        except Exception:
            pass

    def _copy_result(self) -> None:
        if not self._result_rows:
            self.notify("Nothing to copy", timeout=2)
            return
        text = "\n".join(f"{label}: {value}" for label, value in self._result_rows)
        self.app.copy_to_clipboard(text)
        self.notify("Copy requested: result", timeout=2)

    def on_operation_error(self, message: OperationError) -> None:
        self._last_error_message = _strip_ansi(message.message).strip()
        self._progress.on_error(self._last_error_message)
        self._refresh_progress()
        self._append_log_line(f"Error: {message.message}")

    def _failure_resource_line(self) -> str:
        """Describe the resource outcome without inventing cleanup results."""
        status = self._progress.resource_status
        if status == "allocated":
            return "Resource was allocated; it may still be running. Check Manage before redeploying."
        if status in ("unknown", "none"):
            return "Resource state is unknown. Check Manage for stopped or retained resources."
        if status == "retained":
            return "Resource was retained. Stop it in Manage if it is no longer needed."
        if status == "failed":
            return "Cleanup did not confirm. Check Manage for unresolved resources."
        return "Check Manage for the current resource state."

    def _failure_next_steps(self, operation: OperationType) -> str:
        if operation == OperationType.DEPLOY:
            return "Next: review the error, try another placement, or open Manage for logs."
        if operation == OperationType.WARMUP:
            return "Next: the endpoint may still exist; verify in Manage or check logs."
        if operation == OperationType.STOP:
            return "Next: check Manage; a failed stop can leave a billable resource."
        return "Next: check Manage or review the logs above."

    def _show_failure_card(self, message: OperationDone) -> None:
        """Render a concise failure outcome with resource state and next actions."""
        detail = _strip_ansi(message.detail or "").strip() or self._last_error_message
        rows: list[tuple[str, str]] = [
            ("Operation", message.operation.value),
            ("Error", detail or f"exit code {message.exit_code}"),
            ("Resource", self._failure_resource_line()),
            ("Next", self._failure_next_steps(message.operation)),
        ]
        self._failure_rows = rows
        self._failure_title = f"{message.operation.value.capitalize()} failed"
        card = self._mount_outcome_card("failure-card")
        if card is not None:
            self._fill_failure_card()

    def _fill_failure_card(self) -> None:
        """Fill a mounted failure card from retained rows (deferred mount)."""
        if not self._failure_rows:
            return
        try:
            self.query_one("#failure-card-title", Static).update(
                f"[bold]{escape(self._failure_title)}[/]"
            )
            self.query_one("#failure-card-body", Static).update(
                _result_card_markup(self._failure_rows)
            )
            self.query_one("#failure-done-btn", Button).focus()
        except Exception:
            pass

    def _copy_failure(self) -> None:
        if not self._failure_rows:
            self.notify("Nothing to copy", timeout=2)
            return
        text = "\n".join(f"{label}: {value}" for label, value in self._failure_rows)
        self.app.copy_to_clipboard(text)
        self.notify("Copy requested: error", timeout=2)

    def on_log_viewer_status_changed(self, message: LogViewer.StatusChanged) -> None:
        """Keep compact monitor chrome synchronized with the log viewport."""
        self._following = message.following
        self._unseen_lines = message.unseen_lines
        self._line_count = message.line_count
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )

    def on_log_viewer_search_changed(self, message: LogViewer.SearchChanged) -> None:
        """Show search position in compact monitor chrome."""
        self._search_query = message.query
        self._search_current = message.current
        self._search_total = message.total
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "monitor-search":
            self.log_viewer.search(event.value)

    def on_key(self, event: events.Key) -> None:
        search = self.query_one("#monitor-search", Input)
        if search.display and event.key in {"escape", "enter"}:
            search.display = False
            self.log_viewer.log_widget.focus()
            event.prevent_default()
            event.stop()
            return
        super().on_key(event)

    def _selected_text_for_copy(self) -> str | None:
        """Return selected log text before falling back to screen selections."""
        log_widget = self.log_viewer.log_widget
        getter = getattr(log_widget, "get_selected_text", None)
        text = None
        if callable(getter):
            text = getter()
        if not text:
            text = self.get_selected_text()
        return text

    def _show_connection_card(self) -> None:
        payload = self._connection_payload
        if payload is None:
            return
        card = self._mount_outcome_card("connection-card")
        if card is not None:
            self._fill_connection_card()

    def _fill_connection_card(self) -> None:
        """Fill a mounted connection card from the retained payload."""
        payload = self._connection_payload
        if payload is None:
            return
        try:
            self.query_one("#connection-card-title", Static).update(
                "[bold]Endpoint ready[/]"
            )
            self.query_one("#connection-card-body", Static).update(
                _connection_card_markup(payload)
            )
            has_key = bool((payload.get("api_key") or "").strip())
            self.query_one("#copy-key-btn", Button).display = has_key
            self.query_one("#connection-done-btn", Button).focus()
        except Exception:
            pass

    def _copy_connection_field(self, field: str, *, empty_message: str) -> None:
        payload = self._connection_payload
        if payload is None:
            self.notify("Nothing to copy", timeout=2)
            return
        value = (payload.get(field) or "").strip()
        if not value:
            self.notify(empty_message, timeout=2)
            return
        self.app.copy_to_clipboard(value)
        self.notify(f"Copy requested: {field.replace('_', ' ')}", timeout=2)

    def action_copy_base_url(self) -> None:
        self._copy_connection_field("base_url", empty_message="No base URL to copy")

    def action_copy_api_key(self) -> None:
        self._copy_connection_field("api_key", empty_message="No API key to copy")

    def action_copy_connection(self) -> None:
        payload = self._connection_payload
        if payload is None:
            self.notify("Nothing to copy", timeout=2)
            return
        text = _connection_copy_text(payload)
        if not text:
            self.notify("Nothing to copy", timeout=2)
            return
        self.app.copy_to_clipboard(text)
        self.notify("Copy requested: connection details", timeout=2)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-url-btn":
            self.action_copy_base_url()
        elif event.button.id == "copy-key-btn":
            self.action_copy_api_key()
        elif event.button.id == "copy-all-btn":
            self.action_copy_connection()
        elif event.button.id == "connection-manage-btn":
            self.action_open_manage()
        elif event.button.id == "connection-done-btn":
            self.action_finish_success()
        elif event.button.id == "result-copy-btn":
            self._copy_result()
        elif event.button.id == "result-done-btn":
            self.action_finish_success()
        elif event.button.id == "failure-copy-btn":
            self._copy_failure()
        elif event.button.id == "failure-manage-btn":
            self.action_open_manage()
        elif event.button.id == "failure-done-btn":
            self.action_finish_success()

    def action_submit_or_finish(self) -> None:
        """Let Enter activate focused buttons before applying the Done shortcut."""
        if isinstance(self.focused, Button):
            raise SkipAction()
        self.action_finish_success()

    def action_finish_success(self) -> None:
        search = self.query_one("#monitor-search", Input)
        if search.has_focus:
            search.display = False
            self.log_viewer.log_widget.focus()
            return
        if self._success and self._connection_payload:
            self._pop_after_success()
            return
        if self._done:
            # Return to the originating flow after completion or a failure.
            self.app.pop_screen()

    def action_open_manage(self) -> None:
        """Jump to the Manage screen after a successful deploy."""
        popper = getattr(self.app, "pop_to_main_menu", None)
        if callable(popper):
            popper()
        pusher = getattr(self.app, "action_push_manage", None)
        if callable(pusher):
            pusher()

    def _pop_after_success(self) -> None:
        pop_home = getattr(self.app, "pop_to_main_menu", None)
        if callable(pop_home):
            pop_home()
            return
        self.app.pop_screen()

    def action_go_back(self) -> None:
        jobs = getattr(self.app, "deployment_jobs", {})
        if any(job.monitor is self and not job.finished.is_set() for job in jobs.values()):
            self.notify("Deployment continues. Reopen it in Manage → Jobs (Ctrl+O).", timeout=5)
        if self._success and self._connection_payload:
            self._pop_after_success()
            return
        self.app.pop_screen()

    def action_page_up_log(self) -> None:
        self.log_viewer.page_up()

    def action_page_down_log(self) -> None:
        self.log_viewer.page_down()

    def action_resume_follow(self) -> None:
        self.log_viewer.resume_following()

    def action_search_logs(self) -> None:
        search = self.query_one("#monitor-search", Input)
        search.display = True
        search.focus()

    def action_next_search_match(self) -> None:
        self.log_viewer.next_match(1)

    def action_previous_search_match(self) -> None:
        self.log_viewer.next_match(-1)

    def action_toggle_log_view(self) -> None:
        if not self._summary_mode_enabled:
            self.notify("Raw log view is already active.", timeout=2)
            return
        self._view_mode = "raw" if self._view_mode == "summary" else "summary"
        if self._view_mode == "summary":
            self._refresh_summary_view()
        else:
            self.log_viewer.replace_lines(self._raw_log_lines)
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )

    def action_clear_log(self) -> None:
        self.log_viewer.clear()
        self._raw_log_lines = []
        self._summary_log_lines = []
        self._summary_items = []
        self._following = True
        self._unseen_lines = 0
        self._line_count = 0
        self._result_rows = []
        try:
            self.query_one("#result-card", VerticalScroll).remove()
        except Exception:
            pass
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )

    def _tick_summary_spinner(self) -> None:
        """Advance the in-progress spinner without adding log lines."""
        if self._done or self._view_mode != "summary":
            return
        if self._active_in_progress_index() is None:
            return
        self._spinner_index = (self._spinner_index + 1) % len(SUMMARY_SPINNER_FRAMES)
        self._refresh_summary_view()

    def _ingest_summary_line(self, line: str) -> None:
        """Store a summary milestone, replacing percent updates in place."""
        stripped = line.rstrip()
        if not stripped:
            self._summary_items.append(_SummaryRow("", "blank"))
            prune_retained_items(self._summary_items)
            return
        kind = classify_summary_kind(stripped)
        label, _percent = summary_progress_parts(stripped)
        if kind == "step":
            last_step = self._last_summary_step_index()
            if last_step is not None:
                existing_label, _existing_percent = summary_progress_parts(
                    self._summary_items[last_step].text
                )
                if existing_label == label:
                    self._summary_items[last_step] = _SummaryRow(stripped, kind)
                    return
        self._summary_items.append(_SummaryRow(stripped, kind))
        prune_retained_items(self._summary_items)

    def _last_summary_step_index(self) -> int | None:
        for index in range(len(self._summary_items) - 1, -1, -1):
            if self._summary_items[index].kind == "step":
                return index
        return None

    def _active_in_progress_index(self) -> int | None:
        """Return the step still running, so something on screen is moving.

        This used to take the last step-or-done row and spin it only if it was
        a step, which assumes the two alternate. They do not: a Prime deploy
        polls the tunnel and the runtime together, so "Secure endpoint
        connected" lands while the container image is still being built. The
        spinner then vanished and every row took a tick, leaving a build that
        had twenty minutes to run looking like a finished deploy.

        The last step is the one to mark. It can briefly sit on a step that
        has just finished, in the gap before the next one starts -- that reads
        as "still working", which is true, where a static screen does not.
        """

        if self._done:
            return None
        return self._last_summary_step_index()

    def _rendered_summary_lines(self) -> list[str]:
        active_index = self._active_in_progress_index()
        last_step_index = self._last_summary_step_index()
        spinner = SUMMARY_SPINNER_FRAMES[self._spinner_index]
        lines: list[str] = []
        for index, item in enumerate(self._summary_items):
            if item.kind == "blank" or not item.text:
                lines.append("")
                continue
            if item.kind == "error":
                mark = "✗"
            elif item.kind == "info":
                mark = "·"
            elif index == active_index:
                mark = spinner
            elif (
                self._done
                and not self._success
                and item.kind == "step"
                and index == last_step_index
            ):
                mark = "·"
            else:
                mark = "✓"
            lines.append(f"{mark} {item.text}")
        return lines

    def _refresh_summary_view(self) -> None:
        self._summary_log_lines = self._rendered_summary_lines()
        if self._view_mode != "summary":
            return
        self.log_viewer.set_lines(self._summary_log_lines, keep_follow=True)

    def _append_log_line(
        self,
        line: str,
        *,
        raw: bool = True,
        summary: bool = True,
    ) -> None:
        """Append to retained presentations and the currently visible one."""
        if raw:
            self._raw_log_lines.append(line)
            prune_retained_items(self._raw_log_lines)
        if self._summary_mode_enabled and summary:
            self._ingest_summary_line(line)
            if self._view_mode == "summary":
                self._refresh_summary_view()
                return
        visible = (
            (self._view_mode == "raw" and raw)
            or (self._view_mode == "summary" and summary)
        )
        if visible:
            rendered = (
                beautify_summary_line(line)
                if self._view_mode == "summary" and line.strip()
                else line
            )
            self.log_viewer.write_line(rendered)

"""Monitor screen: real-time log streaming and operation status.

Shows the status header + scrolling log output for any running
operation (deploy, warmup, logs, status, stop).
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Input, Static

from ...protocol.enums import BackendType, DeploymentState, OperationType
from ..deploy_log_summary import (
    SUMMARY_SPINNER_FRAMES,
    DeployLogSummarizer,
    beautify_summary_line,
    classify_summary_kind,
    summary_progress_parts,
)
from .copy_enabled import CopyEnabledScreen
from ..widgets.log_viewer import LogViewer, prune_retained_items
from ..widgets.status_header import StatusHeader
from ..workers import ConnectionSummaryReady, LogMessage, OperationDone, OperationError, StateChanged

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text or "")


def _connection_card_markup(payload: dict[str, str]) -> str:
    """Render connection fields for the post-deploy card."""
    base_url = escape((payload.get("base_url") or "").strip() or "(unavailable)")
    model_id = escape((payload.get("model_id") or "").strip() or "(unavailable)")
    display_name = escape((payload.get("display_name") or "").strip() or "(unavailable)")
    api_key = (payload.get("api_key") or "").strip()
    key_line = (
        f"[dim]API key[/dim]     {escape(api_key)}"
        if api_key
        else "[dim]API key[/dim]     none"
    )
    return (
        f"[dim]Base URL[/dim]    {base_url}\n"
        f"[dim]Model ID[/dim]    {model_id}\n"
        f"[dim]Display[/dim]     {display_name}\n"
        f"{key_line}"
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
    return "\n".join(
        f"[dim]{escape(label)}[/dim]  {escape(value)}" for label, value in rows
    )


class MonitorScreen(CopyEnabledScreen):
    """Full-screen operation monitor with streaming logs."""

    AUTO_FOCUS = "#log-output"

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back"),
        Binding(
            "y",
            "copy_text",
            "Copy",
            key_display="y",
            show=True,
        ),
        Binding(
            "ctrl+shift+c,super+c,meta+c,cmd+c,command+c",
            "copy_text_to_clipboard",
            "Copy",
            show=False,
        ),
        Binding("pageup", "page_up_log", "Page up", show=True, priority=True),
        Binding("pagedown", "page_down_log", "Page down", show=True, priority=True),
        Binding("end", "resume_follow", "Follow", show=True, priority=True),
        Binding("/", "search_logs", "Search", show=True, priority=True),
        Binding("n", "next_search_match", show=False),
        Binding("shift+n", "previous_search_match", show=False),
        Binding("v", "toggle_log_view", "Raw/Summary", show=True),
        Binding("ctrl+l", "clear_log", "Clear log", show=True),
        Binding("enter", "finish_success", "Done", show=False, priority=True),
        Binding("u", "copy_base_url", "Copy URL", show=False),
        Binding("k", "copy_api_key", "Copy key", show=False),
    ]

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
            yield Input(
                placeholder="Search logs; Enter closes, n/N navigates",
                id="monitor-search",
                classes="hidden",
            )
            with Vertical(id="connection-card", classes="hidden"):
                yield Static("[bold #7bf168]Connection[/]", id="connection-card-title")
                yield Static("", id="connection-card-body")
                with Horizontal(id="connection-card-actions"):
                    yield Button("Copy URL", id="copy-url-btn")
                    yield Button("Copy API key", id="copy-key-btn")
                    yield Button("Copy all", id="copy-all-btn")
                    yield Button("Manage endpoint", id="connection-manage-btn")
                    yield Button("Done", id="connection-done-btn", variant="primary")
            with Vertical(id="result-card", classes="hidden"):
                yield Static("[bold #7bf168]Result[/]", id="result-card-title")
                yield Static("", id="result-card-body")
                with Horizontal(id="result-card-actions"):
                    yield Button("Copy result", id="result-copy-btn")
                    yield Button("Done", id="result-done-btn", variant="primary")
            yield LogViewer(id="monitor-log-viewer")
        yield Footer()

    def _title_markup(self) -> str:
        """Render a compact operation title and input-mode badge."""
        mouse_enabled = getattr(self.app, "mouse_enabled", True)
        mode = "MOUSE" if mouse_enabled else "TERMINAL SELECT"
        return (
            f"[bold #7bf168]{escape(self._title)}[/]  "
            f"[dim]·[/dim]  [#93a596]{mode}[/]"
        )

    def _view_status_markup(self) -> str:
        """Render live follow and line-count state for the log viewport."""
        line_label = "line" if self._line_count == 1 else "lines"
        if self._following:
            state = "[bold #7bf168]FOLLOWING[/]"
        else:
            new_label = "line" if self._unseen_lines == 1 else "lines"
            state = (
                f"[bold yellow]PAUSED[/]  [yellow]· {self._unseen_lines} new {new_label}[/]"
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
        """Refresh compact chrome after toggling app/terminal mouse mode."""
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
        if self._summary_mode_enabled:
            self.status_header.update_from_event(
                state=DeploymentState.QUEUED,
                backend=self._deploy_backend,
                operation=OperationType.DEPLOY,
                detail="Preparing deployment",
            )
            self._append_log_line("Preparing deployment", summary=True, raw=False)
            self._last_summary_state_detail = "Preparing deployment"
            self.set_interval(0.25, self._tick_summary_spinner)

    def on_log_message(self, message: LogMessage) -> None:
        prefix = "stderr | " if message.stream == "stderr" else ""
        cleaned = _strip_ansi(message.line)
        self._capture_result_lines(cleaned)
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
                self._append_log_line(
                    f"{prefix}{line}" if prefix else line,
                    summary=True,
                    raw=False,
                )
            return

    def on_state_changed(self, message: StateChanged) -> None:
        if message.operation is not None:
            self._current_operation = message.operation
        self.status_header.update_from_event(
            state=message.state,
            operation=message.operation,
            detail=message.detail,
        )
        detail = _strip_ansi(message.detail).strip()
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

    def on_connection_summary_ready(self, message: ConnectionSummaryReady) -> None:
        self._connection_payload = dict(message.payload)
        if self._success:
            self._show_connection_card()

    def on_operation_done(self, message: OperationDone) -> None:
        self._done = True
        self._success = message.success
        self._append_log_line("")
        if message.success:
            self._append_log_line(f"Operation complete ({message.operation.value}).")
            self._show_result_card(message)
        else:
            self._append_log_line(f"Operation failed (exit code {message.exit_code}).")
            if message.detail:
                self._append_log_line(f"Detail: {message.detail}")
            if self._summary_mode_enabled:
                self._append_log_line(
                    'Tip: re-run with "Show debug logs" enabled to see full backend logs.'
                )
            self._append_log_line("Press esc or q to return, or enter to retry.")
            return
        if message.success and self._connection_payload:
            self._append_log_line("Press enter or esc to return home.")
            self._show_connection_card()
        else:
            self._append_log_line("Press esc or q to return.")

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
        card = self.query_one("#result-card", Vertical)
        card.remove_class("hidden")
        self.query_one("#result-card-body", Static).update(_result_card_markup(rows))
        self.query_one("#result-done-btn", Button).focus()

    def _copy_result(self) -> None:
        if not self._result_rows:
            self.notify("Nothing to copy", timeout=2)
            return
        text = "\n".join(f"{label}: {value}" for label, value in self._result_rows)
        self.app.copy_to_clipboard(text)
        self.notify("Copied result", timeout=2)

    def on_operation_error(self, message: OperationError) -> None:
        self._append_log_line(f"Error: {message.message}")

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
        card = self.query_one("#connection-card", Vertical)
        card.remove_class("hidden")
        self.query_one("#connection-card-body", Static).update(
            _connection_card_markup(payload)
        )
        has_key = bool((payload.get("api_key") or "").strip())
        self.query_one("#copy-key-btn", Button).display = has_key
        self.query_one("#connection-done-btn", Button).focus()

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
        self.notify(f"Copied {field.replace('_', ' ')}", timeout=2)

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
        self.notify("Copied connection details", timeout=2)

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

    def action_finish_success(self) -> None:
        if self._success and self._connection_payload:
            self._pop_after_success()
            return
        if self._done and not self._success:
            # Pop back to the form that started the operation so the user can
            # adjust options and retry without navigating from scratch.
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
            self.query_one("#result-card", Vertical).add_class("hidden")
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
        if self._done:
            return None
        active: int | None = None
        for index, item in enumerate(self._summary_items):
            if item.kind in {"step", "done"}:
                active = index
        if active is not None and self._summary_items[active].kind == "step":
            return active
        return None

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

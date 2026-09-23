"""Reopen deployment monitors and deliberately cancel active jobs."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen


class DurableJobsLoaded(Message):
    """Result of one serialized durable-job refresh."""

    def __init__(self, rows: list[tuple[str, str]] | None, error: str = "") -> None:
        super().__init__()
        self.rows = rows
        self.error = error


class DeploymentJobsPanel(VerticalScroll):
    """Keep background deployments and their results reachable."""

    BINDINGS = [
        Binding("d", "deploy_model", "Deploy", show=True),
        Binding("x", "cancel_selected", "Cancel", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._persistent_rows: list[tuple[str, str]] = []
        self._jobs_loaded = False
        self._refresh_inflight = False
        self._reconcile_pending = False
        self._refresh_error = ""

    def compose(self) -> ComposeResult:
        yield OptionList(id="deployment-jobs")
        yield Static("Loading jobs…", id="operations-status")
        yield Static("", id="operations-empty")
        yield Button("Deploy model", id="operations-deploy-btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#operations-empty", Static).update(
            "[dim]No deployments yet. Start one from Deploy.[/dim]"
        )
        self._sync_empty_state(False)
        self._refresh_jobs(reconcile=True)
        self.set_interval(0.5, self._refresh_jobs)
        self.set_interval(
            self._RECONCILE_INTERVAL_SECONDS,
            lambda: self._refresh_jobs(reconcile=True),
        )

    # Reconciliation reads every active job out of SQLite and adopts journal
    # entries, writing as it goes. That is startup work, not repaint work, so
    # the repaint timer does not carry it: at 0.5s it ran those transactions on
    # the event loop twice a second for as long as this screen stayed open.
    _RECONCILE_INTERVAL_SECONDS = 10.0

    def _durable_jobs(self, *, reconcile: bool = False) -> list[tuple[str, str]]:
        """Read durable rows in a worker; failures must not masquerade as empty."""
        rows: list[tuple[str, str]] = []
        store = self.app._get_job_store()  # type: ignore[attr-defined]
        if reconcile:
            reconciler = getattr(store, "reconcile_workers", None)
            if callable(reconciler):
                reconciler()
            importer = getattr(store, "import_journal_entries", None)
            if callable(importer):
                importer()
        for record in store.list_jobs():  # type: ignore[attr-defined]
            try:
                provider_name = record.config.provider.display_name
            except Exception:
                provider_name = record.provider
            rows.append((f"persistent:{record.id}", f"{provider_name} · {record.app_name} · {record.outcome}"))
        return rows

    def _refresh_jobs(self, *, reconcile: bool = False) -> None:
        self._render_jobs()
        self._reconcile_pending |= reconcile
        if self._refresh_inflight:
            return
        self._refresh_inflight = True
        reconcile = self._reconcile_pending
        self._reconcile_pending = False

        def load() -> None:
            try:
                rows = self._durable_jobs(reconcile=reconcile)
            except Exception as exc:
                self.post_message(DurableJobsLoaded(None, str(exc)))
            else:
                self.post_message(DurableJobsLoaded(rows))

        self.run_worker(load, thread=True, name="operations-refresh")

    def on_durable_jobs_loaded(self, message: DurableJobsLoaded) -> None:
        self._refresh_inflight = False
        if message.rows is not None:
            self._persistent_rows = message.rows
            self._jobs_loaded = True
            self._refresh_error = ""
        else:
            self._refresh_error = message.error or "Unknown error"
        self._render_jobs()
        if self._reconcile_pending:
            self._refresh_jobs()

    def _render_jobs(self) -> None:
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        for job in tuple(getattr(self.app, "deployment_jobs", {}).values()):
            rows.append((job.id, f"{job.config.provider.display_name} · {job.config.app_name} · {job.outcome}"))
            if job.persistent_id:
                seen.add(f"persistent:{job.persistent_id}")
        rows.extend(row for row in self._persistent_rows if row[0] not in seen)
        status = "" if self._jobs_loaded else "Loading jobs…"
        if self._refresh_error:
            retained = "Showing last saved results. " if self._jobs_loaded else ""
            status = f"[$warning]{retained}Job refresh failed: {escape(self._refresh_error)}[/$warning]"
        self.query_one("#operations-status", Static).update(status)
        self.query_one("#operations-status", Static).display = bool(status)
        options = self.query_one("#deployment-jobs", OptionList)
        labels = tuple(rows)
        empty_label = "No deployments yet." if self._jobs_loaded else "Loading jobs…"
        render_key = (labels, empty_label)
        if render_key == getattr(self, "_labels", None):
            return
        self._labels = render_key
        highlighted = options.highlighted
        selected = options.highlighted_option
        selected_id = selected.id if selected is not None else None
        options.clear_options()
        options.add_options(Option(escape(label), id=job_id) for job_id, label in labels)
        if not labels:
            options.add_option(Option(empty_label, disabled=True))
        else:
            options.highlighted = next(
                (index for index, (job_id, _) in enumerate(labels) if job_id == selected_id),
                min(highlighted or 0, len(labels) - 1),
            )
        self._sync_empty_state(bool(labels))
        self.query_one("#operations-empty", Static).display = not labels and self._jobs_loaded

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "operations-deploy-btn":
            self.action_deploy_model()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = str(event.option.id or "")
        if option_id.startswith("persistent:"):
            self.app.reopen_persistent_deployment(option_id.removeprefix("persistent:"))  # type: ignore[attr-defined]
            return
        self.app.reopen_deployment(option_id)

    def _sync_empty_state(self, has_jobs: bool) -> None:
        """Show the Deploy action only when there is nothing to reopen.

        The empty OptionList still announces the state for screen readers;
        the button gives it a direct action. Cancel stays enabled but inert
        without a selection (see action_cancel_selected).
        """
        try:
            empty = self.query_one("#operations-empty", Static)
            deploy = self.query_one("#operations-deploy-btn", Button)
        except Exception:
            return
        empty.display = not has_jobs
        deploy.display = not has_jobs

    def action_deploy_model(self) -> None:
        """Leave the empty state directly for the model picker."""
        pusher = getattr(self.app, "action_push_deploy", None)
        if callable(pusher):
            pusher()
            return
        self.app.pop_screen()

    def action_cancel_selected(self) -> None:
        options = self.query_one("#deployment-jobs", OptionList)
        if options.highlighted is None:
            self.app.notify("Nothing to cancel.", timeout=2)
            return
        option_id = str(options.get_option_at_index(options.highlighted).id or "")
        if option_id.startswith("persistent:"):
            persistent_id = option_id.removeprefix("persistent:")
            self.app.push_screen(PersistentCancelDeploymentScreen(persistent_id))  # type: ignore[attr-defined]
            return
        job = self.app.deployment_jobs.get(option_id)
        if job is not None and not job.finished.is_set():
            self.app.push_screen(CancelDeploymentScreen(option_id))

class OperationsScreen(CopyEnabledScreen):
    """Legacy standalone host for the reusable jobs panel."""

    BINDINGS = [Binding("escape", "go_back", "Back", show=True)]

    def compose(self) -> ComposeResult:
        yield DeploymentJobsPanel()
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#deployment-jobs", OptionList).focus()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class CancelDeploymentScreen(CopyEnabledScreen):
    """Confirm resource cleanup, including destructive rental termination."""

    BINDINGS = [Binding("escape", "go_back", "Keep running", show=True)]

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id

    def compose(self) -> ComposeResult:
        job = self.app.deployment_jobs[self.job_id]
        with VerticalScroll(classes="screen-scroll dialog-scroll"):
            with Vertical(id="cancel-deploy-dialog", classes="dialog-panel"):
                yield Static("Cancel deployment?", classes="dialog-title")
                yield Static(f"[bold]{escape(job.config.app_name)}[/bold]")
                yield Static(
                    "This stops the provider resource after the current provider call returns. "
                    "[$warning]For rentals, termination can permanently delete the rental disk "
                    "and cached models.[/$warning]"
                )
                with Horizontal(id="cancel-deploy-actions", classes="dialog-actions"):
                    yield Button("Keep running", id="keep-running")
                    yield Button("Cancel and stop resource", id="cancel-deployment", variant="error")
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#keep-running", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-deployment":
            self.app.cancel_deployment(self.job_id)
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class PersistentCancelDeploymentScreen(CopyEnabledScreen):
    """Cancel a durable job, including ones from previous sessions."""

    BINDINGS = [Binding("escape", "go_back", "Keep running", show=True)]

    def __init__(self, persistent_id: str) -> None:
        super().__init__()
        self.persistent_id = persistent_id

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll dialog-scroll"):
            with Vertical(id="cancel-persistent-dialog", classes="dialog-panel"):
                yield Static("Cancel background deployment?", classes="dialog-title")
                yield Static(f"[bold]{escape(self.persistent_id)}[/bold]")
                yield Static(
                    "This requests cancellation in the background worker. The worker stops "
                    "the provider resource after the current provider call returns. "
                    "Closing the TUI does not cancel it."
                )
                with Horizontal(id="cancel-deploy-actions", classes="dialog-actions"):
                    yield Button("Keep running", id="keep-running")
                    yield Button("Cancel and stop resource", id="cancel-deployment", variant="error")
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#keep-running", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-deployment":
            canceller = getattr(self.app._get_job_store(), "request_cancel", None)  # type: ignore[attr-defined]
            if callable(canceller):
                try:
                    canceller(self.persistent_id)
                except Exception:
                    pass
            self.app.notify("Cancellation requested; the worker cleans up after its current call.", timeout=6)
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()

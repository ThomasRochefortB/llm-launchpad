"""Reopen deployment monitors and deliberately cancel active jobs."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen


class OperationsScreen(CopyEnabledScreen):
    """Keep background deployments and their results reachable."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("x", "cancel_selected", "Cancel deployment"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Deployment operations[/]")
            yield Static("Enter reopens logs and results. Back leaves deployments running.")
            yield OptionList(id="deployment-jobs")
        yield FittedFooter()

    def on_mount(self) -> None:
        self._refresh_jobs(reconcile=True)
        self.query_one("#deployment-jobs", OptionList).focus()
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
        """Session jobs plus durable jobs from previous sessions, newest first."""
        seen_persistent: set[str] = set()
        rows: list[tuple[str, str]] = []
        for job in tuple(self.app.deployment_jobs.values()):
            label = f"{job.config.provider.display_name} · {job.config.app_name} · {job.outcome}"
            rows.append((job.id, label))
            if job.persistent_id:
                seen_persistent.add(job.persistent_id)
        try:
            store = self.app._get_job_store()  # type: ignore[attr-defined]
            if reconcile:
                reconciler = getattr(store, "reconcile_workers", None)
                if callable(reconciler):
                    reconciler()
                importer = getattr(store, "import_journal_entries", None)
                if callable(importer):
                    importer()
            for record in store.list_jobs():  # type: ignore[attr-defined]
                if record.id in seen_persistent:
                    continue
                try:
                    provider_name = record.config.provider.display_name
                except Exception:
                    provider_name = record.provider
                rows.append((f"persistent:{record.id}", f"{provider_name} · {record.app_name} · {record.outcome}"))
        except Exception:
            pass
        return rows

    def _refresh_jobs(self, *, reconcile: bool = False) -> None:
        options = self.query_one("#deployment-jobs", OptionList)
        labels = tuple(self._durable_jobs(reconcile=reconcile))
        if labels == getattr(self, "_labels", None):
            return
        self._labels = labels
        highlighted = options.highlighted
        options.clear_options()
        options.add_options(Option(escape(label), id=job_id) for job_id, label in labels)
        if not labels:
            options.add_option(Option("No deployments yet. Start one from Deploy.", disabled=True))
        else:
            options.highlighted = min(highlighted or 0, len(labels) - 1)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = str(event.option.id or "")
        if option_id.startswith("persistent:"):
            self.app.reopen_persistent_deployment(option_id.removeprefix("persistent:"))  # type: ignore[attr-defined]
            return
        self.app.reopen_deployment(option_id)

    def action_cancel_selected(self) -> None:
        options = self.query_one("#deployment-jobs", OptionList)
        if options.highlighted is None:
            return
        option_id = str(options.get_option_at_index(options.highlighted).id or "")
        if option_id.startswith("persistent:"):
            persistent_id = option_id.removeprefix("persistent:")
            self.app.push_screen(PersistentCancelDeploymentScreen(persistent_id))  # type: ignore[attr-defined]
            return
        job = self.app.deployment_jobs.get(option_id)
        if job is not None and not job.finished.is_set():
            self.app.push_screen(CancelDeploymentScreen(option_id))

    def action_go_back(self) -> None:
        self.app.pop_screen()


class CancelDeploymentScreen(CopyEnabledScreen):
    """Confirm resource cleanup, including destructive rental termination."""

    BINDINGS = [Binding("escape", "go_back", "Keep running")]

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id

    def compose(self) -> ComposeResult:
        job = self.app.deployment_jobs[self.job_id]
        with VerticalScroll(classes="screen-scroll"):
            yield Static(f"Cancel deployment of [bold]{escape(job.config.app_name)}[/bold]?")
            yield Static(
                "This stops the provider resource after the current provider call returns. "
                "For rentals, termination can permanently delete the rental disk and cached models."
            )
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

    BINDINGS = [Binding("escape", "go_back", "Keep running")]

    def __init__(self, persistent_id: str) -> None:
        super().__init__()
        self.persistent_id = persistent_id

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static(f"Cancel background deployment [bold]{escape(self.persistent_id)}[/bold]?")
            yield Static(
                "This requests cancellation in the background worker. The worker stops "
                "the provider resource after the current provider call returns. "
                "Closing the TUI does not cancel it."
            )
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

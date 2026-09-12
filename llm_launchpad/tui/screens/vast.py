"""Account setup and read-only Vast offer browsing."""

from __future__ import annotations

import os

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, DataTable, Input, Static, Switch

from ...core.vast_auth import (
    VastCredentials, clear_vast_api_key, resolve_vast_credentials, save_vast_api_key,
)
from ...core.vast_backend import VastApiError, VastBackend
from ...protocol.models import VastOffer, VastOfferQuery
from ..format import format_money
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen


class VastPreviewResult(Message):
    """Completed account action or offer lookup; contains no credentials."""

    def __init__(self, detail: str, offers: list[VastOffer] | None = None) -> None:
        super().__init__()
        self.detail = detail
        self.offers = offers


class VastPreviewScreen(CopyEnabledScreen):
    """Configure Vast credentials and browse rental costs."""

    BINDINGS = [Binding("escape", "back", "Back", show=True)]
    DEFAULT_CSS = """
    VastPreviewScreen #vast-scroll { padding: 1 2; }
    VastPreviewScreen Input { margin-bottom: 1; }
    VastPreviewScreen .vast-actions { height: auto; }
    VastPreviewScreen Button { margin-right: 1; }
    VastPreviewScreen #vast-feedback { height: auto; min-height: 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="vast-scroll"):
            yield Static("[bold]Vast.ai rentals[/bold]")
            yield Static("Browse verified, on-demand NVIDIA rentals. Use Deploy model to rent one.")
            yield Static("API key (hidden; leave blank to use the configured key)")
            yield Input(password=True, id="vast-key")
            with Horizontal(classes="vast-actions"):
                yield Button("Validate / save key", id="vast-connect")
                yield Button("Forget saved key", id="vast-forget")
            yield Static("GPU name (optional, e.g. RTX 4090)")
            yield Input(id="vast-gpu")
            yield Static("Country code (optional, e.g. US)")
            yield Input(id="vast-country")
            yield Static("Disk allocation for pricing (GiB)")
            yield Input("100", id="vast-disk", type="integer")
            yield Static("Minimum reliability score (0–1; not an uptime guarantee)")
            yield Input("0.99", id="vast-reliability", type="number")
            yield Static("Datacenter only")
            yield Switch(False, id="vast-datacenter")
            yield Button("Refresh offers", id="vast-refresh", variant="primary")
            yield Static("", id="vast-feedback")
            yield DataTable(id="vast-offers", cursor_type="row")
            yield Static(
                "Single-GPU offers, up to 100 results. Hourly totals include the selected disk, "
                "but exclude traffic. Downloads are charged; storage remains billable when stopped. "
                "Unknown prices are not zero. Offers have not been checked against a model."
            )
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#vast-offers", DataTable).add_columns(
            "Offer", "GPU", "VRAM GB", "Location", "Tier", "Reliability",
            "Compute/hr", "Disk/hr", "Total/hr", "Down/GB", "Up/GB", "Max hours",
        )
        try:
            credentials = resolve_vast_credentials()
            text = (
                f"Key configured ({credentials.source}); not verified."
                if credentials.api_key else "Configure a Vast API key to browse offers."
            )
        except ValueError as exc:
            text = str(exc)
        self.query_one("#vast-feedback", Static).update(escape(text))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._busy:
            return
        action = event.button.id
        if action not in {"vast-connect", "vast-forget", "vast-refresh"}:
            return
        key = self.query_one("#vast-key", Input).value.strip()
        if action == "vast-refresh" and key:
            self.query_one("#vast-feedback", Static).update("Validate / save the entered key before refreshing offers.")
            return
        try:
            query = VastOfferQuery(
                gpu_type=self.query_one("#vast-gpu", Input).value or None,
                country=self.query_one("#vast-country", Input).value or None,
                disk_gb=int(self.query_one("#vast-disk", Input).value),
                min_reliability=float(self.query_one("#vast-reliability", Input).value),
                datacenter_only=self.query_one("#vast-datacenter", Switch).value,
            ) if action == "vast-refresh" else None
        except ValueError:
            self.query_one("#vast-feedback", Static).update("Enter a whole-number disk size and numeric reliability score.")
            return
        self.query_one("#vast-key", Input).value = ""
        self._busy = True
        for button in self.query(Button):
            button.disabled = True
        # Clear stale offers before changing credentials or refreshing filters.
        self.query_one("#vast-offers", DataTable).clear()
        self.query_one("#vast-feedback", Static).update("Contacting Vast…" if action != "vast-forget" else "Removing saved key…")
        self.run_worker(lambda: self._run_action(action, key, query), thread=True, name="vast-preview")

    def _run_action(self, action: str, key: str, query: VastOfferQuery | None) -> None:
        offers = None
        try:
            if action == "vast-forget":
                clear_vast_api_key()
                effective = resolve_vast_credentials()
                detail = "Launchpad's saved Vast key has been removed."
                if effective.api_key:
                    detail += f" A key remains configured ({effective.source})."
            elif action == "vast-connect":
                credentials = VastCredentials(key, "provided") if key else resolve_vast_credentials()
                status = VastBackend(credentials).auth_status()
                if not status.authenticated:
                    raise VastApiError(status.error or "Vast authentication failed.")
                if key:
                    save_vast_api_key(key)
                detail = f"Validated Vast account {status.account_id}."
                if key and os.getenv("VAST_API_KEY", "").strip():
                    detail += " VAST_API_KEY takes precedence over the saved key."
            else:
                offers = VastBackend().list_offers(query)
                detail = f"Found {len(offers)} eligible rental offers. Use Deploy model to select one."
        except (ValueError, OSError, VastApiError) as exc:
            detail = f"Error: {exc}"
        self.post_message(VastPreviewResult(detail, offers))

    def on_vast_preview_result(self, message: VastPreviewResult) -> None:
        self._busy = False
        for button in self.query(Button):
            button.disabled = False
        self.query_one("#vast-feedback", Static).update(escape(message.detail))
        table = self.query_one("#vast-offers", DataTable)
        for offer in message.offers or []:
            cost = offer.costs
            table.add_row(
                offer.id, escape(offer.gpu_type), f"{offer.gpu_memory_gb:.2f}",
                escape(offer.location), "Datacenter" if offer.datacenter else "Verified",
                f"{offer.reliability:.4f}",
                *(format_money(value, decimals=4) if value is not None else "unknown" for value in (
                    cost.compute_per_hour_usd, cost.disk_per_hour_usd,
                    cost.total_per_hour_usd, cost.download_per_gb_usd, cost.upload_per_gb_usd,
                )),
                f"{offer.max_duration_hours:.1f}" if offer.max_duration_hours is not None else "unknown",
                key=offer.id,
            )

    def action_back(self) -> None:
        self.app.pop_screen()

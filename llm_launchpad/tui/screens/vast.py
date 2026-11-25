"""Account setup and read-only Vast offer browsing."""

from __future__ import annotations

import os

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, DataTable, Input, Static, Switch

from ...core.vast_auth import (
    VastCredentials, clear_vast_api_key, resolve_vast_credentials, save_vast_api_key,
)
from ...core.vast_backend import VastApiError, VastBackend
from ...protocol.models import VastOffer, VastOfferQuery
from ..format import format_money
from ..visual import screen_title
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen


class VastPreviewResult(Message):
    """Completed account action or offer lookup; contains no credentials."""

    def __init__(
        self,
        detail: str,
        offers: list[VastOffer] | None = None,
        *,
        action: str = "",
        ok: bool = True,
    ) -> None:
        super().__init__()
        self.detail = detail
        self.offers = offers
        self.action = action
        self.ok = ok


class VastPreviewScreen(CopyEnabledScreen):
    """Configure Vast credentials and browse rental costs."""

    BINDINGS = [Binding("escape", "back", "Back", show=True)]
    DEFAULT_CSS = """
    VastPreviewScreen #vast-scroll { padding: 1 2; }
    VastPreviewScreen #vast-title { margin-bottom: 1; }
    VastPreviewScreen .vast-row { width: 100%; height: auto; margin-bottom: 1; }
    VastPreviewScreen .vast-col { width: 1fr; height: auto; margin-right: 2; }
    VastPreviewScreen .vast-col:last-child { margin-right: 0; }
    VastPreviewScreen .vast-actions { height: auto; margin: 1 0; align: left middle; }
    VastPreviewScreen .vast-actions Button { margin-right: 1; }
    VastPreviewScreen .vast-inline-toggle { width: auto; height: 3; margin-right: 2; align: left middle; }
    VastPreviewScreen .vast-inline-toggle Static { width: auto; height: 3; content-align: left middle; padding-right: 1; text-wrap: nowrap; }
    VastPreviewScreen #vast-refresh { margin-top: 1; }
    VastPreviewScreen #vast-feedback { height: auto; color: $text-muted; margin-bottom: 1; }
    VastPreviewScreen #vast-disclaimer { height: auto; color: $text-muted; margin-top: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="vast-scroll"):
            yield Static(
                screen_title("Vast.ai rentals", "Browse verified on-demand offers"),
                id="vast-title",
            )
            yield Static("API key [dim](blank uses configured key)[/dim]", classes="form-label")
            yield Input(password=True, placeholder="Paste Vast API key", id="vast-key")
            with Horizontal(classes="vast-actions"):
                yield Button("Validate / save key", id="vast-connect")
                yield Button("Forget saved key", id="vast-forget")
            with Horizontal(classes="vast-row"):
                with Vertical(classes="vast-col"):
                    yield Static("GPU name", classes="form-label")
                    yield Input(placeholder="Optional, e.g. RTX 4090", id="vast-gpu")
                with Vertical(classes="vast-col"):
                    yield Static("Country code", classes="form-label")
                    yield Input(placeholder="Optional, e.g. US", id="vast-country")
                with Vertical(classes="vast-col"):
                    yield Static("Disk (GiB)", classes="form-label")
                    yield Input("100", id="vast-disk", type="integer")
                with Vertical(classes="vast-col"):
                    yield Static("Min reliability (0–1)", classes="form-label")
                    yield Input("0.99", id="vast-reliability", type="number")
            with Horizontal(classes="vast-actions"):
                with Horizontal(classes="vast-inline-toggle"):
                    yield Static("Datacenter only")
                    yield Switch(False, id="vast-datacenter")
                yield Button("Refresh offers", id="vast-refresh", variant="primary")
            yield Static("", id="vast-feedback")
            yield DataTable(id="vast-offers", cursor_type="row")
            yield Static(
                "Single-GPU offers (max 100) · Totals include disk, exclude traffic · "
                "Storage bills while stopped · Use Deploy model to rent.",
                id="vast-disclaimer",
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
        ok = True
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
            ok = False
            detail = f"Error: {exc}"
        self.post_message(VastPreviewResult(detail, offers, action=action, ok=ok))

    def on_vast_preview_result(self, message: VastPreviewResult) -> None:
        self._busy = False
        for button in self.query(Button):
            button.disabled = False
        self.query_one("#vast-feedback", Static).update(escape(message.detail))
        if message.ok and message.action in {"vast-connect", "vast-forget"}:
            refresher = getattr(self.app, "refresh_provider_readiness", None)
            if callable(refresher):
                refresher()
        table = self.query_one("#vast-offers", DataTable)
        for offer in message.offers or []:
            cost = offer.costs
            table.add_row(
                escape(offer.id), escape(offer.gpu_type), f"{offer.gpu_memory_gb:.2f}",
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

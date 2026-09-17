"""Provider-neutral billing snapshots: one shape for every provider.

Modal invoices a workspace for what it has already spent, while Prime and Vast
draw down a prepaid balance. That difference is real and is kept -- it rides on
:class:`BalanceKind` so a reader can tell money-burned from money-left -- but
everything around it is shared. One status vocabulary, one headline figure, one
list of supporting charges, one place that names the command a provider needs
before it can report anything. Adding a fourth provider is one adapter, not
another bespoke panel section.

Figures stay numeric here: money formatting and markup belong to whatever
renders the snapshot, and ``core`` never imports the TUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..protocol.enums import ComputeProvider
from .coerce import optional_float


class BillingStatus(str, Enum):
    """How much is known about one provider's billing, in shared vocabulary."""

    LOADING = "loading"
    READY = "ready"
    UNCONFIGURED = "unconfigured"
    FAILED = "failed"


class BalanceKind(str, Enum):
    """What the headline figure means, since providers do not agree."""

    SPEND_MTD = "spend_mtd"
    BALANCE = "balance"
    CREDIT = "credit"

    @property
    def label(self) -> str:
        """Return the phrase that tells a reader which way the money runs."""
        return {
            BalanceKind.SPEND_MTD: "spent this month",
            BalanceKind.BALANCE: "balance",
            BalanceKind.CREDIT: "credit",
        }[self]


# The command that makes a provider readable. Shared with the auth block so a
# renamed command cannot be corrected in one panel and left stale in the other.
PROVIDER_SETUP_COMMANDS: dict[ComputeProvider, str] = {
    ComputeProvider.MODAL: "modal setup",
    ComputeProvider.PRIME: "prime login",
    ComputeProvider.VAST: "llm-launchpad vast-auth login",
}

# Rendering order: the panel lists every provider every time, so a missing
# figure reads as "not configured" rather than as an absence to puzzle over.
PROVIDER_BILLING_ORDER: tuple[ComputeProvider, ...] = (
    ComputeProvider.MODAL,
    ComputeProvider.PRIME,
    ComputeProvider.VAST,
)


@dataclass(frozen=True)
class BillingCharge:
    """One named component of a provider's spend, such as gpu or storage."""

    label: str
    amount_usd: float


@dataclass(frozen=True)
class ProviderBilling:
    """Everything one provider can say about its own billing, uniformly.

    ``charges`` and ``notes`` are deliberately independent of ``status``: a
    failed billing call must not take a standing warning about ongoing storage
    spend down with it, because that warning was never sourced from the call
    that failed.
    """

    provider: ComputeProvider
    status: BillingStatus
    kind: BalanceKind | None = None
    amount_usd: float | None = None
    charges: tuple[BillingCharge, ...] = ()
    notes: tuple[str, ...] = ()
    owed_usd: float | None = None
    setup_command: str | None = None
    error: str | None = None

    @classmethod
    def loading(cls, provider: ComputeProvider) -> ProviderBilling:
        return cls(provider=provider, status=BillingStatus.LOADING)

    @classmethod
    def unconfigured(cls, provider: ComputeProvider) -> ProviderBilling:
        return cls(
            provider=provider,
            status=BillingStatus.UNCONFIGURED,
            setup_command=PROVIDER_SETUP_COMMANDS.get(provider),
        )

    @classmethod
    def failed(cls, provider: ComputeProvider, error: str) -> ProviderBilling:
        return cls(
            provider=provider,
            status=BillingStatus.FAILED,
            error=(error or "").strip() or "Unknown error.",
        )

    @classmethod
    def ready(
        cls,
        provider: ComputeProvider,
        kind: BalanceKind,
        amount_usd: float | None,
        *,
        charges: tuple[BillingCharge, ...] = (),
        notes: tuple[str, ...] = (),
        owed_usd: float | None = None,
    ) -> ProviderBilling:
        return cls(
            provider=provider,
            status=BillingStatus.READY,
            kind=kind,
            amount_usd=amount_usd,
            charges=charges,
            notes=notes,
            owed_usd=owed_usd,
        )

    def with_notes(self, *notes: str) -> ProviderBilling:
        """Return a copy carrying extra notes, whatever the status."""
        extra = tuple(note for note in notes if note)
        if not extra:
            return self
        return ProviderBilling(
            provider=self.provider,
            status=self.status,
            kind=self.kind,
            amount_usd=self.amount_usd,
            charges=self.charges,
            notes=self.notes + extra,
            owed_usd=self.owed_usd,
            setup_command=self.setup_command,
            error=self.error,
        )


def money_float(value: Any) -> float | None:
    """Coerce a billing figure that may arrive as ``"$1,234.50"`` text."""
    if isinstance(value, str):
        value = value.strip().replace("$", "").replace(",", "")
        if not value:
            return None
    return optional_float(value)


def _first_money(payload: Any, dotted_keys: tuple[str, ...]) -> float | None:
    """Return the first readable figure among dotted paths into ``payload``."""
    for dotted_key in dotted_keys:
        current = payload
        found = True
        for key in dotted_key.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
                break
        if not found:
            continue
        parsed = money_float(current)
        if parsed is not None:
            return parsed
    return None


def _unwrap(payload: Any) -> Any:
    """Strip the envelope key Modal's CLI has wrapped reports in over time."""
    if isinstance(payload, dict):
        for key in ("report", "data"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


_MODAL_TOTAL_KEYS = (
    "summary.total_usd",
    "summary.cost_usd",
    "summary.spend_usd",
    "totals.total_usd",
    "totals.cost_usd",
    "total_usd",
    "cost_usd",
    "spend_usd",
)
_MODAL_GPU_KEYS = ("summary.gpu_cost_usd", "totals.gpu_cost_usd", "gpu_cost_usd")


def parse_modal_billing(payload: Any) -> ProviderBilling:
    """Read month-to-date workspace spend from any shape the CLI returns."""
    normalized = _unwrap(payload)

    if isinstance(normalized, list):
        rows = [row for row in normalized if isinstance(row, dict)]
        total = 0.0
        has_total = False
        for row in rows:
            cost = money_float(row.get("Cost"))
            if cost is None:
                cost = money_float(row.get("cost"))
            if cost is not None:
                total += cost
                has_total = True
        # An empty report is a real answer -- nothing has billed yet -- while a
        # populated report with no readable cost column is not.
        if not rows:
            return ProviderBilling.ready(
                ComputeProvider.MODAL, BalanceKind.SPEND_MTD, 0.0
            )
        if has_total:
            return ProviderBilling.ready(
                ComputeProvider.MODAL, BalanceKind.SPEND_MTD, total
            )
        return ProviderBilling.ready(
            ComputeProvider.MODAL,
            BalanceKind.SPEND_MTD,
            None,
            notes=("no cost column in billing report",),
        )

    if not isinstance(normalized, dict):
        return ProviderBilling.ready(
            ComputeProvider.MODAL,
            BalanceKind.SPEND_MTD,
            None,
            notes=("unreadable billing report; check `modal billing report --json`",),
        )

    total = _first_money(normalized, _MODAL_TOTAL_KEYS)
    gpu_cost = _first_money(normalized, _MODAL_GPU_KEYS)
    charges = (BillingCharge("gpu", gpu_cost),) if gpu_cost is not None else ()
    notes = () if total is not None else ("no total in billing report",)
    return ProviderBilling.ready(
        ComputeProvider.MODAL,
        BalanceKind.SPEND_MTD,
        total,
        charges=charges,
        notes=notes,
    )


def parse_prime_billing(payload: Any) -> ProviderBilling:
    """Read the Prime wallet balance and group recent charges by resource."""
    if not isinstance(payload, dict):
        return ProviderBilling.ready(
            ComputeProvider.PRIME,
            BalanceKind.BALANCE,
            None,
            notes=("unreadable wallet; check `prime wallet`",),
        )

    balance = money_float(payload.get("balance_usd"))
    totals: dict[str, float] = {}
    rows = payload.get("recent_billings")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            amount = money_float(row.get("amount_usd"))
            if amount is None:
                continue
            resource = str(row.get("resource_type") or "other").strip().casefold()
            resource = resource or "other"
            totals[resource] = totals.get(resource, 0.0) + amount

    charges = tuple(
        BillingCharge(label, amount) for label, amount in sorted(totals.items())
    )
    notes = () if balance is not None else ("no balance in wallet payload",)
    return ProviderBilling.ready(
        ComputeProvider.PRIME,
        BalanceKind.BALANCE,
        balance,
        charges=charges,
        notes=notes,
    )


# Vast rents by the second against credit and bills egress on top of it, so a
# credit figure alone would overstate how long a rental can run.
VAST_BILLING_NOTE = "billed continuously; transfer extra"


def parse_vast_billing(payload: Any) -> ProviderBilling:
    """Read spendable Vast credit, calling out an owed balance separately."""
    if not isinstance(payload, dict):
        return ProviderBilling.ready(
            ComputeProvider.VAST,
            BalanceKind.CREDIT,
            None,
            notes=("unreadable account payload",),
        )

    available = money_float(payload.get("available_usd"))
    balance = money_float(payload.get("balance_usd"))
    notes = (VAST_BILLING_NOTE,) if available is not None else ("no credit in account payload",)
    return ProviderBilling.ready(
        ComputeProvider.VAST,
        BalanceKind.CREDIT,
        available,
        notes=notes,
        owed_usd=abs(balance) if balance is not None and balance < 0 else None,
    )


def load_modal_billing(*, authenticated: bool | None = None) -> ProviderBilling:
    """Fetch Modal's month-to-date spend, or say what stands in the way."""
    if authenticated is False:
        return ProviderBilling.unconfigured(ComputeProvider.MODAL)
    from .backend import ModalBackend

    try:
        payload, error = ModalBackend.billing_report_json()
    except Exception as exc:  # pragma: no cover - defensive around the CLI
        return ProviderBilling.failed(ComputeProvider.MODAL, str(exc))
    if payload is None:
        return ProviderBilling.failed(
            ComputeProvider.MODAL, error or "Could not read billing report."
        )
    return parse_modal_billing(payload)


def load_prime_billing(*, authenticated: bool | None = None) -> ProviderBilling:
    """Fetch the Prime wallet, resolving auth locally before any request."""
    if authenticated is None:
        from .prime_auth import get_prime_auth_status

        try:
            authenticated = get_prime_auth_status().authenticated
        except Exception:
            authenticated = None
    if authenticated is False:
        return ProviderBilling.unconfigured(ComputeProvider.PRIME)
    from .prime_backend import PrimeBackend

    try:
        payload, error = PrimeBackend().billing_wallet()
    except Exception as exc:
        return ProviderBilling.failed(ComputeProvider.PRIME, str(exc))
    if payload is None:
        return ProviderBilling.failed(
            ComputeProvider.PRIME, error or "Could not read Prime billing wallet."
        )
    return parse_prime_billing(payload)


def load_vast_billing() -> ProviderBilling:
    """Fetch Vast credit; an unconfigured key never reaches the network."""
    from .vast_auth import resolve_vast_credentials

    try:
        configured = bool(resolve_vast_credentials().api_key)
    except ValueError:
        configured = False
    if not configured:
        return ProviderBilling.unconfigured(ComputeProvider.VAST)
    from .vast_backend import VastBackend

    try:
        payload, error = VastBackend().billing_credit()
    except Exception as exc:
        return ProviderBilling.failed(ComputeProvider.VAST, str(exc))
    if payload is None:
        return ProviderBilling.failed(
            ComputeProvider.VAST, error or "Could not read Vast credit."
        )
    return parse_vast_billing(payload)

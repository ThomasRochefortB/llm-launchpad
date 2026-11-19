"""One layout for every provider's billing, however differently they bill.

The home panel used to stack three unrelated sections: a heading that never
named Modal above a month-to-date total, a wallet balance, and a credit figure,
each with its own words for "still loading" and "could not read that". Three
providers meant three of everything, and a reader could not tell at a glance
which number was money spent and which was money left.

Every provider is drawn from one template instead -- marker, name, headline
figure, then dim lines saying what the figure means and what makes it up. The
providers still differ where they genuinely differ, and that difference is now
stated in the row rather than implied by its shape.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.markup import escape

from ..core.provider_billing import (
    BillingStatus,
    ProviderBilling,
)
from ..core.storage_costs import (
    MODAL_VOLUME_FREE_TIER_GIB_MONTH,
    estimate_monthly_storage_cost,
)
from ..protocol.enums import ComputeProvider
from ..protocol.models import StorageSnapshot
from .format import clip, format_free_tier, format_gib, format_money
from .markers import provider_marker
from .visual import MONEY_STYLE

# An error is bounded rather than made to fit: the panel is 56 columns and caps
# at half the side column, so an unbounded provider message would reflow into a
# paragraph and push the two providers under it out of view.
_ERROR_WIDTH = 72

# Two spaces between the widest provider name and the figures, so the column of
# amounts lines up without depending on how wide the panel happens to be.
_COLUMN_GAP = 2

_MODAL_USAGE_BILLING_URL = "https://modal.com/settings/usage"

_STATUS_MARKUP: dict[BillingStatus, str] = {
    BillingStatus.LOADING: "[dim]checking...[/dim]",
    BillingStatus.UNCONFIGURED: "[$warning]not configured[/$warning]",
    BillingStatus.FAILED: "[$warning]unavailable[/$warning]",
}
_STATUS_PLAIN: dict[BillingStatus, str] = {
    BillingStatus.LOADING: "checking...",
    BillingStatus.UNCONFIGURED: "not configured",
    BillingStatus.FAILED: "unavailable",
}


def storage_estimate_lines(snapshot: StorageSnapshot | None) -> tuple[str, ...]:
    """Describe what cached models cost to keep, as Modal's own detail lines.

    This is computed from a local snapshot and owes nothing to the billing
    call, so it is assembled separately: a failed report must not take the only
    standing warning about ongoing storage spend down with it.
    """
    if snapshot is None:
        return ()
    estimate = estimate_monthly_storage_cost(snapshot)
    allowance = format_free_tier(MODAL_VOLUME_FREE_TIER_GIB_MONTH)
    return (
        f"cache est. {format_money(estimate.estimated_monthly_cost_usd)}/mo · {allowance} free",
        (
            f"{format_gib(estimate.total_gib_month)} cached · "
            f"{format_gib(estimate.billable_gib_month)} above free tier"
        ),
        "assumes full workspace storage allowance",
    )


def _headline(row: ProviderBilling, spinner: str = "") -> tuple[str, str]:
    """Return the figure column as (markup, plain text) for width measuring."""
    if row.status is BillingStatus.LOADING and spinner:
        # One cell per frame, so the column stays put while it turns.
        return f"[$primary]{spinner}[/] [dim]checking[/dim]", f"{spinner} checking"
    if row.status is BillingStatus.READY:
        if row.amount_usd is None:
            return "[dim]unknown[/dim]", "unknown"
        amount = format_money(row.amount_usd)
        return f"[bold {MONEY_STYLE}]{amount}[/]", amount
    return _STATUS_MARKUP[row.status], _STATUS_PLAIN[row.status]


def _summary_fragments(row: ProviderBilling) -> list[str]:
    """Build the line directly under a figure: what it means, and what it holds."""
    fragments: list[str] = []
    if row.status is BillingStatus.READY and row.kind is not None:
        fragments.append(row.kind.label)
    elif row.status is BillingStatus.UNCONFIGURED and row.setup_command:
        fragments.append(f"run: {escape(row.setup_command)}")
    elif row.status is BillingStatus.FAILED and row.error:
        fragments.append(escape(clip(row.error, _ERROR_WIDTH)))

    if row.owed_usd is not None:
        # An owed balance is the one case worth breaking the dim line for: it
        # is why a healthy-looking credit figure will not start a rental.
        fragments.append(f"[$warning]owed {format_money(row.owed_usd)}[/$warning]")

    fragments.extend(
        f"{escape(charge.label)} {format_money(charge.amount_usd)}"
        for charge in row.charges
    )
    return fragments


def _detail_lines(
    row: ProviderBilling, storage_estimate: tuple[str, ...]
) -> list[str]:
    """Return the dim lines under one figure, each short enough not to wrap.

    Notes get a line each rather than being run onto the summary with a
    separator. The panel is 56 columns and narrows to 40, and a wrapped
    continuation loses the two-space indent that ties a detail to its provider,
    so it reads as a row of its own.
    """
    lines: list[str] = []
    summary = _summary_fragments(row)
    if summary:
        lines.append(" · ".join(summary))
    lines.extend(escape(note) for note in row.notes)
    if row.storage is not None:
        lines.append(f"storage metered this month {format_money(row.storage.metered_usd)}")
        if row.storage.after_allowance_usd is not None:
            lines.append(f"storage after allowance {format_money(row.storage.after_allowance_usd)}")
    if row.provider is ComputeProvider.MODAL:
        if row.credits_applied_usd is not None:
            lines.append(f"credits used this month {format_money(row.credits_applied_usd)}")
        if row.status is BillingStatus.READY:
            lines.append(
                f"[underline link='{_MODAL_USAGE_BILLING_URL}']Credit balance in Modal dashboard[/]"
            )
        lines.extend(storage_estimate)
    return lines


def render_provider_billing(
    rows: tuple[ProviderBilling, ...] | list[ProviderBilling],
    *,
    storage_snapshot: StorageSnapshot | None = None,
    spinner: str = "",
    compact: bool = False,
) -> str:
    """Render every provider's billing row into the shared panel body.

    ``spinner`` is the current frame for rows still loading; without one they
    read "checking...". ``compact`` shows provider totals and storage on Home.
    """
    if not rows:
        return "[dim]No providers configured.[/dim]"

    names = [
        f"{provider_marker(row.provider)} {row.provider.display_name}" for row in rows
    ]
    headlines = [_headline(row, spinner) for row in rows]
    # Both columns are measured against the rows themselves rather than the
    # panel, which is 56 columns wide but narrows to 42 and to 40 in the
    # overlay. Padding to the panel width would wrap every row at the narrow
    # end; padding to the content keeps the figures in one column at any width.
    name_width = max(cell_len(name) for name in names) + _COLUMN_GAP
    value_width = max(cell_len(plain) for _, plain in headlines)
    storage_estimate = storage_estimate_lines(storage_snapshot)

    lines: list[str] = []
    for row, name, (headline_markup, headline_plain) in zip(rows, names, headlines, strict=True):
        name_pad = " " * (name_width - cell_len(name))
        value_pad = " " * (value_width - cell_len(headline_plain))
        line = f"{name}{name_pad}{value_pad}{headline_markup}"
        if compact:
            if row.status is BillingStatus.READY and row.kind is not None:
                line += f"  [dim]{row.kind.label}[/dim]"
            if row.owed_usd is not None:
                line += f"  [$warning]owed {format_money(row.owed_usd)}[/$warning]"
            lines.append(line)
            if row.provider is ComputeProvider.MODAL and storage_estimate:
                lines.append(f"[dim]  {storage_estimate[0]}[/dim]")
            elif row.storage is not None and row.storage.after_allowance_usd is not None:
                lines.append(f"[dim]  storage {format_money(row.storage.after_allowance_usd)} this month[/dim]")
            continue
        lines.append(line)

        lines.extend(
            f"[dim]  {detail}[/dim]" for detail in _detail_lines(row, storage_estimate)
        )
    return "\n".join(lines)

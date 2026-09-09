"""Presentation of model-sized Vast comparisons in Fast Deploy."""

from rich.markup import escape

from ..protocol.models import VastModelOffer
from .format import format_money


def vast_comparison_option(row: VastModelOffer) -> str:
    """Label a rental comparison without presenting it as a deploy action."""
    price = row.costs.total_per_hour_usd
    cost = f"~{format_money(price, decimals=3)}/hr" if price is not None else "price unknown"
    return (
        f"  Vast preview · {cost} · {escape(row.gpu_label)} x{row.offer.gpu_count} "
        f"[dim]{escape(row.recipe.quant or '')}[/dim]"
    )


def vast_comparison_detail(row: VastModelOffer) -> str:
    """Show fit evidence and the storage/traffic costs behind a comparison."""
    cost = row.costs
    hourly = format_money(cost.total_per_hour_usd, decimals=4) if cost.total_per_hour_usd is not None else "unknown"
    disk = format_money(cost.disk_per_hour_usd, decimals=4) if cost.disk_per_hour_usd is not None else "unknown"
    down = format_money(cost.download_per_gb_usd, decimals=4) if cost.download_per_gb_usd is not None else "unknown"
    up = format_money(cost.upload_per_gb_usd, decimals=4) if cost.upload_per_gb_usd is not None else "unknown"
    return (
        f"[bold]Vast.ai · {escape(row.gpu_label)} x{row.offer.gpu_count}[/bold] · "
        f"{escape(row.recipe.quant or '')} · {escape(row.offer.location)}\n"
        f"Estimated {hourly}/hr including {row.disk_gb} GiB disk ({disk}/hr). "
        f"Traffic: down {down}/GB · up {up}/GB.\n"
        f"[dim]Estimated full-context GPU memory fit · offer {escape(row.offer.id)} · "
        "storage stays billable when stopped. Host/runtime not validated.[/dim]\n"
        "[yellow]Comparison only: this offer is outside the deployable runtime.[/yellow]"
    )

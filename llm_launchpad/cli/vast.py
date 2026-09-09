"""Vast account commands and read-only offer presentation."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
import sys

import typer

from ..core.vast_auth import (
    VastCredentials, clear_vast_api_key, normalize_vast_api_key,
    resolve_vast_credentials, save_vast_api_key,
)
from ..core.vast_backend import VastApiError, VastBackend
from ..protocol.models import VastOfferQuery

vast_auth_app = typer.Typer(help="Manage Vast.ai credentials for offer discovery.")
vast_app = typer.Typer(help="Manage local SSH connections to Vast rentals.")


@vast_app.command("connect")
def connect(instance_id: str) -> None:
    """Reconnect a Launchpad rental's local SSH endpoint and test streaming."""
    from ..core.vast_deployment import VastDeploymentBackend

    try:
        endpoint = VastDeploymentBackend().connect(instance_id)
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Local Vast endpoint: {endpoint.web_url}")


@vast_auth_app.command("login")
def login(
    key_stdin: bool = typer.Option(False, "--key-stdin", help="Read the key from standard input."),
) -> None:
    """Validate and store a Vast API key; prompt with hidden input by default."""
    key = sys.stdin.read() if key_stdin else typer.prompt("Vast API key", hide_input=True)
    try:
        key = normalize_vast_api_key(key)
        status = VastBackend(VastCredentials(key, "provided")).auth_status()
        if not status.authenticated:
            raise VastApiError(status.error or "Vast authentication failed.")
        path = save_vast_api_key(key)
    except (ValueError, OSError, VastApiError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Validated Vast account {status.account_id}. Key stored at {path}.")
    if os.getenv("VAST_API_KEY", "").strip():
        typer.echo("VAST_API_KEY takes precedence over the stored key.")


@vast_auth_app.command("status")
def status(
    local: bool = typer.Option(False, "--local", help="Show credential source without contacting Vast."),
) -> None:
    """Show the effective key source, verifying it unless --local is given."""
    try:
        credentials = resolve_vast_credentials()
        if not credentials.api_key:
            raise VastApiError("No Vast key configured. Run llm-launchpad vast-auth login.")
        if local:
            typer.echo(f"Vast key configured ({credentials.source}); not verified.")
            return
        result = VastBackend(credentials).auth_status()
        if not result.authenticated:
            raise VastApiError(result.error or "Vast authentication failed.")
    except (ValueError, VastApiError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Vast account {result.account_id} authenticated ({result.source}).")


@vast_auth_app.command("logout")
def logout() -> None:
    """Remove Launchpad's saved key; do not revoke or change other credentials."""
    try:
        removed = clear_vast_api_key()
        credentials = resolve_vast_credentials()
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo("Removed Launchpad's Vast key." if removed else "No saved Launchpad Vast key to remove.")
    if credentials.api_key:
        typer.echo(f"A key remains configured through {credentials.source}.")


def print_vast_offers(query: VastOfferQuery, *, output_json: bool) -> None:
    """Print normalized rental prices, never implying deployment certification."""
    try:
        rows = VastBackend().list_offers(query)
    except (ValueError, VastApiError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    if output_json:
        typer.echo(json.dumps([asdict(row) for row in rows], indent=2, allow_nan=False))
        return
    typer.echo(f"Vast offers: up to {query.limit} results, {query.disk_gb} GiB disk each. Single-GPU llama.cpp deployment is available (beta).")
    if not rows:
        typer.echo("No eligible Vast offers matched the requested filters.")
        return
    typer.echo("ID       GPU                   VRAM/GPU   Reliability  Compute/hr Disk/hr  Total/hr  Down/GB  Up/GB  Location")
    for row in rows:
        cost = row.costs
        prices = " ".join(
            f"${value:<8.4f}" if value is not None else "unknown  "
            for value in (
                cost.compute_per_hour_usd, cost.disk_per_hour_usd, cost.total_per_hour_usd,
                cost.download_per_gb_usd, cost.upload_per_gb_usd,
            )
        )
        tier = "datacenter" if row.datacenter else "verified"
        typer.echo(
            f"{row.id:<8} {row.gpu_count}x {row.gpu_type:<19} {row.gpu_memory_gb:<10.2f} "
            f"{row.reliability:<12.4f} {prices} {row.location} ({tier})"
        )
    typer.echo("Hourly totals exclude traffic. Model downloads incur transfer charges. Launchpad stop destroys the rental and disk.")

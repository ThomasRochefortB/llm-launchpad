"""Prime persistent-disk commands.

A Prime cache disk deliberately outlives the pod it was attached to, so the
next deploy of the same model skips re-downloading the weights. Stopping a
deployment therefore halves the spend rather than ending it, and until now
nothing in the product could see or remove what was left: ``delete_disk``
existed on the backend, and no command reached it.
"""

from __future__ import annotations

import typer

prime_disks_app = typer.Typer(help="Inspect and remove persistent Prime cache disks.")


def _backend():  # type: ignore[no-untyped-def]
    from ..core.prime_backend import PrimeBackend

    return PrimeBackend()


@prime_disks_app.command("list")
def list_disks() -> None:
    """Show every persistent disk on the Prime account that is still billing."""
    from ..core.prime_disks import list_retained_prime_disks

    try:
        disks = list_retained_prime_disks(_backend())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    if not disks:
        typer.echo("No Prime disks; nothing is billing.")
        return
    typer.echo(f"{len(disks)} Prime disk(s) billing:")
    for disk in disks:
        typer.echo(f"  {disk.describe()}")
    typer.echo("")
    typer.echo("Remove one with: llm-launchpad prime-disks delete <id>")


@prime_disks_app.command("delete")
def delete_disk(
    disk_id: str = typer.Argument(..., help="Disk id from 'prime-disks list'."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Permanently terminate one Prime disk and its cached model weights."""
    from ..core.prime_disks import delete_retained_prime_disk, list_retained_prime_disks

    backend = _backend()
    try:
        match = next(
            (disk for disk in list_retained_prime_disks(backend) if disk.id == disk_id),
            None,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    if match is None:
        typer.echo(f"Error: no Prime disk {disk_id} on this account.", err=True)
        raise typer.Exit(code=1)
    # Deleting a disk destroys the cached weights on it, and the next deploy
    # pays the download again. Name what is being removed before doing it.
    if not yes:
        typer.echo(f"This permanently deletes {match.describe()} and its cached weights.")
        if not typer.confirm("Delete it?", default=False):
            typer.echo("Left alone.")
            raise typer.Exit(code=1)
    try:
        typer.echo(delete_retained_prime_disk(backend, disk_id))
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

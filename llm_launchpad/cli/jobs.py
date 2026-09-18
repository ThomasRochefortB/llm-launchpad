"""Headless durable deployment jobs: list, inspect, follow, cancel."""

from __future__ import annotations

import time

import typer

jobs_app = typer.Typer(help="Inspect and control background deployment jobs.")


def _store() -> object:
    from ..core.job_store import JobStore

    return JobStore()


@jobs_app.command("list")
def list_jobs(
    all_jobs: bool = typer.Option(True, "--all/--active", help="Include finished jobs."),
) -> None:
    """Show durable deployment jobs across TUI sessions."""
    from ..core.job_store import JobStore

    store = JobStore()
    store.reconcile_workers()
    jobs = store.list_jobs(include_terminal=all_jobs)
    if not jobs:
        typer.echo("No deployment jobs.")
        return
    for job in jobs:
        state = job.status
        if job.cleanup_error:
            state += " (cleanup failed)"
        typer.echo(f"{job.id} [{state}] {job.provider}/{job.backend}/{job.app_name} — {job.outcome}")


@jobs_app.command("show")
def show_job(job_id: str = typer.Argument(..., help="Job id from 'jobs list'.")) -> None:
    """Show one job with its retained result and recent log tail."""
    from ..core.job_store import JobStore

    store = JobStore()
    store.reconcile_workers()
    job = store.get_job(job_id)
    if job is None:
        typer.echo(f"Error: unknown job {job_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"id: {job.id}")
    typer.echo(f"status: {job.status}")
    typer.echo(f"target: {job.provider}/{job.backend}/{job.app_name}")
    typer.echo(f"outcome: {job.outcome}")
    if job.resource_app_id:
        typer.echo(f"resource: {job.resource_app_id}")
    if job.cleanup_error:
        typer.echo(f"cleanup-error: {job.cleanup_error}")
    events = store.get_events(job_id)
    tail = events[-30:]
    if tail:
        typer.echo("--- recent events ---")
        for stored in tail:
            typer.echo(f"[{stored.type}] {_event_line(stored.event)}")


def _event_line(event: object) -> str:
    line = getattr(event, "line", None)
    if isinstance(line, str) and line:
        return line
    message = getattr(event, "message", None)
    if isinstance(message, str) and message:
        return message
    detail = getattr(event, "detail", None)
    if isinstance(detail, str) and detail:
        return detail
    return type(event).__name__


@jobs_app.command("follow")
def follow_job(job_id: str = typer.Argument(..., help="Job id from 'jobs list'.")) -> None:
    """Stream a job's retained log until it reaches a terminal state."""
    from ..core.job_store import JobStore

    store = JobStore()
    job = store.get_job(job_id)
    if job is None:
        typer.echo(f"Error: unknown job {job_id}", err=True)
        raise typer.Exit(code=1)
    seen = 0
    try:
        while True:
            store.reconcile_workers()
            for stored in store.get_events(job_id, after_seq=seen):
                seen = max(seen, stored.seq)
                typer.echo(f"[{stored.type}] {_event_line(stored.event)}")
            current = store.get_job(job_id)
            if current is None or current.terminal:
                if current is not None:
                    typer.echo(f"[{current.status}] {current.outcome}")
                return
            time.sleep(0.5)
    except KeyboardInterrupt:
        typer.echo("Stopped following; the job keeps running. Cancel with: llm-launchpad jobs cancel <id>")
        raise typer.Exit(code=130) from None


@jobs_app.command("cancel")
def cancel_job(job_id: str = typer.Argument(..., help="Job id from 'jobs list'.")) -> None:
    """Request cooperative cancellation; the worker stops its resource after the current provider call."""
    from ..core.job_store import JobStore

    store = JobStore()
    if store.request_cancel(job_id):
        typer.echo(f"Cancellation requested for {job_id}. The worker stops its resource after the current provider call returns.")
    else:
        job = store.get_job(job_id)
        if job is None:
            typer.echo(f"Error: unknown job {job_id}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Job {job_id} is already {job.status}; nothing to cancel.")

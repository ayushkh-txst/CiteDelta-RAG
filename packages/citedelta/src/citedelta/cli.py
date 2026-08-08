"""Command-line entry point."""

from __future__ import annotations

import asyncio
from datetime import date

import typer

from citedelta import __version__
from citedelta.config import get_settings
from substrate.obs import configure_logging, new_run_id

app = typer.Typer(
    name="citedelta",
    help="Bitemporal hybrid search over versioned regulations.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the version and exit."""
    typer.echo(f"citedelta {__version__}")


@app.command("plan")
def plan(
    on: list[str] = typer.Option(  # noqa: B008
        None, "--on", help="Specific dates. Omit for every amendment date."
    ),
) -> None:
    """Enqueue one ingestion job per snapshot date."""
    from citedelta.ingest import plan_ingest

    configure_logging(get_settings().log_level)
    new_run_id()
    dates = [date.fromisoformat(d) for d in on] if on else None
    typer.echo(f"enqueued {asyncio.run(plan_ingest(dates))} jobs")


@app.command("work")
def work(
    concurrency: int = typer.Option(2, "--concurrency", "-c"),
    forever: bool = typer.Option(False, "--forever", help="Stay up instead of draining"),
) -> None:
    """Run an ingestion worker."""
    from citedelta.ingest import run_ingest_worker

    configure_logging(get_settings().log_level)
    new_run_id()
    stats = asyncio.run(run_ingest_worker(concurrency=concurrency, drain=not forever))
    typer.echo(
        f"snapshots={stats.snapshots} created={stats.versions_created} "
        f"existing={stats.versions_existing} chunks={stats.chunks_written}"
    )


@app.command("queue-stats")
def queue_stats() -> None:
    """Show the ingest queue's state."""
    from citedelta.ingest import QUEUE_NAME
    from substrate.db import Database
    from substrate.queue import JobQueue

    async def main() -> None:
        async with Database.open(get_settings().database_url) as db:
            typer.echo((await JobQueue(db, queue=QUEUE_NAME).stats()).model_dump_json())

    asyncio.run(main())


if __name__ == "__main__":
    app()

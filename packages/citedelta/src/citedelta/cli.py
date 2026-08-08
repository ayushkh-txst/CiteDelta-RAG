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


@app.command()
def ingest(
    on: list[str] = typer.Option(  # noqa: B008
        ..., "--on", help="Snapshot date, YYYY-MM-DD. Repeatable."
    ),
) -> None:
    """Ingest one or more point-in-time snapshots of 8 CFR Part 214."""
    from citedelta.ingest import ingest_dates

    configure_logging(get_settings().log_level)
    run_id = new_run_id()
    typer.echo(f"run {run_id}")

    stats = asyncio.run(ingest_dates([date.fromisoformat(d) for d in on]))
    typer.echo(
        f"snapshots={stats.snapshots} "
        f"versions_created={stats.versions_created} "
        f"versions_existing={stats.versions_existing} "
        f"chunks={stats.chunks_written} "
        f"sections_skipped={stats.sections_skipped}"
    )


if __name__ == "__main__":
    app()

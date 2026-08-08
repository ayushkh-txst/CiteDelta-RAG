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


index_app = typer.Typer(help="Build and inspect indexes.")
app.add_typer(index_app, name="index")


@index_app.command("build")
def index_build() -> None:
    """Build the lexical index from the corpus."""
    from citedelta.index.build import build_lexical_index

    configure_logging(get_settings().log_level)
    stats = asyncio.run(build_lexical_index())
    typer.echo(
        f"docs={stats.documents} terms={stats.terms} postings={stats.postings}\n"
        f"file={stats.bytes_on_disk / 1e6:.2f} MB  "
        f"postings={stats.postings_bytes_varint / 1e6:.2f} MB varint "
        f"vs {stats.postings_bytes_fixed32 / 1e6:.2f} MB fixed-width "
        f"({stats.compression_ratio:.2f}x)"
    )


@app.command("search")
def search(
    query: str,
    k: int = typer.Option(10, "-k"),
    as_of: str = typer.Option(None, "--as-of", help="YYYY-MM-DD; post-filter (Day 3 fixes this)"),
) -> None:
    """Search the lexical index."""
    from citedelta.index.build import LEXICAL_INDEX_FILENAME
    from citedelta.index.lexical import LexicalIndex
    from substrate.db import Database

    settings = get_settings()

    async def main() -> None:
        allowed: set[int] | None = None
        rows_by_id: dict[int, tuple[str, str]] = {}
        async with Database.open(settings.database_url) as db, db.acquire() as conn:
            if as_of:
                allowed = {
                    int(r["id"])
                    for r in await conn.fetch(
                        """
                        SELECT c.id FROM chunks c
                        JOIN section_versions sv ON sv.id = c.section_version_id
                        WHERE daterange(sv.effective_from, sv.effective_to, '[)')
                              @> $1::date
                          AND sv.superseded_at IS NULL AND NOT sv.removed
                        """,
                        date.fromisoformat(as_of),
                    )
                }

            with LexicalIndex(settings.index_dir / LEXICAL_INDEX_FILENAME) as ix:
                hits = ix.search(query, k=k, allowed=allowed)

            for r in await conn.fetch(
                "SELECT id, citation_path, text FROM chunks WHERE id = ANY($1::bigint[])",
                [h.chunk_id for h in hits],
            ):
                rows_by_id[int(r["id"])] = (str(r["citation_path"]), str(r["text"]))

        for rank, hit in enumerate(hits, 1):
            cite, text = rows_by_id.get(hit.chunk_id, ("?", ""))
            typer.echo(f"{rank:2}. {hit.score:6.2f}  {cite}")
            typer.echo(f"      {text[:160]}...\n")

    asyncio.run(main())


if __name__ == "__main__":
    app()

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


embed_app = typer.Typer(help="Embed the corpus.")
app.add_typer(embed_app, name="embed")


@embed_app.command("run")
def embed_run(batch_size: int = typer.Option(64, "--batch-size")) -> None:
    """Embed every distinct chunk text not already cached. Resumable."""
    from citedelta.embed.corpus import embed_corpus

    configure_logging(get_settings().log_level)
    new_run_id()
    stats = asyncio.run(embed_corpus(batch_size=batch_size))
    typer.echo(
        f"chunks={stats.chunks_total} distinct={stats.distinct_texts} "
        f"cached={stats.already_cached} embedded={stats.newly_embedded} "
        f"dedup={stats.dedup_ratio:.2f}x"
    )


@embed_app.command("status")
def embed_status() -> None:
    """How much of the corpus is embedded."""
    from substrate.db import Database

    async def main() -> None:
        async with Database.open(get_settings().database_url) as db, db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT (SELECT count(*) FROM chunks)                          AS chunks,
                       (SELECT count(DISTINCT content_sha256) FROM chunks)    AS distinct_texts,
                       (SELECT count(*) FROM embeddings WHERE model_id = $1)  AS embedded
                """,
                "BAAI/bge-small-en-v1.5",
            )
            done, total = int(row["embedded"]), int(row["distinct_texts"])
            typer.echo(
                f"chunks={row['chunks']} distinct={total} embedded={done} "
                f"({100 * done / max(total, 1):.1f}%)"
            )

    asyncio.run(main())


vector_app = typer.Typer(help="Vector search.")
app.add_typer(vector_app, name="vector")


@vector_app.command("describe")
def vector_describe() -> None:
    """Measure the corpus's geometry."""
    from citedelta.embed.corpus import load_corpus_vectors
    from citedelta.index.analysis import describe

    configure_logging(get_settings().log_level)
    _, vectors = asyncio.run(load_corpus_vectors())
    g = describe(vectors)
    typer.echo(
        f"vectors={g.n_vectors}  distinct={g.n_distinct} "
        f"(duplicate ratio {g.duplicate_ratio:.2f}x)\n"
        f"ambient_dim={g.ambient_dim}  intrinsic_dim={g.intrinsic_dim:.1f}\n"
        f"mean nearest-neighbour distance={g.mean_nn_distance:.4f}"
    )


@vector_app.command("search")
def vector_search(
    query: str,
    k: int = typer.Option(5, "-k"),
    as_of: str | None = typer.Option(None, "--as-of", help="YYYY-MM-DD; exact temporal filter"),
) -> None:
    """Exact semantic search, optionally restricted to what was in force."""
    import time

    from citedelta.embed.corpus import load_corpus_vectors
    from citedelta.embed.local import LocalEmbeddings
    from citedelta.index.brute import BruteForceIndex
    from citedelta.index.vector import BoolMask
    from citedelta.ingest import EXTERNAL_ID
    from citedelta.store.corpus import CorpusStore
    from citedelta.temporal import AdmissibleSet, AsOf
    from substrate.db import Database

    configure_logging(get_settings().log_level)

    async def main() -> None:
        ids, vectors = await load_corpus_vectors()
        index = BruteForceIndex()
        index.build(ids, vectors)

        mask: BoolMask | None = None
        async with Database.open(get_settings().database_url) as db, db.acquire() as conn:
            store = CorpusStore(conn)
            if as_of:
                doc = await conn.fetchval(
                    "SELECT id FROM documents WHERE external_id = $1", EXTERNAL_ID
                )
                point = AsOf(valid_on=date.fromisoformat(as_of))
                adm = AdmissibleSet.from_as_of(
                    await store.admissible_chunk_ids(int(doc), point), point, len(ids)
                )
                typer.echo(
                    f"admissible {adm.size}/{len(ids)} "
                    f"(selectivity {adm.selectivity:.2%}) at {as_of}\n"
                )
                mask = index.compile_filter(adm.ids)

            q = LocalEmbeddings().embed([query])[0]
            t0 = time.perf_counter()
            hits = index.search(q, k, admissible=mask)
            elapsed = (time.perf_counter() - t0) * 1000

        async with Database.open(get_settings().database_url) as db, db.acquire() as conn:
            rows = {
                int(r["id"]): (str(r["citation_path"]), str(r["text"]))
                for r in await conn.fetch(
                    "SELECT id, citation_path, text FROM chunks WHERE id = ANY($1::bigint[])",
                    [h.id for h in hits],
                )
            }

        typer.echo(f"exact search over {index.size} vectors in {elapsed:.1f} ms\n")
        for rank, hit in enumerate(hits, 1):
            cite, text = rows.get(hit.id, ("?", ""))
            typer.echo(f"{rank:2}. sim={hit.similarity:.3f}  {cite}")
            typer.echo(f"      {text[:170]}…\n")

    asyncio.run(main())


bench_app = typer.Typer(help="Benchmarks.")
app.add_typer(bench_app, name="bench")


@bench_app.command("run")
def bench_run(
    dataset: str = typer.Option(
        "cfr-full", "--dataset", help="cfr-full | cfr-dedup | random-hard | all"
    ),
    k: int = typer.Option(10, "-k"),
    out: str = typer.Option("docs/design/benchmarks/results.json", "--out"),
) -> None:
    """Benchmark every registered index against a dataset."""
    from pathlib import Path

    from citedelta.bench.registry import run_suite

    configure_logging(get_settings().log_level)
    new_run_id()
    results = asyncio.run(run_suite(dataset, k=k))

    from citedelta.bench.runner import as_markdown, save_results

    save_results(results, Path(out))
    typer.echo(as_markdown(results))


@bench_app.command("plot")
def bench_plot(
    results: list[str] = typer.Option(  # noqa: B008
        None, "--results", help="Result JSON files. Defaults to docs/design/benchmarks/*.json"
    ),
    out: str = typer.Option("docs/design/benchmarks/recall-vs-qps.png", "--out"),
) -> None:
    """Render the recall-vs-QPS plot."""
    from pathlib import Path

    from citedelta.bench.plot import plot_results

    paths = (
        [Path(r) for r in results]
        if results
        else sorted(Path("docs/design/benchmarks").glob("*.json"))
    )
    plot_results(paths, Path(out), title="CiteDelta — hand-written ANN indexes")
    typer.echo(f"wrote {out}")


@bench_app.command("collapse")
def bench_collapse(
    as_of: str = typer.Option("2019-06-01", "--as-of"),
    k: int = typer.Option(10, "-k"),
    queries: int = typer.Option(300, "--queries"),
    out: str = typer.Option("docs/design/benchmarks/temporal-collapse.json", "--out"),
) -> None:
    """Measure how badly post-filtering an ANN result set fails."""
    import json
    from pathlib import Path

    from citedelta.bench.temporal import as_markdown, measure_collapse, to_json

    configure_logging(get_settings().log_level)
    new_run_id()
    result = asyncio.run(measure_collapse(date.fromisoformat(as_of), k=k, n_queries=queries))

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(result), indent=2) + "\n")
    typer.echo(as_markdown(result))
    typer.echo(f"wrote {out}")


@bench_app.command("lexical-collapse")
def bench_lexical_collapse(
    as_of: str = typer.Option("2019-06-01", "--as-of"),
    k: int = typer.Option(10, "-k"),
) -> None:
    """BM25 post-filtering vs in-index filtering, same ranker."""
    from citedelta.bench.queries import DOMAIN_QUERIES
    from citedelta.bench.temporal import load_admissible, measure_lexical_collapse
    from citedelta.embed.corpus import load_corpus_vectors
    from citedelta.index.build import LEXICAL_INDEX_FILENAME
    from citedelta.index.lexical import LexicalIndex

    settings = get_settings()
    configure_logging(settings.log_level)

    async def main() -> None:
        ids, _ = await load_corpus_vectors()
        admissible = await load_admissible(date.fromisoformat(as_of), len(ids))
        with LexicalIndex(settings.index_dir / LEXICAL_INDEX_FILENAME) as ix:
            result = measure_lexical_collapse(ix, DOMAIN_QUERIES, admissible, k=k)

        typer.echo(f"\nas_of={as_of}  selectivity={result.selectivity:.2%}\n")
        typer.echo(f"  in-index recall@10    {result.in_index_recall:.3f}  (exact by construction)")
        typer.echo(f"  post-filter recall@10 {result.post_filter_recall:.3f}")
        typer.echo(f"  post-filter zero-rate {result.post_filter_zero_rate:.3f}")
        typer.echo(f"  in-index latency      {result.in_index_ms:.2f} ms")
        typer.echo(f"  unfiltered latency    {result.unfiltered_ms:.2f} ms")

    asyncio.run(main())


def search(
    query: str,
    k: int = typer.Option(10, "-k"),
    as_of: str = typer.Option(
        None, "--as-of", help="YYYY-MM-DD; exact temporal filter inside the postings scan"
    ),
) -> None:
    """Search the lexical index."""
    from citedelta.index.build import LEXICAL_INDEX_FILENAME
    from citedelta.index.lexical import LexicalIndex
    from substrate.db import Database

    settings = get_settings()

    async def main() -> None:
        admissible_ids: set[int] | None = None
        rows_by_id: dict[int, tuple[str, str]] = {}
        async with Database.open(settings.database_url) as db, db.acquire() as conn:
            if as_of:
                admissible_ids = {
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
                mask = ix.compile_filter(admissible_ids) if admissible_ids is not None else None
                hits = ix.search(query, k=k, admissible=mask)

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

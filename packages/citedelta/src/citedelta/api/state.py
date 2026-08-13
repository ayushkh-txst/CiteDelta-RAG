"""Everything expensive, built once at startup and shared read-only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import structlog

from citedelta.answer.service import AnswerService
from citedelta.config import Settings
from citedelta.embed.local import LocalEmbeddings
from citedelta.index.brute import BruteForceIndex
from citedelta.index.build import LEXICAL_INDEX_FILENAME
from citedelta.index.lexical import LexicalIndex
from citedelta.index.vector import VectorIndex
from substrate.db import Database
from substrate.llm import Completions
from substrate.llm.anthropic_adapter import AnthropicCompletions
from substrate.llm.pricing import CostLedger

log = structlog.get_logger(__name__)


@dataclass
class AppState:
    db: Database
    lexical: LexicalIndex
    vector: VectorIndex
    embeddings: LocalEmbeddings
    answers: AnswerService
    resolver_llm: Completions
    ledger: CostLedger
    corpus_size: int
    corpus_since: date
    """Earliest date the corpus can answer from. Dates before this always
    refuse, so the UI's as-of input floors at it rather than promising dates
    the data can't deliver."""
    snapshot_count: int
    """Number of distinct effective dates in the corpus — how many times the
    regulation was captured."""

    amendment_dates: list[date]
    """The 78 real effective dates, sorted, read-only. Every turn's gutter and
    rupture arithmetic needs them, so they load once at startup rather than
    on the request path."""


async def build_state(settings: Settings) -> AppState:
    """Load once. Everything here is read-only afterwards, which is what
    makes it safe to share across concurrent requests without a lock."""
    from citedelta.embed.corpus import load_corpus_vectors

    # Construct and connect explicitly: `await Database.open(dsn).__aenter__()`
    # drops the temporary async context manager, whose aclose() then runs the
    # generator's finally block and closes the pool we just built.
    db = Database(settings.database_url)
    await db.connect()

    async with db.acquire() as conn:
        corpus_since = await conn.fetchval("SELECT min(effective_from) FROM section_versions")
        snapshot_count = await conn.fetchval(
            "SELECT count(DISTINCT effective_from) FROM section_versions"
        )
        rows = await conn.fetch(
            "SELECT DISTINCT effective_from FROM section_versions ORDER BY effective_from"
        )
    if corpus_since is None:
        log.warning("api.corpus_empty")
        corpus_since = date(2016, 1, 1)
    amendment_dates = [r["effective_from"] for r in rows]
    log.info(
        "api.corpus_since",
        since=str(corpus_since),
        snapshots=snapshot_count,
        amendments=len(amendment_dates),
    )

    ids, vectors = await load_corpus_vectors()
    log.info("api.vectors_loaded", count=len(ids), mb=round(vectors.nbytes / 1e6, 1))

    # BruteForceIndex for now: exact, and at 38k vectors it is fast enough
    # to serve while staying obviously correct. Swapping in HNSW is a
    # one-line change here precisely because both satisfy VectorIndex —
    # which is the payoff for the shared index protocol, cashed in.
    vector = BruteForceIndex()
    vector.build(ids, vectors)

    lexical = LexicalIndex(settings.index_dir / LEXICAL_INDEX_FILENAME)
    lexical.__enter__()

    embeddings = LocalEmbeddings()
    # Warm the ONNX session so the first real request doesn't pay for it.
    embeddings.embed(["warmup"])

    # One ledger across both models, so a turn's reported cost is the turn's
    # actual cost: resolver + answer land under the same run_id and the total
    # is what gets persisted.
    ledger = CostLedger()
    llm = AnthropicCompletions(api_key=settings.anthropic_api_key, ledger=ledger)
    resolver_llm = AnthropicCompletions(api_key=settings.anthropic_api_key, ledger=ledger)
    answers = AnswerService(llm, model=settings.llm_model, max_tokens=settings.llm_max_tokens)

    return AppState(
        db=db,
        lexical=lexical,
        vector=vector,
        embeddings=embeddings,
        answers=answers,
        resolver_llm=resolver_llm,
        ledger=ledger,
        corpus_size=len(ids),
        corpus_since=corpus_since,
        snapshot_count=len(amendment_dates),
        amendment_dates=amendment_dates,
    )


async def close_state(state: AppState) -> None:
    state.lexical.__exit__(None, None, None)
    await state.db.__aexit__(None, None, None)

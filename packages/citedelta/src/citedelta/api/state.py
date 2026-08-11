"""Everything expensive, built once at startup and shared read-only."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from citedelta.answer.service import AnswerService
from citedelta.config import Settings
from citedelta.embed.local import LocalEmbeddings
from citedelta.index.brute import BruteForceIndex
from citedelta.index.build import LEXICAL_INDEX_FILENAME
from citedelta.index.lexical import LexicalIndex
from citedelta.index.vector import VectorIndex
from substrate.db import Database
from substrate.llm.anthropic_adapter import AnthropicCompletions

log = structlog.get_logger(__name__)


@dataclass
class AppState:
    db: Database
    lexical: LexicalIndex
    vector: VectorIndex
    embeddings: LocalEmbeddings
    answers: AnswerService
    corpus_size: int


async def build_state(settings: Settings) -> AppState:
    """Load once. Everything here is read-only afterwards, which is what
    makes it safe to share across concurrent requests without a lock."""
    from citedelta.embed.corpus import load_corpus_vectors

    # Construct and connect explicitly: `await Database.open(dsn).__aenter__()`
    # drops the temporary async context manager, whose aclose() then runs the
    # generator's finally block and closes the pool we just built.
    db = Database(settings.database_url)
    await db.connect()

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

    llm = AnthropicCompletions(api_key=settings.anthropic_api_key)
    answers = AnswerService(llm, model=settings.llm_model, max_tokens=settings.llm_max_tokens)

    return AppState(
        db=db,
        lexical=lexical,
        vector=vector,
        embeddings=embeddings,
        answers=answers,
        corpus_size=len(ids),
    )


async def close_state(state: AppState) -> None:
    state.lexical.__exit__(None, None, None)
    await state.db.__aexit__(None, None, None)

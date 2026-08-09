"""The port's guarantees, and the cache's key discipline."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

import numpy as np
import pytest

from citedelta.embed.base import EmbeddingProvider, Vectors
from substrate.db import Database


class FakeEmbeddings:
    """A deterministic stand-in. No test should download a 130 MB model.

    Hashing text into a vector gives stable, distinct, comparable outputs
    without inference — enough to exercise every code path around the
    provider. The real model gets exercised by the corpus run and by the
    recall numbers measured later.
    """

    def __init__(self, dim: int = 8, model_id: str = "fake-v1") -> None:
        self._dim = dim
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> Vectors:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self._dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def test_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeEmbeddings(), EmbeddingProvider)


def test_output_is_unit_normalized() -> None:
    v = FakeEmbeddings().embed(["alpha", "beta", "gamma"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)


def test_output_order_matches_input_order() -> None:
    p = FakeEmbeddings()
    both = p.embed(["alpha", "beta"])
    assert np.allclose(both[0], p.embed(["alpha"])[0])
    assert np.allclose(both[1], p.embed(["beta"])[0])


def test_empty_input_returns_empty_matrix_not_an_error() -> None:
    v = FakeEmbeddings().embed([])
    assert v.shape == (0, 8)


def test_normalized_dot_product_is_cosine_similarity() -> None:
    """The identity every index downstream relies on."""
    v = FakeEmbeddings(dim=16).embed(["a", "b"])
    dot = float(v[0] @ v[1])
    cosine = float(v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1])))
    assert dot == pytest.approx(cosine, abs=1e-6)


def test_identical_text_embeds_identically() -> None:
    """Why content-hash caching is sound in the first place."""
    p = FakeEmbeddings()
    assert np.array_equal(p.embed(["same text"]), p.embed(["same text"]))


@pytest.mark.integration
async def test_cache_embeds_each_distinct_text_once(clean_db: Database) -> None:
    """Two chunks with identical text must produce ONE embeddings row."""
    from citedelta.embed.corpus import embed_corpus

    text = "Optional practical training may be authorized."
    digest = sha256(text.encode()).digest()

    async with clean_db.acquire() as conn, conn.transaction():
        src = await conn.fetchval(
            "INSERT INTO sources (slug, name, base_url) VALUES ('t','T','u') RETURNING id"
        )
        doc = await conn.fetchval(
            """INSERT INTO documents (source_id, external_id, title, citation)
               VALUES ($1,'d','T','C') RETURNING id""",
            src,
        )
        sv = await conn.fetchval(
            """INSERT INTO section_versions
                 (document_id, section, effective_from, issue_date, content_sha256)
               VALUES ($1,'214.2','2020-01-01','2020-01-01',$2) RETURNING id""",
            doc,
            digest,
        )
        # Two chunks, same text, therefore the same content hash.
        for ordinal in (0, 1):
            await conn.execute(
                """INSERT INTO chunks (section_version_id, ordinal, citation_path,
                                       text, char_count, token_count, content_sha256)
                   VALUES ($1,$2,'8 CFR 214.2(f)',$3,$4,$5,$6)""",
                sv,
                ordinal,
                text,
                len(text),
                len(text.split()),
                digest,
            )

    stats = await embed_corpus(FakeEmbeddings())

    assert stats.chunks_total == 2
    assert stats.distinct_texts == 1
    assert stats.newly_embedded == 1  # embedded once, not twice
    assert stats.dedup_ratio == 2.0

    # And it is genuinely idempotent: a second pass does no work.
    again = await embed_corpus(FakeEmbeddings())
    assert again.newly_embedded == 0

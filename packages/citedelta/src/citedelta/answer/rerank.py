"""The reranker seam.

The honest note goes in the docstring rather than in a commit message six
weeks from now:

CiteDelta's corpus has intrinsic dimensionality ~1.1 in a 384-dimensional
space (measured). Retrieval is already near-perfect — filtered HNSW
recall@10 hits 1.000 at ef=16. A cross-encoder reranker reorders a list that
is already essentially correct, so on THIS corpus it buys almost nothing and
adds a model call to every query.

The seam exists anyway because the interesting version of this project is the
one where the corpus is harder, and because Ratchet reuses the protocol. What
does not exist is a cross-encoder implementation nobody measured a need for.
"""

from __future__ import annotations

from typing import Protocol

from citedelta.answer.models import Citation


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Citation], *, k: int) -> list[Citation]: ...


class PassthroughReranker:
    """Keeps fusion order. The default, and — measurably — good enough here."""

    def rerank(self, query: str, candidates: list[Citation], *, k: int) -> list[Citation]:
        return candidates[:k]

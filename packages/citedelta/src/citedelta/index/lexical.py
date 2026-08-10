"""A hand-written inverted index with BM25 ranking.

FILE FORMAT (little-endian throughout)

  header    60 bytes  magic | version | n_docs | n_terms | avgdl | 4 offsets
  doc ids   n_docs x u64      internal index -> external chunk id
  doc lens  n_docs x u32      term count per document, for normalization
  terms     n_terms x (varint len | utf8 | varint df | varint off | varint len)
                              sorted, so the dictionary is scannable
  postings  per term: df x (varint doc_gap, varint tf)

Design notes worth defending:

* The dictionary is read into RAM at open (small: a few hundred KB here). The
  postings blob is NOT — it stays on disk and is reached through mmap, so the
  OS page cache decides what's resident. That's the standard split, and it is
  why a search engine can serve an index far larger than its heap.

* Doc ids inside a postings list are stored as GAPS. Sorted ids yield small
  gaps, small gaps yield short varints. This costs one addition per posting
  during decode and buys roughly half the file.

* External chunk ids are indirected through a table rather than stored in the
  postings, so gaps stay dense (0,1,2,...) instead of tracking Postgres'
  bigint sequence.
"""

from __future__ import annotations

import heapq
import math
import mmap
import os
import struct
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import structlog
from numpy.typing import NDArray

from citedelta.index.tokenize import tokenize
from citedelta.index.varint import read_varint, write_varint

log = structlog.get_logger(__name__)

MAGIC = b"CDLXIDX1"
FORMAT_VERSION = 1
_HEADER = struct.Struct("<8sIIIdQQQQ")

K1 = 1.2
B = 0.75


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: int
    score: float


@dataclass
class BuildStats:
    documents: int = 0
    terms: int = 0
    postings: int = 0
    bytes_on_disk: int = 0
    postings_bytes_varint: int = 0
    postings_bytes_fixed32: int = 0  # what a naive u32 pair array would cost

    @property
    def compression_ratio(self) -> float:
        if not self.postings_bytes_varint:
            return 0.0
        return self.postings_bytes_fixed32 / self.postings_bytes_varint


# --------------------------------------------------------------------- build


def build_index(docs: Iterable[tuple[int, str]], path: Path) -> BuildStats:
    """Build an index over (chunk_id, text) pairs and write it atomically."""
    external_ids: list[int] = []
    doc_lens: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for internal_id, (chunk_id, text) in enumerate(docs):
        terms = tokenize(text)
        external_ids.append(chunk_id)
        doc_lens.append(len(terms))
        # Documents arrive in increasing internal_id, so every postings list
        # is already sorted. No sort pass, and gap encoding just works.
        for term, tf in Counter(terms).items():
            postings[term].append((internal_id, tf))

    n_docs = len(external_ids)
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0

    postings_blob = bytearray()
    dictionary = bytearray()
    total_postings = 0

    for term in sorted(postings):
        plist = postings[term]
        start = len(postings_blob)
        prev = 0
        for internal_id, tf in plist:
            write_varint(internal_id - prev, postings_blob)  # gap, not absolute
            write_varint(tf, postings_blob)
            prev = internal_id
        length = len(postings_blob) - start
        total_postings += len(plist)

        term_bytes = term.encode()
        write_varint(len(term_bytes), dictionary)
        dictionary.extend(term_bytes)
        write_varint(len(plist), dictionary)  # df
        write_varint(start, dictionary)
        write_varint(length, dictionary)

    off_docids = _HEADER.size
    off_doclens = off_docids + n_docs * 8
    off_terms = off_doclens + n_docs * 4
    off_postings = off_terms + len(dictionary)

    header = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        n_docs,
        len(postings),
        avgdl,
        off_docids,
        off_doclens,
        off_terms,
        off_postings,
    )

    payload = b"".join(
        [
            header,
            struct.pack(f"<{n_docs}Q", *external_ids),
            struct.pack(f"<{n_docs}I", *doc_lens),
            bytes(dictionary),
            bytes(postings_blob),
        ]
    )
    _write_atomic(path, payload)

    stats = BuildStats(
        documents=n_docs,
        terms=len(postings),
        postings=total_postings,
        bytes_on_disk=len(payload),
        postings_bytes_varint=len(postings_blob),
        postings_bytes_fixed32=total_postings * 8,  # u32 docid + u32 tf
    )
    log.info(
        "index.built",
        docs=stats.documents,
        terms=stats.terms,
        postings=stats.postings,
        bytes=stats.bytes_on_disk,
        compression=round(stats.compression_ratio, 2),
    )
    return stats


def _write_atomic(path: Path, data: bytes) -> None:
    """Temp file -> fsync -> rename. The index is never observed half-written.

    If the disk fills, the write fails on the TEMP file and the previous index
    is still intact. Writing in place would leave a truncated file that opens
    successfully and returns wrong answers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fh: BinaryIO
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # fsync the directory too, or the rename itself may not survive a power cut.
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------- read


class LexicalIndex:
    """Read-only, mmap-backed."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(self._mm)

        (
            magic,
            version,
            self.n_docs,
            n_terms,
            self.avgdl,
            off_docids,
            off_doclens,
            off_terms,
            self._off_postings,
        ) = _HEADER.unpack_from(self._mm, 0)

        if magic != MAGIC:
            msg = f"not a CiteDelta index: {magic!r}"
            raise ValueError(msg)
        if version != FORMAT_VERSION:
            msg = f"index format v{version}, expected v{FORMAT_VERSION}"
            raise ValueError(msg)

        self._doc_ids = np.asarray(
            struct.unpack_from(f"<{self.n_docs}Q", self._mm, off_docids), dtype=np.int64
        )
        self._doc_lens = np.asarray(
            struct.unpack_from(f"<{self.n_docs}I", self._mm, off_doclens), dtype=np.int32
        )

        # Dictionary into RAM; postings stay on disk behind the mmap.
        self._terms: dict[str, tuple[int, int, int]] = {}
        pos = off_terms
        for _ in range(n_terms):
            length, pos = read_varint(view, pos)
            term = bytes(view[pos : pos + length]).decode()
            pos += length
            df, pos = read_varint(view, pos)
            off, pos = read_varint(view, pos)
            plen, pos = read_varint(view, pos)
            self._terms[term] = (df, off, plen)

    def close(self) -> None:
        self._mm.close()
        self._file.close()

    def __enter__(self) -> LexicalIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def vocabulary_size(self) -> int:
        return len(self._terms)

    def postings(self, term: str) -> Iterator[tuple[int, int]]:
        """Decode one postings list, undoing the gaps."""
        entry = self._terms.get(term)
        if entry is None:
            return
        df, off, plen = entry
        base = self._off_postings + off
        view = memoryview(self._mm)[base : base + plen]
        pos = 0
        doc = 0
        for _ in range(df):
            gap, pos = read_varint(view, pos)
            tf, pos = read_varint(view, pos)
            doc += gap
            yield doc, tf

    def _idf(self, df: int) -> float:
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def compile_filter(self, admissible_ids: Collection[int]) -> NDArray[np.bool_]:
        """External chunk ids → mask over internal document positions."""
        wanted = np.fromiter(admissible_ids, dtype=np.int64, count=len(admissible_ids))
        return np.isin(self._doc_ids, wanted)

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        admissible: NDArray[np.bool_] | None = None,
    ) -> list[Hit]:
        """Top-k by BM25, optionally restricted to an admissible subset.

        The filter is applied INSIDE the postings traversal — before scoring,
        and therefore long before the top-k heap. That makes the result exact:
        the k best admissible documents, not the admissible remainder of the k
        best documents. Those are different sets, and the collapse measurement
        shows how different.

        Skipping early is also strictly cheaper. At ~2% selectivity roughly
        98% of postings are discarded before the BM25 arithmetic runs, so a
        filtered query is FASTER than an unfiltered one. That is the opposite
        of what happens to the graph index, and the contrast is the
        interesting part.
        """
        terms = tokenize(query)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            entry = self._terms.get(term)
            if entry is None:
                continue
            idf = self._idf(entry[0])
            for internal_id, tf in self.postings(term):
                if admissible is not None and not admissible[internal_id]:
                    continue
                dl = self._doc_lens[internal_id]
                denom = tf + K1 * (1.0 - B + B * dl / self.avgdl)
                scores[internal_id] += idf * (tf * (K1 + 1.0)) / denom

        # A bounded heap, not a full sort. Ties break on internal id so
        # results are deterministic — which is what makes the conformance
        # test below meaningful.
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [Hit(chunk_id=int(self._doc_ids[i]), score=s) for i, s in top]

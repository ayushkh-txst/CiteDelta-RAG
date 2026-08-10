"""Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016).

Distances are cosine on unit-normalized vectors, so distance = 1 - dot.

The one performance decision worth stating up front: every place that needs
distances to a SET of nodes computes them in a single matmul rather than in
a Python loop. In pure Python the interpreter overhead per distance dwarfs
the arithmetic, and `search_layer` is nothing but repeated set-distance
queries. Batching is worth roughly an order of magnitude here, and it is why
a from-scratch build of 38k vectors finishes in minutes rather than an hour.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any, Self

import numpy as np
import structlog

from citedelta.index.vector import (
    BoolMask,
    Ids,
    Neighbor,
    Vectors,
    compile_mask,
)

log = structlog.get_logger(__name__)


class HNSWIndex:
    def __init__(
        self,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 42,
        max_visits: int = 20_000,
    ) -> None:
        self._m = m
        # Layer 0 gets twice the budget: it holds every node and does all the
        # fine-grained work, so it can afford — and needs — denser wiring.
        self._m0 = 2 * m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._ml = 1.0 / math.log(m)
        self._seed = seed

        # Latency guardrail. Under a very selective filter the search cannot
        # stop early (see _search_layer), so without a ceiling a pathological
        # query walks the entire graph. Bounding VISITS converts an unbounded
        # latency into bounded latency with degraded recall — the right trade
        # for a request path, and a policy choice, so it is configurable.
        self._max_visits = max_visits
        self._last_visits = 0

        self._ids: Ids = np.zeros(0, dtype=np.int64)
        self._vectors: Vectors = np.zeros((0, 0), dtype=np.float32)
        self._levels: np.ndarray = np.zeros(0, dtype=np.int32)
        self._graph: list[dict[int, list[int]]] = []
        self._entry: int | None = None
        self._max_level: int = 0

    # ------------------------------------------------------------- properties

    @property
    def name(self) -> str:
        return "hnsw"

    @property
    def size(self) -> int:
        return int(self._ids.shape[0])

    @property
    def dimensions(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.size else 0

    @property
    def max_level(self) -> int:
        return self._max_level

    # ---------------------------------------------------------------- helpers

    def _distances(self, query: Vectors, nodes: list[int]) -> np.ndarray:
        """Distance from `query` to every node in `nodes`, in ONE matmul."""
        return 1.0 - (self._vectors[nodes] @ query)

    def _search_layer(
        self,
        query: Vectors,
        entry: list[int],
        ef: int,
        layer: int,
        *,
        admissible: BoolMask | None = None,
        budget: int | None = None,
    ) -> list[tuple[float, int]]:
        """Greedy best-first search in one layer, optionally filtered.

        Exactly two things change when `admissible` is supplied, and the
        asymmetry between them IS the algorithm:

          * Which nodes may enter `results`  — only admissible ones.
          * Which nodes may be EXPANDED      — unchanged. All of them.

        Every inadmissible node is still traversed, because at 1.9%
        selectivity the admissible subgraph has mean degree 1.04 and shatters
        into ~500 fragments (§4.1). Inadmissible nodes are the only paths
        between admissible regions.

        The stopping rule then does the rest of the work by itself. Normally
        we stop once the nearest remaining candidate is worse than the worst
        result held — but that requires `results` to be full. Under a filter
        `results` fills slowly, so the same condition automatically keeps the
        search expanding until it has found ef ADMISSIBLE neighbours. No
        special case; the invariant 'expand while the result set is not yet
        good enough' was already the right one.
        """
        visited = set(entry)
        seed_distances = self._distances(query, entry)

        candidates: list[tuple[float, int]] = [
            (float(d), i) for d, i in zip(seed_distances, entry, strict=True)
        ]
        heapq.heapify(candidates)

        # Only admissible entry points seed the result set. Note `results` may
        # legitimately start EMPTY, which unfiltered search can never do.
        results: list[tuple[float, int]] = []
        for d, i in zip(seed_distances, entry, strict=True):
            if admissible is None or admissible[i]:
                heapq.heappush(results, (-float(d), i))

        adjacency = self._graph[layer]
        visits = 0

        while candidates:
            distance, node = heapq.heappop(candidates)
            if len(results) >= ef and distance > -results[0][0]:
                break

            unvisited = [n for n in adjacency.get(node, ()) if n not in visited]
            if not unvisited:
                continue
            visited.update(unvisited)
            visits += len(unvisited)

            for raw, neighbour in zip(self._distances(query, unvisited), unvisited, strict=True):
                d = float(raw)
                # inf while the result set is underfull, so nothing is pruned
                # from the frontier until we actually have ef good answers.
                worst = -results[0][0] if len(results) >= ef else math.inf
                if d >= worst:
                    continue
                heapq.heappush(candidates, (d, neighbour))
                if admissible is None or admissible[neighbour]:
                    heapq.heappush(results, (-d, neighbour))
                    if len(results) > ef:
                        heapq.heappop(results)

            if budget is not None and visits >= budget:
                break

        self._last_visits = visits
        return [(-negated, node) for negated, node in results]

    def _select_neighbours(
        self, base: Vectors, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Algorithm 4. Diversity, not just proximity — see §5.3."""
        kept: list[int] = []
        for distance, candidate in sorted(candidates):
            if len(kept) >= m:
                break
            if not kept:
                kept.append(candidate)
                continue
            # Batched: distance from this candidate to everything kept so far.
            to_kept = 1.0 - (self._vectors[kept] @ self._vectors[candidate])
            if bool(np.all(to_kept >= distance)):
                kept.append(candidate)

        # Backfill if the heuristic was too strict. An under-connected node is
        # worse than a slightly redundant edge: it can strand a whole region.
        if len(kept) < m:
            chosen = set(kept)
            for _, candidate in sorted(candidates):
                if len(kept) >= m:
                    break
                if candidate not in chosen:
                    kept.append(candidate)
                    chosen.add(candidate)
        return kept

    # ------------------------------------------------------------------ build

    def build(self, ids: Ids, vectors: Vectors) -> None:
        if len(ids) != len(vectors):
            msg = f"{len(ids)} ids but {len(vectors)} vectors"
            raise ValueError(msg)

        self._ids = np.ascontiguousarray(ids, dtype=np.int64)
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(ids)
        self._graph = [{}]
        self._entry = None
        self._max_level = 0
        if n == 0:
            self._levels = np.zeros(0, dtype=np.int32)
            return

        rng = np.random.default_rng(self._seed)
        # Seeded, so a build is reproducible. A benchmark you cannot reproduce
        # is an anecdote.
        self._levels = np.floor(-np.log(rng.random(n)) * self._ml).astype(np.int32)

        for i in range(n):
            self._insert(i)
            if (i + 1) % 5000 == 0:
                log.info("hnsw.progress", inserted=i + 1, total=n, max_level=self._max_level)

        log.info(
            "hnsw.built",
            n=n,
            max_level=self._max_level,
            m=self._m,
            ef_construction=self._ef_construction,
        )

    def _insert(self, node: int) -> None:
        level = int(self._levels[node])
        while len(self._graph) <= level:
            self._graph.append({})

        if self._entry is None:
            for layer in range(level + 1):
                self._graph[layer][node] = []
            self._entry = node
            self._max_level = level
            return

        query = self._vectors[node]
        entry_points = [self._entry]

        # Phase 1 — descend the express lanes with ef=1. No connections made
        # here; this only finds a good place to start.
        for layer in range(self._max_level, level, -1):
            entry_points = [self._search_layer(query, entry_points, 1, layer)[0][1]]

        # Phase 2 — from this node's level down to 0, find neighbours and wire in.
        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(query, entry_points, self._ef_construction, layer)
            budget = self._m0 if layer == 0 else self._m
            neighbours = self._select_neighbours(query, candidates, budget)
            self._graph[layer][node] = list(neighbours)

            # Edges are UNDIRECTED: add the reverse link too. Forget this and
            # the new node can reach its neighbours but nothing can reach it —
            # newly inserted vectors become invisible to search.
            for neighbour in neighbours:
                adjacency = self._graph[layer].setdefault(neighbour, [])
                adjacency.append(node)
                if len(adjacency) > budget:
                    # Over budget: re-run the SAME heuristic from the
                    # neighbour's point of view. Truncating to the nearest few
                    # instead would strip exactly the long-range edges that
                    # make the graph navigable.
                    base = self._vectors[neighbour]
                    scored = [
                        (float(1.0 - base @ self._vectors[other]), other) for other in adjacency
                    ]
                    pruned = set(self._select_neighbours(base, scored, budget))
                    # Keep the graph SYMMETRIC: any member that was just pruned
                    # out of `neighbour`'s list still points to it, so drop
                    # those stale forward edges too — every edge is reverse-able.
                    for other in adjacency:
                        if other not in pruned:
                            stale = self._graph[layer].get(other)
                            if stale and neighbour in stale:
                                stale.remove(neighbour)
                    self._graph[layer][neighbour] = list(pruned)

            entry_points = [node_id for _, node_id in candidates]

        # Layers above the previous maximum: this node is alone up there.
        for layer in range(self._max_level + 1, level + 1):
            self._graph[layer][node] = []

        if level > self._max_level:
            self._max_level = level
            self._entry = node

    # ----------------------------------------------------------------- search

    def compile_filter(self, admissible_ids: Collection[int]) -> BoolMask:
        return compile_mask(self._ids, admissible_ids)

    @property
    def last_visits(self) -> int:
        """Distance computations in the most recent search. Diagnostic only,
        and not thread-safe — same caveat as IVF's probe counter."""
        return self._last_visits

    def search(
        self,
        query: Vectors,
        k: int,
        *,
        effort: int | None = None,
        admissible: BoolMask | None = None,
    ) -> list[Neighbor]:
        """`effort` is ef_search: the size of the layer-0 result set.

        Clamped to at least k — you cannot return k results from a set of
        fewer than k, and silently returning short is worse than the clamp.
        Larger ef explores more of the graph: better recall, lower QPS. That
        is the entire dial.
        """
        if self.size == 0 or k <= 0 or self._entry is None:
            return []

        ef = max(effort or self._ef_search, k)
        entry_points = [self._entry]

        # The descent is DELIBERATELY UNFILTERED.
        #
        # Upper layers exist only to navigate — to land near the query before
        # the real search starts. Filtering here would break it two ways:
        # `_search_layer(..., ef=1)` could return an EMPTY result set (the
        # entry point may be inadmissible), and `[0][1]` would raise; and even
        # if it didn't, restricting navigation to admissible nodes reintroduces
        # exactly the connectivity collapse from §4.1, one layer up.
        #
        # Filtering only ever applies at layer 0, where results are produced.
        for layer in range(self._max_level, 0, -1):
            entry_points = [self._search_layer(query, entry_points, 1, layer)[0][1]]

        found = self._search_layer(
            query, entry_points, ef, 0, admissible=admissible, budget=self._max_visits
        )
        if not found:
            return []

        # Deterministic tie-break on (distance, external id), matching the
        # oracle. Duplicate vectors are everywhere in this corpus.
        found.sort(key=lambda pair: (pair[0], int(self._ids[pair[1]])))

        return [
            Neighbor(id=int(self._ids[node]), distance=float(distance))
            for distance, node in found[: min(k, self.size)]
        ]

    # ------------------------------------------------------------ persistence

    def memory_bytes(self) -> int:
        edges = sum(len(adj) for layer in self._graph for adj in layer.values())
        return int(
            self._vectors.nbytes
            + self._ids.nbytes
            + self._levels.nbytes
            + edges * 4  # int32 per directed edge
        )

    def edge_count(self) -> int:
        return sum(len(adj) for layer in self._graph for adj in layer.values())

    def save(self, path: Path) -> None:
        """CSR per layer — compact, and no pickle in the load path."""
        arrays: dict[str, Any] = {
            "ids": self._ids,
            "vectors": self._vectors,
            "levels": self._levels,
            "meta": np.array(
                [
                    self._m,
                    self._ef_construction,
                    self._ef_search,
                    self._entry if self._entry is not None else -1,
                    self._max_level,
                    len(self._graph),
                ],
                dtype=np.int64,
            ),
        }
        n = self.size
        for layer, adjacency in enumerate(self._graph):
            indptr = np.zeros(n + 1, dtype=np.int64)
            flat: list[int] = []
            for node in range(n):
                neighbours = adjacency.get(node, ())
                flat.extend(neighbours)
                indptr[node + 1] = len(flat)
            arrays[f"l{layer}_indptr"] = indptr
            arrays[f"l{layer}_indices"] = np.asarray(flat, dtype=np.int32)

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        data = np.load(path, allow_pickle=False)
        m, ef_c, ef_s, entry, max_level, n_layers = (int(x) for x in data["meta"])

        index = cls(m=m, ef_construction=ef_c, ef_search=ef_s)
        index._ids = data["ids"]
        index._vectors = data["vectors"]
        index._levels = data["levels"]
        index._entry = None if entry < 0 else entry
        index._max_level = max_level
        index._graph = []

        for layer in range(n_layers):
            indptr = data[f"l{layer}_indptr"]
            indices = data[f"l{layer}_indices"]
            adjacency: dict[int, list[int]] = {}
            for node in range(len(indptr) - 1):
                start, stop = int(indptr[node]), int(indptr[node + 1])
                if stop > start:
                    adjacency[node] = indices[start:stop].tolist()
            index._graph.append(adjacency)

        return index

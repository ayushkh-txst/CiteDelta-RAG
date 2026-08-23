"""OpenRouterEmbeddings against a faked HTTP transport — no live network,
no billing. Same MockTransport approach as the substrate LLM adapter tests.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from citedelta.embed.base import EmbeddingProvider
from citedelta.embed.openrouter import OpenRouterEmbeddings

FULL_DIM = 1536
TRUNCATED_DIM = 512


def _vector(seed: int, dim: int = FULL_DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float64)
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


def _embeddings_response(texts: list[str], *, out_of_order: bool = False) -> dict[str, object]:
    items = [
        {"object": "embedding", "index": i, "embedding": _vector(seed=i)} for i in range(len(texts))
    ]
    if out_of_order:
        items = list(reversed(items))
    return {
        "object": "list",
        "data": items,
        "model": "openai/text-embedding-3-small",
        "usage": {"prompt_tokens": 10 * len(texts), "total_tokens": 10 * len(texts)},
    }


def _provider(
    handler: object, *, dimensions: int = TRUNCATED_DIM, max_attempts: int = 3
) -> OpenRouterEmbeddings:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return OpenRouterEmbeddings(
        api_key="sk-or-test",
        dimensions=dimensions,
        transport=transport,
        max_attempts=max_attempts,
    )


def test_satisfies_the_protocol() -> None:
    provider = _provider(lambda r: httpx.Response(200, json=_embeddings_response([])))
    assert isinstance(provider, EmbeddingProvider)


def test_output_is_truncated_and_renormalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=_embeddings_response(body["input"]))

    vectors = _provider(handler).embed(["alpha", "beta"])

    assert vectors.shape == (2, TRUNCATED_DIM)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_row_order_matches_input_order_even_if_the_api_returns_it_scrambled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=_embeddings_response(body["input"], out_of_order=True))

    provider = _provider(handler)
    both = provider.embed(["alpha", "beta"])
    only_alpha = _provider(handler).embed(["alpha"])
    assert np.allclose(both[0], only_alpha[0], atol=1e-5)


def test_empty_input_returns_empty_matrix_without_a_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for an empty batch")

    vectors = _provider(handler).embed([])
    assert vectors.shape == (0, TRUNCATED_DIM)


def test_model_id_encodes_the_truncated_dimension() -> None:
    provider = _provider(lambda r: httpx.Response(200, json=_embeddings_response([])))
    assert provider.model_id == "openai/text-embedding-3-small@512"
    assert provider.dimensions == TRUNCATED_DIM


def test_batches_respect_batch_size() -> None:
    seen_batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_batch_sizes.append(len(body["input"]))
        return httpx.Response(200, json=_embeddings_response(body["input"]))

    texts = [f"text-{i}" for i in range(5)]
    vectors = _provider(handler).embed(texts, batch_size=2)

    assert seen_batch_sizes == [2, 2, 1]
    assert vectors.shape == (5, TRUNCATED_DIM)


def test_retries_on_429_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        body = json.loads(request.content)
        return httpx.Response(200, json=_embeddings_response(body["input"]))

    vectors = _provider(handler).embed(["alpha"])
    assert vectors.shape == (1, TRUNCATED_DIM)
    assert attempts["n"] == 2


def test_exhausted_retries_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    with pytest.raises(RuntimeError):
        _provider(handler, max_attempts=2).embed(["alpha"])

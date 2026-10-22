"""Text embedding backends.

* :class:`OpenAIEmbedder` - any ``/v1/embeddings`` endpoint (vLLM, llama.cpp,
  Ollama, TEI). Intended for the secondary GPU.
* :class:`HashingEmbedder` - deterministic, dependency-free feature hashing.
  Lexical rather than semantic, but lets the full memory system run offline
  and in CI.

All embedders return L2-normalised vectors so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import OrderedDict
from typing import Protocol, runtime_checkable

import httpx
import numpy as np

from sentientai.config import EmbeddingConfig
from sentientai.errors import ConfigError, ModelError, ModelUnavailableError

_WORD = re.compile(r"[a-z0-9_]+(?:['.\-][a-z0-9_]+)*")
_STOP = frozenset(
    (  # noqa: SIM905 - a word list reads best as one string
        "a an and are as at be by for from has have i in is it its of on or that the this to was "
        "were will with you your me my we our they them what which who how do does did not no"
    ).split()
)


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class HashingEmbedder:
    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions
        self.model_id = f"hashing-v1-{dimensions}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dimensions, (1.0 if (value >> 63) & 1 else -1.0)

    def embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
        features: list[tuple[str, float]] = []
        for w in words:
            features.append((f"w:{w}", 1.0))
            if len(w) > 4:  # light stemming signal via prefixes and char trigrams
                features.append((f"p:{w[:5]}", 0.5))
                features.extend((f"c:{w[i : i + 3]}", 0.15) for i in range(len(w) - 2))
        features.extend((f"b:{a}_{b}", 0.6) for a, b in itertools.pairwise(words))
        for feature, weight in features:
            index, sign = self._bucket(feature)
            vec[index] += sign * weight
        # Sub-linear term frequency keeps long documents from dominating.
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        return [float(x) for x in normalize(vec)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig, *, http: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.dimensions = config.dimensions
        self.model_id = config.model
        headers = {}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"), headers=headers, timeout=config.timeout_s
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.config.batch_size]]
            try:
                response = await self._http.post(
                    "/embeddings", json={"model": self.config.model, "input": batch}
                )
            except httpx.TransportError as exc:
                raise ModelUnavailableError(f"embedding endpoint unreachable: {exc}") from exc
            if response.status_code >= 400:
                raise ModelError(f"embedding HTTP {response.status_code}: {response.text[:300]}")
            items = sorted(response.json()["data"], key=lambda d: d["index"])
            for item in items:
                vec = np.asarray(item["embedding"], dtype=np.float32)
                if vec.shape[0] != self.dimensions:
                    raise ConfigError(
                        f"embedding model returned {vec.shape[0]} dims but "
                        f"models.embedding.dimensions={self.dimensions}"
                    )
                out.append([float(x) for x in normalize(vec)])
        return out

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


class CachedEmbedder:
    """LRU cache in front of an embedder (queries repeat a lot during recall)."""

    def __init__(self, inner: Embedder, max_items: int = 4096) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.dimensions = inner.dimensions
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max = max_items
        self.hits = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        missing = [
            (i, t) for i, (k, t) in enumerate(zip(keys, texts, strict=True)) if k not in self._cache
        ]
        if missing:
            vectors = await self.inner.embed([t for _, t in missing])
            for (i, _), vec in zip(missing, vectors, strict=True):
                self._cache[keys[i]] = vec
        self.hits += len(texts) - len(missing)
        result = []
        for key in keys:
            self._cache.move_to_end(key)
            result.append(self._cache[key])
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return result

    async def aclose(self) -> None:
        closer = getattr(self.inner, "aclose", None)
        if closer is not None:
            await closer()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_embedder(config: EmbeddingConfig) -> Embedder:
    inner: Embedder = (
        OpenAIEmbedder(config)
        if config.provider == "openai"
        else HashingEmbedder(config.dimensions)
    )
    return CachedEmbedder(inner)

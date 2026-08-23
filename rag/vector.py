"""Semantic retrieval via embeddings."""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request

from .corpus import Chunk

DIM = 256


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
        vec[(h >> 64) % DIM] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class EmbeddingClient:
    def __init__(self, backend: str = "auto", model: str = "text-embedding-3-small"):
        if backend == "auto":
            backend = "openai" if os.environ.get("OPENAI_API_KEY") else "hash"
        self.backend = backend
        self.model = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "hash":
            return [_hash_embed(t) for t in texts]
        return self._openai_embed(texts)

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        ordered = sorted(payload["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in ordered]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    def __init__(self, chunks: list[Chunk], client: EmbeddingClient | None = None):
        self.chunks = chunks
        self.client = client or EmbeddingClient()
        self._matrix = self.client.embed_batch([c.text for c in chunks])

    def score(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        qv = self.client.embed_batch([query])[0]
        scored = [(i, cosine(qv, row)) for i, row in enumerate(self._matrix)]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

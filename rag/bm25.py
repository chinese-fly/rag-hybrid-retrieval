"""Okapi BM25 over the chunk corpus.

Exact-lexical matching is what stops the system from confidently returning
topically-similar-but-wrong clauses. Vector search alone does that constantly
on legal text where two sections read almost identically but mean opposite
things.
"""
import math
from collections import Counter

from .corpus import Chunk
from .tokenizer import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize(c.text) for c in chunks]
        self._freqs = [Counter(t) for t in self._tokenized]
        self._lengths = [len(t) for t in self._tokenized]
        self._avg_len = sum(self._lengths) / max(len(self._lengths), 1)
        self._df: Counter = Counter()
        for tf in self._freqs:
            self._df.update(tf.keys())
        self._n = len(chunks)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self._freqs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / self._avg_len)
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    s += self._idf(term) * (f * (self.k1 + 1)) / (f + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

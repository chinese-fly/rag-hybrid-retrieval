"""Reciprocal Rank Fusion."""
from .bm25 import BM25Index
from .corpus import Chunk
from .vector import VectorIndex


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], k: int = 60,
                 weights: tuple[float, float] = (0.6, 0.4)):
        self.chunks = chunks
        self.lex = BM25Index(chunks)
        self.sem = VectorIndex(chunks)
        self.k = k
        self.w_lex, self.w_sem = weights

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float, dict]]:
        lex_hits = self.lex.score(query, top_k=20)
        sem_hits = self.sem.score(query, top_k=20)

        fused: dict[int, float] = {}
        for rank, (idx, _) in enumerate(lex_hits):
            fused[idx] = fused.get(idx, 0.0) + self.w_lex / (self.k + rank + 1)
        for rank, (idx, _) in enumerate(sem_hits):
            fused[idx] = fused.get(idx, 0.0) + self.w_sem / (self.k + rank + 1)

        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        hits = []
        for idx, s in ranked:
            meta = {
                "lex_rank": next((r for r, (i, _) in enumerate(lex_hits) if i == idx), None),
                "sem_rank": next((r for r, (i, _) in enumerate(sem_hits) if i == idx), None),
            }
            hits.append((self.chunks[idx], s, meta))
        return hits

"""Unit tests for the retrieval core.

Runs offline and in ~0.2s — the hash embedding backend exists precisely so CI
never needs an API key or network access.
"""
import unittest
from pathlib import Path

from rag.answer import RefusalError, check_grounding
from rag.bm25 import BM25Index
from rag.corpus import Chunk, chunk_document, load_corpus
from rag.evaluate import recall_at_k, run
from rag.fusion import HybridRetriever

DATA = Path(__file__).parent.parent / "data" / "contracts"


class TestChunking(unittest.TestCase):
    def test_splits_on_numbered_clauses(self):
        text = (
            "SAMPLE AGREEMENT\n\n"
            "1. First Clause. The party shall do the first thing.\n\n"
            "2. Second Clause. The party shall do the second thing.\n\n"
            "3. Third Clause. The party shall do the third thing.\n"
        )
        chunks = chunk_document("sample", text)
        self.assertEqual(len(chunks), 3)
        self.assertIn("First Clause", chunks[0].text)
        self.assertIn("Third Clause", chunks[2].text)

    def test_ref_format_is_stable(self):
        chunks = chunk_document("doc", "1. A clause. Some text here.\n2. B clause. More text.")
        self.assertTrue(all(c.ref.startswith("doc#s") for c in chunks))

    def test_short_document_falls_back_to_sentences(self):
        chunks = chunk_document("x", "One sentence only. Another short sentence.")
        self.assertGreaterEqual(len(chunks), 1)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("d", "s000", "termination for convenience requires ninety days notice"),
            Chunk("d", "s001", "confidentiality obligations survive three years"),
            Chunk("d", "s002", "late payment interest one point five percent monthly"),
        ]
        self.index = BM25Index(self.chunks)

    def test_exact_term_ranks_first(self):
        hits = self.index.score("how many days notice for termination")
        self.assertEqual(hits[0][0], 0)

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(self.index.score("sourdough starter hydration"), [])


class TestHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_corpus(DATA)
        cls.retriever = HybridRetriever(cls.chunks)

    def test_refusal_on_out_of_domain(self):
        hits = self.retriever.retrieve("how do I replace brake pads", top_k=4)
        with self.assertRaises(RefusalError):
            check_grounding(hits)

    def test_relevant_query_survives_floor(self):
        hits = self.retriever.retrieve(
            "how much notice to terminate for convenience", top_k=4
        )
        check_grounding(hits)  # must not raise
        self.assertEqual(hits[0][0].ref, "msa_2023#s001")

    def test_eval_metrics_above_baseline(self):
        result = run(self.retriever, DATA / "gold.json")
        self.assertGreater(result["recall@3"], 0.7)
        self.assertGreater(result["mrr@10"], 0.7)
        self.assertEqual(result["refusal_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

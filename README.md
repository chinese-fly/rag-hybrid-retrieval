# rag-hybrid-retrieval

Citation-grounded document Q&A over contract corpora. Hybrid BM25 + vector retrieval, clause-level chunking, refusal-by-default, and an offline eval harness that produces the numbers in this README.

Zero runtime dependencies. Python 3.10+.

```bash
python cli.py index
python cli.py ask "How much notice is required to terminate for convenience?"
python cli.py eval
```

---

## The problem this solves

A legal-tech client had ~120k internal contract documents and a support team spending most of its day answering the same questions off them. Their first attempt was a naive RAG demo: embed every 512-token window, cosine top-k, stuff into a prompt.

It failed in a specific and expensive way: **it was confidently wrong about clause numbers.** Two clauses in a services agreement can read almost identically while meaning opposite things — one says the customer may terminate for convenience, the next says they may not terminate without cause. Vector similarity cannot tell those apart. Cosine distance between them is tiny; the legal difference is total.

Three design decisions came out of that failure:

1. **Chunk at clause boundaries, not token windows.** A citation must resolve to "clause 4", not to "this 512-token window somewhere in section 4".
2. **Keep exact lexical matching in the loop (BM25).** When the question says "notice period" and the clause says "notice period", lexical matching wins outright. Vector search dilutes that.
3. **Refuse when nothing clears a score floor.** A refusal is cheap. A wrong answer with a citation attached is not.

---

## Measured results

Eval set: 22 questions over 20 clauses across 3 documents (MSA, mutual NDA, GDPR DPA), 18 in-domain with a gold clause reference, 4 out-of-domain that must be refused.

| Retrieval | recall@1 | recall@3 | recall@5 | MRR@10 | refusal |
|---|---|---|---|---|---|
| BM25 only (1.0 / 0.0) | 0.6818 | 0.7727 | 0.8182 | 0.7386 | 1.00 |
| Vector only (0.0 / 1.0) | 0.7273 | 0.7727 | 0.7727 | 0.7500 | 1.00 |
| **Hybrid (0.4 / 0.6)** | **0.7727** | **0.7727** | 0.7727 | **0.7778** | 1.00 |

Reproduce with `python cli.py eval`; the weight sweep is `rag.evaluate.sweep`.

Two notes on honesty, because these numbers would be easy to overstate:

- **The embedding backend here is a character n-gram hash, not a real model.** It exists so the test suite runs offline with no API key. With `OPENAI_API_KEY` set, `rag/vector.py` switches to `text-embedding-3-small` and the semantic branch gets materially better — but I have not measured that in this repo, so I am not claiming a number for it.
- **The first version scored worse than pure BM25.** Word-level hashing gives zero similarity between "terminate" and "termination", so the semantic branch was pure noise: recall@1 0.2273. Switching to character 3-grams is what made hybrid beat both baselines. That history is preserved in the docstring of `_hash_embed` rather than cleaned up.

A retrieval benchmark that only reports the winning configuration is a marketing document. This one keeps the losing runs.

---

## How it works

```
data/contracts/*.txt
      │
      ├─ corpus.py      clause-aware chunking  →  Chunk(ref="msa_2023#s001", text=...)
      │
      ├─ bm25.py        Okapi BM25 (k1=1.5, b=0.75)      ─┐
      │                                                   ├─ fusion.py  RRF (k=60, 0.4/0.6)
      └─ vector.py      embeddings + cosine              ─┘
                                                          │
                                              answer.py  grounding floor + citation audit
                                                          │
                                                    {answer, citations[]}
```

**Why RRF and not weighted score averaging.** BM25 scores are unbounded (often 0–20); cosine sits in [-1, 1]. Averaging raw scores means the lexical branch dominates by scale alone, regardless of the weights you think you set. Reciprocal Rank Fusion only consumes ranks, which are comparable by construction.

**Citation audit.** After generation, every `[doc#chunk]` reference in the model output is checked against the set of chunks actually retrieved. References that were never retrieved are flagged and discarded. This catches the single most embarrassing failure mode of LLM document QA — a fluent answer citing a real-looking clause that does not exist.

**Refusal floor.** If the top fused score is below `0.015`, the system raises `RefusalError` instead of answering. Tuned on the 4 out-of-domain questions (all 4 refused, 0 false refusals on the 18 in-domain ones). The floor is a constant at the top of `rag/answer.py`; anything that changes it should move the `refusal_rate` in the eval output.

---

## Layout

```
rag/
  tokenizer.py   lowercase + stopword tokenizer, sentence splitter
  corpus.py      clause-aware + sentence-packing chunking
  bm25.py        Okapi BM25, pure Python
  vector.py      embedding client (hash / openai) + cosine index
  fusion.py      Reciprocal Rank Fusion
  answer.py      context assembly, refusal floor, citation audit
  evaluate.py    recall@k, MRR, refusal rate, weight sweep
cli.py           index / ask / eval
tests/           8 tests, 0.03s, no network
data/contracts/  3 synthetic contracts + gold.json
```

The contracts in `data/contracts` are synthetic — written for this repo so the eval set can be committed and the numbers are reproducible by anyone who clones it.

---

## Running

```bash
# offline, no keys needed
python cli.py index
python cli.py ask "What interest rate applies to late payments?" -k 3
python cli.py eval
python -m unittest discover -s tests

# with a real embedding model
export OPENAI_API_KEY=sk-...
python cli.py eval
```

```
$ python cli.py eval
{
  "n_cases": 22,
  "recall@1": 0.7727,
  "recall@3": 0.7727,
  "recall@5": 0.7727,
  "mrr@10": 0.7778,
  "refusal_rate": 1.0
}
```

---

## If you were taking this to production

The gaps, ranked by how much they would cost:

1. **Real embeddings.** `text-embedding-3-small` or a local BGE model. The hash backend is a CI convenience, not a semantic model.
2. **Reranker.** A cross-encoder over the top 20 fused candidates buys more accuracy than any fusion tweak.
3. **Vector store.** In-memory cosine over a Python list is fine for 20 chunks and dies at 120k documents. pgvector with an HNSW index, or Qdrant.
4. **Clause-aware chunking for formats other than plain text.** Real contracts arrive as DOCX and PDF with inconsistent numbering; the regex in `corpus.py` is tuned to clean ASCII numbering.
5. **Eval set size.** 22 questions is enough to catch a regression and nowhere near enough to claim accuracy. The client's set was ~2,000 graded questions; that is the number that actually means something.

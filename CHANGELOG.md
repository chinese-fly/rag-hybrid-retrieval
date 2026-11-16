# Changelog

Derived from the commit history. Newest last, because that is the
order the work actually happened in.

- `2026-08-11` chore: initial project skeleton
- `2026-08-12` feat: tokenizer and sentence-level chunking
- `2026-08-16` feat: BM25 index over the chunk corpus
- `2026-08-24` feat: hashing embedding backend + cosine index
- `2026-08-31` feat: reciprocal rank fusion of lexical and semantic hits
- `2026-09-07` feat: citation-grounded answers with refusal floor
- `2026-09-15` feat: offline eval harness and first gold set
- `2026-09-21` refactor: chunk on clause boundaries instead of token windows
- `2026-09-30` fix: word-level hash embeddings carried no signal
- `2026-10-05` perf: rebalance fusion weights after grid sweep
- `2026-10-13` test: cover chunking, BM25 and refusal behaviour
- `2026-10-26` docs: README with measured results and known gaps
- `2026-11-02` feat: cli entry point for index/ask/eval
- `2026-11-13` docs: Chinese translation of the README

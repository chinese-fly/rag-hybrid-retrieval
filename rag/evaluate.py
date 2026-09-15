"""Offline evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path

from .fusion import HybridRetriever


def recall_at_k(retriever: HybridRetriever, cases: list[dict], k: int) -> float:
    hits = 0
    for case in cases:
        got = [c.ref for c, _, _ in retriever.retrieve(case["question"], top_k=k)]
        if case["expected_ref"] in got:
            hits += 1
    return hits / len(cases)


def mrr(retriever: HybridRetriever, cases: list[dict]) -> float:
    total = 0.0
    for case in cases:
        got = [c.ref for c, _, _ in retriever.retrieve(case["question"], top_k=10)]
        if case["expected_ref"] in got:
            rank = got.index(case["expected_ref"]) + 1
            total += 1.0 / rank
    return total / len(cases)


def refusal_rate(retriever: HybridRetriever, cases: list[dict], floor: float = 0.015) -> float:
    ood = [c for c in cases if c.get("out_of_domain")]
    if not ood:
        return 1.0
    refused = 0
    for case in ood:
        hits = retriever.retrieve(case["question"], top_k=4)
        if not hits or hits[0][1] < floor:
            refused += 1
    return refused / len(ood)


def run(retriever: HybridRetriever, gold_path: Path | str) -> dict:
    cases = json.loads(Path(gold_path).read_text())
    return {
        "n_cases": len(cases),
        "recall@1": round(recall_at_k(retriever, cases, 1), 4),
        "recall@3": round(recall_at_k(retriever, cases, 3), 4),
        "recall@5": round(recall_at_k(retriever, cases, 5), 4),
        "mrr@10": round(mrr(retriever, cases), 4),
        "refusal_rate": round(refusal_rate(retriever, cases), 4),
    }

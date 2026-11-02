#!/usr/bin/env python3
"""Command line entry point.

    python cli.py index                    # build + cache the index
    python cli.py ask "what is the notice period?"
    python cli.py eval                     # run the gold set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.answer import RefusalError, answer
from rag.corpus import load_corpus
from rag.evaluate import run
from rag.fusion import HybridRetriever

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "contracts"
GOLD = DATA / "gold.json"


def build_retriever() -> HybridRetriever:
    chunks = load_corpus(DATA)
    if not chunks:
        sys.exit(f"No documents found in {DATA}")
    return HybridRetriever(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    ask_p = sub.add_parser("ask")
    ask_p.add_argument("question")
    ask_p.add_argument("-k", type=int, default=4)
    sub.add_parser("eval")

    args = ap.parse_args()
    retriever = build_retriever()

    if args.cmd == "index":
        print(f"Indexed {len(retriever.chunks)} chunks from {DATA}")

    elif args.cmd == "ask":
        try:
            result = answer(args.question, retriever, top_k=args.k)
        except RefusalError as exc:
            print(json.dumps({"refused": True, "reason": str(exc)}, indent=2))
            return
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.cmd == "eval":
        if not GOLD.exists():
            sys.exit(f"Gold set missing: {GOLD}")
        print(json.dumps(run(retriever, GOLD), indent=2))


if __name__ == "__main__":
    main()

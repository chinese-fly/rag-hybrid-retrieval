"""Citation-grounded answer assembly.

The LLM may only summarise retrieved chunks, and every claim must carry a
chunk reference. If nothing clears the score floor we refuse instead of
guessing — that was a hard requirement from the legal client: a confident
wrong answer costs more than no answer.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .corpus import Chunk
from .fusion import HybridRetriever

SCORE_FLOOR = 0.015

SYSTEM_PROMPT = """You are a document QA assistant. Rules:
1. Answer ONLY from the provided context blocks.
2. Cite the chunk reference [doc#chunk] after every claim.
3. If the context does not contain the answer, say so explicitly.
Never invent references."""


class RefusalError(Exception):
    """Raised when retrieval confidence is too low to answer safely."""


def build_context(hits: list[tuple[Chunk, float, dict]]) -> str:
    return "\n\n".join(
        f"[{c.ref}] (score={s:.4f})\n{c.text}" for c, s, _ in hits
    )


def check_grounding(hits: list[tuple[Chunk, float, dict]]) -> None:
    if not hits or hits[0][1] < SCORE_FLOOR:
        raise RefusalError(
            "No chunk cleared the score floor; refusing to answer rather than guessing."
        )


def _extract_refs(text: str) -> list[str]:
    return re.findall(r"\[[\w\-.]+#c\d+\]", text)


def answer(question: str, retriever: HybridRetriever, top_k: int = 4,
           llm: str = "gpt-4o-mini") -> dict:
    hits = retriever.retrieve(question, top_k=top_k)
    check_grounding(hits)

    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "question": question,
            "answer": None,
            "citations": [h[0].ref for h in hits],
            "mode": "offline",
            "context": build_context(hits),
        }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": llm,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"},
            ],
            "temperature": 0.1,
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.load(resp)
    text = payload["choices"][0]["message"]["content"]

    valid_refs = {h[0].ref for h in hits}
    used = set(_extract_refs(text))
    bad = used - valid_refs
    if bad:
        text += f"\n\n[AUDIT] model cited unknown refs {sorted(bad)} — discarded."

    return {
        "question": question,
        "answer": text,
        "citations": sorted(valid_refs & used) or [h[0].ref for h in hits],
        "mode": "grounded",
    }

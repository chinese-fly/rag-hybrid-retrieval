"""Minimal tokenizer: lowercase, strip punctuation, drop stopwords.

Kept dependency-free on purpose so the retrieval core can be unit tested
without pulling in numpy/torch.
"""
import re

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]*")

STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the to was
were will with this these those there their they we you your our not no but if
""".split())


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

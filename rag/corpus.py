"""Document loading and chunking.

Two strategies, chosen per document:

1. Clause-aware (preferred for contracts): split on numbered clause headings so
   one chunk == one clause. A citation then points at "clause 4" rather than
   at an arbitrary window of text, which is what a lawyer actually needs.
2. Sentence packing (fallback): pack whole sentences up to a token budget.

Sentence packing alone was the first implementation and it graded badly —
retrieval kept returning a chunk that contained the right clause plus three
unrelated ones, which made citations useless even when the hit was correct.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import split_sentences, tokenize

# "2. Termination for Convenience." at the start or mid-paragraph
_CLAUSE_RE = re.compile(r"(?=(?:\A|(?<=[\s.]))\d{1,2}\.\s+[A-Z])")
# an all-caps document title line, e.g. "MASTER SERVICES AGREEMENT"
_TITLE_RE = re.compile(r"\A\s*([A-Z][A-Z \-,]{8,})\s*\n")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


def load_document(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _split_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in _CLAUSE_RE.split(text) if p.strip()]
    return [p for p in parts if len(tokenize(p)) >= 3]


def _pack_sentences(text: str, max_tokens: int) -> list[str]:
    out, buf, count = [], [], 0
    for sentence in split_sentences(text):
        n = len(tokenize(sentence))
        if buf and count + n > max_tokens:
            out.append(" ".join(buf))
            buf, count = [], 0
        buf.append(sentence)
        count += n
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_document(doc_id: str, text: str, max_tokens: int = 90) -> list[Chunk]:
    body = text
    title = ""
    m = _TITLE_RE.match(text)
    if m:
        title = m.group(1).strip().title()
        body = text[m.end():]

    pieces = _split_clauses(body)
    if len(pieces) < 2:
        pieces = _pack_sentences(body, max_tokens)

    return [
        Chunk(doc_id, f"s{i:03d}", f"{title} — {p}" if title else p)
        for i, p in enumerate(pieces)
    ]


def load_corpus(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.txt")):
        chunks.extend(chunk_document(path.stem, load_document(path)))
    return chunks

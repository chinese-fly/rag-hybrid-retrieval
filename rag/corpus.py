"""Document loading and chunking."""
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import split_sentences, tokenize


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


def chunk_document(doc_id: str, text: str, max_tokens: int = 90) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    count = 0
    idx = 0

    for sentence in split_sentences(text):
        n = len(tokenize(sentence))
        if buf and count + n > max_tokens:
            chunks.append(Chunk(doc_id, f"c{idx:03d}", " ".join(buf)))
            idx += 1
            buf, count = [], 0
        buf.append(sentence)
        count += n

    if buf:
        chunks.append(Chunk(doc_id, f"c{idx:03d}", " ".join(buf)))
    return chunks


def load_corpus(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.txt")):
        chunks.extend(chunk_document(path.stem, load_document(path)))
    return chunks

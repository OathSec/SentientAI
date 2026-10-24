"""Structure-aware chunking.

* Markdown: split on headings and keep the heading path as chunk context.
* Code: split on top-level definitions (Python precisely, others by blank-line blocks).
* Prose: split on paragraphs, then sentences.

Pieces are packed up to a target size with a small overlap so facts that span
a boundary stay retrievable.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from sentientai.models.types import estimate_tokens

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PY_TOP = re.compile(r"^(async\s+def|def|class)\s+\w+|^@\w+", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    heading: str | None = None

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def _split_oversized(piece: str, target: int) -> list[str]:
    if estimate_tokens(piece) <= target:
        return [piece]
    sentences = _SENTENCE.split(piece)
    if len(sentences) == 1:
        sentences = piece.splitlines() or [piece]
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        if estimate_tokens(sentence) > target:  # a single enormous line: hard wrap
            if buf:
                out.append(buf)
                buf = ""
            width = target * 4
            out.extend(sentence[i : i + width] for i in range(0, len(sentence), width))
            continue
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if estimate_tokens(candidate) > target and buf:
            out.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


def _pack(pieces: list[str], target: int, overlap: int, sep: str = "\n\n") -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for piece in (p for raw in pieces for p in _split_oversized(raw.strip(), target) if p.strip()):
        tokens = estimate_tokens(piece)
        if buf and size + tokens > target:
            chunks.append(sep.join(buf))
            # carry a tail of the previous chunk forward as overlap
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(buf):
                t = estimate_tokens(prev)
                if tail_size + t > overlap:
                    break
                tail.insert(0, prev)
                tail_size += t
            buf, size = tail, tail_size
        buf.append(piece)
        size += tokens
    if buf:
        chunks.append(sep.join(buf))
    return chunks


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    path: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append((" > ".join(path) or None, body))
        current.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = [*path[: level - 1], match.group(2).strip()]
        else:
            current.append(line)
    flush()
    return sections


def chunk_text(
    text: str, *, fmt: str = "prose", target_tokens: int = 400, overlap_tokens: int = 60
) -> list[Chunk]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    results: list[Chunk] = []
    if fmt == "markdown":
        for heading, body in _markdown_sections(text):
            for packed in _pack(re.split(r"\n\s*\n", body), target_tokens, overlap_tokens):
                prefix = f"[{heading}]\n" if heading else ""
                results.append(Chunk(prefix + packed, len(results), heading))
        return results
    if fmt == "code":
        starts = [m.start() for m in _PY_TOP.finditer(text)]
        if starts:
            bounds = [0, *[s for s in starts if s > 0], len(text)]
            pieces = [text[a:b] for a, b in itertools.pairwise(bounds)]
        else:
            pieces = re.split(r"\n\s*\n", text)
        return [Chunk(p, i) for i, p in enumerate(_pack(pieces, target_tokens, 0, sep="\n\n"))]
    return [
        Chunk(p, i)
        for i, p in enumerate(_pack(re.split(r"\n\s*\n", text), target_tokens, overlap_tokens))
    ]

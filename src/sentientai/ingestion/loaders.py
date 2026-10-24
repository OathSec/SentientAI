"""Turn files and URLs into plain text documents."""

from __future__ import annotations

import io
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentientai.errors import ToolError

TEXT_SUFFIXES = frozenset(
    (  # noqa: SIM905 - suffix lists read best as strings
        ".txt .md .markdown .rst .log .csv .tsv .json .jsonl .yaml .yml .toml .ini .cfg .conf "
        ".xml .tex .org .adoc"
    ).split()
)
CODE_SUFFIXES = frozenset(
    (  # noqa: SIM905
        ".py .rs .go .js .ts .tsx .jsx .java .kt .c .h .cpp .hpp .cc .cs .rb .php .swift .scala "
        ".sh .bash .sql .lua .zig .proto .css .scss .vue .svelte"
    ).split()
)
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}


@dataclass
class LoadedDocument:
    source: str
    title: str
    text: str
    mime: str
    raw: bytes
    format: str  # "markdown" | "code" | "prose"
    metadata: dict[str, Any] = field(default_factory=dict)


def pdf_bytes_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ToolError(
            "PDF support requires the 'documents' extra (pip install sentientai[documents])"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[page {number}]\n{text}")
    return "\n\n".join(pages)


def supported(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in TEXT_SUFFIXES | CODE_SUFFIXES | HTML_SUFFIXES | {".pdf"}


def load_bytes(data: bytes, *, source: str, name: str) -> LoadedDocument:
    suffix = Path(name).suffix.lower()
    mime = mimetypes.guess_type(name)[0] or "text/plain"
    title = Path(name).stem.replace("_", " ").replace("-", " ").strip() or name
    if suffix == ".pdf":
        return LoadedDocument(
            source, title, pdf_bytes_to_text(data), "application/pdf", data, "prose"
        )
    if b"\x00" in data[:8192]:
        raise ToolError(f"{name}: binary file type is not supported for ingestion")
    text = data.decode("utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        from sentientai.tools.web import html_to_text

        html_title, body = html_to_text(text)
        return LoadedDocument(source, html_title or title, body, "text/html", data, "markdown")
    if suffix in CODE_SUFFIXES:
        return LoadedDocument(
            source, name, text, mime, data, "code", {"language": suffix.lstrip(".")}
        )
    fmt = "markdown" if suffix in {".md", ".markdown", ".rst", ".org", ".adoc"} else "prose"
    if fmt == "markdown":
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    return LoadedDocument(source, title, text, mime, data, fmt)


def load_path(path: Path) -> LoadedDocument:
    if not path.is_file():
        raise ToolError(f"{path} is not a file")
    if not supported(path):
        raise ToolError(f"{path.name}: unsupported file type {path.suffix or '(none)'}")
    return load_bytes(path.read_bytes(), source=str(path), name=path.name)

"""Continuous knowledge ingestion: load -> dedupe -> archive -> chunk -> embed -> store."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.errors import ToolError
from sentientai.ingestion.chunking import chunk_text
from sentientai.ingestion.loaders import LoadedDocument, load_bytes, load_path, supported
from sentientai.memory.system import MemorySystem
from sentientai.storage.records import DocumentRecord, MemoryKind, MemoryRecord
from sentientai.tools.filesystem import SKIP_DIRS
from sentientai.tools.web import WebClient

log = get_logger(__name__)


@dataclass
class IngestResult:
    source: str
    title: str = ""
    document_id: str | None = None
    chunks: int = 0
    skipped: bool = False
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None or self.skipped


class IngestionPipeline:
    def __init__(
        self,
        memory: MemorySystem,
        archive_dir: Path,
        *,
        web: WebClient | None = None,
        bus: EventBus | None = None,
        target_tokens: int = 400,
        overlap_tokens: int = 60,
        max_file_bytes: int = 100_000_000,
    ) -> None:
        self.memory = memory
        self.db = memory.db
        self.archive_dir = archive_dir
        self.web = web
        self.bus = bus
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_file_bytes = max_file_bytes

    def _archive(self, doc: LoadedDocument, sha: str) -> Path:
        suffix = Path(doc.source).suffix.lower() if "." in Path(doc.source).name else ".txt"
        target = self.archive_dir / sha[:2] / f"{sha}{suffix[:10]}"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(doc.raw)
            os.replace(tmp, target)
        return target

    async def ingest_document(self, doc: LoadedDocument) -> IngestResult:
        sha = hashlib.sha256(doc.raw or doc.text.encode()).hexdigest()
        existing = await self.db.documents.get_by_sha(sha)
        if existing is not None:
            return IngestResult(
                doc.source,
                existing.title,
                existing.id,
                existing.chunk_count,
                skipped=True,
                reason="already ingested",
            )
        if not doc.text.strip():
            return IngestResult(doc.source, doc.title, reason="no extractable text")

        chunks = chunk_text(
            doc.text,
            fmt=doc.format,
            target_tokens=self.target_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        archive = self._archive(doc, sha)
        record = DocumentRecord(
            source=doc.source,
            sha256=sha,
            title=doc.title,
            mime=doc.mime,
            size_bytes=len(doc.raw),
            chunk_count=len(chunks),
            archive_path=str(archive),
            metadata=doc.metadata,
        )
        await self.db.documents.insert(record)
        memories = [
            MemoryRecord(
                kind=MemoryKind.DOCUMENT,
                content=f"From \u201c{doc.title}\u201d:\n{chunk.text}",
                importance=0.4,
                confidence=0.9,
                source=f"document:{record.id}",
                metadata={
                    "document_id": record.id,
                    "chunk": chunk.index,
                    "title": doc.title,
                    "source": doc.source,
                    "heading": chunk.heading,
                },
            )
            for chunk in chunks
        ]
        await self.memory.store_batch(memories)
        log.info("ingested document", extra={"title": doc.title, "chunks": len(chunks)})
        if self.bus:
            await self.bus.publish(
                "ingest.completed",
                document_id=record.id,
                title=doc.title,
                chunks=len(chunks),
                source=doc.source,
            )
        return IngestResult(doc.source, doc.title, record.id, len(chunks))

    async def ingest_path(self, path: Path) -> IngestResult:
        try:
            if path.stat().st_size > self.max_file_bytes:
                return IngestResult(str(path), reason=f"file exceeds {self.max_file_bytes} bytes")
            return await self.ingest_document(load_path(path))
        except (ToolError, OSError) as exc:
            return IngestResult(str(path), reason=str(exc))

    async def ingest_directory(
        self, root: Path, *, pattern: str = "*", recursive: bool = True
    ) -> list[IngestResult]:
        results = []
        walker = os.walk(root) if recursive else [(str(root), [], os.listdir(root))]
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if (
                    name.startswith(".")
                    or not fnmatch.fnmatch(name, pattern)
                    or not supported(path)
                ):
                    continue
                results.append(await self.ingest_path(path))
        return results

    async def ingest_url(self, url: str) -> IngestResult:
        if self.web is None:
            return IngestResult(url, reason="web access is not configured")
        try:
            title, text, ctype = await self.web.fetch_text(url)
        except ToolError as exc:
            return IngestResult(url, reason=str(exc))
        doc = LoadedDocument(
            source=url,
            title=title,
            text=text,
            mime=ctype,
            raw=text.encode(),
            format="markdown" if ctype == "text/html" else "prose",
        )
        return await self.ingest_document(doc)

    async def ingest_text(
        self,
        text: str,
        *,
        title: str,
        source: str = "inline",
        name: str = "note.md",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        doc = load_bytes(text.encode(), source=source, name=name)
        doc.title = title
        doc.metadata.update(metadata or {})
        return await self.ingest_document(doc)

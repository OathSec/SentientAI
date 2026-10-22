"""Tools that let the agent use its own long-term memory and knowledge base."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from sentientai.config import RiskLevel
from sentientai.ingestion.pipeline import IngestionPipeline
from sentientai.memory.system import MemorySystem
from sentientai.memory.types import KIND_LABELS
from sentientai.storage.records import MemoryKind
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult
from sentientai.tools.filesystem import Workspace

KindName = Literal["fact", "episode", "procedure", "document", "summary", "reflection", "thought"]


class RecallArgs(ToolArgs):
    query: str = Field(min_length=2, description="What to search your memory for.")
    kinds: list[KindName] | None = Field(default=None, description="Restrict to memory kinds.")
    k: int = Field(default=6, ge=1, le=20)


class MemoryRecall(Tool[RecallArgs]):
    name = "memory_recall"
    description = "Search your long-term memory (facts, past experiences, procedures, documents)."
    args_model = RecallArgs
    tags = ("memory", "read")

    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory

    async def run(self, args: RecallArgs, ctx: ToolContext) -> ToolResult:
        kinds = [MemoryKind(k) for k in args.kinds] if args.kinds else None
        hits = await self.memory.recall(args.query, kinds=kinds, k=args.k)
        if not hits:
            return ToolResult.success("nothing relevant in memory")
        lines = [
            f"[{h.record.id}] ({KIND_LABELS[h.record.kind]}, relevance {h.similarity:.2f}) "
            f"{h.record.content}"
            for h in hits
        ]
        return ToolResult.success("\n\n".join(lines), data=[h.record.id for h in hits])


class RememberArgs(ToolArgs):
    content: str = Field(min_length=3, description="A self-contained statement worth keeping.")
    kind: Literal["fact", "reflection", "thought"] = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    subject_key: str | None = Field(
        default=None,
        description="Topic key (e.g. 'operator.timezone'); newer facts replace older ones.",
    )


class MemoryRemember(Tool[RememberArgs]):
    name = "memory_remember"
    description = "Store something in long-term memory so you can recall it in future sessions."
    args_model = RememberArgs
    risk = RiskLevel.LOW
    tags = ("memory", "write")

    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory

    async def run(self, args: RememberArgs, ctx: ToolContext) -> ToolResult:
        record = await self.memory.remember(
            args.content,
            MemoryKind(args.kind),
            importance=args.importance,
            subject_key=args.subject_key,
            source=f"task:{ctx.task_id}" if ctx.task_id else "agent",
        )
        return ToolResult.success(f"remembered as {record.id}")


class ForgetArgs(ToolArgs):
    memory_id: str
    reason: str = Field(min_length=3)


class MemoryForget(Tool[ForgetArgs]):
    name = "memory_forget"
    description = "Retire a memory that is wrong or obsolete (soft delete)."
    args_model = ForgetArgs
    risk = RiskLevel.MEDIUM
    tags = ("memory", "write")

    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory

    async def run(self, args: ForgetArgs, ctx: ToolContext) -> ToolResult:
        if await self.memory.forget(args.memory_id):
            return ToolResult.success(f"forgot {args.memory_id}")
        return ToolResult.failure(f"no memory {args.memory_id}")


class IngestArgs(ToolArgs):
    source: str = Field(description="Workspace file/directory path, or an http(s) URL.")


class IngestKnowledge(Tool[IngestArgs]):
    name = "ingest"
    description = "Add a document, directory or web page to your searchable knowledge base."
    args_model = IngestArgs
    risk = RiskLevel.LOW
    timeout_s = 900.0
    tags = ("memory", "write")

    def __init__(self, pipeline: IngestionPipeline, workspace: Workspace) -> None:
        self.pipeline = pipeline
        self.ws = workspace

    async def run(self, args: IngestArgs, ctx: ToolContext) -> ToolResult:
        if args.source.startswith(("http://", "https://")):
            results = [await self.pipeline.ingest_url(args.source)]
        else:
            path = self.ws.resolve(args.source, must_exist=True)
            results = (
                await self.pipeline.ingest_directory(path)
                if path.is_dir()
                else [await self.pipeline.ingest_path(path)]
            )
        lines = []
        for r in results:
            if r.skipped:
                lines.append(f"skipped {r.source}: {r.reason}")
            elif r.reason:
                lines.append(f"failed {r.source}: {r.reason}")
            else:
                lines.append(f"ingested {r.title!r} ({r.chunks} chunks)")
        ok = any(r.ok for r in results) or not results
        return ToolResult(
            ok=ok,
            output="\n".join(lines) or "nothing to ingest",
            error=None if ok else "ingestion failed",
        )


def memory_tools(
    memory: MemorySystem, pipeline: IngestionPipeline, workspace: Workspace
) -> list[Tool[Any]]:
    return [
        MemoryRecall(memory),
        MemoryRemember(memory),
        MemoryForget(memory),
        IngestKnowledge(pipeline, workspace),
    ]

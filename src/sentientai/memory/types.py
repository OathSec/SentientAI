"""Structured memory payloads stored inside :class:`MemoryRecord` metadata."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from sentientai.core.clock import utcnow
from sentientai.storage.records import MemoryKind, MemoryRecord

Outcome = Literal["success", "partial", "failure", "aborted"]


def slugify(text: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "unnamed"


class Episode(BaseModel):
    """A chronological record of one experience (usually one task run)."""

    task_id: str | None = None
    goal: str
    outcome: Outcome
    summary: str
    steps: int = 0
    tools_used: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    tokens: int = 0
    lessons: list[str] = Field(default_factory=list)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    source: str = "user"
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime = Field(default_factory=utcnow)

    def render(self) -> str:
        text = f"[{self.outcome}] Goal: {self.goal}\nWhat happened: {self.summary}"
        if self.tools_used:
            text += f"\nTools: {', '.join(sorted(set(self.tools_used)))}"
        if self.lessons:
            text += "\nLessons: " + " | ".join(self.lessons)
        return text

    @classmethod
    def from_record(cls, record: MemoryRecord) -> Episode:
        return cls.model_validate(record.metadata["episode"])


class Procedure(BaseModel):
    """A reusable strategy distilled from successful experience."""

    name: str
    when_to_use: str
    steps: list[str]
    successes: int = 0
    failures: int = 0
    source_tasks: list[str] = Field(default_factory=list)

    @property
    def subject_key(self) -> str:
        return f"procedure:{slugify(self.name)}"

    @property
    def reliability(self) -> float:
        """Laplace-smoothed success rate."""

        return (self.successes + 1) / (self.successes + self.failures + 2)

    def render(self) -> str:
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.steps, 1))
        return (
            f"Procedure: {self.name} (reliability {self.reliability:.0%})\n"
            f"Use when: {self.when_to_use}\nSteps:\n{steps}"
        )

    @classmethod
    def from_record(cls, record: MemoryRecord) -> Procedure:
        return cls.model_validate(record.metadata["procedure"])


def memory_metadata(record: MemoryRecord) -> dict[str, Any]:
    return {"id": record.id, "kind": record.kind.value, "importance": record.importance}


KIND_LABELS = {
    MemoryKind.FACT: "fact",
    MemoryKind.EPISODE: "past experience",
    MemoryKind.PROCEDURE: "known procedure",
    MemoryKind.DOCUMENT: "document excerpt",
    MemoryKind.SUMMARY: "conversation summary",
    MemoryKind.REFLECTION: "reflection",
    MemoryKind.THOUGHT: "thought",
}

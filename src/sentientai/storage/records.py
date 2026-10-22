"""Typed records persisted by the runtime. Shared by storage, domain layers and the API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from sentientai.config import RiskLevel
from sentientai.core.clock import new_id, utcnow

# Stored as an integer (orderable in SQL), shown by name in JSON ("HIGH"), accepted either way.
Risk = Annotated[
    RiskLevel,
    BeforeValidator(RiskLevel.parse),
    PlainSerializer(lambda r: r.name, return_type=str, when_used="json"),
]


class _Record(BaseModel):
    model_config = ConfigDict(use_enum_values=False, validate_assignment=True)


# --------------------------------------------------------------------- memory
class MemoryKind(StrEnum):
    FACT = "fact"
    EPISODE = "episode"
    PROCEDURE = "procedure"
    DOCUMENT = "document"
    SUMMARY = "summary"
    REFLECTION = "reflection"
    THOUGHT = "thought"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("mem"))
    kind: MemoryKind
    content: str
    subject_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = 0
    source: str = "agent"
    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: str | None = None
    embedding_model: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime = Field(default_factory=utcnow)


class ScoredMemory(BaseModel):
    record: MemoryRecord
    score: float
    similarity: float


# ---------------------------------------------------------------------- tasks
class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class TaskSource(StrEnum):
    USER = "user"
    API = "api"
    SCHEDULE = "schedule"
    WATCHER = "watcher"
    INTRINSIC = "intrinsic"
    SYSTEM = "system"

    @property
    def autonomous(self) -> bool:
        """Self-initiated work is held to a stricter approval threshold."""

        return self in {TaskSource.INTRINSIC, TaskSource.SCHEDULE, TaskSource.WATCHER}


class TaskRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("task"))
    goal: str
    kind: str = "task"
    source: TaskSource = TaskSource.USER
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5
    session_id: str | None = None
    parent_id: str | None = None
    goal_id: str | None = None
    attempts: int = 0
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ------------------------------------------------------------------ approvals
class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class ApprovalRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("apr"))
    task_id: str | None = None
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk: Risk
    reasons: list[str] = Field(default_factory=list)
    autonomous: bool = False
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------- goals
class GoalStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalOrigin(StrEnum):
    OPERATOR = "operator"
    INTRINSIC = "intrinsic"
    SCHEDULE = "schedule"


class GoalRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("goal"))
    description: str
    origin: GoalOrigin = GoalOrigin.OPERATOR
    status: GoalStatus = GoalStatus.ACTIVE
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    parent_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    attempts: int = 0
    deadline: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# -------------------------------------------------------- tool invocations
class ToolInvocationRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("inv"))
    task_id: str | None = None
    tool: str
    args_digest: str
    ok: bool
    duration_ms: float
    risk: Risk = RiskLevel.NONE
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ToolStats(BaseModel):
    tool: str
    calls: int
    failures: int
    success_rate: float
    p50_ms: float
    p95_ms: float
    top_errors: list[str]


# ------------------------------------------------------------------ proposals
class ProposalStatus(StrEnum):
    DRAFT = "draft"
    STAGED = "staged"
    STAGING_FAILED = "staging_failed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


class ProposalRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("prop"))
    title: str
    rationale: str
    diff: str
    files: list[str] = Field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    report: dict[str, Any] = Field(default_factory=dict)
    branch: str | None = None
    decided_by: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------------------------ documents
class DocumentRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("doc"))
    source: str
    sha256: str
    title: str
    mime: str
    size_bytes: int
    chunk_count: int = 0
    archive_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------- conversation
class ConversationMessage(_Record):
    id: str = Field(default_factory=lambda: new_id("msg"))
    session_id: str
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

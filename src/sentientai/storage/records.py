"""Typed records persisted by the runtime. Shared by storage, domain layers and the API."""

from __future__ import annotations

from dataclasses import dataclass
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
    CONVERSATION = "conversation"


GLOBAL_SCOPE = "global"


def scope_key(user_id: str | None) -> str:
    """Vector-index scope label for a memory owned by ``user_id`` (or shared)."""

    return f"u:{user_id}" if user_id else GLOBAL_SCOPE


@dataclass(frozen=True)
class Scope:
    """Which memories a query may see.

    The default is *shared memories only*: a call site that forgets to pass a scope
    misses private memories instead of leaking one person's memories to another.
    """

    user_id: str | None = None
    include_global: bool = True
    everything: bool = False

    @classmethod
    def system(cls) -> Scope:
        return cls(everything=True)

    @classmethod
    def for_user(cls, user_id: str | None) -> Scope:
        return cls(user_id=user_id)

    def vector_scopes(self) -> list[str] | None:
        if self.everything:
            return None
        scopes = [GLOBAL_SCOPE] if self.include_global else []
        if self.user_id:
            scopes.append(scope_key(self.user_id))
        return scopes

    def allows(self, user_id: str | None) -> bool:
        if self.everything:
            return True
        if user_id is None:
            return self.include_global
        return user_id == self.user_id


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("mem"))
    kind: MemoryKind
    content: str
    user_id: str | None = None  # owner of a private memory; None = shared/self
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
    user_id: str | None = None
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
    user_id: str | None = None  # a goal about one person stays in their compartment


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
    user_id: str | None = None
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------------------------- accounts
class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"


class UserStatus(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    DISABLED = "disabled"


class UserRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("usr"))
    username: str
    username_key: str  # case-folded, unique
    display_name: str
    password_hash: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    can_teach: bool = False
    totp_secret: str | None = None
    totp_last_step: int = 0
    failed_logins: int = 0
    locked_until: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_login_at: datetime | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role.value,
            "can_teach": self.can_teach or self.role is UserRole.ADMIN,
            "totp_enabled": self.totp_secret is not None,
            "created_at": self.created_at.isoformat(),
        }


class SessionRecord(_Record):
    id: str  # SHA-256 of the session token; the token itself is never stored
    user_id: str
    csrf_token: str
    mfa_pending: bool = False
    ip: str | None = None
    user_agent: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime


class InviteRecord(_Record):
    id: str  # SHA-256 of the invite code
    created_by: str
    role: UserRole = UserRole.USER
    can_teach: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    used_by: str | None = None
    used_at: datetime | None = None


class UsageRecord(_Record):
    user_id: str
    day: str  # YYYY-MM-DD (UTC)
    messages: int = 0
    images: int = 0
    teach_bytes: int = 0


# --------------------------------------------------------------- social mind
class RelationshipRecord(_Record):
    """How Fuzzy sees one person. Numbers and words are Fuzzy's own appraisals."""

    user_id: str
    familiarity: float = Field(default=0.0, ge=0.0, le=1.0)
    affinity: float = Field(default=0.0, ge=-1.0, le=1.0)
    trust: float = Field(default=0.3, ge=0.0, le=1.0)
    opinion: str = ""
    feelings: str = ""
    nickname: str | None = None
    interactions: int = 0
    first_met_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime = Field(default_factory=utcnow)
    last_reflected_at: datetime | None = None
    typical_gap_s: float | None = None  # learned rhythm of this person's visits
    history: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)


class BeliefStatus(StrEnum):
    ACTIVE = "active"
    REVISED = "revised"
    CONSIDERING = "considering"  # a possible change Fuzzy hasn't thought through yet
    KEPT = "kept"  # a considered change Fuzzy decided against


class SelfBeliefRecord(_Record):
    """Something Fuzzy has decided about itself (identity, tastes, values...)."""

    id: str = Field(default_factory=lambda: new_id("blf"))
    aspect: str  # normalized key, e.g. "gender", "favourite_colour"
    value: str
    reasoning: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    status: BeliefStatus = BeliefStatus.ACTIVE
    supersedes: str | None = None
    formed_at: datetime = Field(default_factory=utcnow)
    revised_at: datetime | None = None
    source: str = "self"


class EmotionEvent(_Record):
    """A feeling Fuzzy recognised in itself, in its own words."""

    id: str = Field(default_factory=lambda: new_id("emo"))
    emotion: str
    intensity: float = Field(ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    lingers_s: float = Field(default=3600.0, gt=0)  # Fuzzy's own sense of how long it lasts
    cause: str
    user_id: str | None = None  # about this person: private to them
    created_at: datetime = Field(default_factory=utcnow)
    # "beat" = noticed in a beat (a complete report of how it feels right now); anything else
    # (or None, from before this was recorded) is a reaction to something that happened.
    source: str | None = None

    def intensity_at(self, now: datetime) -> float:
        age = max(0.0, (now - self.created_at).total_seconds())
        return float(self.intensity * 0.5 ** (age / self.lingers_s))


class ActivityStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ActivityRecord(_Record):
    """Something Fuzzy chose to do on its own, and why."""

    id: str = Field(default_factory=lambda: new_id("act"))
    kind: str
    intent: str
    reason: str = ""
    outcome: str = ""
    status: ActivityStatus = ActivityStatus.RUNNING
    user_id: str | None = None  # involves this person: private to them
    detail: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class NoteRecord(_Record):
    """A message Fuzzy decided to leave for someone."""

    id: str = Field(default_factory=lambda: new_id("note"))
    user_id: str
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    read_at: datetime | None = None


class MediaRecord(_Record):
    id: str = Field(default_factory=lambda: new_id("img"))
    kind: str = "image"
    prompt: str
    path: str
    mime: str = "image/png"
    width: int = 0
    height: int = 0
    user_id: str | None = None  # requested by (private to) this user; None = Fuzzy's own
    public: bool = False
    caption: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class WorldObject(_Record):
    """Something in Fuzzy's world, built from simple parts."""

    id: str = Field(default_factory=lambda: new_id("obj"))
    name: str
    description: str = ""
    parts: list[dict[str, Any]] = Field(default_factory=list)
    x: float = 0.0
    z: float = 0.0
    rotation: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

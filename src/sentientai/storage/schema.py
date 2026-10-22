"""Relational schema shared by PostgreSQL and SQLite."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Dialect,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

from sentientai.core.clock import ensure_utc

SCHEMA_VERSION = 1


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes on every backend (SQLite has no native tz)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        value = ensure_utc(value)
        if dialect.name == "sqlite":
            return value.replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


JsonType: Any = JSON().with_variant(JSONB(), "postgresql")

metadata = MetaData()

schema_meta = Table(
    "schema_meta",
    metadata,
    Column("key", String(64), primary_key=True),
    Column("value", String(256), nullable=False),
)

memories = Table(
    "memories",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("kind", String(24), nullable=False),
    Column("content", Text, nullable=False),
    Column("subject_key", String(256)),
    Column("metadata", JsonType, nullable=False),
    Column("importance", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("strength", Float, nullable=False),
    Column("access_count", Integer, nullable=False, default=0),
    Column("source", String(128), nullable=False),
    Column("status", String(16), nullable=False),
    Column("superseded_by", String(48)),
    Column("embedding_model", String(128)),
    Column("created_at", UTCDateTime, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
    Column("last_accessed_at", UTCDateTime, nullable=False),
    Index("ix_memories_kind_status", "kind", "status"),
    Index("ix_memories_subject", "subject_key"),
    Index("ix_memories_created", "created_at"),
)

documents = Table(
    "documents",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("source", Text, nullable=False),
    Column("sha256", String(64), nullable=False, unique=True),
    Column("title", Text, nullable=False),
    Column("mime", String(128), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("chunk_count", Integer, nullable=False),
    Column("archive_path", Text),
    Column("metadata", JsonType, nullable=False),
    Column("ingested_at", UTCDateTime, nullable=False),
)

tasks = Table(
    "tasks",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("goal", Text, nullable=False),
    Column("kind", String(32), nullable=False),
    Column("source", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("priority", Integer, nullable=False),
    Column("session_id", String(128)),
    Column("parent_id", String(48)),
    Column("goal_id", String(48)),
    Column("attempts", Integer, nullable=False),
    Column("result", Text),
    Column("error", Text),
    Column("metadata", JsonType, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    Column("started_at", UTCDateTime),
    Column("finished_at", UTCDateTime),
    Index("ix_tasks_status_priority", "status", "priority"),
)

approvals = Table(
    "approvals",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("task_id", String(48)),
    Column("tool", String(128), nullable=False),
    Column("arguments", JsonType, nullable=False),
    Column("risk", Integer, nullable=False),
    Column("reasons", JsonType, nullable=False),
    Column("autonomous", Boolean, nullable=False),
    Column("status", String(16), nullable=False),
    Column("requested_at", UTCDateTime, nullable=False),
    Column("expires_at", UTCDateTime),
    Column("decided_at", UTCDateTime),
    Column("decided_by", String(128)),
    Column("note", Text),
    Index("ix_approvals_status", "status"),
)

goals = Table(
    "goals",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("description", Text, nullable=False),
    Column("origin", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("priority", Float, nullable=False),
    Column("progress", Float, nullable=False),
    Column("parent_id", String(48)),
    Column("notes", JsonType, nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("deadline", UTCDateTime),
    Column("created_at", UTCDateTime, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
)

tool_invocations = Table(
    "tool_invocations",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("task_id", String(48)),
    Column("tool", String(128), nullable=False),
    Column("args_digest", String(64), nullable=False),
    Column("ok", Boolean, nullable=False),
    Column("duration_ms", Float, nullable=False),
    Column("risk", Integer, nullable=False),
    Column("error", Text),
    Column("created_at", UTCDateTime, nullable=False),
    Index("ix_tool_invocations_tool_created", "tool", "created_at"),
)

proposals = Table(
    "proposals",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("title", Text, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("diff", Text, nullable=False),
    Column("files", JsonType, nullable=False),
    Column("status", String(24), nullable=False),
    Column("report", JsonType, nullable=False),
    Column("branch", String(256)),
    Column("decided_by", String(128)),
    Column("created_at", UTCDateTime, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
)

kv_state = Table(
    "kv_state",
    metadata,
    Column("key", String(128), primary_key=True),
    Column("value", JsonType, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
)

conversation_messages = Table(
    "conversation_messages",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("session_id", String(128), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("metadata", JsonType, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    Index("ix_conversation_session_created", "session_id", "created_at"),
)

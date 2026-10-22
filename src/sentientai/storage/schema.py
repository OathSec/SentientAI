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

SCHEMA_VERSION = 2


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
    Column("user_id", String(48)),
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
    Index("ix_memories_user", "user_id"),
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
    Column("user_id", String(48)),
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
    Column("user_id", String(48)),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("metadata", JsonType, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    Index("ix_conversation_session_created", "session_id", "created_at"),
)

# ------------------------------------------------------------------ v2: people
users = Table(
    "users",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("username", String(64), nullable=False),
    Column("username_key", String(64), nullable=False, unique=True),
    Column("display_name", String(80), nullable=False),
    Column("password_hash", String(256), nullable=False),
    Column("role", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("can_teach", Boolean, nullable=False),
    Column("totp_secret", String(64)),
    Column("totp_last_step", Integer, nullable=False),
    Column("failed_logins", Integer, nullable=False),
    Column("locked_until", UTCDateTime),
    Column("created_at", UTCDateTime, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
    Column("last_login_at", UTCDateTime),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(48), nullable=False),
    Column("csrf_token", String(64), nullable=False),
    Column("mfa_pending", Boolean, nullable=False),
    Column("ip", String(64)),
    Column("user_agent", String(256)),
    Column("created_at", UTCDateTime, nullable=False),
    Column("last_seen_at", UTCDateTime, nullable=False),
    Column("expires_at", UTCDateTime, nullable=False),
    Index("ix_sessions_user", "user_id"),
)

invites = Table(
    "invites",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("created_by", String(48), nullable=False),
    Column("role", String(16), nullable=False),
    Column("can_teach", Boolean, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    Column("expires_at", UTCDateTime, nullable=False),
    Column("used_by", String(48)),
    Column("used_at", UTCDateTime),
)

usage = Table(
    "usage",
    metadata,
    Column("user_id", String(48), primary_key=True),
    Column("day", String(10), primary_key=True),
    Column("messages", Integer, nullable=False),
    Column("images", Integer, nullable=False),
    Column("teach_bytes", Integer, nullable=False),
)

relationships = Table(
    "relationships",
    metadata,
    Column("user_id", String(48), primary_key=True),
    Column("familiarity", Float, nullable=False),
    Column("affinity", Float, nullable=False),
    Column("trust", Float, nullable=False),
    Column("opinion", Text, nullable=False),
    Column("feelings", Text, nullable=False),
    Column("nickname", String(80)),
    Column("interactions", Integer, nullable=False),
    Column("first_met_at", UTCDateTime, nullable=False),
    Column("last_seen_at", UTCDateTime, nullable=False),
    Column("last_reflected_at", UTCDateTime),
    Column("history", JsonType, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
)

self_beliefs = Table(
    "self_beliefs",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("aspect", String(96), nullable=False),
    Column("value", Text, nullable=False),
    Column("reasoning", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("status", String(16), nullable=False),
    Column("supersedes", String(48)),
    Column("formed_at", UTCDateTime, nullable=False),
    Column("revised_at", UTCDateTime),
    Column("source", String(64), nullable=False),
    Index("ix_self_beliefs_aspect_status", "aspect", "status"),
)

emotion_events = Table(
    "emotion_events",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("emotion", String(48), nullable=False),
    Column("intensity", Float, nullable=False),
    Column("cause", Text, nullable=False),
    Column("user_id", String(48)),
    Column("created_at", UTCDateTime, nullable=False),
    Index("ix_emotion_events_created", "created_at"),
)

notes = Table(
    "notes",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("user_id", String(48), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    Column("read_at", UTCDateTime),
    Index("ix_notes_user", "user_id"),
)

media = Table(
    "media",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("kind", String(16), nullable=False),
    Column("prompt", Text, nullable=False),
    Column("path", Text, nullable=False),
    Column("mime", String(64), nullable=False),
    Column("width", Integer, nullable=False),
    Column("height", Integer, nullable=False),
    Column("user_id", String(48)),
    Column("public", Boolean, nullable=False),
    Column("caption", Text, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
)

world_objects = Table(
    "world_objects",
    metadata,
    Column("id", String(48), primary_key=True),
    Column("name", String(120), nullable=False),
    Column("description", Text, nullable=False),
    Column("parts", JsonType, nullable=False),
    Column("x", Float, nullable=False),
    Column("z", Float, nullable=False),
    Column("rotation", Float, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
)

# Columns added after v1 (tables created by create_all; columns need ALTER on upgrade).
ADDED_COLUMNS: dict[int, list[tuple[str, str, str]]] = {
    2: [
        ("memories", "user_id", "VARCHAR(48)"),
        ("tasks", "user_id", "VARCHAR(48)"),
        ("conversation_messages", "user_id", "VARCHAR(48)"),
    ],
}

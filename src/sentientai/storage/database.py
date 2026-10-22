"""Async database access: engine lifecycle plus one repository per aggregate."""

from __future__ import annotations

import builtins
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import Table, and_, delete, event, func, or_, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from sentientai.core.clock import ensure_utc, utcnow
from sentientai.errors import StorageError
from sentientai.storage import schema
from sentientai.storage.records import (
    ApprovalRecord,
    ApprovalStatus,
    ConversationMessage,
    DocumentRecord,
    GoalRecord,
    GoalStatus,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    ProposalRecord,
    ProposalStatus,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolStats,
)

R = TypeVar("R", bound=BaseModel)


def _row_dict(row: Row[Any]) -> dict[str, Any]:
    return dict(row._mapping)


class _Repository(Generic[R]):
    table: Table
    model: type[R]

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    def _to_row(self, record: R) -> dict[str, Any]:
        return record.model_dump(mode="python")

    def _from_row(self, row: Row[Any]) -> R:
        return self.model.model_validate(_row_dict(row))

    async def insert(self, record: R) -> R:
        async with self.engine.begin() as conn:
            await conn.execute(self.table.insert().values(**self._to_row(record)))
        return record

    async def insert_many(self, records: Sequence[R]) -> None:
        if not records:
            return
        async with self.engine.begin() as conn:
            await conn.execute(self.table.insert(), [self._to_row(r) for r in records])

    async def get(self, id_: str) -> R | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(self.table).where(self.table.c.id == id_))).first()
        return self._from_row(row) if row else None

    async def save(self, record: R) -> R:
        """Full-row update by primary key."""

        values = self._to_row(record)
        id_ = values.pop("id")
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(self.table).where(self.table.c.id == id_).values(**values)
            )
            if result.rowcount == 0:
                raise StorageError(f"{self.table.name}: no row with id {id_}")
        return record

    async def _select(self, stmt: Any) -> builtins.list[R]:
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [self._from_row(r) for r in rows]


class MemoryRepository(_Repository[MemoryRecord]):
    table = schema.memories
    model = MemoryRecord

    async def get_many(self, ids: Sequence[str]) -> dict[str, MemoryRecord]:
        if not ids:
            return {}
        records = await self._select(select(self.table).where(self.table.c.id.in_(list(ids))))
        return {r.id: r for r in records}

    async def list(
        self,
        kinds: Iterable[MemoryKind] | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> builtins.list[MemoryRecord]:
        t = self.table
        stmt = select(t)
        if kinds is not None:
            stmt = stmt.where(t.c.kind.in_([k.value for k in kinds]))
        if status is not None:
            stmt = stmt.where(t.c.status == status.value)
        if since is not None:
            stmt = stmt.where(t.c.created_at >= since)
        order = t.c.created_at.desc() if newest_first else t.c.created_at.asc()
        return await self._select(stmt.order_by(order, t.c.id).limit(limit).offset(offset))

    async def find_by_subject(
        self, subject_key: str, kind: MemoryKind | None = None
    ) -> builtins.list[MemoryRecord]:
        t = self.table
        stmt = select(t).where(
            and_(t.c.subject_key == subject_key, t.c.status == MemoryStatus.ACTIVE.value)
        )
        if kind is not None:
            stmt = stmt.where(t.c.kind == kind.value)
        return await self._select(stmt.order_by(t.c.updated_at.desc()))

    async def search_text(
        self, query: str, kinds: Iterable[MemoryKind] | None = None, limit: int = 50
    ) -> builtins.list[tuple[MemoryRecord, float]]:
        """Keyword search used to complement dense retrieval (hybrid recall)."""

        terms = [t for t in re.findall(r"[\w\-\.]{3,}", query.lower()) if len(t) >= 3][:12]
        if not terms:
            return []
        t = self.table
        content = func.lower(t.c.content)
        stmt = select(t).where(
            and_(
                t.c.status == MemoryStatus.ACTIVE.value, or_(*[content.contains(x) for x in terms])
            )
        )
        if kinds is not None:
            stmt = stmt.where(t.c.kind.in_([k.value for k in kinds]))
        records = await self._select(stmt.limit(limit * 4))
        scored = []
        for record in records:
            text = record.content.lower()
            hits = sum(1 for term in terms if term in text)
            scored.append((record, hits / len(terms)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    async def mark_accessed(self, ids: Sequence[str], when: datetime, boost: float) -> None:
        if not ids:
            return
        t = self.table
        async with self.engine.begin() as conn:
            await conn.execute(
                update(t)
                .where(t.c.id.in_(list(ids)))
                .values(
                    access_count=t.c.access_count + 1,
                    last_accessed_at=when,
                    strength=func.min(1.0, t.c.strength + boost)
                    if self.engine.dialect.name == "sqlite"
                    else func.least(1.0, t.c.strength + boost),
                )
            )

    async def set_status(
        self, ids: Sequence[str], status: MemoryStatus, superseded_by: str | None = None
    ) -> int:
        if not ids:
            return 0
        t = self.table
        values: dict[str, Any] = {"status": status.value, "updated_at": utcnow()}
        if superseded_by is not None:
            values["superseded_by"] = superseded_by
        async with self.engine.begin() as conn:
            result = await conn.execute(update(t).where(t.c.id.in_(list(ids))).values(**values))
        return int(result.rowcount)

    async def update_strengths(self, strengths: dict[str, float]) -> None:
        if not strengths:
            return
        t = self.table
        async with self.engine.begin() as conn:
            for id_, strength in strengths.items():
                await conn.execute(update(t).where(t.c.id == id_).values(strength=strength))

    async def count_by_kind(self, status: MemoryStatus = MemoryStatus.ACTIVE) -> dict[str, int]:
        t = self.table
        stmt = select(t.c.kind, func.count()).where(t.c.status == status.value).group_by(t.c.kind)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return {str(kind): int(count) for kind, count in rows}

    async def purge(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        async with self.engine.begin() as conn:
            result = await conn.execute(delete(self.table).where(self.table.c.id.in_(list(ids))))
        return int(result.rowcount)


class TaskRepository(_Repository[TaskRecord]):
    table = schema.tasks
    model = TaskRecord

    async def list(
        self, statuses: Iterable[TaskStatus] | None = None, limit: int = 50
    ) -> builtins.list[TaskRecord]:
        t = self.table
        stmt = select(t)
        if statuses is not None:
            stmt = stmt.where(t.c.status.in_([s.value for s in statuses]))
        return await self._select(stmt.order_by(t.c.created_at.desc()).limit(limit))

    async def claim_next(self) -> TaskRecord | None:
        """Atomically move the highest-priority pending task to RUNNING."""

        t = self.table
        async with self.engine.begin() as conn:
            for _ in range(5):
                row = (
                    await conn.execute(
                        select(t)
                        .where(t.c.status == TaskStatus.PENDING.value)
                        .order_by(t.c.priority.desc(), t.c.created_at.asc())
                        .limit(1)
                    )
                ).first()
                if row is None:
                    return None
                record = self._from_row(row)
                now = utcnow()
                result = await conn.execute(
                    update(t)
                    .where(and_(t.c.id == record.id, t.c.status == TaskStatus.PENDING.value))
                    .values(
                        status=TaskStatus.RUNNING.value,
                        started_at=now,
                        attempts=t.c.attempts + 1,
                    )
                )
                if result.rowcount == 1:
                    record.status = TaskStatus.RUNNING
                    record.started_at = now
                    record.attempts += 1
                    return record
        return None

    async def claim(self, id_: str) -> TaskRecord | None:
        """Atomically move a specific pending task to RUNNING (None if already taken)."""

        t = self.table
        now = utcnow()
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(t)
                .where(and_(t.c.id == id_, t.c.status == TaskStatus.PENDING.value))
                .values(status=TaskStatus.RUNNING.value, started_at=now, attempts=t.c.attempts + 1)
            )
        if result.rowcount != 1:
            return None
        return await self.get(id_)

    async def pending(self, limit: int = 50) -> builtins.list[TaskRecord]:
        t = self.table
        return await self._select(
            select(t)
            .where(t.c.status == TaskStatus.PENDING.value)
            .order_by(t.c.priority.desc(), t.c.created_at.asc())
            .limit(limit)
        )

    async def finish(
        self,
        id_: str,
        status: TaskStatus,
        result: str | None = None,
        error: str | None = None,
        *,
        only_if_running: bool = False,
    ) -> bool:
        t = self.table
        condition = t.c.id == id_
        if only_if_running:
            condition = and_(condition, t.c.status == TaskStatus.RUNNING.value)
        async with self.engine.begin() as conn:
            outcome = await conn.execute(
                update(t)
                .where(condition)
                .values(status=status.value, result=result, error=error, finished_at=utcnow())
            )
        return bool(outcome.rowcount)

    async def requeue(self, id_: str, reason: str) -> bool:
        """Return a RUNNING task to the queue (e.g. interrupted by an orderly shutdown)."""

        t = self.table
        async with self.engine.begin() as conn:
            outcome = await conn.execute(
                update(t)
                .where(and_(t.c.id == id_, t.c.status == TaskStatus.RUNNING.value))
                .values(status=TaskStatus.PENDING.value, error=reason, started_at=None)
            )
        return bool(outcome.rowcount)

    async def recover_interrupted(self, max_attempts: int = 3) -> builtins.list[TaskRecord]:
        """At boot: tasks left RUNNING by a crash are re-queued (or failed if exhausted)."""

        t = self.table
        running = await self._select(select(t).where(t.c.status == TaskStatus.RUNNING.value))
        async with self.engine.begin() as conn:
            for task in running:
                exhausted = task.attempts >= max_attempts
                await conn.execute(
                    update(t)
                    .where(t.c.id == task.id)
                    .values(
                        status=(TaskStatus.FAILED if exhausted else TaskStatus.PENDING).value,
                        error="interrupted by shutdown; retry budget exhausted"
                        if exhausted
                        else None,
                    )
                )
        return running

    async def count_by_status(self) -> dict[str, int]:
        t = self.table
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(select(t.c.status, func.count()).group_by(t.c.status))
            ).fetchall()
        return {str(s): int(c) for s, c in rows}


class ApprovalRepository(_Repository[ApprovalRecord]):
    table = schema.approvals
    model = ApprovalRecord

    async def list(
        self, status: ApprovalStatus | None = None, limit: int = 100
    ) -> builtins.list[ApprovalRecord]:
        t = self.table
        stmt = select(t)
        if status is not None:
            stmt = stmt.where(t.c.status == status.value)
        return await self._select(stmt.order_by(t.c.requested_at.desc()).limit(limit))

    async def decide(
        self, id_: str, status: ApprovalStatus, decided_by: str, note: str | None = None
    ) -> bool:
        """Resolve a pending approval. Returns False if it was already decided."""

        t = self.table
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(t)
                .where(and_(t.c.id == id_, t.c.status == ApprovalStatus.PENDING.value))
                .values(status=status.value, decided_at=utcnow(), decided_by=decided_by, note=note)
            )
        return bool(result.rowcount == 1)


class GoalRepository(_Repository[GoalRecord]):
    table = schema.goals
    model = GoalRecord

    async def list(self, statuses: Iterable[GoalStatus] | None = None) -> builtins.list[GoalRecord]:
        t = self.table
        stmt = select(t)
        if statuses is not None:
            stmt = stmt.where(t.c.status.in_([s.value for s in statuses]))
        return await self._select(stmt.order_by(t.c.priority.desc(), t.c.created_at.asc()))


class ToolInvocationRepository(_Repository[ToolInvocationRecord]):
    table = schema.tool_invocations
    model = ToolInvocationRecord

    async def stats(self, since: datetime | None = None) -> builtins.list[ToolStats]:
        t = self.table
        stmt = select(t.c.tool, t.c.ok, t.c.duration_ms, t.c.error)
        if since is not None:
            stmt = stmt.where(t.c.created_at >= since)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        durations: dict[str, list[float]] = defaultdict(list)
        failures: Counter[str] = Counter()
        errors: dict[str, Counter[str]] = defaultdict(Counter)
        for tool, ok, duration, error in rows:
            durations[tool].append(float(duration))
            if not ok:
                failures[tool] += 1
                if error:
                    errors[tool][str(error)[:120]] += 1
        result = []
        for tool, values in sorted(durations.items()):
            values.sort()
            calls = len(values)
            p95_index = max(0, min(calls - 1, round(0.95 * (calls - 1))))
            result.append(
                ToolStats(
                    tool=tool,
                    calls=calls,
                    failures=failures[tool],
                    success_rate=(calls - failures[tool]) / calls,
                    p50_ms=statistics.median(values),
                    p95_ms=values[p95_index],
                    top_errors=[e for e, _ in errors[tool].most_common(3)],
                )
            )
        return result


class ProposalRepository(_Repository[ProposalRecord]):
    table = schema.proposals
    model = ProposalRecord

    async def list(
        self, status: ProposalStatus | None = None, limit: int = 50
    ) -> builtins.list[ProposalRecord]:
        t = self.table
        stmt = select(t)
        if status is not None:
            stmt = stmt.where(t.c.status == status.value)
        return await self._select(stmt.order_by(t.c.created_at.desc()).limit(limit))


class DocumentRepository(_Repository[DocumentRecord]):
    table = schema.documents
    model = DocumentRecord

    async def get_by_sha(self, sha256: str) -> DocumentRecord | None:
        records = await self._select(select(self.table).where(self.table.c.sha256 == sha256))
        return records[0] if records else None

    async def list(self, limit: int = 100) -> builtins.list[DocumentRecord]:
        t = self.table
        return await self._select(select(t).order_by(t.c.ingested_at.desc()).limit(limit))


class ConversationRepository(_Repository[ConversationMessage]):
    table = schema.conversation_messages
    model = ConversationMessage

    async def recent(self, session_id: str, limit: int = 50) -> builtins.list[ConversationMessage]:
        t = self.table
        records = await self._select(
            select(t)
            .where(t.c.session_id == session_id)
            .order_by(t.c.created_at.desc(), t.c.id.desc())
            .limit(limit)
        )
        return list(reversed(records))

    async def since(self, when: datetime, limit: int = 500) -> builtins.list[ConversationMessage]:
        t = self.table
        return await self._select(
            select(t)
            .where(t.c.created_at > when)
            .order_by(t.c.created_at.asc(), t.c.id.asc())
            .limit(limit)
        )

    async def sessions(self, limit: int = 20) -> builtins.list[tuple[str, datetime]]:
        t = self.table
        stmt = (
            select(t.c.session_id, func.max(t.c.created_at).label("last"))
            .group_by(t.c.session_id)
            .order_by(func.max(t.c.created_at).desc())
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        result = []
        for session_id, last in rows:
            if isinstance(last, str):  # SQLite may return aggregate datetimes as text
                last = datetime.fromisoformat(last)
            result.append((str(session_id), ensure_utc(last)))
        return result


class StateRepository:
    """Small JSON key/value store for checkpoints (affect, self-model, scheduler...)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def get(self, key: str) -> Any | None:
        t = schema.kv_state
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(t.c.value).where(t.c.key == key))).first()
        return row[0] if row else None

    async def set(self, key: str, value: Any) -> None:
        t = schema.kv_state
        values = {"key": key, "value": value, "updated_at": utcnow()}
        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            pg_stmt = postgresql.insert(t).values(**values)
            stmt: Any = pg_stmt.on_conflict_do_update(
                index_elements=[t.c.key],
                set_={"value": pg_stmt.excluded.value, "updated_at": pg_stmt.excluded.updated_at},
            )
        elif dialect == "sqlite":
            sl_stmt = sqlite.insert(t).values(**values)
            stmt = sl_stmt.on_conflict_do_update(
                index_elements=[t.c.key],
                set_={"value": sl_stmt.excluded.value, "updated_at": sl_stmt.excluded.updated_at},
            )
        else:  # pragma: no cover - other dialects
            raise StorageError(f"unsupported dialect {dialect}")
        async with self.engine.begin() as conn:
            await conn.execute(stmt)

    async def delete(self, key: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(delete(schema.kv_state).where(schema.kv_state.c.key == key))


class Database:
    """Owns the engine and exposes repositories. Works on PostgreSQL and SQLite."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        kwargs: dict[str, Any] = {"echo": echo}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"timeout": 30}
        else:
            kwargs.update(pool_size=10, max_overflow=10, pool_pre_ping=True)
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine.sync_engine, "connect", _sqlite_pragmas)
        self.memories = MemoryRepository(self.engine)
        self.tasks = TaskRepository(self.engine)
        self.approvals = ApprovalRepository(self.engine)
        self.goals = GoalRepository(self.engine)
        self.tool_invocations = ToolInvocationRepository(self.engine)
        self.proposals = ProposalRepository(self.engine)
        self.documents = DocumentRepository(self.engine)
        self.conversations = ConversationRepository(self.engine)
        self.state = StateRepository(self.engine)

    @property
    def dialect(self) -> str:
        return str(self.engine.dialect.name)

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(schema.metadata.create_all)
            row = (
                await conn.execute(
                    select(schema.schema_meta.c.value).where(schema.schema_meta.c.key == "version")
                )
            ).first()
            if row is None:
                await conn.execute(
                    schema.schema_meta.insert().values(
                        key="version", value=str(schema.SCHEMA_VERSION)
                    )
                )
            elif int(row[0]) > schema.SCHEMA_VERSION:
                raise StorageError(
                    f"database schema v{row[0]} is newer than this build (v{schema.SCHEMA_VERSION})"
                )

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(select(1))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self.engine.dispose()


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()

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
from sqlalchemy import (
    Table,
    and_,
    delete,
    event,
    false,
    func,
    inspect,
    or_,
    select,
    text,
    true,
    update,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from sentientai.core.clock import ensure_utc, utcnow
from sentientai.errors import StorageError
from sentientai.storage import schema
from sentientai.storage.records import (
    ActivityRecord,
    ActivityStatus,
    ApprovalRecord,
    ApprovalStatus,
    BeliefStatus,
    ConversationMessage,
    DocumentRecord,
    EmotionEvent,
    GoalRecord,
    GoalStatus,
    InviteRecord,
    MediaRecord,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    NoteRecord,
    ProposalRecord,
    ProposalStatus,
    RelationshipRecord,
    Scope,
    SelfBeliefRecord,
    SessionRecord,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolStats,
    UsageRecord,
    UserRecord,
    WorldObject,
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


def _scope_clause(column: Any, scope: Scope) -> Any:
    """SQL filter implementing :class:`Scope` (shared and/or one person's private memories)."""

    if scope.everything:
        return true()
    clauses = []
    if scope.include_global:
        clauses.append(column.is_(None))
    if scope.user_id:
        clauses.append(column == scope.user_id)
    return or_(*clauses) if clauses else false()


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
        scope: Scope = Scope(),
    ) -> builtins.list[MemoryRecord]:
        t = self.table
        stmt = select(t).where(_scope_clause(t.c.user_id, scope))
        if kinds is not None:
            stmt = stmt.where(t.c.kind.in_([k.value for k in kinds]))
        if status is not None:
            stmt = stmt.where(t.c.status == status.value)
        if since is not None:
            stmt = stmt.where(t.c.created_at >= since)
        order = t.c.created_at.desc() if newest_first else t.c.created_at.asc()
        return await self._select(stmt.order_by(order, t.c.id).limit(limit).offset(offset))

    async def by_source(self, source: str, limit: int = 500) -> builtins.list[MemoryRecord]:
        t = self.table
        return await self._select(
            select(t)
            .where(and_(t.c.source == source, t.c.status == MemoryStatus.ACTIVE.value))
            .order_by(t.c.created_at.asc(), t.c.id)
            .limit(limit)
        )

    async def sample(
        self, n: int, *, kinds: Iterable[MemoryKind] | None = None, scope: Scope = Scope()
    ) -> builtins.list[MemoryRecord]:
        """Random active memories (a mind wandering)."""

        t = self.table
        stmt = select(t).where(
            and_(t.c.status == MemoryStatus.ACTIVE.value, _scope_clause(t.c.user_id, scope))
        )
        if kinds is not None:
            stmt = stmt.where(t.c.kind.in_([k.value for k in kinds]))
        return await self._select(stmt.order_by(func.random()).limit(n))

    async def count_since(
        self, since: datetime, *, kinds: Iterable[MemoryKind] | None = None
    ) -> int:
        t = self.table
        stmt = select(func.count()).select_from(t).where(t.c.created_at >= since)
        if kinds is not None:
            stmt = stmt.where(t.c.kind.in_([k.value for k in kinds]))
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar_one())

    async def find_by_subject(
        self, subject_key: str, kind: MemoryKind | None = None, *, user_id: str | None = None
    ) -> builtins.list[MemoryRecord]:
        """Active memories about ``subject_key`` owned by exactly ``user_id`` (None = shared)."""

        t = self.table
        owner = t.c.user_id.is_(None) if user_id is None else t.c.user_id == user_id
        stmt = select(t).where(
            and_(t.c.subject_key == subject_key, t.c.status == MemoryStatus.ACTIVE.value, owner)
        )
        if kind is not None:
            stmt = stmt.where(t.c.kind == kind.value)
        return await self._select(stmt.order_by(t.c.updated_at.desc()))

    async def search_text(
        self,
        query: str,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 50,
        *,
        scope: Scope = Scope(),
    ) -> builtins.list[tuple[MemoryRecord, float]]:
        """Keyword search used to complement dense retrieval (hybrid recall)."""

        terms = [t for t in re.findall(r"[\w\-\.]{3,}", query.lower()) if len(t) >= 3][:12]
        if not terms:
            return []
        t = self.table
        content = func.lower(t.c.content)
        stmt = select(t).where(
            and_(
                t.c.status == MemoryStatus.ACTIVE.value,
                _scope_clause(t.c.user_id, scope),
                or_(*[content.contains(x) for x in terms]),
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

    async def count_by_kind(
        self, status: MemoryStatus = MemoryStatus.ACTIVE, *, scope: Scope = Scope.system()
    ) -> dict[str, int]:
        t = self.table
        stmt = (
            select(t.c.kind, func.count())
            .where(and_(t.c.status == status.value, _scope_clause(t.c.user_id, scope)))
            .group_by(t.c.kind)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return {str(kind): int(count) for kind, count in rows}

    async def purge(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        async with self.engine.begin() as conn:
            result = await conn.execute(delete(self.table).where(self.table.c.id.in_(list(ids))))
        return int(result.rowcount)

    async def ids_for_user(self, user_id: str) -> builtins.list[str]:
        t = self.table
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(t.c.id).where(t.c.user_id == user_id))).fetchall()
        return [str(r[0]) for r in rows]


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

    async def delete_user(self, user_id: str) -> int:
        return await _delete_where(self.engine, self.table.c.user_id, user_id)

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

    async def list(
        self, statuses: Iterable[GoalStatus] | None = None, *, scope: Scope | None = None
    ) -> builtins.list[GoalRecord]:
        t = self.table
        stmt = select(t)
        if statuses is not None:
            stmt = stmt.where(t.c.status.in_([s.value for s in statuses]))
        clause = _scope_clause(t.c.user_id, scope or Scope.system())
        if clause is not None:
            stmt = stmt.where(clause)
        return await self._select(stmt.order_by(t.c.priority.desc(), t.c.created_at.asc()))

    async def delete_user(self, user_id: str) -> None:
        await _delete_where(self.engine, self.table.c.user_id, user_id)


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

    async def delete(self, id_: str) -> None:
        await _delete_where(self.engine, self.table.c.id, id_)


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

    async def for_user(
        self, user_id: str, limit: int = 10_000
    ) -> builtins.list[ConversationMessage]:
        t = self.table
        return await self._select(
            select(t).where(t.c.user_id == user_id).order_by(t.c.created_at.asc()).limit(limit)
        )

    async def delete_user(self, user_id: str) -> int:
        return await _delete_where(self.engine, self.table.c.user_id, user_id)

    async def page(
        self, user_id: str, *, limit: int = 40, before: datetime | None = None
    ) -> builtins.list[ConversationMessage]:
        """One person's messages, oldest first, ending just before ``before``."""

        t = self.table
        stmt = select(t).where(t.c.user_id == user_id)
        if before is not None:
            stmt = stmt.where(t.c.created_at < before)
        records = await self._select(
            stmt.order_by(t.c.created_at.desc(), t.c.id.desc()).limit(limit)
        )
        return list(reversed(records))

    async def count_since(self, when: datetime, *, role: str | None = None) -> int:
        t = self.table
        stmt = select(func.count()).select_from(t).where(t.c.created_at >= when)
        if role is not None:
            stmt = stmt.where(t.c.role == role)
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar_one())

    async def search(
        self, user_id: str | None, query: str, limit: int = 20
    ) -> builtins.list[ConversationMessage]:
        """Keyword search over one person's conversations (None: the operator's)."""

        terms = re.findall(r"[\w\-']{3,}", query.lower())[:8]
        if not terms:
            return []
        t = self.table
        owner = t.c.user_id.is_(None) if user_id is None else t.c.user_id == user_id
        content = func.lower(t.c.content)
        return await self._select(
            select(t)
            .where(and_(owner, or_(*[content.contains(x) for x in terms])))
            .order_by(t.c.created_at.desc())
            .limit(limit)
        )

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


# --------------------------------------------------------------------- people
class UserRepository(_Repository[UserRecord]):
    table = schema.users
    model = UserRecord

    async def by_username(self, username: str) -> UserRecord | None:
        t = self.table
        records = await self._select(
            select(t).where(t.c.username_key == username.strip().casefold())
        )
        return records[0] if records else None

    async def list(self, limit: int = 500) -> builtins.list[UserRecord]:
        t = self.table
        return await self._select(select(t).order_by(t.c.created_at.asc()).limit(limit))

    async def count(self) -> int:
        async with self.engine.connect() as conn:
            return int(
                (await conn.execute(select(func.count()).select_from(self.table))).scalar_one()
            )

    async def delete(self, id_: str) -> None:
        await _delete_where(self.engine, self.table.c.id, id_)


class SessionRepository(_Repository[SessionRecord]):
    table = schema.sessions
    model = SessionRecord

    async def touch(self, id_: str, when: datetime, expires_at: datetime) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(self.table)
                .where(self.table.c.id == id_)
                .values(last_seen_at=when, expires_at=expires_at)
            )

    async def delete(self, id_: str) -> None:
        await _delete_where(self.engine, self.table.c.id, id_)

    async def delete_for_user(self, user_id: str, *, keep: str | None = None) -> int:
        t = self.table
        condition = t.c.user_id == user_id
        if keep is not None:
            condition = and_(condition, t.c.id != keep)
        async with self.engine.begin() as conn:
            result = await conn.execute(delete(t).where(condition))
        return int(result.rowcount)

    async def purge_expired(self, now: datetime) -> int:
        async with self.engine.begin() as conn:
            result = await conn.execute(delete(self.table).where(self.table.c.expires_at < now))
        return int(result.rowcount)


class InviteRepository(_Repository[InviteRecord]):
    table = schema.invites
    model = InviteRecord

    async def consume(self, id_: str, user_id: str, now: datetime) -> bool:
        """Atomically mark an unused, unexpired invite as used."""

        t = self.table
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(t)
                .where(and_(t.c.id == id_, t.c.used_by.is_(None), t.c.expires_at > now))
                .values(used_by=user_id, used_at=now)
            )
        return bool(result.rowcount == 1)

    async def list(self, limit: int = 100) -> builtins.list[InviteRecord]:
        t = self.table
        return await self._select(select(t).order_by(t.c.created_at.desc()).limit(limit))


class UsageRepository:
    """Per-user daily usage counters (quotas for a shared GPU box)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def get(self, user_id: str, day: str) -> UsageRecord:
        t = schema.usage
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(select(t).where(and_(t.c.user_id == user_id, t.c.day == day)))
            ).first()
        return (
            UsageRecord.model_validate(_row_dict(row))
            if row
            else UsageRecord(user_id=user_id, day=day)
        )

    async def add(
        self, user_id: str, day: str, *, messages: int = 0, images: int = 0, teach_bytes: int = 0
    ) -> UsageRecord:
        t = schema.usage
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(t)
                .where(and_(t.c.user_id == user_id, t.c.day == day))
                .values(
                    messages=t.c.messages + messages,
                    images=t.c.images + images,
                    teach_bytes=t.c.teach_bytes + teach_bytes,
                )
            )
            if result.rowcount == 0:
                await conn.execute(
                    t.insert().values(
                        user_id=user_id,
                        day=day,
                        messages=messages,
                        images=images,
                        teach_bytes=teach_bytes,
                    )
                )
        return await self.get(user_id, day)

    async def delete_user(self, user_id: str) -> None:
        await _delete_where(self.engine, schema.usage.c.user_id, user_id)


# ---------------------------------------------------------------- social mind
class RelationshipRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def get(self, user_id: str) -> RelationshipRecord | None:
        t = schema.relationships
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(t).where(t.c.user_id == user_id))).first()
        return RelationshipRecord.model_validate(_row_dict(row)) if row else None

    async def upsert(self, record: RelationshipRecord) -> RelationshipRecord:
        t = schema.relationships
        values = record.model_dump(mode="python")
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(t).where(t.c.user_id == record.user_id).values(**values)
            )
            if result.rowcount == 0:
                await conn.execute(t.insert().values(**values))
        return record

    async def list(self, limit: int = 500) -> builtins.list[RelationshipRecord]:
        t = schema.relationships
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(select(t).order_by(t.c.last_seen_at.desc()).limit(limit))
            ).fetchall()
        return [RelationshipRecord.model_validate(_row_dict(r)) for r in rows]

    async def delete(self, user_id: str) -> None:
        await _delete_where(self.engine, schema.relationships.c.user_id, user_id)


class SelfBeliefRepository(_Repository[SelfBeliefRecord]):
    table = schema.self_beliefs
    model = SelfBeliefRecord

    async def active(self, aspect: str | None = None) -> builtins.list[SelfBeliefRecord]:
        t = self.table
        stmt = select(t).where(t.c.status == BeliefStatus.ACTIVE.value)
        if aspect is not None:
            stmt = stmt.where(t.c.aspect == aspect)
        return await self._select(stmt.order_by(t.c.formed_at.asc()))

    async def with_status(self, status: BeliefStatus) -> builtins.list[SelfBeliefRecord]:
        t = self.table
        return await self._select(
            select(t).where(t.c.status == status.value).order_by(t.c.formed_at.asc())
        )

    async def recent_changes(self, limit: int = 30) -> builtins.list[SelfBeliefRecord]:
        t = self.table
        return await self._select(select(t).order_by(t.c.formed_at.desc()).limit(limit))

    async def history(self, aspect: str) -> builtins.list[SelfBeliefRecord]:
        t = self.table
        return await self._select(
            select(t).where(t.c.aspect == aspect).order_by(t.c.formed_at.asc())
        )

    async def replace(
        self, old_id: str | None, new: SelfBeliefRecord, when: datetime
    ) -> SelfBeliefRecord:
        """Retire the current belief (if any) and record the new one atomically."""

        t = self.table
        async with self.engine.begin() as conn:
            if old_id is not None:
                await conn.execute(
                    update(t)
                    .where(and_(t.c.id == old_id, t.c.status == BeliefStatus.ACTIVE.value))
                    .values(status=BeliefStatus.REVISED.value, revised_at=when)
                )
            await conn.execute(t.insert().values(**new.model_dump(mode="python")))
        return new

    async def activate(
        self, proposal: SelfBeliefRecord, old_id: str | None, reasoning: str, when: datetime
    ) -> None:
        """Adopt a considered change: the old belief is retired, the proposal takes its place."""

        t = self.table
        proposal.status = BeliefStatus.ACTIVE
        proposal.reasoning = reasoning
        proposal.formed_at = when
        async with self.engine.begin() as conn:
            if old_id is not None:
                await conn.execute(
                    update(t)
                    .where(and_(t.c.id == old_id, t.c.status == BeliefStatus.ACTIVE.value))
                    .values(status=BeliefStatus.REVISED.value, revised_at=when)
                )
            await conn.execute(
                update(t)
                .where(t.c.id == proposal.id)
                .values(
                    status=proposal.status.value,
                    reasoning=reasoning,
                    formed_at=when,
                    confidence=proposal.confidence,
                )
            )


class EmotionRepository(_Repository[EmotionEvent]):
    table = schema.emotion_events
    model = EmotionEvent

    async def recent(
        self, limit: int = 50, since: datetime | None = None, *, scope: Scope | None = None
    ) -> builtins.list[EmotionEvent]:
        t = self.table
        stmt = select(t)
        if since is not None:
            stmt = stmt.where(t.c.created_at >= since)
        clause = _scope_clause(t.c.user_id, scope or Scope.system())
        if clause is not None:
            stmt = stmt.where(clause)
        return await self._select(stmt.order_by(t.c.created_at.desc()).limit(limit))

    async def delete_user(self, user_id: str) -> None:
        await _delete_where(self.engine, self.table.c.user_id, user_id)


class ActivityRepository(_Repository[ActivityRecord]):
    table = schema.activities
    model = ActivityRecord

    async def recent(
        self, limit: int = 50, *, scope: Scope | None = None, kind: str | None = None
    ) -> builtins.list[ActivityRecord]:
        t = self.table
        stmt = select(t)
        clause = _scope_clause(t.c.user_id, scope or Scope.system())
        if clause is not None:
            stmt = stmt.where(clause)
        if kind is not None:
            stmt = stmt.where(t.c.kind == kind)
        return await self._select(stmt.order_by(t.c.started_at.desc()).limit(limit))

    async def finish(self, id_: str, status: ActivityStatus, outcome: str, when: datetime) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(self.table)
                .where(self.table.c.id == id_)
                .values(status=status.value, outcome=outcome, finished_at=when)
            )

    async def fail_running(self, when: datetime) -> int:
        """Activities interrupted by a restart never finished."""

        t = self.table
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(t)
                .where(t.c.status == ActivityStatus.RUNNING.value)
                .values(status=ActivityStatus.FAILED.value, outcome="interrupted", finished_at=when)
            )
        return int(result.rowcount)

    async def delete_user(self, user_id: str) -> None:
        await _delete_where(self.engine, self.table.c.user_id, user_id)


class NoteRepository(_Repository[NoteRecord]):
    table = schema.notes
    model = NoteRecord

    async def for_user(
        self, user_id: str, *, unread_only: bool = False
    ) -> builtins.list[NoteRecord]:
        t = self.table
        stmt = select(t).where(t.c.user_id == user_id)
        if unread_only:
            stmt = stmt.where(t.c.read_at.is_(None))
        return await self._select(stmt.order_by(t.c.created_at.desc()).limit(200))

    async def count_since(self, user_id: str, since: datetime) -> int:
        t = self.table
        async with self.engine.connect() as conn:
            return int(
                (
                    await conn.execute(
                        select(func.count())
                        .select_from(t)
                        .where(and_(t.c.user_id == user_id, t.c.created_at >= since))
                    )
                ).scalar_one()
            )

    async def mark_read(self, user_id: str, when: datetime) -> None:
        t = self.table
        async with self.engine.begin() as conn:
            await conn.execute(
                update(t)
                .where(and_(t.c.user_id == user_id, t.c.read_at.is_(None)))
                .values(read_at=when)
            )

    async def delete_user(self, user_id: str) -> None:
        await _delete_where(self.engine, self.table.c.user_id, user_id)


class MediaRepository(_Repository[MediaRecord]):
    table = schema.media
    model = MediaRecord

    async def visible_to(
        self, user_id: str | None, *, admin: bool = False, limit: int = 100
    ) -> builtins.list[MediaRecord]:
        """Fuzzy's own images and public ones, plus the viewer's private requests."""

        t = self.table
        stmt = select(t)
        if not admin:
            allowed = or_(t.c.public.is_(True), t.c.user_id.is_(None))
            if user_id:
                allowed = or_(allowed, t.c.user_id == user_id)
            stmt = stmt.where(allowed)
        return await self._select(stmt.order_by(t.c.created_at.desc()).limit(limit))

    async def for_user(self, user_id: str) -> builtins.list[MediaRecord]:
        t = self.table
        return await self._select(select(t).where(t.c.user_id == user_id))

    async def delete(self, id_: str) -> None:
        await _delete_where(self.engine, self.table.c.id, id_)


class WorldRepository(_Repository[WorldObject]):
    table = schema.world_objects
    model = WorldObject

    async def all(self) -> builtins.list[WorldObject]:
        t = self.table
        return await self._select(select(t).order_by(t.c.created_at.asc()))

    async def delete(self, id_: str) -> None:
        await _delete_where(self.engine, self.table.c.id, id_)

    async def count(self) -> int:
        async with self.engine.connect() as conn:
            return int(
                (await conn.execute(select(func.count()).select_from(self.table))).scalar_one()
            )


async def _delete_where(engine: AsyncEngine, column: Any, value: Any) -> int:
    async with engine.begin() as conn:
        result = await conn.execute(delete(column.table).where(column == value))
    return int(result.rowcount)


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
        self.users = UserRepository(self.engine)
        self.sessions = SessionRepository(self.engine)
        self.invites = InviteRepository(self.engine)
        self.usage = UsageRepository(self.engine)
        self.relationships = RelationshipRepository(self.engine)
        self.self_beliefs = SelfBeliefRepository(self.engine)
        self.emotions = EmotionRepository(self.engine)
        self.activities = ActivityRepository(self.engine)
        self.notes = NoteRepository(self.engine)
        self.media = MediaRepository(self.engine)
        self.world = WorldRepository(self.engine)

    @property
    def dialect(self) -> str:
        return str(self.engine.dialect.name)

    async def create_all(self) -> None:
        """Create missing tables and upgrade older schemas in place."""

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
                return
            current = int(row[0])
            if current > schema.SCHEMA_VERSION:
                raise StorageError(
                    f"database schema v{current} is newer than this build "
                    f"(v{schema.SCHEMA_VERSION})"
                )
            for version in range(current + 1, schema.SCHEMA_VERSION + 1):
                for table, column, ddl in schema.ADDED_COLUMNS.get(version, []):
                    existing = await conn.run_sync(_column_names, table)
                    if column not in existing:
                        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
            if current < schema.SCHEMA_VERSION:
                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS ix_memories_user ON memories (user_id)")
                )
                await conn.execute(
                    update(schema.schema_meta)
                    .where(schema.schema_meta.c.key == "version")
                    .values(value=str(schema.SCHEMA_VERSION))
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


def _column_names(sync_conn: Any, table: str) -> set[str]:
    return {str(c["name"]) for c in inspect(sync_conn).get_columns(table)}


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()

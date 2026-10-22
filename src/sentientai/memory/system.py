"""The memory system: one facade over working, semantic, episodic and procedural memory.

Retrieval is hybrid (dense vectors + keyword overlap) and re-ranked with the
Generative-Agents style blend of similarity, recency, importance and strength.

Forgetting follows a spaced-repetition curve: retention halves every
``half_life * (1 + 2*importance) * (1 + ln(1 + accesses))`` days since the
memory was last recalled. Recall resets the clock, so memories that keep
proving useful persist while noise fades. Decay is computed from timestamps,
so running it repeatedly is idempotent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sentientai.config import MemoryConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.core.metrics import Metrics
from sentientai.memory.types import Episode, Procedure
from sentientai.memory.vector import VectorItem, VectorStore
from sentientai.memory.working import Turn, WorkingMemory
from sentientai.models.embeddings import Embedder
from sentientai.storage.database import Database
from sentientai.storage.records import (
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    Scope,
    ScoredMemory,
    scope_key,
)

log = get_logger(__name__)


def _item(record: MemoryRecord, vector: list[float]) -> VectorItem:
    return VectorItem(record.id, vector, record.kind.value, scope_key(record.user_id))


_SECONDS_PER_DAY = 86400.0


@dataclass
class DecayReport:
    scanned: int = 0
    updated: int = 0
    archived: list[str] = field(default_factory=list)


class MemorySystem:
    def __init__(
        self,
        db: Database,
        vectors: VectorStore,
        embedder: Embedder,
        working: WorkingMemory,
        config: MemoryConfig,
        *,
        bus: EventBus | None = None,
        metrics: Metrics | None = None,
        clock: Clock | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.working = working
        self.config = config
        self.bus = bus
        self.metrics = metrics
        self.clock = clock or SystemClock()
        # Asymmetric embedding models (nomic, e5, bge) expect task prefixes.
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embedder.embed([self.document_prefix + t for t in texts])

    async def _embed_query(self, text: str) -> list[float]:
        [vector] = await self.embedder.embed([self.query_prefix + text])
        return vector

    async def open(self) -> None:
        await self.vectors.open()

    async def aclose(self) -> None:
        await self.vectors.aclose()
        await self.working.aclose()
        closer = getattr(self.embedder, "aclose", None)
        if closer is not None:
            await closer()

    # ------------------------------------------------------------ helpers
    def _count(self, op: str, kind: MemoryKind | str) -> None:
        if self.metrics:
            self.metrics.memory_ops.inc(op=op, kind=str(kind))

    async def _emit(self, type_: str, **payload: Any) -> None:
        if self.bus:
            await self.bus.publish(type_, **payload)

    async def _index(self, records: Sequence[MemoryRecord]) -> None:
        if not records:
            return
        vectors = await self._embed_documents([r.content for r in records])
        await self.vectors.upsert([_item(r, v) for r, v in zip(records, vectors, strict=True)])

    # ------------------------------------------------------------- writes
    async def remember(
        self,
        content: str,
        kind: MemoryKind = MemoryKind.FACT,
        *,
        importance: float = 0.5,
        confidence: float = 0.8,
        subject_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
        dedupe: bool = True,
        user_id: str | None = None,
    ) -> MemoryRecord:
        """Store a memory. ``user_id`` makes it private to that person (and to Fuzzy)."""

        content = content.strip()
        if not content:
            raise ValueError("cannot remember empty content")
        now = self.clock.now()
        [vector] = await self._embed_documents([content])

        superseded: list[MemoryRecord] = []
        if subject_key:
            for existing in await self.db.memories.find_by_subject(
                subject_key, kind, user_id=user_id
            ):
                if existing.content.strip() == content:
                    return await self._reinforce(existing, importance, now)
                superseded.append(existing)
        elif dedupe:
            for hit in await self.vectors.search(
                vector, k=3, kinds=[kind.value], scopes=[scope_key(user_id)]
            ):
                if hit.score < self.config.dedupe_similarity:
                    break
                candidate = await self.db.memories.get(hit.id)
                if candidate is not None and candidate.status is MemoryStatus.ACTIVE:
                    return await self._reinforce(candidate, importance, now)

        record = MemoryRecord(
            kind=kind,
            content=content,
            user_id=user_id,
            subject_key=subject_key,
            metadata=metadata or {},
            importance=importance,
            confidence=confidence,
            source=source,
            embedding_model=self.embedder.model_id,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )
        await self.db.memories.insert(record)
        await self.vectors.upsert([_item(record, vector)])
        if superseded:
            ids = [r.id for r in superseded]
            await self.db.memories.set_status(ids, MemoryStatus.SUPERSEDED, superseded_by=record.id)
            await self.vectors.delete(ids)
            self._count("supersede", kind)
        self._count("store", kind)
        await self._emit(
            "memory.stored",
            memory_id=record.id,
            kind=kind.value,
            superseded=[r.id for r in superseded],
        )
        return record

    async def store_batch(self, records: Sequence[MemoryRecord], *, batch_size: int = 64) -> int:
        """Bulk insert without dedupe; used by document ingestion."""

        for start in range(0, len(records), batch_size):
            batch = list(records[start : start + batch_size])
            vectors = await self._embed_documents([r.content for r in batch])
            for record in batch:
                record.embedding_model = self.embedder.model_id
            await self.db.memories.insert_many(batch)
            await self.vectors.upsert([_item(r, v) for r, v in zip(batch, vectors, strict=True)])
            for record in batch:
                self._count("store", record.kind)
        await self.vectors.flush()
        return len(records)

    async def _reinforce(
        self, record: MemoryRecord, importance: float, now: datetime
    ) -> MemoryRecord:
        record.importance = max(record.importance, importance)
        record.strength = 1.0
        record.access_count += 1
        record.updated_at = now
        record.last_accessed_at = now
        await self.db.memories.save(record)
        self._count("reinforce", record.kind)
        return record

    async def forget(self, memory_id: str, *, hard: bool = False) -> bool:
        record = await self.db.memories.get(memory_id)
        if record is None:
            return False
        await self.vectors.delete([memory_id])
        if hard:
            await self.db.memories.purge([memory_id])
        else:
            await self.db.memories.set_status([memory_id], MemoryStatus.DELETED)
        self._count("forget", record.kind)
        await self._emit("memory.forgotten", memory_id=memory_id, hard=hard)
        return True

    # -------------------------------------------------------------- reads
    def _recency(self, record: MemoryRecord, now: datetime) -> float:
        age_days = max(0.0, (now - record.last_accessed_at).total_seconds() / _SECONDS_PER_DAY)
        return float(0.5 ** (age_days / max(self.config.decay_half_life_days / 4, 1e-6)))

    async def recall(
        self,
        query: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        k: int | None = None,
        min_similarity: float | None = None,
        touch: bool = True,
        scope: Scope = Scope(),
    ) -> list[ScoredMemory]:
        """Hybrid recall limited to ``scope`` (default: shared memories only)."""

        if not query.strip():
            return []
        k = k or self.config.recall_k
        floor = self.config.min_relevance if min_similarity is None else min_similarity
        kind_list = list(kinds) if kinds is not None else None
        kind_values = [x.value for x in kind_list] if kind_list is not None else None

        qvec = await self._embed_query(query)
        hits = await self.vectors.search(
            qvec, k=k * 4, kinds=kind_values, scopes=scope.vector_scopes()
        )
        dense = {h.id: h.score for h in hits}
        lexical = {
            r.id: s
            for r, s in await self.db.memories.search_text(
                query, kind_list, limit=k * 2, scope=scope
            )
        }
        records = await self.db.memories.get_many(list(dense.keys() | lexical.keys()))

        now = self.clock.now()
        c = self.config
        scored: list[ScoredMemory] = []
        for id_, record in records.items():
            # Defence in depth: the indexes already filter, but never trust them alone.
            if record.status is not MemoryStatus.ACTIVE or not scope.allows(record.user_id):
                continue
            dense_sim = max(0.0, dense.get(id_, 0.0))
            lex = lexical.get(id_, 0.0)
            similarity = dense_sim + 0.25 * lex * (1 - dense_sim) if id_ in dense else 0.6 * lex
            if similarity < floor:
                continue
            score = (
                c.similarity_weight * similarity
                + c.recency_weight * self._recency(record, now)
                + c.importance_weight * record.importance
                + c.strength_weight * record.strength
            )
            scored.append(ScoredMemory(record=record, score=score, similarity=similarity))
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:k]
        if touch and top:
            await self.db.memories.mark_accessed([s.record.id for s in top], now, boost=0.2)
        for s in top:
            self._count("recall", s.record.kind)
        await self._emit("memory.recalled", query=query[:200], hits=[s.record.id for s in top])
        return top

    async def get(self, memory_id: str) -> MemoryRecord | None:
        return await self.db.memories.get(memory_id)

    async def timeline(
        self,
        kinds: Iterable[MemoryKind] | None = (MemoryKind.EPISODE,),
        *,
        since: datetime | None = None,
        limit: int = 20,
        scope: Scope = Scope(),
    ) -> list[MemoryRecord]:
        """Chronological episodic access (newest first)."""

        return await self.db.memories.list(kinds=kinds, since=since, limit=limit, scope=scope)

    # ----------------------------------------------------------- episodic
    async def record_episode(
        self, episode: Episode, *, importance: float | None = None, user_id: str | None = None
    ) -> MemoryRecord:
        if importance is None:
            importance = 0.7 if episode.outcome in ("failure", "aborted") else 0.5
            importance = min(1.0, importance + 0.1 * len(episode.lessons))
        return await self.remember(
            episode.render(),
            MemoryKind.EPISODE,
            importance=importance,
            metadata={"episode": episode.model_dump(mode="json")},
            source=episode.source,
            dedupe=False,
            user_id=user_id,
        )

    # --------------------------------------------------------- procedural
    async def learn_procedure(self, procedure: Procedure, *, success: bool = True) -> MemoryRecord:
        existing = await self.db.memories.find_by_subject(
            procedure.subject_key, MemoryKind.PROCEDURE
        )
        if existing:
            record = existing[0]
            current = Procedure.from_record(record)
            current.successes += 1 if success else 0
            current.failures += 0 if success else 1
            current.source_tasks = (current.source_tasks + procedure.source_tasks)[-20:]
            if success and len(procedure.steps) >= 1:
                current.steps = procedure.steps
                current.when_to_use = procedure.when_to_use
            record.metadata = {"procedure": current.model_dump(mode="json")}
            record.content = current.render()
            record.confidence = current.reliability
            record.updated_at = self.clock.now()
            await self.db.memories.save(record)
            await self._index([record])
            self._count("procedure_update", MemoryKind.PROCEDURE)
            return record
        if success:
            procedure.successes = max(procedure.successes, 1)
        else:
            procedure.failures = max(procedure.failures, 1)
        return await self.remember(
            procedure.render(),
            MemoryKind.PROCEDURE,
            importance=0.6,
            confidence=procedure.reliability,
            subject_key=procedure.subject_key,
            metadata={"procedure": procedure.model_dump(mode="json")},
            dedupe=False,
        )

    async def record_procedure_outcome(self, memory_id: str, *, success: bool) -> None:
        record = await self.db.memories.get(memory_id)
        if record is None or record.kind is not MemoryKind.PROCEDURE:
            return
        proc = Procedure.from_record(record)
        await self.learn_procedure(proc.model_copy(update={"source_tasks": []}), success=success)

    async def find_procedures(self, query: str, k: int = 3) -> list[ScoredMemory]:
        """Procedures are Fuzzy's own skills, always shared."""

        return await self.recall(query, kinds=[MemoryKind.PROCEDURE], k=k, min_similarity=0.3)

    async def forget_source(self, source: str) -> int:
        """Remove every memory that came from one source (e.g. a taught document)."""

        ids = [m.id for m in await self.db.memories.by_source(source, limit=100_000)]
        await self.vectors.delete(ids)
        await self.db.memories.purge(ids)
        await self.vectors.flush()
        await self._emit("memory.source_forgotten", source=source, count=len(ids))
        return len(ids)

    async def forget_user(self, user_id: str) -> int:
        """Erase every private memory about a person (account deletion)."""

        ids = await self.db.memories.ids_for_user(user_id)
        await self.vectors.delete(ids)
        await self.db.memories.purge(ids)
        await self.vectors.flush()
        await self._emit("memory.user_erased", user_id=user_id, count=len(ids))
        return len(ids)

    # ------------------------------------------------------------ working
    async def add_turn(self, session_id: str, role: str, content: str, **metadata: Any) -> int:
        return await self.working.append(
            session_id, Turn(role=role, content=content, metadata=metadata)
        )

    async def conversation(self, session_id: str, limit: int | None = None) -> list[Turn]:
        return await self.working.turns(session_id, limit)

    # --------------------------------------------------------------- decay
    def retention(self, record: MemoryRecord, now: datetime) -> float:
        age_days = max(0.0, (now - record.last_accessed_at).total_seconds() / _SECONDS_PER_DAY)
        stability = (
            self.config.decay_half_life_days
            * (1 + 2 * record.importance)
            * (1 + math.log1p(record.access_count))
        )
        return float(0.5 ** (age_days / stability))

    async def decay(self, *, batch: int = 500) -> DecayReport:
        report = DecayReport()
        now = self.clock.now()
        offset = 0
        while True:
            page = await self.db.memories.list(
                limit=batch, offset=offset, newest_first=False, scope=Scope.system()
            )
            if not page:
                break
            offset += len(page)
            strengths: dict[str, float] = {}
            for record in page:
                report.scanned += 1
                strength = round(self.retention(record, now), 4)
                pinned = record.importance >= self.config.pin_importance
                if strength < self.config.prune_strength_threshold and not pinned:
                    report.archived.append(record.id)
                elif abs(strength - record.strength) > 1e-4:
                    strengths[record.id] = strength
            await self.db.memories.update_strengths(strengths)
            report.updated += len(strengths)
        if report.archived:
            await self.db.memories.set_status(report.archived, MemoryStatus.ARCHIVED)
            await self.vectors.delete(report.archived)
        await self.vectors.flush()
        self._count("decay_archive", "all")
        await self._emit("memory.decayed", scanned=report.scanned, archived=len(report.archived))
        return report

    # ---------------------------------------------------------- maintenance
    async def reindex(self, *, batch: int = 256) -> int:
        """Rebuild the vector index from the relational source of truth."""

        total = 0
        offset = 0
        while True:
            page = await self.db.memories.list(
                limit=batch, offset=offset, newest_first=False, scope=Scope.system()
            )
            if not page:
                break
            offset += len(page)
            await self._index(page)
            for record in page:
                if record.embedding_model != self.embedder.model_id:
                    record.embedding_model = self.embedder.model_id
                    await self.db.memories.save(record)
            total += len(page)
        await self.vectors.flush()
        return total

    async def stats(self) -> dict[str, Any]:
        counts = await self.db.memories.count_by_kind()
        if self.metrics:
            for kind, count in counts.items():
                self.metrics.memory_items.set(count, kind=kind)
        return {
            "active_by_kind": counts,
            "vectors": await self.vectors.count(),
            "embedding_model": self.embedder.model_id,
            "working_sessions": len(await self.working.sessions()),
        }

from __future__ import annotations

from datetime import timedelta

from sentientai.config import RiskLevel
from sentientai.core.clock import utcnow
from sentientai.storage.database import Database
from sentientai.storage.records import (
    ApprovalRecord,
    ApprovalStatus,
    ConversationMessage,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    TaskRecord,
    TaskSource,
    TaskStatus,
    ToolInvocationRecord,
)


async def test_memory_roundtrip_and_queries(db: Database) -> None:
    fact = MemoryRecord(
        kind=MemoryKind.FACT,
        content="Operator prefers Rust for systems code",
        subject_key="operator.language",
        metadata={"tags": ["pref"]},
        importance=0.8,
    )
    note = MemoryRecord(kind=MemoryKind.THOUGHT, content="I wonder how the GPU scheduler behaves")
    await db.memories.insert(fact)
    await db.memories.insert(note)

    loaded = await db.memories.get(fact.id)
    assert loaded is not None
    assert loaded.metadata == {"tags": ["pref"]}
    assert loaded.created_at.tzinfo is not None
    assert loaded.kind is MemoryKind.FACT

    assert [m.id for m in await db.memories.find_by_subject("operator.language")] == [fact.id]
    facts = await db.memories.list(kinds=[MemoryKind.FACT])
    assert [m.id for m in facts] == [fact.id]

    hits = await db.memories.search_text("which language for systems code?")
    assert hits and hits[0][0].id == fact.id

    await db.memories.mark_accessed([fact.id], utcnow(), boost=0.5)
    again = await db.memories.get(fact.id)
    assert again is not None and again.access_count == 1 and again.strength == 1.0

    await db.memories.set_status([note.id], MemoryStatus.ARCHIVED)
    assert await db.memories.count_by_kind() == {"fact": 1}


async def test_task_queue_claims_by_priority_and_recovers(db: Database) -> None:
    low = await db.tasks.insert(TaskRecord(goal="low", priority=1))
    high = await db.tasks.insert(TaskRecord(goal="high", priority=9, source=TaskSource.API))
    claimed = await db.tasks.claim_next()
    assert claimed is not None and claimed.id == high.id and claimed.status is TaskStatus.RUNNING

    # Simulate a crash: the RUNNING task is re-queued on recovery.
    recovered = await db.tasks.recover_interrupted()
    assert [t.id for t in recovered] == [high.id]
    requeued = await db.tasks.get(high.id)
    assert requeued is not None and requeued.status is TaskStatus.PENDING

    second = await db.tasks.claim_next()
    assert second is not None and second.id == high.id and second.attempts == 2
    await db.tasks.finish(second.id, TaskStatus.SUCCEEDED, result="done")
    third = await db.tasks.claim_next()
    assert third is not None and third.id == low.id
    assert await db.tasks.claim_next() is None


async def test_approval_decide_is_single_shot(db: Database) -> None:
    approval = await db.approvals.insert(
        ApprovalRecord(tool="shell", arguments={"cmd": "ls"}, risk=RiskLevel.HIGH, reasons=["x"])
    )
    assert await db.approvals.decide(approval.id, ApprovalStatus.APPROVED, "op")
    assert not await db.approvals.decide(approval.id, ApprovalStatus.DENIED, "op")
    stored = await db.approvals.get(approval.id)
    assert stored is not None and stored.status is ApprovalStatus.APPROVED
    assert stored.risk is RiskLevel.HIGH


async def test_state_upsert(db: Database) -> None:
    assert await db.state.get("affect") is None
    await db.state.set("affect", {"curiosity": 0.4})
    await db.state.set("affect", {"curiosity": 0.9})
    assert await db.state.get("affect") == {"curiosity": 0.9}


async def test_tool_stats(db: Database) -> None:
    for i in range(10):
        await db.tool_invocations.insert(
            ToolInvocationRecord(
                tool="fs_read",
                args_digest="x",
                ok=i != 3,
                duration_ms=float(i),
                error="boom" if i == 3 else None,
            )
        )
    [stats] = await db.tool_invocations.stats(since=utcnow() - timedelta(hours=1))
    assert stats.calls == 10 and stats.failures == 1 and stats.top_errors == ["boom"]
    assert stats.p95_ms == 9.0


async def test_conversation_ordering(db: Database) -> None:
    for i in range(5):
        await db.conversations.insert(
            ConversationMessage(session_id="s1", role="user", content=str(i))
        )
    recent = await db.conversations.recent("s1", limit=3)
    assert [m.content for m in recent] == ["2", "3", "4"]
    sessions = await db.conversations.sessions()
    assert sessions[0][0] == "s1"

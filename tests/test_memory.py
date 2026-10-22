from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from sentientai.config import MemoryConfig, RoutingConfig
from sentientai.core.clock import FakeClock
from sentientai.memory.consolidation import MemoryConsolidator, valid_fact
from sentientai.memory.system import MemorySystem
from sentientai.memory.types import Episode, Procedure
from sentientai.memory.vector import (
    InMemoryVectorStore,
    PgVectorStore,
    QdrantVectorStore,
    VectorItem,
    VectorStore,
)
from sentientai.memory.working import InMemoryWorkingMemory, RedisWorkingMemory, Turn, WorkingMemory
from sentientai.models.embeddings import HashingEmbedder
from sentientai.models.router import ModelRouter
from sentientai.models.scripted import ScriptedLLM
from sentientai.models.types import ModelTier
from sentientai.storage.database import Database
from sentientai.storage.records import ConversationMessage, MemoryKind, MemoryStatus
from sentientai.storage.schema import metadata
from tests.conftest import POSTGRES_URL, REDIS_URL, postgres_available, redis_available

DIM = 256

# ------------------------------------------------------------- vector stores


@pytest.fixture(
    params=[
        "memory",
        "qdrant",
        pytest.param("pgvector", marks=pytest.mark.integration),
    ]
)
async def vector_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[VectorStore]:
    store: VectorStore
    db: Database | None = None
    if request.param == "memory":
        store = InMemoryVectorStore(DIM, tmp_path / "v.npz")
    elif request.param == "qdrant":
        store = QdrantVectorStore(DIM, collection="test")
    else:
        if not postgres_available():
            pytest.skip("PostgreSQL not reachable")
        db = Database(POSTGRES_URL)
        async with db.engine.begin() as conn:
            from sqlalchemy import text

            await conn.execute(text("DROP TABLE IF EXISTS memory_vectors"))
        store = PgVectorStore(db.engine, DIM)
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()
        if db is not None:
            await db.close()


async def test_vector_store_contract(vector_store: VectorStore) -> None:
    emb = HashingEmbedder(DIM)
    texts = {
        "a": ("GPU inference with llama.cpp on ROCm", "document"),
        "b": ("The operator's cat is named Turing", "fact"),
        "c": ("Running vLLM tensor parallel inference on GPUs", "document"),
    }
    vectors = await emb.embed([t for t, _ in texts.values()])
    await vector_store.upsert(
        [VectorItem(k, v, kind) for (k, (_, kind)), v in zip(texts.items(), vectors)]
    )
    assert await vector_store.count() == 3

    [q] = await emb.embed(["GPU inference"])
    hits = await vector_store.search(q, k=2)
    assert [h.id for h in hits][:1] in (["a"], ["c"]) and len(hits) == 2
    assert hits[0].score >= hits[1].score

    only_facts = await vector_store.search(q, k=5, kinds=["fact"])
    assert [h.id for h in only_facts] == ["b"]
    assert await vector_store.search(q, k=5, kinds=[]) == []

    # upsert replaces
    await vector_store.upsert([VectorItem("b", vectors[0], "fact")])
    assert await vector_store.count() == 3
    await vector_store.delete(["a", "missing"])
    assert await vector_store.count() == 2


async def test_in_memory_vectors_persist(tmp_path: Path) -> None:
    path = tmp_path / "v.npz"
    store = InMemoryVectorStore(8, path)
    await store.upsert([VectorItem("x", [1.0] + [0.0] * 7, "fact")])
    await store.aclose()
    reopened = InMemoryVectorStore(8, path)
    await reopened.open()
    assert await reopened.count() == 1
    with pytest.raises(Exception, match="dims"):
        await InMemoryVectorStore(16, path).open()


# ------------------------------------------------------------ working memory


@pytest.fixture(params=["memory", pytest.param("redis", marks=pytest.mark.integration)])
async def working(request: pytest.FixtureRequest) -> AsyncIterator[WorkingMemory]:
    wm: WorkingMemory
    if request.param == "redis":
        if not redis_available():
            pytest.skip("Redis not reachable")
        wm = RedisWorkingMemory(REDIS_URL, max_turns=5, prefix=f"test-{uuid.uuid4().hex[:8]}")
    else:
        wm = InMemoryWorkingMemory(max_turns=5)
    try:
        yield wm
    finally:
        await wm.aclose()


async def test_working_memory_contract(working: WorkingMemory) -> None:
    for i in range(7):
        await working.append("s1", Turn(role="user", content=f"m{i}"))
    turns = await working.turns("s1")
    assert [t.content for t in turns] == ["m2", "m3", "m4", "m5", "m6"]  # trimmed to max
    assert [t.content for t in await working.turns("s1", limit=2)] == ["m5", "m6"]
    await working.append("s2", Turn(role="user", content="x"))
    assert set(await working.sessions()) == {"s1", "s2"}
    await working.replace("s1", [Turn(role="system", content="summary")])
    assert [t.content for t in await working.turns("s1")] == ["summary"]
    await working.set("focus", {"task": "t1"})
    assert await working.get("focus") == {"task": "t1"}
    await working.clear("s2")
    assert "s2" not in await working.sessions()
    assert await working.ping()


# ------------------------------------------------------------- memory system


@pytest.fixture
async def memory(sqlite_db: Database, tmp_path: Path) -> AsyncIterator[MemorySystem]:
    clock = FakeClock()
    system = MemorySystem(
        sqlite_db,
        InMemoryVectorStore(DIM, tmp_path / "vectors.npz"),
        HashingEmbedder(DIM),
        InMemoryWorkingMemory(),
        MemoryConfig(decay_half_life_days=10, prune_strength_threshold=0.1),
        clock=clock,
    )
    await system.open()
    system.clock = clock
    yield system
    await system.aclose()


async def test_remember_dedupes_and_supersedes(memory: MemorySystem) -> None:
    first = await memory.remember("The operator's favourite editor is Neovim", importance=0.4)
    again = await memory.remember("The operator's favourite editor is Neovim", importance=0.7)
    assert again.id == first.id and again.access_count == 1 and again.importance == 0.7

    old = await memory.remember("Operator lives in Lisbon", subject_key="operator.city")
    new = await memory.remember("Operator lives in Porto", subject_key="operator.city")
    old_now = await memory.get(old.id)
    assert old_now is not None and old_now.status is MemoryStatus.SUPERSEDED
    assert old_now.superseded_by == new.id
    hits = await memory.recall("where does the operator live?")
    assert hits[0].record.id == new.id
    assert old.id not in [h.record.id for h in hits]


async def test_recall_ranking_and_kind_filter(memory: MemorySystem) -> None:
    await memory.remember("SentientAI stores working memory in Redis", MemoryKind.DOCUMENT)
    target = await memory.remember(
        "The heavy model runs on two Radeon Pro V340 GPUs", importance=0.9
    )
    await memory.remember("Banana bread needs ripe bananas")
    hits = await memory.recall("which GPUs run the heavy model?")
    assert hits[0].record.id == target.id
    assert all(h.similarity >= memory.config.min_relevance for h in hits)
    docs = await memory.recall("Redis working memory", kinds=[MemoryKind.DOCUMENT])
    assert docs and all(h.record.kind is MemoryKind.DOCUMENT for h in docs)
    touched = await memory.get(target.id)
    assert touched is not None and touched.access_count == 1


async def test_forget(memory: MemorySystem) -> None:
    rec = await memory.remember("temporary secret plan for testing forget")
    assert await memory.forget(rec.id)
    assert not await memory.recall("secret plan forget")
    assert await memory.forget(rec.id, hard=True)
    assert await memory.get(rec.id) is None


async def test_decay_is_idempotent_and_archives(memory: MemorySystem) -> None:
    clock = memory.clock
    assert isinstance(clock, FakeClock)
    trivial = await memory.remember("Weather was cloudy on Tuesday", importance=0.0)
    core = await memory.remember("Operator name is Dana", importance=0.95)
    recalled = await memory.remember("Project codename is Prometheus", importance=0.0)

    clock.advance(20 * 86400)
    await memory.recall("project codename prometheus")  # spaced retrieval resets the clock
    clock.advance(20 * 86400)

    report = await memory.decay()
    assert trivial.id in report.archived
    assert core.id not in report.archived  # pinned by importance
    assert recalled.id not in report.archived
    again = await memory.decay()
    assert again.archived == []  # idempotent
    stored = await memory.get(recalled.id)
    assert stored is not None and 0.1 < stored.strength < 1.0


async def test_episodes_and_procedures(memory: MemorySystem) -> None:
    episode = Episode(
        goal="Summarize the README",
        outcome="success",
        summary="Read and summarized",
        tools_used=["fs_read"],
        lessons=["Read the whole file first"],
    )
    rec = await memory.record_episode(episode)
    assert Episode.from_record(rec).goal == "Summarize the README"
    assert rec.importance == pytest.approx(0.6)

    proc = Procedure(
        name="Summarize a file",
        when_to_use="asked to summarize a document",
        steps=["fs_read the file", "summarize key points"],
        source_tasks=["t1"],
    )
    p1 = await memory.learn_procedure(proc)
    p2 = await memory.learn_procedure(proc.model_copy(update={"source_tasks": ["t2"]}))
    assert p1.id == p2.id
    assert Procedure.from_record(p2).successes == 2
    await memory.record_procedure_outcome(p2.id, success=False)
    final = Procedure.from_record(await memory.get(p2.id))  # type: ignore[arg-type]
    assert (final.successes, final.failures) == (2, 1)
    found = await memory.find_procedures("summarize this document for me")
    assert found and found[0].record.id == p1.id
    assert [r.id for r in await memory.timeline()] == [rec.id]


async def test_reindex_rebuilds_vectors(memory: MemorySystem) -> None:
    await memory.remember("alpha fact about reindexing")
    await memory.remember("beta fact about rebuilding")
    await memory.vectors.delete([r.id for r in await memory.db.memories.list()])
    assert await memory.vectors.count() == 0
    assert await memory.reindex() == 2
    assert await memory.vectors.count() == 2
    stats = await memory.stats()
    assert stats["active_by_kind"] == {"fact": 2}


# ------------------------------------------------------------- consolidation


def test_valid_fact_filters_junk() -> None:
    assert valid_fact("Operator prefers dark mode", "operator.ui.theme", 0.9, 0.6)
    assert not valid_fact("The user asked a question", "x.y", 0.9, 0.6)
    assert not valid_fact("short", "x", 0.9, 0.6)
    assert not valid_fact("Operator prefers dark mode", "Bad Key!", 0.9, 0.6)
    assert not valid_fact("Operator prefers dark mode", "a.b", 0.3, 0.6)


async def test_consolidation_compresses_extracts_and_decays(memory: MemorySystem) -> None:
    memory.config.summarize_after_turns = 4
    for i in range(6):
        await memory.add_turn("s1", "user" if i % 2 == 0 else "assistant", f"turn {i}")
    await memory.db.conversations.insert(
        ConversationMessage(
            session_id="s1", role="user", content="My name is Dana and I use Arch Linux"
        )
    )
    facts = {
        "facts": [
            {
                "statement": "Operator's name is Dana",
                "subject_key": "operator.name",
                "importance": 0.9,
                "confidence": 0.95,
            },
            {
                "statement": "The user asked a question",
                "subject_key": "junk",
                "importance": 0.1,
                "confidence": 0.9,
            },
            {
                "statement": "Operator uses Arch Linux",
                "subject_key": "operator.os",
                "importance": 0.6,
                "confidence": 0.4,
            },
        ]
    }
    llm = ScriptedLLM(["Dana discussed turns 0-3.", json.dumps(facts)])
    router = ModelRouter({ModelTier.FAST: llm}, RoutingConfig())
    report = await MemoryConsolidator(memory, router, memory.config).run()

    assert report.sessions_compressed == 1 and report.errors == []
    turns = await memory.conversation("s1")
    assert turns[0].metadata.get("summary") and turns[0].content == "Dana discussed turns 0-3."
    assert len(report.facts_stored) == 1 and report.facts_rejected == 2
    [name] = await memory.db.memories.find_by_subject("operator.name")
    assert name.content == "Operator's name is Dana"
    assert await memory.db.state.get("consolidation.last_run") is not None

    # A second run sees no new conversation and makes no extraction call.
    report2 = await MemoryConsolidator(memory, router, memory.config).run()
    assert report2.facts_stored == [] and llm.remaining == 0


@pytest.mark.integration
async def test_memory_system_on_postgres_pgvector(tmp_path: Path) -> None:
    if not postgres_available():
        pytest.skip("PostgreSQL not reachable")
    db = Database(POSTGRES_URL)
    async with db.engine.begin() as conn:
        from sqlalchemy import text

        await conn.run_sync(metadata.drop_all)
        await conn.execute(text("DROP TABLE IF EXISTS memory_vectors"))
    await db.create_all()
    system = MemorySystem(
        db,
        PgVectorStore(db.engine, DIM),
        HashingEmbedder(DIM),
        InMemoryWorkingMemory(),
        MemoryConfig(),
    )
    await system.open()
    try:
        rec = await system.remember("pgvector backs semantic recall in production", importance=0.8)
        hits = await system.recall("what backs semantic recall?")
        assert hits[0].record.id == rec.id
        report = await system.decay()
        assert report.scanned == 1
    finally:
        await system.aclose()
        await db.close()


async def test_embedding_task_prefixes(sqlite_db: Database) -> None:
    seen: list[str] = []

    class Recording(HashingEmbedder):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            seen.extend(texts)
            return await super().embed(texts)

    system = MemorySystem(
        sqlite_db,
        InMemoryVectorStore(DIM),
        Recording(DIM),
        InMemoryWorkingMemory(),
        MemoryConfig(),
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    await system.remember("The GPU cluster has four Vega devices")
    await system.recall("how many GPUs?")
    assert seen == [
        "search_document: The GPU cluster has four Vega devices",
        "search_query: how many GPUs?",
    ]

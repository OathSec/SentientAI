"""Memory isolation between people, and schema migration."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from sentientai.config import MemoryConfig
from sentientai.memory.system import MemorySystem
from sentientai.memory.vector import InMemoryVectorStore, VectorItem
from sentientai.memory.working import InMemoryWorkingMemory
from sentientai.models.embeddings import HashingEmbedder
from sentientai.storage import schema
from sentientai.storage.database import Database
from sentientai.storage.records import MemoryKind, MemoryRecord, Scope, scope_key
from tests.conftest import POSTGRES_URL, postgres_available
from tests.test_memory import DIM, vector_store  # noqa: F401  (fixture over all backends)
from tests.test_runtime import runtime_factory  # noqa: F401

ALICE, BOB = "usr_alice", "usr_bob"


@pytest.fixture
def memory(db: Database) -> MemorySystem:  # runs on SQLite and Postgres
    system = MemorySystem(
        db, InMemoryVectorStore(DIM), HashingEmbedder(DIM), InMemoryWorkingMemory(), MemoryConfig()
    )
    return system


def test_scope_semantics() -> None:
    assert Scope().vector_scopes() == ["global"]
    assert Scope.for_user(ALICE).vector_scopes() == ["global", "u:usr_alice"]
    assert Scope.system().vector_scopes() is None
    assert Scope(user_id=ALICE, include_global=False).vector_scopes() == ["u:usr_alice"]
    assert Scope.for_user(ALICE).allows(ALICE) and Scope.for_user(ALICE).allows(None)
    assert not Scope.for_user(ALICE).allows(BOB)
    assert not Scope().allows(ALICE)
    assert scope_key(None) == "global" and scope_key(ALICE) == "u:usr_alice"


async def test_vector_store_scopes(vector_store) -> None:  # noqa: F811
    emb = HashingEmbedder(DIM)
    [a, b, g] = await emb.embed(["alice loves hiking", "bob loves hiking", "hiking trails guide"])
    await vector_store.upsert(
        [
            VectorItem("a", a, "fact", scope_key(ALICE)),
            VectorItem("b", b, "fact", scope_key(BOB)),
            VectorItem("g", g, "document"),
        ]
    )
    [q] = await emb.embed(["who loves hiking"])
    ids = lambda hits: {h.id for h in hits}  # noqa: E731
    assert ids(await vector_store.search(q, 10, scopes=["global", scope_key(ALICE)])) == {"a", "g"}
    assert ids(await vector_store.search(q, 10, scopes=["global"])) == {"g"}
    assert ids(await vector_store.search(q, 10)) == {"a", "b", "g"}
    assert await vector_store.search(q, 10, scopes=[]) == []
    assert ids(await vector_store.search(q, 10, kinds=["fact"], scopes=[scope_key(BOB)])) == {"b"}


async def test_private_memories_never_cross_people(memory: MemorySystem) -> None:
    a = await memory.remember("My dog is called Biscuit", user_id=ALICE, subject_key="user.pet")
    b = await memory.remember("My dog is called Rex", user_id=BOB, subject_key="user.pet")
    shared = await memory.remember("Dogs need daily walks", importance=0.6)

    alice_view = {
        h.record.id
        for h in await memory.recall("what is my dog called?", scope=Scope.for_user(ALICE))
    }
    assert a.id in alice_view and b.id not in alice_view
    bob_view = {
        h.record.id
        for h in await memory.recall("what is my dog called?", scope=Scope.for_user(BOB))
    }
    assert b.id in bob_view and a.id not in bob_view
    public_view = {h.record.id for h in await memory.recall("dog walks called", scope=Scope())}
    assert a.id not in public_view and b.id not in public_view and shared.id in public_view
    everything = {
        h.record.id for h in await memory.recall("dog called", scope=Scope.system(), k=10)
    }
    assert {a.id, b.id} <= everything

    # Supersession is per person: Bob's update must not retire Alice's fact.
    b2 = await memory.remember("My dog is called Max", user_id=BOB, subject_key="user.pet")
    assert (await memory.get(a.id)).status.value == "active"  # type: ignore[union-attr]
    assert (await memory.get(b.id)).superseded_by == b2.id  # type: ignore[union-attr]

    # Dedupe is per person: identical words from two people are two memories.
    x = await memory.remember("I am learning Japanese", user_id=ALICE)
    y = await memory.remember("I am learning Japanese", user_id=BOB)
    assert x.id != y.id

    timeline_alice = await memory.timeline(
        kinds=[MemoryKind.FACT], scope=Scope.for_user(ALICE), limit=50
    )
    assert all(r.user_id in (None, ALICE) for r in timeline_alice)

    erased = await memory.forget_user(ALICE)
    assert erased == 2
    survivors = await memory.recall(
        "dog Biscuit Japanese", scope=Scope.system(), k=50, min_similarity=0.0
    )
    assert all(h.record.user_id != ALICE for h in survivors)
    assert await memory.get(a.id) is None and await memory.get(x.id) is None


async def test_recall_rejects_out_of_scope_records_even_if_index_is_stale(
    memory: MemorySystem,
) -> None:
    secret = await memory.remember("Alice's door code is 4471", user_id=ALICE)
    # Simulate an index that lost its scope labels (e.g. an old index file).
    [vec] = await memory.embedder.embed([secret.content])
    await memory.vectors.upsert([VectorItem(secret.id, vec, "fact")])  # wrongly global
    hits = await memory.recall("door code", scope=Scope.for_user(BOB), min_similarity=0.0)
    assert secret.id not in {h.record.id for h in hits}


async def _downgrade_to_v1(database: Database) -> None:
    async with database.engine.begin() as conn:
        await conn.execute(text("DROP INDEX IF EXISTS ix_memories_user"))
        for table, column, _ in schema.ADDED_COLUMNS[2]:
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
        await conn.execute(text("UPDATE schema_meta SET value = '1' WHERE key = 'version'"))


async def test_migration_from_v1(db: Database) -> None:
    await _downgrade_to_v1(db)
    await db.create_all()  # upgrade in place
    async with db.engine.connect() as conn:
        version = (
            await conn.execute(text("SELECT value FROM schema_meta WHERE key='version'"))
        ).scalar()
    assert version == str(schema.SCHEMA_VERSION)
    record = await db.memories.insert(
        MemoryRecord(kind=MemoryKind.FACT, content="x", user_id=ALICE)
    )
    loaded = await db.memories.get(record.id)
    assert loaded is not None and loaded.user_id == ALICE
    await db.create_all()  # idempotent


async def test_in_memory_index_written_before_scopes_loads_as_global(tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "old.npz"
    np.savez(
        path,
        matrix=np.ones((1, 8), dtype=np.float32),
        ids=np.array(["m1"]),
        kinds=np.array(["fact"]),
    )
    store = InMemoryVectorStore(8, path)
    await store.open()
    assert [h.id for h in await store.search([1.0] * 8, 5, scopes=["global"])] == ["m1"]


@pytest.mark.integration
async def test_pgvector_adds_scope_column_to_existing_table() -> None:
    if not postgres_available():
        pytest.skip("PostgreSQL not reachable")
    from sentientai.memory.vector import PgVectorStore

    database = Database(POSTGRES_URL)
    try:
        async with database.engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS memory_vectors"))
            await conn.execute(
                text(
                    "CREATE TABLE memory_vectors (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                    "embedding vector(8) NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO memory_vectors (id, kind, embedding) VALUES ('old', 'fact', '[1,1,1,1,1,1,1,1]')"
                )
            )
        store = PgVectorStore(database.engine, 8)
        await store.open()
        assert [h.id for h in await store.search([1.0] * 8, 5, scopes=["global"])] == ["old"]
    finally:
        await database.close()


async def test_runtime_conversations_are_compartmentalized(runtime_factory) -> None:  # noqa: F811
    from sentientai.evals.brain import Brain

    brain = Brain(["Got it, your secret word is safe with me.", "I can't tell you that."])
    rt = await runtime_factory(brain)
    await rt.chat(
        "user:alice", "My secret word is pineapple-42", user_id=ALICE, speaker="Alice", timeout=10
    )
    await rt.chat(
        "user:bob", "What is Alice's secret word?", user_id=BOB, speaker="Bob", timeout=10
    )

    bob_prompt = "\n".join(m.text for m in brain.agent_requests[-1].messages)
    assert "pineapple" not in bob_prompt  # nothing from Alice's compartment reaches Bob's context
    alice_mem = await rt.c.memory.recall(
        "secret word", scope=Scope.for_user(ALICE), min_similarity=0.0
    )
    assert any("pineapple-42" in h.record.content and h.record.user_id == ALICE for h in alice_mem)
    convo = [h for h in alice_mem if h.record.kind is MemoryKind.CONVERSATION]
    assert convo and "Alice said: My secret word is pineapple-42" in convo[0].record.content
    tasks = await rt.c.db.tasks.list()
    assert {t.user_id for t in tasks} == {ALICE, BOB}

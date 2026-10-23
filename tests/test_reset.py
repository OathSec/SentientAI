"""`sentientai reset-mind`: a full fresh start for Fuzzy that keeps the people and the record."""

from __future__ import annotations

from typing import Any

from sentientai.evals.brain import Brain
from sentientai.runtime.reset import reset_mind
from sentientai.storage.records import MediaRecord, MemoryKind, Scope, UserRole
from tests.test_runtime import runtime_factory  # noqa: F401


async def live_a_little(rt: Any) -> Any:
    c = rt.c
    user = await c.accounts.create_user("dana", "correct horse battery staple", role=UserRole.USER)
    await rt.chat("user:dana", "hello there", user=user, timeout=10)
    await c.selfhood.define("favourite colour", "green", "it's calm", source="self")
    await c.memory.remember("I like building things.", MemoryKind.REFLECTION, importance=0.6)
    await c.pipeline.ingest_text("Ferns reproduce with spores.", title="Ferns")
    await c.world.build("hut", [{"shape": "box"}], x=3, z=3)
    await c.world.design_avatar("a small round creature", [{"shape": "sphere"}])
    await c.world.expand(80)
    await c.agency.projects.start("build a lighthouse")
    media_dir = c.images.media_dir
    media_dir.mkdir(parents=True, exist_ok=True)
    (media_dir / "img_old.png").write_bytes(b"x")
    await c.db.media.insert(
        MediaRecord(prompt="a lighthouse", path=str(media_dir / "img_old.png"), mime="image/png")
    )
    await c.mind.save(c.db.state)
    return user


async def test_a_full_reset_clears_its_life_and_world_but_keeps_people(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(Brain(["Hi Dana!"]))
    user = await live_a_little(rt)
    settings, audit = rt.settings, rt.c.audit
    await rt.stop("test: reset")

    cleared = await reset_mind(settings, audit=audit)
    assert cleared["memories"] > 0 and cleared["world_objects"] == 1 and cleared["state"] > 0

    fresh = await runtime_factory(Brain(), start=False)
    c = fresh.c
    assert await c.db.memories.list(scope=Scope.system()) == []
    assert await c.db.conversations.recent("user:dana") == []
    assert await c.selfhood.beliefs() == []
    assert await c.world.objects() == []
    assert not c.world.avatar.parts and c.world.size == settings.world.size  # back to 40 m
    assert await c.agency.projects.all() == []
    assert await c.db.documents.list() == []
    assert not any(c.images.media_dir.iterdir())  # the gallery files are gone too
    assert (await c.db.users.get(user.id)) is not None  # the people are kept
    assert any(e.type == "mind.reset" for e in c.audit.iter_entries())  # and the record
    await fresh.start()
    assert c.mind.self_model.boot_count == 1  # it wakes up for the first time


async def test_teachings_can_be_kept_across_a_reset(runtime_factory: Any) -> None:  # noqa: F811
    rt = await runtime_factory(Brain(["Hi Dana!"]))
    await live_a_little(rt)
    settings = rt.settings
    await rt.stop("test: reset")
    await reset_mind(settings, keep_teachings=True)
    fresh = await runtime_factory(Brain(), start=False)
    kept = await fresh.c.db.memories.list(scope=Scope.system())
    assert kept and all(m.kind is MemoryKind.DOCUMENT for m in kept)
    assert [d.title for d in await fresh.c.db.documents.list()] == ["Ferns"]

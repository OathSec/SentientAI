"""The world, images, quotas and name privacy."""

from __future__ import annotations

import asyncio
import base64
import math
import struct
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from sentientai.accounts.quotas import QuotaExceeded, Quotas
from sentientai.cognition.privacy import Redactor
from sentientai.config import AccountsConfig, ImagesConfig, WorldConfig
from sentientai.core.clock import FakeClock
from sentientai.core.events import EventBus
from sentientai.evals.brain import Brain, call
from sentientai.media.images import (
    Automatic1111Backend,
    ImageError,
    ImageService,
    OpenAIImagesBackend,
    screen_prompt,
    sniff_image,
)
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import MediaRecord, RelationshipRecord
from sentientai.world.model import clean_parts
from sentientai.world.service import World, WorldError
from tests.test_accounts import GOOD, service
from tests.test_runtime import runtime_factory  # noqa: F401

PNG = (
    b"\x89PNG\r\n\x1a\n"
    + struct.pack(">I", 13)
    + b"IHDR"
    + struct.pack(">II", 64, 32)
    + b"\x08\x02\x00\x00\x00"
    + b"\x00" * 16
)
JPEG = (
    b"\xff\xd8\xff\xc0"
    + struct.pack(">H", 17)
    + b"\x08"
    + struct.pack(">HH", 48, 96)
    + b"\x03"
    + b"\x00" * 9
)
WEBP = b"RIFF" + struct.pack("<I", 20) + b"WEBP" + b"VP8 " + b"\x00" * 16


# -------------------------------------------------------------------- parts


def test_parts_are_validated_and_clamped() -> None:
    parts = clean_parts(
        [
            {
                "shape": "box",
                "size": [1e9, -5, "x"],
                "position": [999, float("nan"), 1],
                "color": "red",
            },
            {"shape": "teapot"},  # not a shape
            "not a dict",
            {"shape": "sphere", "radius": 3, "color": "javascript:alert(1)"},
            {"shape": "cone", "color": "#abc", "rotation": [720, 0, 0]},
            {"shape": "torus", "tube": 1e6},
        ],
        max_parts=10,
    )
    assert [p["shape"] for p in parts] == ["box", "sphere", "cone", "torus"]
    box = parts[0]
    assert box["size"] == [20.0, 0.02, 1.0] and box["position"] == [20.0, 0.0, 1.0]
    assert box["color"] == "#c0392b"
    assert parts[1]["color"] == "#8a8a8a"  # never passed through to the renderer
    assert parts[2]["color"] == "#aabbcc" and parts[2]["rotation"][0] == 360.0
    assert parts[3]["tube"] == 5.0
    assert len(clean_parts([{"shape": "box"}] * 50, max_parts=7)) == 7
    assert clean_parts("nope", max_parts=5) == [] and clean_parts(None, max_parts=5) == []


# -------------------------------------------------------------------- world


def make_world(db: Database, tmp_path: Path, **config: Any) -> tuple[World, EventBus]:
    bus = EventBus()
    world = World(
        db,
        WorldConfig(**config),
        audit=AuditLog(tmp_path / "audit", fsync=False),
        bus=bus,
        clock=FakeClock(),
    )
    return world, bus


async def test_building_changing_and_removing(sqlite_db: Database, tmp_path: Path) -> None:
    world, bus = make_world(sqlite_db, tmp_path, size=20, max_objects=2)
    box = [{"shape": "box", "size": [1, 1, 1]}]
    obj = await world.build("  A <b>bench</b>  ", box, x=999, z=-999, rotation=1000)
    assert (obj.x, obj.z) == (9.5, -9.5)  # kept inside the world
    assert obj.rotation == 360.0 and obj.name == "A <b>bench</b>"
    assert [e.payload["op"] for e in bus.recent("world.object")] == ["built"]

    changed = await world.modify(obj.id, name="Better bench", x=2, parts=[{"shape": "cone"}])
    assert changed.name == "Better bench" and changed.x == 2 and changed.parts[0]["shape"] == "cone"
    with pytest.raises(WorldError, match="no valid parts"):
        await world.modify(obj.id, parts=[{"shape": "teapot"}])
    with pytest.raises(WorldError, match="no object"):
        await world.modify("obj_missing", name="x")

    await world.build("second", box)
    with pytest.raises(WorldError, match="already holds 2"):
        await world.build("third", box)
    with pytest.raises(WorldError, match="no valid parts"):
        await world.build("empty", [{"shape": "teapot"}])
    await world.remove(obj.id)
    assert [o.name for o in await world.objects()] == ["second"]
    await world.build("third", box)  # room again
    assert "second" in await world.describe()
    with pytest.raises(WorldError):
        await world.apply({"action": "explode"})


async def test_avatar_is_designed_and_survives_restart(sqlite_db: Database, tmp_path: Path) -> None:
    world, _ = make_world(sqlite_db, tmp_path)
    assert "plain marker" in world.describe_avatar()
    with pytest.raises(WorldError):
        await world.design_avatar("nothing", [])
    await world.design_avatar(
        "a tall lantern", [{"shape": "cylinder", "radius": 0.3, "height": 1.8}]
    )
    await world.design_avatar(
        "a short lantern", [{"shape": "cylinder", "radius": 0.3, "height": 1.0}]
    )
    assert world.avatar.versions == 2
    await world.walk_to(3, 4)
    await world.set_action("thinking", "x" * 500)
    assert len(world.body.thought) == 240

    again, _ = make_world(sqlite_db, tmp_path)
    await again.load()
    assert again.avatar.description == "a short lantern" and again.avatar.versions == 2
    assert again.body.target is None and again.body.action == "standing"  # no stale walk


async def test_it_walks_where_it_decided_to_go(sqlite_db: Database, tmp_path: Path) -> None:
    world, bus = make_world(sqlite_db, tmp_path, walk_speed=50)
    runner = asyncio.create_task(world.run())
    try:
        assert await world.arrived(0.1)  # nowhere to go
        await world.walk_to(3, 4)
        assert world.body.action == "walking"
        assert await world.arrived(5)
        assert (world.body.x, world.body.z) == (3.0, 4.0) and world.body.action == "standing"
        assert world.body.heading == pytest.approx(math.degrees(math.atan2(3, 4)))
        await world.walk_to(10_000, 10_000)  # clamped into the world
        assert world.body.target == [world.half - 0.5, world.half - 0.5]
        await world.set_action("thinking")  # changing its mind about walking stops the walk
        assert world.body.target is None
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
    assert any(e.payload["body"]["action"] == "standing" for e in bus.recent("world.body"))
    reloaded, _ = make_world(sqlite_db, tmp_path)
    await reloaded.load()
    assert (reloaded.body.x, reloaded.body.z) == (3.0, 4.0)


# ------------------------------------------------------------------- images


def test_images_are_checked_by_content() -> None:
    assert sniff_image(PNG) == ("image/png", "png", 64, 32)
    assert sniff_image(JPEG) == ("image/jpeg", "jpg", 96, 48)
    assert sniff_image(WEBP)[:2] == ("image/webp", "webp")
    for bogus in (b"<html><script>alert(1)</script>", b"<svg onload=alert(1)>", b"GIF89a....", b""):
        with pytest.raises(ImageError, match="isn't an image"):
            sniff_image(bogus)


def test_prompt_screening() -> None:
    assert screen_prompt("a green hill at dawn") is None
    assert screen_prompt("two kids flying a kite") is None
    assert (
        screen_prompt("a nude figure")
        and screen_prompt("NSFW photo")
        and screen_prompt("erotic art")
    )
    assert screen_prompt("a child in a bath")  # conservative around minors


@respx.mock
async def test_automatic1111_backend() -> None:
    config = ImagesConfig(backend="automatic1111", base_url="http://sd.test", width=256, height=128)
    route = respx.post("http://sd.test/sdapi/v1/txt2img").respond(
        200, json={"images": [base64.b64encode(PNG).decode()]}
    )
    backend = Automatic1111Backend(config)
    assert await backend.generate("a hill", config=config) == PNG
    body = httpx.Request("POST", "http://x", content=route.calls[0].request.content).read()
    assert b'"width":256' in body.replace(b" ", b"") and b'"prompt":"a hill"' in body.replace(
        b'": "', b'":"'
    )
    respx.post("http://sd.test/sdapi/v1/txt2img").respond(500, text="CUDA out of memory")
    with pytest.raises(ImageError, match="500"):
        await backend.generate("a hill", config=config)
    respx.post("http://sd.test/sdapi/v1/txt2img").respond(200, json={"images": []})
    with pytest.raises(ImageError, match="no image"):
        await backend.generate("a hill", config=config)
    respx.post("http://sd.test/sdapi/v1/txt2img").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ImageError, match="unreachable"):
        await backend.generate("a hill", config=config)
    await backend.aclose()


@respx.mock
async def test_openai_compatible_backend() -> None:
    config = ImagesConfig(
        backend="openai",
        base_url="http://img.test",
        api_key="sk-test",
        model="sdxl",  # type: ignore[arg-type]
    )
    route = respx.post("http://img.test/v1/images/generations").respond(
        200, json={"data": [{"b64_json": base64.b64encode(JPEG).decode()}]}
    )
    backend = OpenAIImagesBackend(config)
    assert await backend.generate("a hill", config=config) == JPEG
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer sk-test"
    assert b'"model":"sdxl"' in request.content.replace(b" ", b"")
    respx.post("http://img.test/v1/images/generations").respond(200, json={"data": [{}]})
    with pytest.raises(ImageError, match="no image"):
        await backend.generate("a hill", config=config)
    await backend.aclose()


class FakeBackend:
    name = "fake"

    def __init__(self, data: bytes = PNG) -> None:
        self.data = data
        self.prompts: list[str] = []
        self.active = 0
        self.peak = 0

    async def generate(self, prompt: str, *, config: ImagesConfig) -> bytes:
        self.prompts.append(prompt)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return self.data

    async def aclose(self) -> None:
        return None


def make_images(db: Database, tmp_path: Path, fake: Any = None, **cfg: Any) -> ImageService:
    return ImageService(
        ImagesConfig(**cfg),
        db,
        tmp_path / "media",
        audit=AuditLog(tmp_path / "audit", fsync=False),
        bus=EventBus(),
        clock=FakeClock(),
        backend=fake,
    )


async def test_image_service_stores_and_records(sqlite_db: Database, tmp_path: Path) -> None:
    backend = FakeBackend()
    images = make_images(sqlite_db, tmp_path, backend, backend="openai")
    record = await images.create(
        "  a green hill   at dawn ", user_id="usr_a", public=False, caption="calm"
    )
    assert record.prompt == "a green hill at dawn" and (record.width, record.height) == (64, 32)
    path = images.path_for(record)
    assert path.read_bytes() == PNG and path.suffix == ".png" and path.parent == tmp_path / "media"
    stored = await sqlite_db.media.get(record.id)
    assert stored is not None and stored.user_id == "usr_a" and stored.public is False

    # The GPU is shared: simultaneous requests are made one at a time.
    await asyncio.gather(
        *(images.create(f"picture {n}", user_id=None, public=True) for n in range(3))
    )
    assert backend.peak == 1

    escaped = MediaRecord(prompt="x", path="/etc/passwd")
    with pytest.raises(ImageError, match="outside the media directory"):
        images.path_for(escaped)


async def test_image_service_refuses_and_reports(sqlite_db: Database, tmp_path: Path) -> None:
    backend = FakeBackend()
    images = make_images(sqlite_db, tmp_path, backend, backend="openai")
    with pytest.raises(ImageError, match="aren't something I'll make"):
        await images.create("a nude figure", user_id="u", public=False)
    with pytest.raises(ImageError, match="describe the image"):
        await images.create("a", user_id="u", public=False)
    assert backend.prompts == []  # refused before reaching the model
    assert any(e.type == "media.refused" for e in images.audit.iter_entries())

    backend.data = b"<svg onload=alert(1)>"
    with pytest.raises(ImageError, match="isn't an image"):
        await images.create("a hill", user_id="u", public=False)
    backend.data = PNG + b"\x00" * 100
    small = make_images(sqlite_db, tmp_path, backend, backend="openai", max_bytes=50)
    with pytest.raises(ImageError, match="too large"):
        await small.create("a hill", user_id="u", public=False)
    assert not list((tmp_path / "media").glob("*")) if (tmp_path / "media").exists() else True

    nothing = make_images(sqlite_db, tmp_path)
    assert not nothing.available
    with pytest.raises(ImageError, match="isn't set up"):
        await nothing.create("a hill", user_id=None, public=True)


# ------------------------------------------------------------------- quotas


async def test_daily_quotas(sqlite_db: Database) -> None:
    clock = FakeClock()
    quotas = Quotas(sqlite_db.usage, AccountsConfig(daily_images=2, daily_messages=3), clock=clock)
    await quotas.take("u1", images=1)
    await quotas.take("u1", images=1)
    with pytest.raises(QuotaExceeded, match="image limit"):
        await quotas.take("u1", images=1)
    assert (await quotas.usage("u1")).images == 2  # the refused one wasn't counted
    await quotas.take("u2", images=1)  # per person
    await quotas.take("u1", images=5, privileged=True)  # admins aren't limited
    await quotas.take(None, images=99)  # the operator isn't a person
    await quotas.take("u1", images=-1, privileged=True)  # refunds are possible
    for _ in range(3):
        await quotas.take("u3", messages=1)
    with pytest.raises(QuotaExceeded, match="message limit"):
        await quotas.take("u3", messages=1)
    clock.advance(86400)
    await quotas.take("u3", messages=1)  # a new day
    unlimited = Quotas(sqlite_db.usage, AccountsConfig(daily_images=0), clock=clock)
    for _ in range(20):
        await unlimited.take("u4", images=1)


async def test_images_in_conversation_are_private_and_limited(
    runtime_factory: Any,  # noqa: F811
    tmp_path: Path,
) -> None:
    img = call("image_generate", prompt="a green hill at dawn", caption="for you")
    brain = Brain([img, img, "Here's your picture!", "Here again."])
    rt = await runtime_factory(brain, accounts={"daily_images": 1})
    rt.c.images.backend = FakeBackend()
    dana = await rt.c.accounts.create_user("dana", GOOD)
    reply = await rt.chat("user:dana", "Draw me a hill", user=dana, timeout=10)
    assert reply.answer == "Here's your picture!" and len(reply.media) == 1
    [record] = await rt.c.db.media.for_user(dana.id)
    assert reply.media == [record.id] and record.public is False and record.caption == "for you"
    second_result = next(m.text for m in brain.agent_requests[1].messages if m.role.value == "tool")
    assert "Made the image" in second_result
    third_result = [m.text for m in brain.agent_requests[2].messages if m.role.value == "tool"]
    assert "image limit" in third_result[-1]
    assert len(await rt.c.db.media.for_user(dana.id)) == 1
    history = await rt.c.db.conversations.recent("user:dana")
    assert history[-1].metadata["media"] == [record.id]
    # Others don't see it in their gallery; the owner and admins do.
    assert record.id not in [m.id for m in await rt.c.db.media.visible_to("usr_other")]
    assert record.id in [m.id for m in await rt.c.db.media.visible_to(dana.id)]
    assert record.id in [m.id for m in await rt.c.db.media.visible_to(None, admin=True)]


# ----------------------------------------------------------------- privacy


async def test_other_peoples_names_are_removed(sqlite_db: Database, tmp_path: Path) -> None:
    accounts = service(sqlite_db, tmp_path)
    alice = await accounts.create_user("alice", GOOD, display_name="Alice Smith")
    bob = await accounts.create_user("bobby_t", GOOD, display_name="Bob")
    await sqlite_db.relationships.upsert(RelationshipRecord(user_id=bob.id, nickname="Bobster"))
    redactor = Redactor(sqlite_db, ttl_s=0)
    text = "Bob told Alice Smith that Bobster likes @BOBBY_T's jokes; bobsled and bob stay."
    for_alice = await redactor.for_viewer(alice.id)
    assert for_alice(text) == (
        "someone told Alice Smith that someone likes someone's jokes; bobsled and bob stay."
    )
    for_bob = await redactor.for_viewer(bob.id)
    assert for_bob("Bob and Alice Smith") == "Bob and someone"
    assert (await redactor.for_viewer(alice.id, privileged=True))(text) == text
    assert (await redactor.shared())("Alice Smith met Bob") == "someone met someone"
    assert for_alice("") == ""

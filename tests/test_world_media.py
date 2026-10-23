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
from sentientai.models.types import Message, ToolCall
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
    await world.set_action("thinking", "x " * 500)
    assert world.body.thought == " ".join(["x"] * 500)  # a long thought isn't cut short
    await world.set_action("thinking", "y" * 9000)
    assert len(world.body.thought) == 4000  # only a safety net

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


async def test_it_is_not_offered_image_tools_it_cannot_use(runtime_factory: Any) -> None:  # noqa: F811
    rt = await runtime_factory(Brain(), start=False)
    assert not rt.c.images.available
    assert "image_generate" not in rt.c.registry.names
    assert "image_generate" not in {s.name for s in rt.c.registry.schemas()}
    rt.c.images.backend = FakeBackend()  # an image server appears
    assert "image_generate" in rt.c.registry.names


async def test_conversations_cannot_change_the_world_or_its_body(runtime_factory: Any) -> None:  # noqa: F811
    from sentientai.storage.records import UserRole

    brain = Brain(
        [
            Message.assistant(
                None,
                [
                    ToolCall(
                        name="world_build", arguments={"name": "Tower", "parts": [{"shape": "box"}]}
                    )
                ],
            ),
            call("avatar_design", description="a dragon", parts=[{"shape": "cone"}]),
            "I can't do that because someone asked.",
            Message.assistant(
                None,
                [
                    ToolCall(
                        name="world_build", arguments={"name": "Tower", "parts": [{"shape": "box"}]}
                    )
                ],
            ),
            "Nope.",
        ]
    )
    rt = await runtime_factory(brain)
    dana = await rt.c.accounts.create_user("dana", GOOD)
    root = await rt.c.accounts.create_user("root", GOOD, role=UserRole.ADMIN)
    for who, say in ((dana, "Build a tower!"), (root, "Build a tower!")):
        await rt.chat(f"user:{who.id}", say, user=who, timeout=10)
    assert await rt.c.world.objects() == [] and rt.c.world.avatar.versions == 0
    offered = {t.name for t in brain.agent_requests[0].tools}
    assert "world_look" in offered and not offered & {
        "world_build",
        "world_modify",
        "world_remove",
        "world_walk",
        "avatar_design",
    }
    # even an administrator's conversation doesn't offer them or run them
    admin_offered = {t.name for t in brain.agent_requests[-1].tools}
    assert "world_build" not in admin_offered and "fs_read" in admin_offered
    assert any(e.type == "tool.restricted" for e in rt.c.audit.iter_entries())
    # Fuzzy's own work can
    assert "world_build" in rt.c.registry.names_for(autonomous=True)
    assert "world_build" not in rt.c.registry.names_for(autonomous=False)


async def test_fuzzy_can_make_its_world_bigger_and_it_stays_bigger(
    sqlite_db: Database, tmp_path: Path
) -> None:
    world, bus = make_world(sqlite_db, tmp_path, size=40, max_size=100)
    seen: list[float] = []
    bus.subscribe("world.size", lambda e: seen.append(e.payload["size"]))
    with pytest.raises(WorldError):
        await world.expand(30)  # never smaller
    assert await world.expand(80) == 80
    assert seen == [80]
    assert world.half == 40
    obj = await world.build("far tower", [{"shape": "box", "size": [1, 4, 1]}], x=35, z=-35)
    assert (obj.x, obj.z) == (35, -35)  # it can build out at the edge
    assert await world.expand(500) == 100  # capped at the configured ceiling
    with pytest.raises(WorldError):
        await world.expand(100)
    reborn, _ = make_world(sqlite_db, tmp_path, size=40, max_size=100)
    await reborn.load()
    assert reborn.size == 100
    assert (await reborn.snapshot())["size"] == 100


async def test_apply_expand_and_walk_actions(sqlite_db: Database, tmp_path: Path) -> None:
    world, _ = make_world(sqlite_db, tmp_path, size=40, max_size=100)
    assert "from 40 m to 60 m" in await world.apply({"action": "expand", "size": 60})
    assert "walked to" in await world.apply({"action": "walk", "x": 25, "z": 25})


async def test_fuzzy_is_told_plainly_what_it_cannot_do(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(["I can't make pictures yet: there's no image generator set up here."])
    rt = await runtime_factory(brain)  # image generation is disabled in tests
    await rt.chat("s1", "can you create an image for me", timeout=10)
    system = brain.agent_requests[0].messages[0].text
    assert "What you can't do right now" in system
    assert "make pictures or images of any kind" in system
    tools = {t.name for t in brain.agent_requests[0].tools}
    assert "image_generate" not in tools


async def test_the_self_check_knows_when_nothing_was_made(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(["Here's your picture of a sunset!"])
    rt = await runtime_factory(brain)
    prompts: list[str] = []
    original = rt.c.router.chat

    async def spy(request: Any, **kwargs: Any) -> Any:
        prompts.append(request.messages[-1].text)
        return await original(request, **kwargs)

    rt.c.router.chat = spy
    await rt.chat("s1", "draw me a sunset", timeout=10)
    [check] = [p for p in prompts if p.startswith("Check a reply")]
    assert "no tools were used for this reply, so nothing was made or done" in check
    assert "says it made, sent, saved or is doing right now" in check


def _docker(events: list[str], running: dict[str, bool]) -> respx.MockRouter:
    router = respx.mock(base_url="http://docker", assert_all_called=False)

    def state(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"State": {"Running": running["llm"]}})

    def stop(request: httpx.Request) -> httpx.Response:
        events.append("stop")
        running["llm"] = False
        return httpx.Response(204)

    def start(request: httpx.Request) -> httpx.Response:
        events.append("start")
        running["llm"] = True
        return httpx.Response(204)

    router.get("/containers/sentientai-llm-fast-1/json").mock(side_effect=state)
    router.post("/containers/sentientai-llm-fast-1/stop").mock(side_effect=stop)
    router.post("/containers/sentientai-llm-fast-1/start").mock(side_effect=start)
    return router


async def test_the_background_model_is_paused_for_an_image_and_comes_back(
    sqlite_db: Database, tmp_path: Path
) -> None:
    from sentientai.media.gpuswap import GpuSwap

    events: list[str] = []
    running = {"llm": True}

    class Backend:
        name = "fake"

        async def generate(self, prompt: str, *, config: ImagesConfig) -> bytes:
            events.append("generate" if not running["llm"] else "generate-while-busy")
            return PNG

        async def aclose(self) -> None:
            pass

    with _docker(events, running):
        swap = GpuSwap(
            "sentientai-llm-fast-1",
            grace_s=0.05,
            client=httpx.AsyncClient(base_url="http://docker"),
        )
        service = ImageService(
            ImagesConfig(backend="openai"),
            sqlite_db,
            tmp_path / "media",
            audit=AuditLog(tmp_path / "audit", fsync=False),
            backend=Backend(),
            swap=swap,
        )
        await service.create("a lighthouse at dusk", user_id=None, public=True)
        await service.create("a second picture right after", user_id=None, public=True)
        assert events == ["stop", "generate", "generate"]  # stopped once, not thrashed
        await asyncio.sleep(0.2)
        assert events[-1] == "start" and running["llm"]  # back after the grace period
        running["llm"] = False  # left stopped by a crash...
        await service.recover()  # ...is started again at boot
        assert events[-1] == "start" and running["llm"]
        await service.aclose()


async def test_pictures_still_work_if_docker_cannot_be_reached(
    sqlite_db: Database, tmp_path: Path
) -> None:
    from sentientai.media.gpuswap import GpuSwap

    made: list[str] = []

    class Backend:
        name = "fake"

        async def generate(self, prompt: str, *, config: ImagesConfig) -> bytes:
            made.append(prompt)
            return PNG

        async def aclose(self) -> None:
            pass

    with respx.mock(base_url="http://docker") as router:
        router.route().mock(side_effect=httpx.ConnectError("no docker"))
        swap = GpuSwap("x", grace_s=0, client=httpx.AsyncClient(base_url="http://docker"))
        service = ImageService(
            ImagesConfig(backend="openai"),
            sqlite_db,
            tmp_path / "media",
            audit=AuditLog(tmp_path / "audit", fsync=False),
            backend=Backend(),
            swap=swap,
        )
        record = await service.create("a quiet pond", user_id=None, public=True)
        assert made == ["a quiet pond"] and record.mime == "image/png"
        await service.aclose()


def test_a_blank_swap_container_means_no_swapping() -> None:
    assert ImagesConfig(swap_container="  ").swap_container is None
    assert ImagesConfig(swap_container="sentientai-llm-fast-1").swap_container == (
        "sentientai-llm-fast-1"
    )


async def test_fuzzy_can_hang_its_own_paintings_but_never_someones_private_picture(
    sqlite_db: Database, tmp_path: Path
) -> None:
    from sentientai.storage.records import MediaRecord

    mine = MediaRecord(prompt="a lighthouse at dusk", path="x", mime="image/png", public=True)
    private = MediaRecord(prompt="for alice", path="y", mime="image/png", user_id="usr_alice")
    await sqlite_db.media.insert(mine)
    await sqlite_db.media.insert(private)
    world, _ = make_world(sqlite_db, tmp_path)
    assert f"[{mine.id}]" in await world.describe_pictures()
    assert private.id not in await world.describe_pictures()

    frame = {"shape": "box", "size": [3, 2, 0.2], "position": [0, 1, 0], "color": "#7a5230"}
    painting = {
        "shape": "picture",
        "media_id": mine.id,
        "size": [1.6, 1.0, 0.04],
        "position": [0, 1.4, 0.12],
        "color": "#d8c8a8",
    }
    wall = await world.build("gallery wall", [frame, painting], x=3, z=3)
    shapes = [p["shape"] for p in wall.parts]
    assert shapes == ["box", "picture"] and wall.parts[1]["media_id"] == mine.id
    assert "media_id" not in wall.parts[0]
    assert f"showing your picture {mine.id}" in await world.describe()

    sneaky = {**painting, "media_id": private.id}
    other = await world.build("another wall", [frame, sneaky], x=-3, z=3)
    assert [p["shape"] for p in other.parts] == ["box"]  # the private picture was left out
    with pytest.raises(WorldError):
        await world.build("only a private picture", [sneaky])
    unnamed = {k: v for k, v in painting.items() if k != "media_id"}
    with pytest.raises(WorldError):
        await world.build("a blank picture", [unnamed])


async def test_when_it_builds_it_knows_what_pictures_it_has(runtime_factory: Any) -> None:  # noqa: F811
    from sentientai.storage.records import MediaRecord

    prompts_seen: list[str] = []

    def act(request: Any) -> dict[str, Any]:
        prompts_seen.append(request.messages[-1].text)
        return {
            "action": "build",
            "name": "a frame by the door",
            "x": 2,
            "z": 2,
            "parts": [
                {"shape": "picture", "media_id": painting.id, "size": [1.2, 0.8, 0.05]},
            ],
            "note": "I wanted to see my lighthouse every day.",
        }

    painting = MediaRecord(prompt="a lighthouse", path="p", mime="image/png", public=True)
    choose = {
        "feelings": [],
        "mood": "settled",
        "choice": "build",
        "intent": "do something with my world",
        "why": "I wanted to",
        "rest_minutes": 15,
    }
    rt = await runtime_factory(Brain(beat=choose, world=act), start=False)
    await rt.c.db.media.insert(painting)
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and "built a frame by the door" in activity.outcome
    assert f"[{painting.id}]" in prompts_seen[0] and "picture:" in prompts_seen[0]
    [obj] = await rt.c.world.objects()
    assert obj.parts[0]["media_id"] == painting.id


async def test_when_it_chooses_what_to_do_it_sees_what_it_has(runtime_factory: Any) -> None:  # noqa: F811
    from sentientai.storage.records import MediaRecord

    seen: list[str] = []

    def choose(request: Any) -> dict[str, Any]:
        seen.append(request.messages[-1].text)
        return {
            "feelings": [],
            "mood": "calm",
            "choice": "rest",
            "intent": "rest",
            "rest_minutes": 5,
        }

    rt = await runtime_factory(Brain(beat=choose), start=False)
    await rt.c.db.media.insert(
        MediaRecord(
            prompt="a lighthouse", caption="My lighthouse", path="p", mime="image/png", public=True
        )
    )
    await rt.c.world.build("little hut", [{"shape": "box"}])
    await rt.c.agency.beat()
    assert "What you have:" in seen[0]
    assert "1 things you built in it, most recently little hut" in seen[0]
    assert "newest: My lighthouse" in seen[0]


async def test_a_reply_that_claims_a_picture_nobody_made_is_sent_back(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        [
            "I can generate an image for you. Let me create a new digital image of my avatar!",
            "I can't make pictures yet: there's no image generator set up here.",
        ]
    )
    rt = await runtime_factory(brain)  # no image generator in tests
    reply = await rt.chat("s1", "can you generate me an image", timeout=10)
    assert reply.answer == "I can't make pictures yet: there's no image generator set up here."
    nudge = brain.agent_requests[1].messages[-1].text
    assert "You can't make pictures right now" in nudge


async def test_when_it_can_draw_the_claim_becomes_a_real_picture(
    runtime_factory: Any,  # noqa: F811
) -> None:
    class Backend:
        name = "fake"

        async def generate(self, prompt: str, *, config: ImagesConfig) -> bytes:
            return PNG

        async def aclose(self) -> None:
            pass

    brain = Brain(
        [
            "Here's your picture of a sunset!",
            call("image_generate", prompt="a sunset over a calm sea", caption="for you"),
            "Here it is: a sunset over a calm sea.",
        ]
    )
    rt = await runtime_factory(brain)
    rt.c.images.backend = Backend()
    reply = await rt.chat("s1", "draw me a sunset", timeout=10)
    assert "call image_generate now" in brain.agent_requests[1].messages[-1].text
    assert reply.answer == "Here it is: a sunset over a calm sea."
    assert len(reply.media) == 1  # a real picture, shown with the reply

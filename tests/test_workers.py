"""Computers that lend Fuzzy their power: trust rules, the HTTP path, fallbacks."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from sentientai.config import ImagesConfig, ModelEndpointConfig
from sentientai.core.clock import FakeClock
from sentientai.evals.brain import Brain
from sentientai.media.images import ImageService
from sentientai.models.types import ChatRequest, Message
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import MemoryKind, UserRole
from sentientai.workers.api import build_worker_router
from sentientai.workers.client import WorkerFirstClient
from sentientai.workers.hub import WorkerError, WorkerHub
from sentientai.workers.registry import WorkerRegistry
from tests.test_runtime import runtime_factory  # noqa: F401
from tests.test_web import web_factory  # noqa: F401


def png() -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    raw = zlib.compress(b"\x00\xff\x00\x00")
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


@pytest.fixture
async def stack(sqlite_db: Database, tmp_path: Path) -> Any:
    registry = WorkerRegistry(sqlite_db.state, clock=FakeClock())
    hub = WorkerHub()
    audit = AuditLog(tmp_path / "audit", fsync=False)
    app = FastAPI()
    app.include_router(build_worker_router(hub, registry, audit, wait_s=0.3))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fuzzy")
    yield registry, hub, http, audit, sqlite_db
    await http.aclose()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def serve_one(
    http: httpx.AsyncClient,
    token: str,
    answer: dict[str, Any],
    caps: list[str],
    info: dict[str, Any] | None = None,
) -> None:
    """A worker: ask for a job, do it, hand back the result."""

    while True:
        polled = (
            await http.post(
                "/worker/poll",
                json={"capabilities": caps, "info": info or {}},
                headers=bearer(token),
            )
        ).json()
        if polled["job"]:
            await http.post(
                "/worker/result",
                json={"job_id": polled["job"]["id"], "ok": True, "data": answer},
                headers=bearer(token),
            )
            return


async def test_tokens_are_hashed_checked_and_revocable(stack: Any) -> None:
    registry, _hub, http, *_ = stack
    record, token = await registry.create("owens-pc", "private")
    assert token.startswith("fzw_") and token not in record.model_dump_json()
    assert (await registry.authenticate(token)).id == record.id  # type: ignore[union-attr]
    assert await registry.authenticate("fzw_nope") is None
    assert (await http.post("/worker/poll", json={})).status_code == 401
    assert (await http.post("/worker/poll", json={}, headers=bearer("fzw_nope"))).status_code == 401
    await registry.revoke("owens-pc")
    assert await registry.authenticate(token) is None
    assert (await http.post("/worker/poll", json={}, headers=bearer(token))).status_code == 401


async def test_a_job_goes_to_a_free_worker_and_the_answer_comes_back(stack: Any) -> None:
    registry, hub, http, *_ = stack
    _, token = await registry.create("owens-pc", "private")
    worker = asyncio.create_task(serve_one(http, token, {"choices": [{"x": 1}]}, ["llm"]))
    await asyncio.sleep(0.05)  # it is polling now
    assert hub.available("llm")
    result = await hub.run("llm", {"messages": []}, timeout=5)
    assert result == {"choices": [{"x": 1}]}
    await worker
    [online] = hub.online()
    assert online["name"] == "owens-pc" and online["jobs_done"] == 1 and not online["busy"]


async def test_private_things_never_go_to_a_friends_computer(stack: Any) -> None:
    registry, hub, http, *_ = stack
    _, friend = await registry.create("zanders-pc", "shared")
    poller = asyncio.create_task(
        http.post("/worker/poll", json={"capabilities": ["llm", "image"]}, headers=bearer(friend))
    )
    await asyncio.sleep(0.05)
    assert not hub.available("llm")  # thinking, with private conversations in it
    assert not hub.available("image", private_only=True)  # a private picture's prompt
    assert hub.available("image", private_only=False)  # Fuzzy's own public art
    with pytest.raises(WorkerError):
        await hub.run("llm", {"messages": []}, timeout=1)
    await poller


async def test_a_busy_or_vanished_worker_is_not_waited_on(stack: Any) -> None:
    registry, hub, http, *_ = stack
    _, token = await registry.create("owens-pc", "private")
    first = asyncio.create_task(
        http.post("/worker/poll", json={"capabilities": ["llm"]}, headers=bearer(token))
    )
    await asyncio.sleep(0.05)
    job = asyncio.create_task(hub.run("llm", {"messages": []}, timeout=0.2))
    await asyncio.sleep(0.01)
    assert not hub.available("llm")  # it has been given work
    with pytest.raises(WorkerError, match="too long"):
        await job  # it never answered
    assert hub.available("llm")  # and it isn't left stuck as busy
    await first
    stale = WorkerHub(clock=lambda: 1e6)
    assert not stale.available("llm")  # nobody connected


async def test_a_bad_result_makes_the_caller_use_its_own_model(stack: Any) -> None:
    registry, hub, http, *_ = stack
    _, token = await registry.create("owens-pc", "private")
    httpx_mock = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "from keystone"}}],
                "usage": {},
            },
        )
    )
    client = WorkerFirstClient(
        "fast",
        ModelEndpointConfig(base_url="http://keystone/v1", model="m"),
        hub=hub,
        http=httpx.AsyncClient(base_url="http://keystone/v1", transport=httpx_mock),
    )
    request = ChatRequest(messages=[Message.user("hi")], background=True)
    assert (await client.chat(request)).message.text == "from keystone"  # no worker: as before

    async def bad_worker() -> None:
        await serve_one(http, token, {"nonsense": True}, ["llm"])

    task = asyncio.create_task(bad_worker())
    await asyncio.sleep(0.05)
    assert (await client.chat(request)).message.text == "from keystone"  # worker sent junk
    await task

    task = asyncio.create_task(
        serve_one(
            http,
            token,
            {
                "choices": [{"message": {"role": "assistant", "content": "from the PC"}}],
                "usage": {},
            },
            ["llm"],
        )
    )
    await asyncio.sleep(0.05)
    assert (await client.chat(request)).message.text == "from the PC"
    await task


async def test_pictures_go_to_workers_by_trust(stack: Any, tmp_path: Path) -> None:
    registry, hub, http, audit, db = stack
    _, owner = await registry.create("owens-pc", "private")
    _, friend = await registry.create("zanders-pc", "shared")
    picture = {"b64_json": base64.b64encode(png()).decode()}
    local_drawn: list[str] = []

    class Local:
        name = "local"

        async def generate(self, prompt: str, *, config: Any) -> bytes:
            local_drawn.append(prompt)
            return png()

        async def aclose(self) -> None:
            return None

    images = ImageService(
        ImagesConfig(backend="openai"),
        db,
        tmp_path / "media",
        audit=audit,
        backend=Local(),
        workers=hub,
    )

    task = asyncio.create_task(serve_one(http, friend, picture, ["image"]))
    await asyncio.sleep(0.05)
    private = await images.create("a secret picture for Alf", user_id="u1", public=False)
    assert local_drawn == ["a secret picture for Alf"]  # a friend's PC never sees it
    public = await images.create("a lighthouse at dusk", user_id=None, public=True)
    await task
    assert local_drawn == ["a secret picture for Alf"] and public.public and not private.public

    task = asyncio.create_task(serve_one(http, owner, picture, ["image"]))
    await asyncio.sleep(0.05)
    await images.create("another secret", user_id="u1", public=False)
    await task
    assert local_drawn == ["a secret picture for Alf"]  # the owner's own PC may draw private ones


def test_the_operator_cli_prints_a_connection_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from fuzzy_worker.setup import parse_code
    from sentientai.cli import app

    (tmp_path / "data").mkdir()
    monkeypatch.setenv("SENTIENT_STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SENTIENT_CONFIG", raising=False)
    result = CliRunner().invoke(
        app, ["workers", "create", "owens-pc", "--private", "--url", "https://fuzzy.example/"]
    )
    assert result.exit_code == 0, result.output
    token = next(w for w in result.output.split() if w.startswith("fzw_"))
    code = next(w for w in result.output.split() if w.startswith("fuzzy-worker:"))
    assert parse_code(code) == ("https://fuzzy.example", token)


async def test_a_person_links_a_computer_to_their_account_and_sees_what_it_did(
    web_factory: Any,  # noqa: F811
) -> None:
    from fuzzy_worker.setup import parse_code
    from tests.test_web import GOOD, post, sign_in

    web = await web_factory()
    alex = await web.rt.c.accounts.create_user("alex", GOOD, display_name="Alex")
    browser = web.client()
    csrf = await sign_in(browser, "alex")

    linked = await post(browser, "/api/compute/computers", csrf, json={"name": "Alex's PC"})
    assert linked.status_code == 200, linked.text
    server, token = parse_code(linked.json()["code"])  # type: ignore[misc]
    assert server == "https://test" and token.startswith("fzw_")  # a public name means https

    program = web.client(token=token)  # what the Windows program sends
    hello = await program.post("/worker/hello")
    assert hello.json() == {
        "linked": True,
        "name": "Alex's PC",
        "trust": "shared",  # an ordinary account never gets a private computer
        "account_linked": True,
        "protocol": 1,
    }
    poll = asyncio.create_task(
        program.post(
            "/worker/poll",
            json={
                "capabilities": ["image"],
                "wait": 2,
                "info": {"gpu": "RTX 3060", "gpu_memory_mb": 12288, "gpu_percent": 7},
            },
        )
    )
    await asyncio.sleep(0.1)
    job = asyncio.create_task(
        web.rt.c.workers.run("image", {"prompt": "a fox"}, private_only=False, timeout=5)
    )
    handed = (await poll).json()["job"]
    await asyncio.sleep(0.2)
    await program.post(
        "/worker/result", json={"job_id": handed["id"], "ok": True, "data": {"b64_json": "x"}}
    )
    await job
    await asyncio.sleep(0.2)  # the lifetime totals are written in the background

    page = (await browser.get("/api/compute")).json()
    [mine] = page["mine"]
    assert mine["online"] and mine["gpu"] == "RTX 3060" and mine["memory_total_mb"] == 12288
    assert mine["jobs"] == 1 and mine["picture_jobs"] == 1 and mine["busy_s"] >= 0
    assert page["my_totals"]["jobs"] == 1 and page["my_totals"]["rank"] == 1
    assert page["leaders"][0]["name"] == "Alex" and page["pool"]["online"] == 1

    other = await web.rt.c.accounts.create_user("sam", GOOD, display_name="Sam")
    sam = web.client()
    sam_csrf = await sign_in(sam, "sam")
    assert (await sam.get("/api/compute")).json()["mine"] == []  # Sam doesn't see Alex's
    gone = await sam.request(
        "DELETE",
        "/api/compute/computers/" + mine["id"],
        headers={"X-CSRF-Token": sam_csrf, "Origin": "http://test"},
        json={},
    )
    assert gone.status_code == 404  # and can't unlink it either
    removed = await browser.request(
        "DELETE",
        "/api/compute/computers/" + mine["id"],
        headers={"X-CSRF-Token": csrf, "Origin": "http://test"},
        json={},
    )
    assert removed.status_code == 200
    assert (await program.post("/worker/hello")).status_code == 401  # the token stops working
    assert alex.id and other.id


async def test_embeddings_go_to_the_operators_computers_only_for_big_batches(stack: Any) -> None:
    from sentientai.config import EmbeddingConfig
    from sentientai.models.embeddings import WorkerFirstEmbedder

    registry, hub, http, *_ = stack
    _, friend = await registry.create("zanders-pc", "shared")
    _, owner = await registry.create("owens-pc", "private")
    config = EmbeddingConfig(provider="openai", dimensions=16, base_url="http://keystone/v1")
    local_calls: list[int] = []

    def unit(k: int) -> list[float]:
        return [1.0 if i == k else 0.0 for i in range(16)]

    def keystone(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        local_calls.append(len(texts))
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": unit(0)} for i in range(len(texts))]}
        )

    embedder = WorkerFirstEmbedder(config, hub=hub)
    embedder._http = httpx.AsyncClient(
        base_url="http://keystone/v1", transport=httpx.MockTransport(keystone)
    )
    texts = [f"chunk {i}" for i in range(10)]

    # a friend's computer never sees memory text, even if it offers its processor
    poll = asyncio.create_task(
        http.post(
            "/worker/poll", json={"capabilities": ["embed"], "wait": 1}, headers=bearer(friend)
        )
    )
    await asyncio.sleep(0.05)
    assert len(await embedder.embed(texts)) == 10 and local_calls == [10]
    await poll

    # the operator's own computer takes a big batch...
    answer = {"embeddings": [unit(1)] * 10}
    task = asyncio.create_task(serve_one(http, owner, answer, ["embed"]))
    await asyncio.sleep(0.05)
    vectors = await embedder.embed(texts)
    await task
    assert vectors[0] == unit(1) and local_calls == [10]
    # ...but a single search stays on the server (a round trip would be slower)
    task = asyncio.create_task(serve_one(http, owner, answer, ["embed"]))
    await asyncio.sleep(0.05)
    assert (await embedder.embed(["one query"]))[0] == unit(0) and local_calls == [10, 1]
    task.cancel()
    # junk from a worker falls back to the server's own embedder
    task = asyncio.create_task(serve_one(http, owner, {"embeddings": [[1.0, 0.0]]}, ["embed"]))
    await asyncio.sleep(0.05)
    assert len(await embedder.embed(texts)) == 10 and local_calls == [10, 1, 10]
    await task


async def test_replies_to_people_never_go_to_a_lent_computer_but_background_thinking_does(
    stack: Any,
) -> None:
    from sentientai.config import RoutingConfig
    from sentientai.models.router import ModelRouter
    from sentientai.models.types import ModelTier, TaskKind

    registry, hub, http, *_ = stack
    _, token = await registry.create("owens-pc", "private")
    asked: list[str] = []

    def keystone(request: httpx.Request) -> httpx.Response:
        asked.append("keystone")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "server"}}],
                "usage": {},
            },
        )

    llm = WorkerFirstClient(
        "big",
        ModelEndpointConfig(base_url="http://keystone/v1", model="m"),
        hub=hub,
        http=httpx.AsyncClient(
            base_url="http://keystone/v1", transport=httpx.MockTransport(keystone)
        ),
    )
    router = ModelRouter({ModelTier.HEAVY: llm, ModelTier.FAST: llm}, RoutingConfig())
    answer = {"choices": [{"message": {"role": "assistant", "content": "helper"}}], "usage": {}}
    helper = asyncio.create_task(serve_one(http, token, answer, ["llm"]))
    await asyncio.sleep(0.05)
    request = ChatRequest(messages=[Message.user("hello there")])
    for kind in (TaskKind.CHAT, TaskKind.REASON, TaskKind.REFLECT):  # replies, projects, decisions
        assert (await router.chat(request, kind=kind)).message.text == "server"
    assert asked == ["keystone"] * 3 and hub.available("llm")  # the helper was never asked
    assert (await router.chat(request, kind=TaskKind.PONDER)).message.text == "helper"
    await helper


async def test_helpers_hand_fuzzy_thought_patches_its_own_model_then_considers(
    runtime_factory: Any,  # noqa: F811
) -> None:
    from tests.test_inner_life import beat

    seen: list[Any] = []

    def choose(request: Any) -> dict[str, Any]:
        seen.append(request)
        return beat("rest")

    brain = Brain(
        beat=choose,
        ponder={"thoughts": ["Maybe silence is a kind of memory.", "What do rivers remember?"]},
    )
    rt = await runtime_factory(brain, start=False)
    agency = rt.c.agency
    await rt.c.memory.remember("Tides follow the moon.", MemoryKind.FACT, source="teach")

    assert await agency.ponder() == 0  # no helper connected: it doesn't use the server's model
    assert brain.calls.get("ponder", 0) == 0
    agency.helper_free = lambda: True
    assert await agency.ponder() == 2 and brain.calls["ponder"] == 1
    assert await agency.ponder() == 0  # the same thoughts again aren't piled up

    await agency.beat()
    prompt = seen[0].messages[-1].text
    assert "What do rivers remember?" in prompt and "quiet background mind" in prompt
    assert await agency.patches.active() == []  # seen once, then retired
    kept = await rt.c.memory.recall("rivers remember", kinds=[MemoryKind.THOUGHT])
    assert kept and "rivers" in kept[0].record.content  # but not lost


async def test_only_a_computer_running_the_same_model_takes_pieces_of_a_conversation(
    stack: Any,
) -> None:
    registry, hub, http, *_ = stack
    _, small = await registry.create("owens-pc", "private")
    _, same = await registry.create("big-pc", "private")

    def keystone(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "server"}}],
                "usage": {},
            },
        )

    llm = WorkerFirstClient(
        "big",
        ModelEndpointConfig(base_url="http://keystone/v1", model="m"),
        hub=hub,
        chat_model="qwen3:14b",
        http=httpx.AsyncClient(
            base_url="http://keystone/v1", transport=httpx.MockTransport(keystone)
        ),
    )
    answer = {"choices": [{"message": {"role": "assistant", "content": "worker"}}], "usage": {}}

    async def offer(token: str, model: str) -> asyncio.Task[None]:
        task = asyncio.create_task(
            http.post(
                "/worker/poll",
                json={"capabilities": ["llm"], "wait": 1, "info": {"base_model": model}},
                headers=bearer(token),
            )
        )
        await asyncio.sleep(0.05)
        return task  # type: ignore[return-value]

    snippet = ChatRequest(messages=[Message.user("check this reply")], snippet=True)
    plain = ChatRequest(messages=[Message.user("a reply for someone")])

    await offer(small, "qwen3:8b")  # a smaller model is online and free
    assert (await llm.chat(snippet)).message.text == "server"  # not for conversations
    assert hub.available("llm") and not hub.available("llm", model="qwen3:14b")

    serving = asyncio.create_task(  # the same model as the server's chat model
        serve_one(http, same, answer, ["llm"], {"base_model": "qwen3:14b"})
    )
    await asyncio.sleep(0.1)
    assert hub.available("llm", model="qwen3:14b")
    assert (await llm.chat(plain)).message.text == "server"  # replies are written on the server
    assert (await llm.chat(snippet)).message.text == "worker"  # but the checks around them aren't
    await serving


async def test_the_steps_around_a_reply_are_marked_as_pieces_of_a_conversation(
    runtime_factory: Any,  # noqa: F811
) -> None:
    from tests.test_conversation_quality import make_user

    marks: dict[str, bool] = {}

    def engage(request: Any) -> dict[str, Any]:
        marks["engage"] = request.snippet
        return {"choice": "talk", "say": ""}

    def ground(request: Any) -> dict[str, Any]:
        marks["grounding"] = request.snippet
        return {"unsupported": [], "new_self_decisions": []}

    def appraise(request: Any) -> dict[str, Any]:
        marks["appraisal"] = request.snippet
        return {"opinion": "Fine.", "feelings": "ok", "felt": []}

    brain = Brain(["Hello!"], engage=engage, grounding=ground, appraisal=appraise)
    rt = await runtime_factory(brain, mind={"appraise_after_quiet_s": 0})
    alex = await make_user(rt, "Alex")
    await rt.chat("s", "hi", user=alex, timeout=10)
    await asyncio.sleep(0.5)
    assert marks == {"engage": True, "grounding": True, "appraisal": True}
    assert brain.agent_requests and not brain.agent_requests[0].snippet  # the reply itself isn't


async def test_an_administrator_can_say_a_computer_is_their_own_but_others_cannot(
    web_factory: Any,  # noqa: F811
) -> None:
    from tests.test_web import GOOD, post, sign_in

    web = await web_factory()
    await web.rt.c.accounts.create_user("boss", GOOD, display_name="Boss", role=UserRole.ADMIN)
    await web.rt.c.accounts.create_user("alex", GOOD, display_name="Alex")
    boss, alex = web.client(), web.client()
    boss_csrf, alex_csrf = await sign_in(boss, "boss"), await sign_in(alex, "alex")
    made = await post(alex, "/api/compute/computers", alex_csrf, json={"name": "Alex's PC"})
    other = await post(boss, "/api/compute/computers", boss_csrf, json={"name": "Boss PC"})
    mine = (await boss.get("/api/compute")).json()["mine"][0]
    assert made.status_code == other.status_code == 200 and not mine["own"]  # shared by default
    refused = await post(
        alex, f"/api/compute/computers/{mine['id']}/own", alex_csrf, json={"own": True}
    )
    assert refused.status_code in (401, 403)  # only an administrator
    ok = await post(boss, f"/api/compute/computers/{mine['id']}/own", boss_csrf, json={"own": True})
    assert ok.json() == {"ok": True, "trust": "private"}
    assert (await boss.get("/api/compute")).json()["mine"][0]["own"]


async def test_each_person_can_invite_a_few_friends_and_the_friend_can_join(
    web_factory: Any,  # noqa: F811
) -> None:
    from tests.test_web import GOOD, post, sign_in

    web = await web_factory(accounts={"registration": "invite"})
    await web.rt.c.accounts.create_user("alex", GOOD, display_name="Alex")
    alex = web.client()
    csrf = await sign_in(alex, "alex")
    assert (await alex.get("/api/me/invites")).json()["left"] == 3
    codes = []
    for _ in range(3):
        made = await post(alex, "/api/me/invites", csrf, json={})
        assert made.status_code == 201
        codes.append(made.json()["code"])
    assert (await post(alex, "/api/me/invites", csrf, json={})).status_code == 409  # only three

    friend = web.client()
    joined = await friend.post(
        "/api/auth/register",
        json={"username": "sam", "password": GOOD, "display_name": "Sam", "invite_code": codes[0]},
    )
    assert joined.status_code in (200, 201), joined.text
    mine = (await alex.get("/api/me/invites")).json()
    assert mine["left"] == 0
    assert sorted(i["status"] for i in mine["invites"]) == ["joined", "waiting", "waiting"]
    assert [i["joined"] for i in mine["invites"] if i["status"] == "joined"] == ["Sam"]
    # invites made this way never grant teaching rights or admin
    sam = web.client()
    await sign_in(sam, "sam")
    assert (await sam.get("/api/auth/state")).json()["admin"] is False


async def test_the_public_numbers_are_safe_to_show_and_need_no_login(
    web_factory: Any,  # noqa: F811
) -> None:
    from tests.test_web import GOOD

    web = await web_factory()
    alex = await web.rt.c.accounts.create_user("alex", GOOD, display_name="Alex")
    await web.rt.c.worker_registry.create("Alex's PC", "shared", owner_id=alex.id)
    await web.rt.c.worker_registry.record_job(
        (await web.rt.c.worker_registry.all())[0].id, "image", 3600, ok=True
    )
    await web.rt.c.memory.remember("a private thing Alex said", MemoryKind.FACT, user_id=alex.id)
    reply = await web.client().get("/api/public/stats")  # no sign-in
    assert reply.status_code == 200 and reply.headers["access-control-allow-origin"] == "*"
    data = reply.json()
    assert data["compute"]["jobs"] == 1 and data["compute"]["hours"] == 1.0
    assert data["leaders"] == [{"name": "Alex", "jobs": 1, "hours": 1.0}]
    assert data["fuzzy"]["memories"] >= 1 and data["fuzzy"]["people"] == 1
    assert "private thing" not in reply.text  # counts only, never content


async def test_pages_ask_for_versioned_scripts_so_nobody_gets_a_stale_copy(
    web_factory: Any,  # noqa: F811
) -> None:
    import re

    web = await web_factory()
    page = (await web.client().get("/compute")).text
    assert re.search(r'src="/static/compute\.js\?v=[0-9a-f]{10}"', page)
    assert re.search(r'href="/static/style\.css\?v=[0-9a-f]{10}"', page)
    asset = await web.client().get("/static/app.js")
    assert asset.headers["cache-control"] == "no-cache"


async def test_unused_invite_codes_can_be_reviewed_and_revoked(
    web_factory: Any,  # noqa: F811
) -> None:
    from tests.test_web import GOOD, post, sign_in

    web = await web_factory(accounts={"registration": "invite"})
    await web.rt.c.accounts.create_user("alex", GOOD, display_name="Alex")
    alex = web.client()
    csrf = await sign_in(alex, "alex")
    codes = [(await post(alex, "/api/me/invites", csrf, json={})).json()["code"] for _ in range(3)]
    listed = (await alex.get("/api/me/invites")).json()
    assert listed["left"] == 0 and [i["status"] for i in listed["invites"]] == ["waiting"] * 3
    assert all(code not in str(listed) for code in codes)  # the codes themselves are never listed

    one = listed["invites"][0]["ref"]
    gone = await alex.request(
        "DELETE",
        f"/api/me/invites/{one}",
        headers={"X-CSRF-Token": csrf, "Origin": "http://test"},
        json={},
    )
    assert gone.status_code == 200
    assert (await alex.get("/api/me/invites")).json()["left"] == 1  # its place came back
    friend = web.client()
    refused = await friend.post(
        "/api/auth/register",
        json={"username": "sam", "password": GOOD, "display_name": "Sam", "invite_code": codes[0]},
    )
    assert refused.status_code == 403  # a revoked code no longer works
    swept = await post(alex, "/api/me/invites/revoke-unused", csrf, json={})
    assert (
        swept.json() == {"revoked": 2} and (await alex.get("/api/me/invites")).json()["left"] == 3
    )


async def test_fuzzy_can_only_show_addresses_it_actually_found() -> None:
    from sentientai.tools.base import ToolContext
    from sentientai.tools.embed import ShareLink, remember_urls

    tool, ctx = ShareLink(), ToolContext()

    async def show(url: str) -> Any:
        return await tool.run(tool.args_model(url=url, title="A thing", note="why"), ctx)

    assert not (await show("https://www.youtube.com/watch?v=abcdefghijk")).ok  # nothing found yet
    remember_urls(ctx, "Top result: https://www.youtube.com/watch?v=abcdefghijk (Markiplier)")
    video = await show("https://www.youtube.com/watch?v=abcdefghijk")
    assert video.ok and video.data["embed"]["kind"] == "youtube"
    assert video.data["embed"]["player"] == "https://www.youtube-nocookie.com/embed/abcdefghijk"
    remember_urls(ctx, "see https://example.org/article/")
    card = await show("https://www.example.org/article")
    assert card.ok and card.data["embed"]["kind"] == "link" and "player" not in card.data["embed"]
    for bad in (
        "http://example.org/article/",
        "https://127.0.0.1/x",
        "https://user:pw@example.org/a",
    ):
        remember_urls(ctx, bad)
        assert not (await show(bad)).ok  # plain http, raw addresses and logins are never shown

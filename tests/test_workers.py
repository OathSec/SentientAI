"""Computers that lend Fuzzy their power: trust rules, the HTTP path, fallbacks."""

from __future__ import annotations

import asyncio
import base64
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from sentientai.config import ImagesConfig, ModelEndpointConfig
from sentientai.core.clock import FakeClock
from sentientai.media.images import ImageService
from sentientai.models.types import ChatRequest, Message
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.workers.api import build_worker_router
from sentientai.workers.client import WorkerFirstClient
from sentientai.workers.hub import WorkerError, WorkerHub
from sentientai.workers.registry import WorkerRegistry


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
    http: httpx.AsyncClient, token: str, answer: dict[str, Any], caps: list[str]
) -> None:
    """A worker: ask for a job, do it, hand back the result."""

    while True:
        polled = (
            await http.post("/worker/poll", json={"capabilities": caps}, headers=bearer(token))
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
    request = ChatRequest(messages=[Message.user("hi")])
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

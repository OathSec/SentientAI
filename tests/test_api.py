from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from sentientai.api.server import create_app
from sentientai.cli import app as cli_app
from sentientai.core.clock import FakeClock
from sentientai.evals.brain import Brain, call
from sentientai.models.embeddings import HashingEmbedder
from sentientai.models.types import ModelTier
from sentientai.runtime.runtime import Runtime
from tests.conftest import wait_for
from tests.test_runtime import make_settings

TOKEN = "s3cret-token"


@pytest.fixture
async def stack(tmp_path: Path) -> AsyncIterator[tuple[Runtime, httpx.AsyncClient, Brain]]:
    brain = Brain(default_answer="Hello from the API.")
    settings = make_settings(tmp_path, api={"auth_token": TOKEN})
    rt = await Runtime.create(
        settings,
        clients={ModelTier.FAST: brain.llm(), ModelTier.HEAVY: brain.llm()},
        embedder=HashingEmbedder(256),
        clock=FakeClock(),
    )
    await rt.start()
    transport = httpx.ASGITransport(app=create_app(rt))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        yield rt, client, brain
    await rt.stop("test")


async def test_auth_and_probes(stack: tuple[Runtime, httpx.AsyncClient, Brain]) -> None:
    _rt, client, _ = stack
    async with httpx.AsyncClient(transport=client._transport, base_url="http://test") as anon:
        assert (await anon.get("/health")).json()["status"] == "ok"
        assert (await anon.get("/v1/state")).status_code == 401
        assert (await anon.get("/metrics")).status_code == 401
        bad = await anon.get("/v1/state", headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401
    ready = await client.get("/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] is True
    metrics = await client.get("/metrics")
    assert "sentient_affect" in metrics.text
    dashboard = await client.get("/")
    assert dashboard.status_code == 200 and "SentientAI Console" in dashboard.text


async def test_chat_tasks_and_state(stack: tuple[Runtime, httpx.AsyncClient, Brain]) -> None:
    _rt, client, _ = stack
    reply = (await client.post("/v1/chat", json={"message": "hi", "session_id": "web"})).json()
    assert reply["answer"] == "Hello from the API." and reply["status"] == "succeeded"
    assert reply["session_id"] == "web"

    task = (await client.post("/v1/tasks", json={"goal": "background job", "priority": 7})).json()
    done = (await client.get(f"/v1/tasks/{task['id']}", params={"wait_s": 10})).json()
    assert done["status"] == "succeeded" and done["result"] == "Hello from the API."
    listed = (await client.get("/v1/tasks", params={"status": "succeeded"})).json()
    assert {t["id"] for t in listed} >= {task["id"], reply["task_id"]}
    assert (await client.get("/v1/tasks/nope")).status_code == 404

    state = (await client.get("/v1/state")).json()
    assert state["name"] == "Sentient" and state["track_record"]["succeeded"] >= 2
    tools = (await client.get("/v1/tools")).json()
    assert {"fs_read", "memory_recall", "goal_add"} <= {t["name"] for t in tools}


async def test_approval_via_api(stack: tuple[Runtime, httpx.AsyncClient, Brain]) -> None:
    rt, client, brain = stack
    brain.agent.extend([call("fs_delete", path="notes.md"), "Deleted as approved."])
    task = (await client.post("/v1/tasks", json={"goal": "delete notes.md"})).json()

    async def pending() -> list[dict[str, Any]]:
        return (await client.get("/v1/approvals")).json()

    await wait_for(pending, timeout=10)
    [req] = await pending()
    assert req["tool"] == "fs_delete" and req["risk"] == "HIGH"
    decided = (
        await client.post(f"/v1/approvals/{req['id']}", json={"approve": True, "note": "ok"})
    ).json()
    assert decided["status"] == "approved" and decided["decided_by"] == "operator:api"
    done = (await client.get(f"/v1/tasks/{task['id']}", params={"wait_s": 10})).json()
    assert done["result"] == "Deleted as approved."
    assert not (rt.settings.safety.workspace_roots[0] / "notes.md").exists()
    assert (await client.post("/v1/approvals/missing", json={"approve": True})).status_code == 404


async def test_memory_ingest_goals(stack: tuple[Runtime, httpx.AsyncClient, Brain]) -> None:
    _rt, client, _ = stack
    created = (
        await client.post(
            "/v1/memory",
            json={"content": "Operator prefers metric units", "subject_key": "operator.units"},
        )
    ).json()
    hits = (
        await client.get("/v1/memory/search", params={"q": "which units does the operator prefer"})
    ).json()
    assert hits[0]["id"] == created["id"]
    stats = (await client.get("/v1/memory/stats")).json()
    assert stats["active_by_kind"]["fact"] == 1
    assert (await client.delete(f"/v1/memory/{created['id']}")).status_code == 200
    assert (await client.delete("/v1/memory/nope")).status_code == 404

    ingested = (await client.post("/v1/ingest", json={"source": "notes.md"})).json()
    assert ingested[0]["chunks"] >= 1 and ingested[0]["reason"] is None
    assert (await client.post("/v1/ingest", json={"source": "../etc/passwd"})).status_code == 400

    goal = (
        await client.post("/v1/goals", json={"description": "Index the research papers"})
    ).json()
    assert [g["id"] for g in (await client.get("/v1/goals")).json()] == [goal["id"]]
    consolidated = (await client.post("/v1/memory/consolidate")).json()
    assert consolidated["errors"] == []


async def test_kill_resume_audit_proposals(stack: tuple[Runtime, httpx.AsyncClient, Brain]) -> None:
    rt, client, _ = stack
    assert (await client.post("/v1/kill", json={"reason": "test stop"})).json()["engaged"] is True
    assert rt.c.kill_switch.engaged and (await client.get("/health")).json()["kill_switch"] is True
    await client.post("/v1/resume")
    assert not rt.c.kill_switch.engaged
    verify = (await client.get("/v1/audit/verify")).json()
    assert verify["ok"] is True and verify["entries"] > 0
    tail = (await client.get("/v1/audit/tail", params={"n": 5, "prefix": "safety."})).json()
    assert [e["type"] for e in tail] == [
        "safety.kill_switch.engaged",
        "safety.kill_switch.released",
    ]
    assert (await client.get("/v1/proposals")).json() == []
    assert (await client.post("/v1/proposals/x/stage")).status_code == 409


# ------------------------------------------------- real server + CLI + SSE


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.integration
async def test_real_server_cli_and_events(tmp_path: Path) -> None:
    import uvicorn
    import yaml

    port = _free_port()
    brain = Brain(default_answer="CLI says hi.")
    settings = make_settings(tmp_path, api={"auth_token": TOKEN, "port": port})
    rt = await Runtime.create(
        settings,
        clients={ModelTier.FAST: brain.llm(), ModelTier.HEAVY: brain.llm()},
        embedder=HashingEmbedder(256),
    )
    await rt.start()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(rt), host="127.0.0.1", port=port, log_level="warning", lifespan="off"
        )
    )
    serving = asyncio.create_task(server.serve())
    try:
        await wait_for(lambda: server.started, timeout=10)
        config = tmp_path / "sentient.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "storage": {"data_dir": str(settings.storage.data_dir)},
                    "safety": {"workspace_roots": [str(settings.safety.workspace_roots[0])]},
                    "api": {"port": port, "auth_token": TOKEN},
                }
            )
        )
        runner = CliRunner()

        def cli(*args: str) -> Any:
            return runner.invoke(cli_app, ["-c", str(config), *args])

        events: list[dict[str, Any]] = []

        async def listen() -> None:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=30,
                headers={"Authorization": f"Bearer {TOKEN}"},
            ) as http:
                async with http.stream("GET", "/v1/events", params={"pattern": "task.*"}) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
                            if any(e["type"] == "task.finished" for e in events):
                                return

        listener = asyncio.create_task(listen())
        await asyncio.sleep(0.2)
        result = await asyncio.to_thread(cli, "ask", "hello there")
        assert result.exit_code == 0, result.output
        assert "CLI says hi." in result.output
        await asyncio.wait_for(listener, 10)
        assert {"task.submitted", "task.started", "task.finished"} <= {e["type"] for e in events}

        state = await asyncio.to_thread(cli, "state")
        assert json.loads(state.output)["name"] == "Sentient"
        tasks = await asyncio.to_thread(cli, "tasks")
        assert "succeeded" in tasks.output
        remembered = await asyncio.to_thread(
            cli, "memory", "remember", "The lab is in Porto", "--key", "lab.city"
        )
        assert remembered.exit_code == 0
        search = await asyncio.to_thread(cli, "memory", "search", "where is the lab?")
        assert "Porto" in search.output
        killed = await asyncio.to_thread(cli, "kill", "cli test")
        assert "engaged via API" in killed.output and rt.c.kill_switch.engaged
        await asyncio.to_thread(cli, "resume")
        assert not rt.c.kill_switch.engaged
        verify = await asyncio.to_thread(cli, "audit", "verify")
        assert verify.exit_code == 0 and "intact" in verify.output
    finally:
        server.should_exit = True
        await serving
        await rt.stop("test")


def test_cli_offline_paths(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "sentient.yaml"
    assert runner.invoke(cli_app, ["init", "-o", str(out)]).exit_code == 0
    assert "models:" in out.read_text()
    assert runner.invoke(cli_app, ["init", "-o", str(out)]).exit_code == 1  # no clobber

    import yaml

    config = tmp_path / "offline.yaml"
    data_dir = tmp_path / "data"
    config.write_text(
        yaml.safe_dump({"storage": {"data_dir": str(data_dir)}, "api": {"port": _free_port()}})
    )
    # with no daemon running, kill falls back to the sentinel file
    killed = runner.invoke(cli_app, ["-c", str(config), "kill", "panic"])
    assert "wrote sentinel" in killed.output and (data_dir / "KILL").exists()
    runner.invoke(cli_app, ["-c", str(config), "resume"])
    assert not (data_dir / "KILL").exists()
    down = runner.invoke(cli_app, ["-c", str(config), "state"])
    assert down.exit_code == 2 and "sentientai run" in down.output
    assert runner.invoke(cli_app, ["version"]).output.strip() == "0.1.0"
    missing = runner.invoke(cli_app, ["-c", str(tmp_path / "nope.yaml"), "state"])
    assert missing.exit_code == 1 and "not found" in missing.output


def test_cli_doctor(tmp_path: Path) -> None:
    import yaml

    config = tmp_path / "doctor.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "storage": {"data_dir": str(tmp_path / "data")},
                "safety": {"workspace_roots": [str(tmp_path / "ws")]},
                "models": {
                    "embedding": {"provider": "hashing"},
                    "fast": {"base_url": "http://127.0.0.1:9/v1"},
                    "heavy": {"base_url": "http://127.0.0.1:9/v1"},
                },
                "sandbox": {"backend": "disabled"},
            }
        )
    )
    result = CliRunner().invoke(cli_app, ["-c", str(config), "doctor"])
    assert result.exit_code == 1  # model servers are not running
    for line in ["data directory", "database", "embeddings", "vector index", "audit log"]:
        row = next(r for r in result.output.splitlines() if line in r)
        assert row.startswith("ok"), row
    assert "unreachable" in result.output


def test_cli_bench(tmp_path: Path) -> None:
    import respx
    import yaml

    config = tmp_path / "bench.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "storage": {"data_dir": str(tmp_path / "data")},
                "models": {
                    "fast": {"base_url": "http://fast.test/v1", "model": "tiny-8b"},
                    "heavy": {
                        "base_url": "http://heavy.test/v1",
                        "model": "big-70b",
                        "max_retries": 0,
                    },
                },
            }
        )
    )
    completion = {
        "model": "tiny-8b",
        "choices": [{"message": {"content": "WAL..."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 120},
    }
    with respx.mock:
        respx.post("http://fast.test/v1/chat/completions").respond(200, json=completion)
        respx.post("http://heavy.test/v1/chat/completions").respond(500, text="CUDA OOM")
        result = CliRunner().invoke(cli_app, ["-c", str(config), "bench", "--requests", "3"])
    assert result.exit_code == 0, result.output
    fast_row = next(line for line in result.output.splitlines() if line.startswith("fast"))
    heavy_row = next(line for line in result.output.splitlines() if line.startswith("heavy"))
    assert "tiny-8b" in fast_row and " 3 " in fast_row
    assert "big-70b" in heavy_row and "CUDA OOM" in heavy_row

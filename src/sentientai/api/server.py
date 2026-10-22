"""HTTP API and operator dashboard.

All ``/v1`` routes (and ``/metrics``) require ``Authorization: Bearer <token>``
when ``api.auth_token`` is configured. ``/health`` is always open for probes.

Note: no ``from __future__ import annotations`` here. FastAPI must evaluate the
locally defined ``Operator`` dependency alias at definition time; with
postponed annotations it would silently degrade to a query parameter.
"""

import asyncio
import contextlib
import json
import secrets
from collections.abc import AsyncIterator
from importlib import resources
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from sentientai import __version__
from sentientai.core.clock import new_id
from sentientai.errors import ConfigError, ToolError
from sentientai.runtime.runtime import Runtime
from sentientai.storage.records import (
    ApprovalStatus,
    GoalOrigin,
    MemoryKind,
    ProposalStatus,
    Scope,
    TaskSource,
    TaskStatus,
)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    session_id: str | None = None
    timeout_s: float = Field(default=600, gt=0, le=3600)


class TaskIn(BaseModel):
    goal: str = Field(min_length=1, max_length=100_000)
    priority: int = Field(default=5, ge=0, le=10)
    kind: str = "task"
    session_id: str | None = None


class DecisionIn(BaseModel):
    approve: bool
    note: str | None = None


class RememberIn(BaseModel):
    content: str = Field(min_length=3)
    kind: Literal["fact", "reflection", "thought"] = "fact"
    importance: float = Field(default=0.6, ge=0, le=1)
    subject_key: str | None = None


class IngestIn(BaseModel):
    source: str


class GoalIn(BaseModel):
    description: str = Field(min_length=3)
    priority: float = Field(default=0.6, ge=0, le=1)


class KillIn(BaseModel):
    reason: str = "operator request via API"


class RejectIn(BaseModel):
    reason: str = Field(min_length=3)


class ProposeIn(BaseModel):
    insight: str | None = None
    files: list[str] | None = None


def create_app(runtime: Runtime) -> FastAPI:
    settings = runtime.settings
    c = runtime.c
    token = settings.api.auth_token.get_secret_value() if settings.api.auth_token else None

    app = FastAPI(title="SentientAI", version=__version__, docs_url="/docs", redoc_url=None)
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def auth(request: Request) -> str:
        if token is None:
            return "operator"
        header = request.headers.get("authorization", "")
        supplied = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        if not supplied:
            supplied = request.query_params.get("token", "")  # EventSource cannot set headers
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        return "operator"

    Operator = Annotated[str, Depends(auth)]

    # -------------------------------------------------------------- probes
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "kill_switch": c.kill_switch.engaged}

    @app.get("/ready")
    async def ready(_: Operator) -> dict[str, Any]:
        checks = {"database": await c.db.ping(), "working_memory": await c.memory.working.ping()}
        checks.update({f"model:{k}": v for k, v in (await c.router.health()).items()})
        ok = (
            checks["database"]
            and checks["working_memory"]
            and any(v for k, v in checks.items() if k.startswith("model:"))
        )
        if not ok:
            raise HTTPException(status_code=503, detail=checks)
        return {"ready": True, "checks": checks}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(_: Operator) -> str:
        await c.memory.stats()
        return c.metrics.render()

    # ---------------------------------------------------------------- chat
    @app.post("/v1/chat")
    async def chat(body: ChatIn, _: Operator) -> dict[str, Any]:
        session = body.session_id or new_id("session")
        try:
            reply = await runtime.chat(session, body.message, timeout=body.timeout_s)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        return {
            "session_id": session,
            "task_id": reply.task.id,
            "status": reply.task.status.value,
            "answer": reply.answer,
            "confidence": reply.confidence,
        }

    # --------------------------------------------------------------- tasks
    @app.post("/v1/tasks", status_code=202)
    async def submit_task(body: TaskIn, _: Operator) -> dict[str, Any]:
        record = await runtime.submit(
            body.goal,
            source=TaskSource.API,
            priority=body.priority,
            kind=body.kind,
            session_id=body.session_id,
        )
        return record.model_dump(mode="json")

    @app.get("/v1/tasks")
    async def list_tasks(
        _: Operator, status: TaskStatus | None = None, limit: int = Query(50, le=500)
    ) -> list[dict[str, Any]]:
        records = await c.db.tasks.list([status] if status else None, limit=limit)
        return [r.model_dump(mode="json") for r in records]

    @app.get("/v1/tasks/{task_id}")
    async def get_task(
        task_id: str, _: Operator, wait_s: float = Query(0, ge=0, le=600)
    ) -> dict[str, Any]:
        if wait_s:
            with contextlib.suppress(TimeoutError, KeyError):
                await runtime.wait(task_id, timeout=wait_s)
        record = await c.db.tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="no such task")
        return record.model_dump(mode="json")

    @app.post("/v1/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, _: Operator) -> dict[str, Any]:
        return {"cancelled": await runtime.cancel(task_id)}

    # ----------------------------------------------------------- approvals
    @app.get("/v1/approvals")
    async def list_approvals(
        _: Operator, status: ApprovalStatus | None = ApprovalStatus.PENDING
    ) -> list[dict[str, Any]]:
        return [a.model_dump(mode="json") for a in await c.db.approvals.list(status)]

    @app.post("/v1/approvals/{approval_id}")
    async def decide(approval_id: str, body: DecisionIn, who: Operator) -> dict[str, Any]:
        try:
            record = await c.approvals.decide(
                approval_id, approved=body.approve, by=f"{who}:api", note=body.note
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="no such approval") from exc
        return record.model_dump(mode="json")

    # -------------------------------------------------------------- memory
    @app.get("/v1/memory/search")
    async def search_memory(
        _: Operator,
        q: str = Query(min_length=2),
        kinds: list[MemoryKind] | None = Query(None),
        k: int = Query(8, ge=1, le=50),
    ) -> list[dict[str, Any]]:
        hits = await c.memory.recall(q, kinds=kinds, k=k, touch=False, scope=Scope.system())
        return [
            {
                "score": round(h.score, 4),
                "similarity": round(h.similarity, 4),
                **h.record.model_dump(mode="json"),
            }
            for h in hits
        ]

    @app.post("/v1/memory", status_code=201)
    async def remember(body: RememberIn, who: Operator) -> dict[str, Any]:
        record = await c.memory.remember(
            body.content,
            MemoryKind(body.kind),
            importance=body.importance,
            subject_key=body.subject_key,
            source=who,
        )
        return record.model_dump(mode="json")

    @app.delete("/v1/memory/{memory_id}")
    async def forget(memory_id: str, _: Operator, hard: bool = False) -> dict[str, Any]:
        if not await c.memory.forget(memory_id, hard=hard):
            raise HTTPException(status_code=404, detail="no such memory")
        return {"forgotten": memory_id}

    @app.get("/v1/memory/stats")
    async def memory_stats(_: Operator) -> dict[str, Any]:
        return await c.memory.stats()

    @app.post("/v1/memory/consolidate")
    async def consolidate(_: Operator) -> dict[str, Any]:
        report = await c.consolidator.run()
        return report.__dict__

    @app.post("/v1/ingest")
    async def ingest(body: IngestIn, _: Operator) -> list[dict[str, Any]]:
        source = body.source
        if source.startswith(("http://", "https://")):
            results = [await c.pipeline.ingest_url(source)]
        else:
            try:
                path = c.workspace.resolve(source, must_exist=True)
            except ToolError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            results = (
                await c.pipeline.ingest_directory(path)
                if path.is_dir()
                else [await c.pipeline.ingest_path(path)]
            )
        return [r.__dict__ for r in results]

    # ------------------------------------------------------- mind & goals
    @app.get("/v1/state")
    async def state(_: Operator) -> dict[str, Any]:
        return await runtime.state()

    @app.get("/v1/goals")
    async def goals(_: Operator) -> list[dict[str, Any]]:
        return [g.model_dump(mode="json") for g in await c.goals.active()]

    @app.post("/v1/goals", status_code=201)
    async def add_goal(body: GoalIn, _: Operator) -> dict[str, Any]:
        goal = await c.goals.add(
            body.description, origin=GoalOrigin.OPERATOR, priority=body.priority
        )
        assert goal is not None
        return goal.model_dump(mode="json")

    @app.get("/v1/tools")
    async def tools(_: Operator) -> list[dict[str, Any]]:
        return c.registry.describe()

    # -------------------------------------------------------------- safety
    @app.post("/v1/kill")
    async def kill(body: KillIn, who: Operator) -> dict[str, Any]:
        await c.kill_switch.engage(body.reason, by=f"{who}:api")
        return {"engaged": True, "reason": body.reason}

    @app.post("/v1/resume")
    async def resume(who: Operator) -> dict[str, Any]:
        await c.kill_switch.release(by=f"{who}:api")
        return {"engaged": False}

    @app.get("/v1/audit/verify")
    async def audit_verify(_: Operator) -> dict[str, Any]:
        result = await asyncio.to_thread(c.audit.verify)
        return result.__dict__

    @app.get("/v1/audit/tail")
    async def audit_tail(
        _: Operator, n: int = Query(50, ge=1, le=1000), prefix: str | None = None
    ) -> list[dict[str, Any]]:
        entries = await asyncio.to_thread(c.audit.tail, n, prefix)
        return [e.model_dump(mode="json") for e in entries]

    # ----------------------------------------------------------- proposals
    def engine() -> Any:
        if c.improvement is None:
            raise HTTPException(status_code=409, detail="self-improvement is disabled")
        return c.improvement

    @app.get("/v1/proposals")
    async def proposals(_: Operator, status: ProposalStatus | None = None) -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in await c.db.proposals.list(status)]

    @app.post("/v1/proposals", status_code=201)
    async def propose(body: ProposeIn, who: Operator) -> dict[str, Any]:
        try:
            record = await engine().propose(body.insight, body.files, by=who)
        except (ToolError, ConfigError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return dict(record.model_dump(mode="json"))

    @app.post("/v1/proposals/{proposal_id}/stage")
    async def stage(proposal_id: str, who: Operator) -> dict[str, Any]:
        try:
            record = await engine().stage(proposal_id, by=who)
        except (KeyError, ToolError, ConfigError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return dict(record.model_dump(mode="json"))

    @app.post("/v1/proposals/{proposal_id}/apply")
    async def apply(proposal_id: str, who: Operator) -> dict[str, Any]:
        try:
            record = await engine().apply(proposal_id, by=who)
        except (KeyError, ToolError, ConfigError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return dict(record.model_dump(mode="json"))

    @app.post("/v1/proposals/{proposal_id}/reject")
    async def reject(proposal_id: str, body: RejectIn, who: Operator) -> dict[str, Any]:
        try:
            record = await engine().reject(proposal_id, by=who, reason=body.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="no such proposal") from exc
        return dict(record.model_dump(mode="json"))

    # -------------------------------------------------------------- events
    @app.get("/v1/events")
    async def events(_: Operator, pattern: str = "*") -> StreamingResponse:
        async def stream() -> AsyncIterator[bytes]:
            yield b": connected\n\n"
            for event in c.bus.recent(pattern, limit=20):
                yield f"data: {json.dumps(event.to_dict(), default=str)}\n\n".encode()
            iterator = c.bus.stream(pattern).__aiter__()
            while True:
                try:
                    event = await asyncio.wait_for(iterator.__anext__(), timeout=15)
                except TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event.to_dict(), default=str)}\n\n".encode()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----------------------------------------------------------- dashboard
    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return (
            resources.files("sentientai.api").joinpath("dashboard.html").read_text(encoding="utf-8")
        )

    return app

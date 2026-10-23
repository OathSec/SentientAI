"""The endpoints a worker program talks to: ``/worker/poll``, ``/worker/result``, ``/worker/bye``.

These are separate from people's sessions: a worker authenticates with its own bearer token.
"""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from sentientai.safety.audit import AuditLog
from sentientai.workers.hub import KINDS, POLL_WAIT_S, WorkerHub
from sentientai.workers.registry import WorkerRecord, WorkerRegistry

PROTOCOL = 1
SEEN_EVERY_S = 300.0  # how often a worker's "last seen" is written to the database


class PollIn(BaseModel):
    capabilities: list[str] = Field(default_factory=list, max_length=8)
    info: dict[str, Any] = Field(default_factory=dict)
    protocol: int = PROTOCOL


class ResultIn(BaseModel):
    job_id: str = Field(max_length=64)
    ok: bool
    data: dict[str, Any] | None = None
    error: str = Field(default="", max_length=500)


def _small_info(info: dict[str, Any]) -> dict[str, Any]:
    """What a worker says about itself (GPU name, version): short strings only."""

    return {
        str(k)[:30]: str(v)[:120]
        for k, v in list(info.items())[:10]
        if isinstance(v, (str, int, float, bool))
    }


def build_worker_router(
    hub: WorkerHub, registry: WorkerRegistry, audit: AuditLog, *, wait_s: float = POLL_WAIT_S
) -> APIRouter:
    router = APIRouter(prefix="/worker", tags=["worker"])
    last_written: dict[str, float] = {}

    async def worker(authorization: Annotated[str | None, Header()] = None) -> WorkerRecord:
        scheme, _, token = (authorization or "").partition(" ")
        record = await registry.authenticate(token.strip()) if scheme.lower() == "bearer" else None
        if record is None:
            raise HTTPException(status_code=401, detail="unknown or revoked worker token")
        return record

    Worker = Annotated[WorkerRecord, Depends(worker)]

    @router.post("/poll")
    async def poll(body: PollIn, record: Worker) -> dict[str, Any]:
        if body.protocol != PROTOCOL:
            raise HTTPException(status_code=426, detail="update the Fuzzy worker program")
        caps = {c for c in body.capabilities if c in KINDS}
        now = time.monotonic()
        if now - last_written.get(record.id, -1e9) > SEEN_EVERY_S:
            last_written[record.id] = now
            await registry.seen(record.id)
        job = await hub.poll(record, caps, _small_info(body.info), wait_s=wait_s)
        if job is None:
            return {"job": None}
        await audit.record(
            "worker.job", actor=f"worker:{record.name}", job_id=job.id, kind=job.kind
        )
        return {"job": {"id": job.id, "kind": job.kind, "payload": job.payload}}

    @router.post("/result")
    async def result(body: ResultIn, record: Worker) -> dict[str, bool]:
        accepted = hub.result(record, body.job_id, ok=body.ok, data=body.data, error=body.error)
        return {"accepted": accepted}

    @router.post("/bye")
    async def bye(record: Worker) -> dict[str, bool]:
        hub.goodbye(record)
        return {"ok": True}

    return router

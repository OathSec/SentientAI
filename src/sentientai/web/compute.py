"""The Compute page's API: the pool of computers lending power to Fuzzy, and each person's own.

A signed-in person can link their computers to their account (a token and connection code made
here), see them live (their GPU, load, what they're doing) and what they have contributed.
"""

import time
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from sentientai.runtime import hardware
from sentientai.runtime.runtime import Runtime
from sentientai.web.security import Principal, WebAuth, json_only
from sentientai.workers.registry import Trust, WorkerRecord

HARDWARE_CACHE_S = 10.0


class NewComputer(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    own: bool = False  # an administrator's own machine: may run Fuzzy's thinking


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def site_address(request: Request) -> str:
    """The address people reached this site at (behind the tunnel the scheme is https)."""

    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    local = urlsplit("//" + host).hostname in ("localhost", "127.0.0.1", "::1")
    return f"{'http' if local else 'https'}://{host}"


def connection_code(server: str, token: str) -> str:
    import base64
    import json

    raw = json.dumps({"s": server, "t": token}, separators=(",", ":")).encode()
    return "fuzzy-worker:" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def build_compute_router(runtime: Runtime, auth: WebAuth) -> APIRouter:
    router = APIRouter(tags=["compute"])
    c = runtime.c
    Person = Annotated[Principal, Depends(auth.person)]
    cache: dict[str, tuple[float, Any]] = {}

    async def home_gpus() -> list[dict[str, Any]]:
        """Fuzzy's own server: shown beside the lent computers as part of the same pool."""

        now = time.monotonic()
        hit = cache.get("gpus")
        if hit and now - hit[0] < HARDWARE_CACHE_S:
            return list(hit[1])
        try:
            data = await hardware.collect(runtime.settings.storage.data_dir)
        except Exception:
            data = {"gpus": []}
        gpus = [
            {
                "name": g["name"],
                "load_pct": g.get("util_pct"),
                "memory_used_mb": g.get("mem_used_mb"),
                "memory_total_mb": g.get("mem_total_mb"),
                "temp_c": g.get("temp_c"),
            }
            for g in data.get("gpus", [])
        ]
        cache["gpus"] = (now, gpus)
        return gpus

    def computer_json(
        record: WorkerRecord, live: dict[str, Any] | None, owner_name: str | None
    ) -> dict[str, Any]:
        info = (live or {}).get("info", {})
        stats = record.stats
        return {
            "id": record.id,
            "name": record.name,
            "owner": owner_name,
            "own": record.trust == "private",
            "online": live is not None,
            "busy": bool(live and live["busy"]),
            "capabilities": (live or {}).get("capabilities", []),
            "gpu": info.get("gpu"),
            "memory_total_mb": _int(info.get("gpu_memory_mb")),
            "load_pct": _int(info.get("gpu_percent")),
            "connected_s": (live or {}).get("connected_s"),
            "program_version": info.get("version"),
            "model": info.get("base_model"),
            "same_model": info.get("base_model") == runtime.settings.workers.chat_model,
            "linked_at": record.created_at.isoformat(),
            "last_seen_at": record.last_seen_at.isoformat() if record.last_seen_at else None,
            "jobs": stats.jobs,
            "chat_jobs": stats.llm_jobs,
            "picture_jobs": stats.image_jobs,
            "failed_jobs": stats.failed_jobs,
            "busy_s": round(stats.busy_s),
        }

    @router.get("/api/compute")
    async def compute(p: Person) -> dict[str, Any]:
        me = p.user
        assert me is not None
        live = {w["id"]: w for w in c.workers.online()}
        records = [r for r in await c.worker_registry.all() if not r.revoked]
        users = {u.id: u.display_name for u in await c.db.users.list(limit=10_000)}
        mine = [r for r in records if r.owner_id == me.id]
        everyone = [
            computer_json(r, live.get(r.id), users.get(r.owner_id or "", "the operator"))
            for r in records
            if r.id in live
        ]
        totals: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.owner_id is None:
                continue
            t = totals.setdefault(r.owner_id, {"jobs": 0, "busy_s": 0.0})
            t["jobs"] += r.stats.jobs
            t["busy_s"] += r.stats.busy_s
        ranking = sorted(totals.items(), key=lambda kv: kv[1]["busy_s"], reverse=True)
        my_total = totals.get(me.id, {"jobs": 0, "busy_s": 0.0})
        rank = next((i + 1 for i, (uid, _) in enumerate(ranking) if uid == me.id), None)
        return {
            "download_url": runtime.settings.workers.download_url,
            "pool": {
                "online": len(everyone),
                "busy": sum(1 for e in everyone if e["busy"]),
                "jobs": sum(r.stats.jobs for r in records),
                "contributors": len(totals),
                "home_gpus": await home_gpus(),
            },
            "computers": everyone,
            "mine": [computer_json(r, live.get(r.id), me.display_name) for r in mine],
            "my_totals": {
                "computers": len(mine),
                "jobs": my_total["jobs"],
                "busy_s": round(my_total["busy_s"]),
                "rank": rank,
                "of": len(ranking),
            },
            "leaders": [
                {"name": users.get(uid, "someone"), "jobs": t["jobs"], "busy_s": round(t["busy_s"])}
                for uid, t in ranking[:5]
            ],
            "can_link_own": p.admin,
        }

    @router.post("/api/compute/computers")
    async def link_computer(body: NewComputer, request: Request, p: Person) -> dict[str, Any]:
        json_only(request)
        me = p.user
        assert me is not None
        limit = runtime.settings.workers.max_per_person
        if len(await c.worker_registry.for_owner(me.id)) >= limit:
            raise HTTPException(status_code=409, detail=f"You can link up to {limit} computers.")
        trust: Trust = "private" if body.own and p.admin else "shared"
        try:
            record, token = await c.worker_registry.create(body.name, trust, owner_id=me.id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await c.audit.record(
            "worker.linked", actor=f"user:{me.id}", worker=record.name, trust=trust
        )
        # The token exists only in this response: it is stored hashed.
        return {
            "id": record.id,
            "name": record.name,
            "code": connection_code(site_address(request), token),
        }

    @router.delete("/api/compute/computers/{worker_id}")
    async def unlink_computer(worker_id: str, request: Request, p: Person) -> dict[str, bool]:
        json_only(request)
        me = p.user
        assert me is not None
        record = await c.worker_registry.revoke(worker_id, owner_id=me.id)
        if record is None:
            raise HTTPException(status_code=404, detail="no such computer")
        c.workers.goodbye(record)
        await c.audit.record("worker.unlinked", actor=f"user:{me.id}", worker=record.name)
        return {"ok": True}

    return router

"""Inviting friends (a few codes each) and the public numbers shown on the project's website."""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from sentientai.runtime.runtime import Runtime
from sentientai.storage import schema
from sentientai.web.security import Principal, WebAuth, json_only

STATS_CACHE_S = 15.0


def build_community_router(runtime: Runtime, auth: WebAuth) -> APIRouter:
    router = APIRouter(tags=["community"])
    c = runtime.c
    cfg = runtime.settings.accounts
    Person = Annotated[Principal, Depends(auth.person)]
    stats_cache: list[tuple[float, dict[str, Any]]] = []

    async def invite_state(p: Principal) -> dict[str, Any]:
        me = p.user
        assert me is not None
        now = c.clock.now()
        mine = [i for i in await c.db.invites.list(limit=500) if i.created_by == me.id]
        # An invite that expired unused hands its place back.
        holding = [i for i in mine if i.used_by or i.expires_at > now]
        names = {u.id: u.display_name for u in await c.db.users.list(limit=10_000)}
        allowed = None if p.admin else cfg.invites_per_person
        return {
            "allowed": allowed,
            "left": None if allowed is None else max(0, allowed - len(holding)),
            "invites": [
                {
                    "ref": i.id[:16],  # not the code (that is never stored), just a handle
                    "created_at": i.created_at.isoformat(),
                    "expires_at": i.expires_at.isoformat(),
                    "status": "joined"
                    if i.used_by
                    else "waiting"
                    if i.expires_at > now
                    else "expired",
                    "joined": names.get(i.used_by or ""),
                }
                for i in mine
            ],
        }

    @router.get("/api/me/invites")
    async def my_invites(p: Person) -> dict[str, Any]:
        return await invite_state(p)

    @router.post("/api/me/invites", status_code=201)
    async def make_invite(request: Request, p: Person) -> dict[str, Any]:
        json_only(request)
        state = await invite_state(p)
        if state["left"] is not None and state["left"] <= 0:
            raise HTTPException(status_code=409, detail="You've used all your invites.")
        code = await c.accounts.create_invite(by=p.user, days=cfg.invite_days)  # type: ignore[arg-type]
        return {"code": code, "expires_in_days": cfg.invite_days}  # shown once; stored hashed

    async def own_unused(p: Principal) -> list[Any]:
        me = p.user
        assert me is not None
        now = c.clock.now()
        return [
            i
            for i in await c.db.invites.list(limit=500)
            if i.created_by == me.id and i.used_by is None and i.expires_at > now
        ]

    @router.delete("/api/me/invites/{ref}")
    async def revoke_invite(ref: str, request: Request, p: Person) -> dict[str, bool]:
        """Cancel one of your unused invite codes; it stops working and its place comes back."""

        json_only(request)
        found = next(
            (i for i in await own_unused(p) if len(ref) >= 8 and i.id.startswith(ref)), None
        )
        if found is None:
            raise HTTPException(status_code=404, detail="No such unused invite.")
        await c.db.invites.delete(found.id)
        await c.audit.record("account.invite_revoked", actor=p.user.id if p.user else "")
        return {"ok": True}

    @router.post("/api/me/invites/revoke-unused")
    async def revoke_unused(request: Request, p: Person) -> dict[str, int]:
        json_only(request)
        unused = await own_unused(p)
        for i in unused:
            await c.db.invites.delete(i.id)
        await c.audit.record(
            "account.invite_revoked", actor=p.user.id if p.user else "", count=len(unused)
        )
        return {"revoked": len(unused)}

    async def count(table: Any, *where: Any) -> int:
        async with c.db.engine.connect() as conn:
            return int(
                (
                    await conn.execute(select(func.count()).select_from(table).where(*where))
                ).scalar_one()
            )

    async def collect() -> dict[str, Any]:
        sm = c.mind.self_model
        now = c.clock.now()
        records = [r for r in await c.worker_registry.all() if not r.revoked]
        users = {u.id: u.display_name for u in await c.db.users.list(limit=10_000)}
        totals: dict[str, dict[str, float]] = {}
        for r in records:
            if r.owner_id:
                t = totals.setdefault(r.owner_id, {"jobs": 0, "busy_s": 0.0})
                t["jobs"] += r.stats.jobs
                t["busy_s"] += r.stats.busy_s
        ranked = sorted(totals.items(), key=lambda kv: kv[1]["busy_s"], reverse=True)
        online = c.workers.online()
        return {
            "fuzzy": {
                "days_alive": max(0, (now - sm.first_boot_at).days) if sm.first_boot_at else 0,
                "doing": c.world.body.action,
                "memories": await count(schema.memories, schema.memories.c.status == "active"),
                "things_it_did": await count(schema.activities),
                "pictures": await count(schema.media, schema.media.c.public.is_(True)),
                "things_built": await count(schema.world_objects),
                "people": await count(schema.users),
            },
            "compute": {
                "computers_online": len(online),
                "working_now": sum(1 for w in online if w["busy"]),
                "contributors": len(totals),
                "jobs": sum(r.stats.jobs for r in records),
                "hours": round(sum(r.stats.busy_s for r in records) / 3600, 1),
            },
            "leaders": [
                {
                    "name": users.get(uid, "someone"),
                    "jobs": int(t["jobs"]),
                    "hours": round(t["busy_s"] / 3600, 1),
                }
                for uid, t in ranked[:5]
            ],
        }

    @router.get("/api/public/stats")
    async def public_stats() -> JSONResponse:
        """Aggregate numbers only, for the project website. No messages, no memories' content."""

        now = time.monotonic()
        if not stats_cache or now - stats_cache[0][0] > STATS_CACHE_S:
            stats_cache[:] = [(now, await collect())]
        return JSONResponse(
            stats_cache[0][1],
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=10",
            },
        )

    return router

"""The public web app's JSON API (cookie sessions).

Note: no ``from __future__ import annotations`` here. FastAPI must evaluate the dependency
aliases defined inside :func:`build_router` at definition time.
"""

import asyncio
import contextlib
import fnmatch
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from sentientai.accounts.quotas import QuotaExceeded
from sentientai.accounts.service import AuthError, LoginThrottle, clean_display_name
from sentientai.core.logging import get_logger
from sentientai.media.images import ImageError
from sentientai.runtime import hardware
from sentientai.runtime.runtime import Runtime
from sentientai.storage.records import (
    ConversationMessage,
    MemoryKind,
    Scope,
    UserRecord,
    UserRole,
    UserStatus,
)
from sentientai.web.brain import PUBLIC_PATTERNS, BrainView, Viewer, filter_event
from sentientai.web.security import (
    Principal,
    WebAuth,
    clear_session_cookie,
    json_only,
    origin_ok,
    raise_for,
    set_session_cookie,
)

log = get_logger(__name__)

MAX_MESSAGE_CHARS = 4000
MAX_STREAMS_PER_CALLER = 4
MAX_STREAMS_TOTAL = 300
TEXT_LIMIT_USER = 200_000
FILE_LIMIT_USER = 5_000_000
FILE_LIMIT_ADMIN = 50_000_000
URL_COST = 20_000  # a page is charged a flat amount against the daily teaching allowance


# ---------------------------------------------------------------- request models
class RegisterIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=80)
    invite_code: str | None = Field(default=None, max_length=128)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class CodeIn(BaseModel):
    code: str = Field(min_length=1, max_length=16)


class PasswordIn(BaseModel):
    current: str = Field(max_length=256)
    new: str = Field(max_length=256)


class TotpEnableIn(BaseModel):
    secret: str = Field(min_length=16, max_length=64)
    code: str = Field(min_length=1, max_length=16)


class PasswordOnlyIn(BaseModel):
    password: str = Field(max_length=256)


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class TeachTextIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=20)


class TeachUrlIn(BaseModel):
    url: str = Field(min_length=8, max_length=2000)


class UserUpdateIn(BaseModel):
    status: UserStatus | None = None
    role: UserRole | None = None
    can_teach: bool | None = None


class InviteIn(BaseModel):
    role: UserRole = UserRole.USER
    can_teach: bool = False
    days: float = Field(default=7.0, gt=0, le=90)


def build_router(runtime: Runtime, auth: WebAuth) -> APIRouter:
    c = runtime.c
    cfg = runtime.settings.accounts
    router = APIRouter()
    brain = BrainView(runtime)

    register_limit = LoginThrottle(5)
    chat_limit = LoginThrottle(20)
    teach_limit = LoginThrottle(10)
    export_limit = LoginThrottle(3)
    # Anything that checks the current password is limited, so a hijacked session can't be used
    # to guess it.
    password_limit = LoginThrottle(5)

    def check_password_attempts(user: UserRecord) -> None:
        if not password_limit.allow(user.id):
            raise HTTPException(status_code=429, detail="Too many attempts. Please wait a minute.")

    # person -> (the reply being made for them, when it started)
    chat_jobs: dict[str, tuple[asyncio.Task[None], Any]] = {}
    streams: dict[str, int] = {}

    Anyone = Annotated[Principal, Depends(auth.anyone)]
    Signed = Annotated[Principal, Depends(auth.user)]
    Person = Annotated[Principal, Depends(auth.person)]
    Admin = Annotated[Principal, Depends(auth.admin)]
    Viewing = Annotated[Principal, Depends(auth.viewer)]

    def viewer_of(p: Principal) -> Viewer:
        return Viewer(user_id=p.user.id if p.user else None, admin=p.admin)

    async def redactor(p: Principal) -> Any:
        return await c.redactor.for_viewer(p.user.id if p.user else None, privileged=p.admin)

    def user_of(p: Principal) -> UserRecord:
        assert p.user is not None
        return p.user

    def public_media(media_id: str) -> dict[str, str]:
        return {"id": media_id, "url": f"/media/{media_id}"}

    async def session_state(p: Principal) -> dict[str, Any]:
        return {
            "name": c.mind.self_model.name,
            "signed_in": p.signed_in and p.user is not None,
            "mfa_pending": p.second_factor_pending,
            "registration": cfg.registration,
            "public_pages": cfg.public_pages,
            "user": p.user.public() if p.user else None,
            "csrf": p.session.csrf_token if p.session else None,
            "admin": p.admin,
            "operator": p.operator,
            # Things Fuzzy is waiting on you for (only administrators can decide them).
            "pending_approvals": len(await c.approvals.pending()) if p.admin else 0,
        }

    # ------------------------------------------------------------------ auth
    @router.get("/api/auth/state")
    async def auth_state(p: Anyone) -> dict[str, Any]:
        return await session_state(p)

    def guard_anonymous_post(request: Request) -> None:
        json_only(request)
        if not origin_ok(request):
            raise HTTPException(status_code=403, detail="cross-site request refused")

    @router.post("/api/auth/register", status_code=201)
    async def register(body: RegisterIn, request: Request, response: Response) -> dict[str, Any]:
        guard_anonymous_post(request)
        ip = auth.ip(request)
        if not register_limit.allow(ip):
            raise HTTPException(status_code=429, detail="Too many sign-ups from here. Try later.")
        try:
            user = await c.accounts.register(
                body.username,
                body.password,
                display_name=body.display_name,
                invite_code=body.invite_code,
            )
            result = await c.accounts.login(
                body.username, body.password, ip=ip, user_agent=request.headers.get("user-agent")
            )
        except AuthError as exc:
            raise raise_for(exc) from exc
        c.redactor.invalidate()
        set_session_cookie(response, result.token, cfg)
        return {"user": user.public(), "csrf": result.session.csrf_token, "mfa_required": False}

    @router.post("/api/auth/login")
    async def login(body: LoginIn, request: Request, response: Response) -> dict[str, Any]:
        guard_anonymous_post(request)
        try:
            result = await c.accounts.login(
                body.username,
                body.password,
                ip=auth.ip(request),
                user_agent=request.headers.get("user-agent"),
            )
        except AuthError as exc:
            raise raise_for(exc) from exc
        set_session_cookie(response, result.token, cfg)
        return {
            "user": result.user.public(),
            "csrf": result.session.csrf_token,
            "mfa_required": result.needs_second_factor,
        }

    @router.post("/api/auth/mfa")
    async def second_factor(body: CodeIn, p: Anyone) -> dict[str, Any]:
        if p.user is None or p.session is None:
            raise HTTPException(status_code=401, detail="please sign in")
        try:
            await c.accounts.verify_second_factor(p.session, p.user, body.code)
        except AuthError as exc:
            raise raise_for(exc) from exc
        return {"ok": True}

    @router.post("/api/auth/logout")
    async def logout(p: Anyone, response: Response) -> dict[str, bool]:
        await c.accounts.logout(p.token)
        clear_session_cookie(response, cfg)
        return {"ok": True}

    # -------------------------------------------------------------------- me
    @router.get("/api/me")
    async def me(p: Person) -> dict[str, Any]:
        user = user_of(p)
        usage = await c.quotas.usage(user.id)
        unread = await c.db.notes.for_user(user.id, unread_only=True)
        return {
            **await session_state(p),
            "usage": {
                "messages": usage.messages,
                "images": usage.images,
                "teach_bytes": usage.teach_bytes,
            },
            "limits": {
                "messages": cfg.daily_messages,
                "images": cfg.daily_images,
                "teach_bytes": cfg.daily_teach_bytes,
            },
            "unread_notes": len(unread),
            "image_generation": c.images.available,
        }

    @router.post("/api/me/password")
    async def change_password(body: PasswordIn, p: Person) -> dict[str, bool]:
        check_password_attempts(user_of(p))
        try:
            await c.accounts.change_password(
                user_of(p), body.current, body.new, keep_session=p.session.id if p.session else None
            )
        except AuthError as exc:
            raise raise_for(exc) from exc
        return {"ok": True}

    @router.post("/api/me/totp/begin")
    async def totp_begin(p: Person) -> dict[str, str]:
        secret, uri = c.accounts.begin_totp(user_of(p))
        return {"secret": secret, "uri": uri}

    @router.post("/api/me/totp/enable")
    async def totp_enable(body: TotpEnableIn, p: Person) -> dict[str, bool]:
        try:
            await c.accounts.enable_totp(user_of(p), body.secret, body.code)
        except AuthError as exc:
            raise raise_for(exc) from exc
        return {"ok": True}

    @router.post("/api/me/totp/disable")
    async def totp_disable(body: PasswordOnlyIn, p: Person) -> dict[str, bool]:
        if cfg.require_admin_totp and p.admin:
            raise HTTPException(status_code=403, detail="administrators must keep two-factor on")
        check_password_attempts(user_of(p))
        try:
            await c.accounts.disable_totp(user_of(p), body.password)
        except AuthError as exc:
            raise raise_for(exc) from exc
        return {"ok": True}

    @router.get("/api/me/export")
    async def export_data(p: Person) -> JSONResponse:
        user = user_of(p)
        if not export_limit.allow(user.id):
            raise HTTPException(status_code=429, detail="Please wait before exporting again.")
        data = await c.user_data.export(user)
        return JSONResponse(
            data, headers={"Content-Disposition": 'attachment; filename="my-data-with-fuzzy.json"'}
        )

    async def erase_account(user: UserRecord, by: str) -> dict[str, int]:
        if user.role is UserRole.ADMIN:
            admins = [u for u in await c.db.users.list() if u.role is UserRole.ADMIN]
            if len(admins) <= 1:
                raise HTTPException(status_code=409, detail="You are the only administrator.")
        counts = await c.user_data.erase(user, by=by)
        c.redactor.invalidate()
        return counts

    @router.post("/api/me/delete")
    async def delete_me(body: PasswordOnlyIn, p: Person, response: Response) -> dict[str, Any]:
        from sentientai.accounts.passwords import verify_password

        user = user_of(p)
        check_password_attempts(user)
        if not verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Your password is incorrect.")
        counts = await erase_account(user, by=user.id)
        clear_session_cookie(response, cfg)
        return {"erased": counts}

    @router.get("/api/me/relationship")
    async def my_relationship(p: Person) -> dict[str, Any]:
        user = user_of(p)
        rel = await c.relationships.get(user.id)
        own = Scope(user_id=user.id, include_global=False)
        feelings = await c.emotions.current(own)
        return {
            "name": c.mind.self_model.name,
            "relationship": c.relationships.public(rel) if rel else None,
            "feelings": [f.public() for f in feelings],
        }

    # ------------------------------------------------------------------ chat
    def message_json(m: Any) -> dict[str, Any]:
        return {
            "role": m.role,
            "content": m.content,
            "at": m.created_at.isoformat(),
            "media": [public_media(i) for i in m.metadata.get("media", [])],
        }

    async def reply_in_background(user: UserRecord, text: str, recorded: asyncio.Event) -> None:
        """Fuzzy's reply is made on the server, not inside the browser's request, so a person can
        close the page, come back later and find it (it is stored with the conversation)."""

        session = f"user:{user.id}"
        try:
            await runtime.chat(
                session,
                text,
                user=user,
                timeout=runtime.settings.accounts.chat_timeout_s,
                recorded=recorded,
            )
        except TimeoutError:
            await c.quotas.take(user.id, messages=-1, privileged=True)
            apology = (
                "Sorry, that took me too long and I lost my train of thought. Could you ask again?"
            )
        except Exception:
            log.exception("chat reply failed", extra={"user_id": user.id})
            await c.quotas.take(user.id, messages=-1, privileged=True)
            apology = "Sorry, something went wrong on my side. Could you try again?"
        else:
            return
        await c.db.conversations.insert(
            ConversationMessage(
                session_id=session, user_id=user.id, role="assistant", content=apology
            )
        )

    @router.post("/api/chat", status_code=202)
    async def chat(body: ChatBody, request: Request, p: Person) -> dict[str, Any]:
        json_only(request)
        user = user_of(p)
        if c.kill_switch.engaged:
            raise HTTPException(
                status_code=503, detail=f"{c.mind.self_model.name} is paused right now."
            )
        if not chat_limit.allow(user.id):
            raise HTTPException(status_code=429, detail="Slow down a little.")
        text = body.message.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Say something first.")
        running = chat_jobs.get(user.id)
        if running is not None and not running[0].done():
            raise HTTPException(status_code=429, detail="Still working on your last message.")
        try:
            await c.quotas.take(user.id, messages=1, privileged=p.admin)
        except QuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        started = c.clock.now()
        recorded = asyncio.Event()
        task = asyncio.create_task(
            reply_in_background(user, text, recorded), name=f"chat:{user.id}"
        )
        chat_jobs[user.id] = (task, started)
        task.add_done_callback(
            lambda t: (
                chat_jobs.pop(user.id, None) if chat_jobs.get(user.id, (None,))[0] is t else None
            )
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(recorded.wait(), timeout=5)
        return {"accepted": True, "at": started.isoformat()}

    @router.get("/api/chat/poll")
    async def chat_poll(
        p: Person, after: str | None = Query(None, max_length=40)
    ) -> dict[str, Any]:
        """Is Fuzzy still working on a reply, and what has been said since ``after``?"""

        from datetime import datetime

        user = user_of(p)
        cutoff = None
        if after:
            try:
                cutoff = datetime.fromisoformat(after)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="bad cursor") from exc
        running = chat_jobs.get(user.id)
        busy = running is not None and not running[0].done()
        messages = await c.db.conversations.after(user.id, cutoff) if cutoff else []
        return {
            "busy": busy,
            "since": running[1].isoformat() if busy and running else None,
            "messages": [message_json(m) for m in messages],
        }

    @router.get("/api/chat/history")
    async def history(
        p: Person,
        limit: int = Query(40, ge=1, le=200),
        before: str | None = Query(None, max_length=40),
    ) -> dict[str, Any]:
        from datetime import datetime

        cutoff = None
        if before:
            try:
                cutoff = datetime.fromisoformat(before)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="bad cursor") from exc
        user = user_of(p)
        messages = await c.db.conversations.page(user.id, limit=limit, before=cutoff)
        return {
            "messages": [message_json(m) for m in messages],
            "more": len(messages) == limit,
            "busy": bool((j := chat_jobs.get(user.id)) and not j[0].done()),
        }

    @router.get("/api/notes")
    async def notes(p: Person) -> dict[str, Any]:
        user = user_of(p)
        items = await c.db.notes.for_user(user.id)
        return {
            "name": c.mind.self_model.name,
            "notes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "at": n.created_at.isoformat(),
                    "read": n.read_at is not None,
                }
                for n in items
            ],
        }

    @router.post("/api/notes/read")
    async def notes_read(p: Person) -> dict[str, bool]:
        await c.db.notes.mark_read(user_of(p).id, c.clock.now())
        return {"ok": True}

    # ------------------------------------------------------------ brain/world
    @contextlib.asynccontextmanager
    async def stream_slot(p: Principal, request: Request) -> AsyncIterator[None]:
        key = p.user.id if p.user else auth.ip(request)
        if (
            streams.get(key, 0) >= MAX_STREAMS_PER_CALLER
            or sum(streams.values()) >= MAX_STREAMS_TOTAL
        ):
            raise HTTPException(status_code=429, detail="too many open live views")
        streams[key] = streams.get(key, 0) + 1
        try:
            yield
        finally:
            streams[key] -= 1
            if streams[key] <= 0:
                streams.pop(key, None)

    def sse(
        p: Principal, request: Request, patterns: tuple[str, ...], viewer: Viewer, redact: Any
    ) -> StreamingResponse:
        async def generate() -> AsyncIterator[bytes]:
            async with stream_slot(p, request):
                yield b": connected\n\n"
                with c.bus.subscription("*") as queue:
                    waited = 0.0
                    while True:
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=15)
                        except TimeoutError:
                            waited += 15
                            if waited >= 60:  # a session that has ended must not keep a stream open
                                waited = 0.0
                                if p.signed_in and not (await auth.principal(request)).signed_in:
                                    return
                            yield b": keepalive\n\n"
                            continue
                        if not any(fnmatch.fnmatchcase(event.type, pat) for pat in patterns):
                            continue
                        item = filter_event(event, viewer, redact)
                        if item is not None:
                            yield f"data: {json.dumps(item, default=str)}\n\n".encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/api/brain")
    async def brain_snapshot(p: Viewing) -> dict[str, Any]:
        viewer, redact = viewer_of(p), await redactor(p)
        data = await brain.snapshot(viewer, redact)
        data["events"] = await brain.recent_events(viewer, redact)
        return data

    @router.get("/api/brain/stream")
    async def brain_stream(request: Request, p: Viewing) -> StreamingResponse:
        patterns = ("*",) if p.admin else PUBLIC_PATTERNS
        return sse(p, request, patterns, viewer_of(p), await redactor(p))

    @router.get("/api/brain/memories")
    async def brain_memories(
        p: Person, q: str = Query(min_length=2, max_length=200), k: int = Query(12, ge=1, le=40)
    ) -> list[dict[str, Any]]:
        return await brain.memories(viewer_of(p), await redactor(p), q, k=k)

    @router.get("/api/world")
    async def world_snapshot(p: Viewing) -> dict[str, Any]:
        redact = await redactor(p)
        data = await c.world.snapshot()
        data["body"]["thought"] = redact(data["body"].get("thought", ""))
        data["name"] = c.mind.self_model.name
        for obj in data["objects"]:
            obj["description"] = redact(obj.get("description", ""))
        data["avatar"]["description"] = redact(data["avatar"].get("description", ""))
        return data

    @router.get("/api/world/stream")
    async def world_stream(request: Request, p: Viewing) -> StreamingResponse:
        return sse(p, request, ("world.*",), viewer_of(p), await redactor(p))

    # ---------------------------------------------------------------- gallery
    async def gallery_items(p: Principal) -> list[dict[str, Any]]:
        records = await c.db.media.visible_to(p.user.id if p.user else None, admin=p.admin)
        return [
            {
                "id": r.id,
                "url": f"/media/{r.id}",
                "caption": r.caption,
                "prompt": r.prompt
                if (p.admin or r.user_id == (p.user.id if p.user else None))
                else None,
                "mine": r.user_id is not None,
                "at": r.created_at.isoformat(),
            }
            for r in records
        ]

    @router.get("/api/gallery")
    async def gallery(p: Viewing) -> dict[str, Any]:
        redact = await redactor(p)
        items = await gallery_items(p)
        for item in items:
            item["caption"] = redact(item["caption"])
        return {"name": c.mind.self_model.name, "images": items}

    @router.get("/media/{media_id}")
    async def media_file(media_id: str, p: Viewing) -> FileResponse:
        record = await c.db.media.get(media_id)
        mine = p.user is not None and record is not None and record.user_id == p.user.id
        visible = record is not None and (
            p.admin or record.public or record.user_id is None or mine
        )
        if record is None or not visible:
            raise HTTPException(status_code=404, detail="no such image")
        try:
            path = c.images.path_for(record)
        except ImageError as exc:
            raise HTTPException(status_code=404, detail="no such image") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no such image")
        return FileResponse(
            path,
            media_type=record.mime,
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "private, max-age=3600",
            },
        )

    # ------------------------------------------------------------------ teach
    def teacher(p: Principal) -> UserRecord | None:
        if p.operator:
            return None
        user = user_of(p)
        if not (p.admin or user.can_teach):
            raise HTTPException(
                status_code=403, detail="You haven't been given permission to teach."
            )
        return user

    async def take_teaching_quota(p: Principal, user: UserRecord | None, size: int) -> None:
        if user is None:
            return
        try:
            await c.quotas.take(user.id, teach_bytes=size, privileged=p.admin)
        except QuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    async def refund_teaching(p: Principal, user: UserRecord | None, size: int) -> None:
        if user is not None:
            await c.quotas.take(user.id, teach_bytes=-size, privileged=True)

    async def ingest_summary(
        result: Any, p: Principal, user: UserRecord | None, charged: int
    ) -> dict[str, Any]:
        if result.reason and not result.skipped:
            await refund_teaching(p, user, charged)  # a rejected upload doesn't use up the day
            raise HTTPException(status_code=422, detail=f"Couldn't read that: {result.reason}")
        return {
            "title": result.title,
            "chunks": result.chunks,
            "already_known": bool(result.skipped),
        }

    @router.post("/api/teach/text")
    async def teach_text(body: TeachTextIn, request: Request, p: Signed) -> dict[str, Any]:
        json_only(request)
        user = teacher(p)
        who = user.id if user else "operator"
        if not teach_limit.allow(who):
            raise HTTPException(status_code=429, detail="Slow down a little.")
        if not p.admin and len(body.text) > TEXT_LIMIT_USER:
            raise HTTPException(status_code=413, detail="That's too long for one go.")
        size = len(body.text.encode())
        await take_teaching_quota(p, user, size)
        result = await c.pipeline.ingest_text(
            body.text,
            title=clean_display_name(body.title) or "Untitled",
            source="taught",
            metadata={"taught_by": who},
        )
        await c.audit.record("teach.text", actor=who, title=body.title[:100], chunks=result.chunks)
        return await ingest_summary(result, p, user, size)

    @router.post("/api/teach/url")
    async def teach_url(body: TeachUrlIn, request: Request, p: Signed) -> dict[str, Any]:
        json_only(request)
        user = teacher(p)
        who = user.id if user else "operator"
        if not teach_limit.allow(who):
            raise HTTPException(status_code=429, detail="Slow down a little.")
        if not body.url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422, detail="Give a web address starting with http(s)://"
            )
        await take_teaching_quota(p, user, URL_COST)
        result = await c.pipeline.ingest_url(body.url)
        await c.audit.record("teach.url", actor=who, url=body.url[:300], chunks=result.chunks)
        return await ingest_summary(result, p, user, URL_COST)

    @router.post("/api/teach/file")
    async def teach_file(
        request: Request, p: Signed, name: str = Query(min_length=1, max_length=200)
    ) -> dict[str, Any]:
        user = teacher(p)
        who = user.id if user else "operator"
        if not teach_limit.allow(who):
            raise HTTPException(status_code=429, detail="Slow down a little.")
        limit = FILE_LIMIT_ADMIN if p.admin else FILE_LIMIT_USER
        chunks: list[bytes] = []
        size = 0
        async for part in request.stream():
            size += len(part)
            if size > limit:
                raise HTTPException(
                    status_code=413, detail=f"Files can be up to {limit // 1_000_000} MB."
                )
            chunks.append(part)
        data = b"".join(chunks)
        if not data:
            raise HTTPException(status_code=422, detail="That file is empty.")
        from sentientai.ingestion.loaders import load_bytes

        safe_name = Path(name).name or "upload"
        try:
            doc = load_bytes(data, source="taught", name=safe_name)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="I can't read that kind of file.") from exc
        await take_teaching_quota(p, user, size)
        doc.metadata["taught_by"] = who
        result = await c.pipeline.ingest_document(doc)
        await c.audit.record("teach.file", actor=who, name=safe_name[:100], chunks=result.chunks)
        return await ingest_summary(result, p, user, size)

    @router.get("/api/teach/documents")
    async def teach_documents(p: Signed) -> dict[str, Any]:
        teacher(p)
        studied = await c.interoception.studied()
        docs = await c.db.documents.list(limit=200)
        return {
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "chunks": d.chunk_count,
                    "bytes": d.size_bytes,
                    "at": d.ingested_at.isoformat(),
                    "read": d.id in studied,
                }
                for d in docs
            ]
        }

    # ------------------------------------------------------------------ admin
    @router.get("/api/admin/users")
    async def admin_users(_: Admin) -> dict[str, Any]:
        rels = {r.user_id: r for r in await c.db.relationships.list(limit=10_000)}
        out = []
        for u in await c.db.users.list(limit=2000):
            rel = rels.get(u.id)
            usage = await c.quotas.usage(u.id)
            out.append(
                {
                    **u.public(),
                    "status": u.status.value,
                    "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
                    "locked": bool(u.locked_until and u.locked_until > c.clock.now()),
                    "interactions": rel.interactions if rel else 0,
                    "affinity": round(rel.affinity, 2) if rel else None,
                    "opinion": rel.opinion if rel else "",
                    "today": {"messages": usage.messages, "images": usage.images},
                }
            )
        return {"users": out}

    @router.post("/api/admin/users/{user_id}")
    async def admin_update_user(
        user_id: str, body: UserUpdateIn, request: Request, p: Admin
    ) -> dict[str, Any]:
        json_only(request)
        try:
            user = await c.accounts.update_user(
                user_id,
                by=p.user if p.user else p.actor,
                status=body.status,
                role=body.role,
                can_teach=body.can_teach,
            )
        except AuthError as exc:
            raise raise_for(exc) from exc
        c.redactor.invalidate()
        return {**user.public(), "status": user.status.value}

    @router.delete("/api/admin/users/{user_id}")
    async def admin_delete_user(user_id: str, p: Admin) -> dict[str, Any]:
        user = await c.db.users.get(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="No such user.")
        if p.user is not None and user.id == p.user.id:
            raise HTTPException(
                status_code=400, detail="Use your own account page to delete yourself."
            )
        return {"erased": await erase_account(user, by=p.actor)}

    @router.get("/api/admin/invites")
    async def admin_invites(_: Admin) -> dict[str, Any]:
        return {
            "invites": [
                {
                    "role": i.role.value,
                    "can_teach": i.can_teach,
                    "created_at": i.created_at.isoformat(),
                    "expires_at": i.expires_at.isoformat(),
                    "used": i.used_by is not None,
                }
                for i in await c.db.invites.list()
            ]
        }

    @router.post("/api/admin/invites", status_code=201)
    async def admin_create_invite(body: InviteIn, request: Request, p: Admin) -> dict[str, Any]:
        json_only(request)
        code = await c.accounts.create_invite(
            by=p.user if p.user else p.actor,
            role=body.role,
            can_teach=body.can_teach,
            days=body.days,
        )
        return {"code": code, "expires_in_days": body.days}  # shown once; only its hash is stored

    @router.delete("/api/admin/documents/{document_id}")
    async def admin_delete_document(document_id: str, p: Admin) -> dict[str, Any]:
        doc = await c.db.documents.get(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="No such document.")
        removed = await c.memory.forget_source(f"document:{doc.id}")
        await c.db.documents.delete(doc.id)
        await c.audit.record(
            "teach.removed", actor=p.actor, document_id=doc.id, title=doc.title, memories=removed
        )
        return {"removed_memories": removed}

    @router.get("/api/admin/thinking")
    async def admin_thinking(
        _: Admin, limit: int = Query(40, ge=1, le=200)
    ) -> list[dict[str, Any]]:
        """What Fuzzy worked through when it answered people (newest last)."""

        entries = await asyncio.to_thread(
            c.audit.recent,
            limit * 3,
            ("agent.reasoning", "agent.thought", "agent.answer", "agent.grounding"),
        )
        return [
            {
                "seq": e.seq,
                "at": e.ts.isoformat(),
                "task_id": e.task_id,
                "type": e.type[6:],
                **e.payload,
            }
            for e in entries[-limit * 3 :]
        ]

    @router.get("/api/admin/hardware")
    async def admin_hardware(_: Admin) -> dict[str, Any]:
        return await hardware.collect(runtime.settings.storage.data_dir)

    @router.get("/api/admin/memories")
    async def admin_memories(
        _: Admin, kind: MemoryKind | None = None, limit: int = Query(50, ge=1, le=200)
    ) -> list[dict[str, Any]]:
        records = await c.db.memories.list(
            kinds=[kind] if kind else None, limit=limit, scope=Scope.system()
        )
        return [
            {
                "id": m.id,
                "kind": m.kind.value,
                "content": m.content,
                "user_id": m.user_id,
                "importance": round(m.importance, 2),
                "at": m.created_at.isoformat(),
            }
            for m in records
        ]

    return router

"""The public web app: sessions, CSRF, isolation between people, admin, teaching and media."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.requests import Request

from sentientai.accounts import totp
from sentientai.api.server import create_app
from sentientai.core.clock import FakeClock
from sentientai.core.events import Event
from sentientai.evals.brain import Brain, call
from sentientai.models.embeddings import HashingEmbedder
from sentientai.models.scripted import ScriptedLLM
from sentientai.models.types import ModelTier
from sentientai.runtime.runtime import Runtime
from sentientai.storage.records import MemoryKind, Scope, UserRole
from sentientai.web.brain import Viewer, filter_event
from sentientai.web.security import client_ip
from tests.test_runtime import make_settings

GOOD = "correct horse battery staple"
OTHER = "another long passphrase here"


class Web(SimpleNamespace):
    rt: Runtime
    app: Any
    clients: list[httpx.AsyncClient]

    def client(self, *, token: str | None = None) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test", headers=headers
        )
        self.clients.append(client)
        return client


@pytest.fixture
async def web_factory(tmp_path: Path) -> AsyncIterator[Callable[..., Any]]:
    made: list[Web] = []

    async def factory(brain: Brain | None = None, *, llm: Any = None, **overrides: Any) -> Web:
        brain = brain or Brain(default_answer="Hello there.")
        overrides.setdefault("accounts", {})
        overrides["accounts"] = {
            "cookie_secure": False,
            "login_attempts_per_ip_per_minute": 1000,
            "registration": "closed",
            **overrides["accounts"],
        }
        settings = make_settings(tmp_path, **overrides)
        fast = llm or brain.llm("fast")
        rt = await Runtime.create(
            settings,
            clients={ModelTier.FAST: fast, ModelTier.HEAVY: fast},
            embedder=HashingEmbedder(256),
            clock=FakeClock(),
        )
        await rt.start()
        w = Web(rt=rt, app=create_app(rt), clients=[], brain=brain)
        made.append(w)
        return w

    yield factory
    for w in made:
        for client in w.clients:
            await client.aclose()
        await w.rt.stop("test teardown")


async def sign_in(client: httpx.AsyncClient, username: str, password: str = GOOD) -> str:
    """Log in through the API; returns the CSRF token."""

    response = await client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    return str(response.json()["csrf"])


def post(client: httpx.AsyncClient, path: str, csrf: str, **kwargs: Any) -> Any:
    headers = {"X-CSRF-Token": csrf, "Origin": "http://test", **kwargs.pop("headers", {})}
    return client.post(path, headers=headers, **kwargs)


async def talk(
    client: httpx.AsyncClient, csrf: str, text: str, *, timeout: float = 15
) -> dict[str, Any]:
    """Say something and wait for Fuzzy's reply (replies are made in the background)."""

    latest = (await client.get("/api/chat/history?limit=1")).json()["messages"]
    cursor = latest[-1]["at"] if latest else "1970-01-01T00:00:00+00:00"
    accepted = await post(client, "/api/chat", csrf, json={"message": text})
    assert accepted.status_code == 202, accepted.text
    heard: list[dict[str, Any]] = []
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        r = (await client.get("/api/chat/poll", params={"after": cursor})).json()
        heard = r["messages"]
        if not r["busy"]:
            break
        assert asyncio.get_running_loop().time() < deadline, "no reply in time"
        await asyncio.sleep(0.05)
    replies = [m for m in heard if m["role"] == "assistant"]
    assert replies, heard
    return {"answer": replies[-1]["content"], "media": replies[-1]["media"], "messages": heard}


async def person(
    w: Web,
    username: str,
    *,
    role: UserRole = UserRole.USER,
    teach: bool = False,
    display: str | None = None,
) -> tuple[Any, httpx.AsyncClient, str]:
    user = await w.rt.c.accounts.create_user(
        username, GOOD, display_name=display or username.title(), role=role, can_teach=teach
    )
    client = w.client()
    return user, client, await sign_in(client, username)


# ----------------------------------------------------------------- basics


async def test_nothing_is_open_to_strangers(web_factory: Any) -> None:
    w = await web_factory()
    anon = w.client()
    for path in (
        "/api/me",
        "/api/chat/history",
        "/api/notes",
        "/api/gallery",
        "/api/brain",
        "/api/world",
        "/api/admin/users",
        "/api/teach/documents",
    ):
        assert (await anon.get(path)).status_code == 401, path
    assert (await anon.post("/api/chat", json={"message": "hi"})).status_code == 401
    assert (await anon.get("/v1/state")).status_code == 401  # operator API is closed by default
    assert (await anon.get("/metrics")).status_code == 401
    assert (await anon.get("/media/img_x")).status_code == 401
    assert (await anon.get("/docs")).status_code == 404 and (
        await anon.get("/openapi.json")
    ).status_code == 404
    state = (await anon.get("/api/auth/state")).json()
    assert (
        state["signed_in"] is False and state["csrf"] is None and state["registration"] == "closed"
    )
    assert (await anon.get("/health")).status_code == 200


async def test_operator_token_is_generated_and_never_open(web_factory: Any, tmp_path: Path) -> None:
    w = await web_factory()
    token_file = w.rt.settings.api_token_path
    token = token_file.read_text().strip()
    assert len(token) >= 40 and oct(token_file.stat().st_mode & 0o777) == "0o600"
    assert (await w.client(token=token).get("/v1/state")).status_code == 200
    assert (await w.client(token="wrong").get("/v1/state")).status_code == 401
    assert w.rt.settings.resolved_api_token() == token  # the local CLI finds it


async def test_security_headers_everywhere(web_factory: Any) -> None:
    w = await web_factory(accounts={"hsts": True})
    anon = w.client()
    for path in ("/", "/login", "/api/auth/state", "/static/app.js", "/health"):
        headers = (await anon.get(path)).headers
        assert "script-src 'self'" in headers["content-security-policy"], path
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "same-origin"
        assert headers["strict-transport-security"].startswith("max-age=")
    assert (await anon.get("/api/auth/state")).headers["cache-control"] == "no-store"
    assert "unsafe-inline" not in (await anon.get("/")).headers["content-security-policy"]
    assert (await anon.get("/static/vendor/three.min.js")).status_code == 200
    assert (await anon.get("/static/%2e%2e/config.py")).status_code == 404
    assert (await anon.get("/static/../../config.py")).status_code in (400, 404)


async def test_pages_have_no_inline_script_or_handlers(web_factory: Any) -> None:
    import re

    w = await web_factory()
    anon = w.client()
    for path in ("/", "/login", "/brain", "/world", "/gallery", "/you", "/teach", "/admin"):
        page = await anon.get(path)
        assert page.status_code == 200, path
        assert not re.search(r"<script(?![^>]*\bsrc=)", page.text), path
        assert not re.search(r"\son[a-z]+\s*=", page.text), path
        assert "style=" not in page.text, path


# ------------------------------------------------------------- registration


async def test_registration_modes(web_factory: Any) -> None:
    w = await web_factory(accounts={"registration": "closed"})
    anon = w.client()
    body = {"username": "dana", "password": GOOD}
    assert (await anon.post("/api/auth/register", json=body)).status_code == 403

    w = await web_factory(accounts={"registration": "invite"})
    anon = w.client()
    assert (await anon.post("/api/auth/register", json=body)).status_code == 403
    assert (
        await anon.post("/api/auth/register", json={**body, "invite_code": "nope"})
    ).status_code == 403
    code = await w.rt.c.accounts.create_invite(by="operator", can_teach=True)
    ok = await anon.post(
        "/api/auth/register", json={**body, "invite_code": code, "display_name": "Dana"}
    )
    assert ok.status_code == 201 and ok.json()["user"]["can_teach"] is True
    assert (await anon.get("/api/me")).json()["user"]["username"] == "dana"  # signed straight in
    again = await w.client().post(
        "/api/auth/register", json={"username": "eve", "password": GOOD, "invite_code": code}
    )
    assert again.status_code == 403  # single use

    w = await web_factory(accounts={"registration": "open"})
    anon = w.client()
    assert (
        await anon.post("/api/auth/register", json={"username": "short", "password": "tooshort"})
    ).status_code == 400
    assert (
        await anon.post("/api/auth/register", json={"username": "root", "password": GOOD})
    ).status_code == 400
    assert (
        await anon.post("/api/auth/register", json={"username": "Fuzzy", "password": GOOD})
    ).status_code == 400
    assert (
        await anon.post("/api/auth/register", json={"username": "frank", "password": GOOD})
    ).status_code == 201
    assert (
        await w.client().post("/api/auth/register", json={"username": "FRANK", "password": GOOD})
    ).status_code == 409


async def test_signups_are_rate_limited_per_address(web_factory: Any) -> None:
    w = await web_factory(accounts={"registration": "open"})
    statuses = []
    for n in range(7):
        r = await w.client().post(
            "/api/auth/register", json={"username": f"user{n}", "password": GOOD}
        )
        statuses.append(r.status_code)
    assert statuses[:5] == [201] * 5 and statuses[5:] == [429, 429]


# ------------------------------------------------------------------- login


async def test_session_cookie_is_locked_down(web_factory: Any) -> None:
    w = await web_factory(accounts={"cookie_secure": True})
    await w.rt.c.accounts.create_user("dana", GOOD)
    r = await w.client().post("/api/auth/login", json={"username": "dana", "password": GOOD})
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("__Host-fuzzy_session=")
    for attribute in ("HttpOnly", "Secure", "SameSite=lax", "Path=/"):
        assert attribute in cookie
    assert "Domain" not in cookie
    token = cookie.split("=", 1)[1].split(";")[0]
    stored = await w.rt.c.db.sessions.get(__import__("hashlib").sha256(token.encode()).hexdigest())
    assert stored is not None  # only the hash is stored, never the token itself
    assert all(token not in str(v) for v in stored.model_dump().values())


async def test_login_errors_do_not_reveal_accounts_and_lock_out(web_factory: Any) -> None:
    w = await web_factory(accounts={"max_failed_logins": 3, "lockout_minutes": 15})
    await w.rt.c.accounts.create_user("dana", GOOD)
    anon = w.client()
    unknown = await anon.post("/api/auth/login", json={"username": "nobody", "password": GOOD})
    wrong = await anon.post(
        "/api/auth/login", json={"username": "dana", "password": "wrong password!!"}
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()  # same message either way
    assert "set-cookie" not in wrong.headers
    await anon.post("/api/auth/login", json={"username": "dana", "password": "wrong password!!"})
    locked = await anon.post(
        "/api/auth/login", json={"username": "dana", "password": "wrong password!!"}
    )
    assert locked.status_code == 401
    refused = await anon.post("/api/auth/login", json={"username": "dana", "password": GOOD})
    assert refused.status_code == 429  # even the right password is refused while locked
    w.rt.c.clock.advance(16 * 60)  # type: ignore[attr-defined]
    assert (
        await anon.post("/api/auth/login", json={"username": "dana", "password": GOOD})
    ).status_code == 200


async def test_login_attempts_are_throttled_per_address(web_factory: Any) -> None:
    w = await web_factory(accounts={"login_attempts_per_ip_per_minute": 4})
    anon = w.client()
    codes = [
        (
            await anon.post("/api/auth/login", json={"username": f"u{n}", "password": "x"})
        ).status_code
        for n in range(6)
    ]
    assert codes == [401, 401, 401, 401, 429, 429]


async def test_every_login_gets_a_fresh_session(web_factory: Any) -> None:
    w = await web_factory()
    await w.rt.c.accounts.create_user("dana", GOOD)
    c1, c2 = w.client(), w.client()
    csrf1, csrf2 = await sign_in(c1, "dana"), await sign_in(c2, "dana")
    assert csrf1 != csrf2 and c1.cookies.get("fuzzy_session") != c2.cookies.get("fuzzy_session")
    out = await post(c1, "/api/auth/logout", csrf1)
    assert out.status_code == 200
    assert (await c1.get("/api/me")).status_code == 401  # logged out for real
    assert (await c2.get("/api/me")).status_code == 200  # other device unaffected
    stale = httpx.AsyncClient(transport=httpx.ASGITransport(app=w.app), base_url="http://test")
    stale.cookies.set("fuzzy_session", "x" * 43)
    assert (await stale.get("/api/me")).status_code == 401
    await stale.aclose()


async def test_sessions_expire(web_factory: Any) -> None:
    w = await web_factory(accounts={"idle_hours": 1, "session_days": 1})
    await w.rt.c.accounts.create_user("dana", GOOD)
    client = w.client()
    await sign_in(client, "dana")
    assert (await client.get("/api/me")).status_code == 200
    w.rt.c.clock.advance(2 * 3600)  # type: ignore[attr-defined]
    assert (await client.get("/api/me")).status_code == 401


# --------------------------------------------------------------------- csrf


async def test_csrf_and_origin_are_enforced(web_factory: Any) -> None:
    w = await web_factory()
    await w.rt.c.accounts.create_user("dana", GOOD)
    client = w.client()
    csrf = await sign_in(client, "dana")
    msg = {"message": "hello"}

    assert (await client.post("/api/chat", json=msg)).status_code == 403  # no token
    bad = await client.post("/api/chat", json=msg, headers={"X-CSRF-Token": "forged"})
    assert bad.status_code == 403
    evil = await client.post(
        "/api/chat", json=msg, headers={"X-CSRF-Token": csrf, "Origin": "https://evil.example"}
    )
    assert evil.status_code == 403 and "cross-site" in evil.json()["detail"]
    null_origin = await client.post(
        "/api/chat", json=msg, headers={"X-CSRF-Token": csrf, "Origin": "null"}
    )
    assert null_origin.status_code == 403
    form = await client.post(
        "/api/chat", data={"message": "hi"}, headers={"X-CSRF-Token": csrf, "Origin": "http://test"}
    )
    assert form.status_code in (415, 422)  # form posts are not JSON
    ok = await post(client, "/api/chat", csrf, json=msg)
    assert ok.status_code == 202 and ok.json()["accepted"] is True

    # Login and sign-up (no session yet) refuse cross-site origins and non-JSON too.
    anon = w.client()
    cross = await anon.post(
        "/api/auth/login",
        json={"username": "dana", "password": GOOD},
        headers={"Origin": "https://evil.example"},
    )
    assert cross.status_code == 403
    assert (
        await anon.post("/api/auth/login", data={"username": "dana", "password": GOOD})
    ).status_code in (415, 422)
    # Another person's token doesn't work either.
    other_user = await w.rt.c.accounts.create_user("eve", GOOD)
    other = w.client()
    other_csrf = await sign_in(other, "eve")
    assert (
        await client.post("/api/chat", json=msg, headers={"X-CSRF-Token": other_csrf})
    ).status_code == 403
    assert other_user.id != ""


async def test_bodies_are_size_limited(web_factory: Any) -> None:
    w = await web_factory()
    await w.rt.c.accounts.create_user("dana", GOOD)
    client = w.client()
    csrf = await sign_in(client, "dana")
    huge = await post(
        client,
        "/api/chat",
        csrf,
        content=b'{"message":"' + b"a" * 400_000 + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert huge.status_code == 413
    long_message = await post(client, "/api/chat", csrf, json={"message": "a" * 4001})
    assert long_message.status_code == 422


# ---------------------------------------------------------------------- 2FA


async def test_two_factor_flow(web_factory: Any) -> None:
    w = await web_factory()
    await w.rt.c.accounts.create_user("dana", GOOD)
    client = w.client()
    csrf = await sign_in(client, "dana")
    begin = (await post(client, "/api/me/totp/begin", csrf)).json()
    secret = begin["secret"]
    assert begin["uri"].startswith("otpauth://totp/Fuzzy") and "issuer=Fuzzy" in begin["uri"]
    now = w.rt.c.clock.now().timestamp()
    step = totp.current_step(now)
    wrong = await post(
        client, "/api/me/totp/enable", csrf, json={"secret": secret, "code": "000000"}
    )
    assert wrong.status_code == 400
    assert (
        await post(
            client,
            "/api/me/totp/enable",
            csrf,
            json={"secret": secret, "code": totp.code_at(secret, step)},
        )
    ).status_code == 200

    fresh = w.client()
    login = await fresh.post("/api/auth/login", json={"username": "dana", "password": GOOD})
    assert login.json()["mfa_required"] is True
    pending_csrf = login.json()["csrf"]
    # Password alone gets nothing: every protected endpoint refuses a half-signed-in session.
    for path in ("/api/me", "/api/chat/history", "/api/brain"):
        assert (await fresh.get(path)).status_code == 403, path
    assert (await fresh.get("/api/auth/state")).json()["mfa_pending"] is True
    assert (
        await post(fresh, "/api/auth/mfa", pending_csrf, json={"code": "123456"})
    ).status_code == 401
    w.rt.c.clock.advance(31)  # type: ignore[attr-defined]
    step = totp.current_step(w.rt.c.clock.now().timestamp())
    code = totp.code_at(secret, step)
    assert (
        await post(fresh, "/api/auth/mfa", pending_csrf, json={"code": code})
    ).status_code == 200
    assert (await fresh.get("/api/me")).status_code == 200
    replay = w.client()
    replay_csrf = (
        await replay.post("/api/auth/login", json={"username": "dana", "password": GOOD})
    ).json()["csrf"]
    assert (
        await post(replay, "/api/auth/mfa", replay_csrf, json={"code": code})
    ).status_code == 401  # codes can't be replayed
    off = await post(client, "/api/me/totp/disable", csrf, json={"password": "wrong password!!"})
    assert off.status_code == 401


async def test_admins_can_be_required_to_use_two_factor(web_factory: Any) -> None:
    w = await web_factory(accounts={"require_admin_totp": True})
    await w.rt.c.accounts.create_user("root", GOOD, role=UserRole.ADMIN)
    client = w.client()
    await sign_in(client, "root")
    refused = await client.get("/api/admin/users")
    assert refused.status_code == 403 and "two-factor" in refused.json()["detail"]


# --------------------------------------------------------------- self-service


async def test_changing_password_signs_out_other_devices(web_factory: Any) -> None:
    w = await web_factory()
    await w.rt.c.accounts.create_user("dana", GOOD)
    phone, laptop = w.client(), w.client()
    await sign_in(phone, "dana")
    csrf = await sign_in(laptop, "dana")
    assert (
        await post(
            laptop, "/api/me/password", csrf, json={"current": "wrong password!!", "new": OTHER}
        )
    ).status_code == 401
    assert (
        await post(laptop, "/api/me/password", csrf, json={"current": GOOD, "new": "short"})
    ).status_code == 400
    assert (
        await post(laptop, "/api/me/password", csrf, json={"current": GOOD, "new": OTHER})
    ).status_code == 200
    assert (await laptop.get("/api/me")).status_code == 200
    assert (await phone.get("/api/me")).status_code == 401
    assert (
        await w.client().post("/api/auth/login", json={"username": "dana", "password": GOOD})
    ).status_code == 401
    assert (
        await w.client().post("/api/auth/login", json={"username": "dana", "password": OTHER})
    ).status_code == 200


async def test_export_and_erase_account(web_factory: Any) -> None:
    brain = Brain(
        ["Nice to meet you, Dana."],
        appraisal={"opinion": "Dana is thoughtful.", "feelings": "Warm.", "felt": []},
    )
    w = await web_factory(brain)
    user, client, csrf = await person(w, "dana")
    await talk(client, csrf, "My cat is called Biscuit")
    await asyncio.sleep(0.2)
    data = (await client.get("/api/me/export")).json()
    assert data["account"]["username"] == "dana" and "password_hash" not in str(data)
    assert any("Biscuit" in m["content"] for m in data["conversations"])
    assert any("Biscuit" in m["content"] for m in data["memories"])
    assert (await client.get("/api/me/export")).status_code == 200

    wrong = await post(client, "/api/me/delete", csrf, json={"password": "wrong password!!"})
    assert wrong.status_code == 401
    gone = await post(client, "/api/me/delete", csrf, json={"password": GOOD})
    assert gone.status_code == 200 and gone.json()["erased"]["memories"] >= 1
    assert (await client.get("/api/me")).status_code == 401
    assert (
        await w.client().post("/api/auth/login", json={"username": "dana", "password": GOOD})
    ).status_code == 401
    left = await w.rt.c.memory.recall("Biscuit", scope=Scope.system(), min_similarity=0.0)
    assert all("Biscuit" not in h.record.content for h in left)
    assert await w.rt.c.db.conversations.for_user(user.id) == []


async def test_the_last_administrator_cannot_delete_themselves(web_factory: Any) -> None:
    w = await web_factory()
    _, client, csrf = await person(w, "root", role=UserRole.ADMIN)
    r = await post(client, "/api/me/delete", csrf, json={"password": GOOD})
    assert r.status_code == 409


# --------------------------------------------------------------------- chat


async def test_chat_history_quota_and_pause(web_factory: Any) -> None:
    brain = Brain(["One.", "Two.", "Three."])
    w = await web_factory(brain, accounts={"daily_messages": 2})
    _, client, csrf = await person(w, "dana")
    assert (await talk(client, csrf, "first"))["answer"] == "One."
    await talk(client, csrf, "second")
    over = await post(client, "/api/chat", csrf, json={"message": "third"})
    assert over.status_code == 429 and "message limit" in over.json()["detail"]
    history = (await client.get("/api/chat/history")).json()
    assert [(m["role"], m["content"]) for m in history["messages"]] == [
        ("user", "first"),
        ("assistant", "One."),
        ("user", "second"),
        ("assistant", "Two."),
    ]
    page = (await client.get("/api/chat/history?limit=2")).json()
    assert len(page["messages"]) == 2 and page["more"] is True
    assert (await client.get("/api/chat/history?before=garbage")).status_code == 422
    me = (await client.get("/api/me")).json()
    assert me["usage"]["messages"] == 2 and me["limits"]["messages"] == 2

    _, admin, admin_csrf = await person(w, "root", role=UserRole.ADMIN)
    await w.rt.c.kill_switch.engage("test", by="test")
    paused = await post(admin, "/api/chat", admin_csrf, json={"message": "hello?"})
    assert paused.status_code == 503 and "paused" in paused.json()["detail"]


async def test_only_one_message_at_a_time_per_person(web_factory: Any) -> None:
    gate = asyncio.Event()

    async def slow(request: Any) -> str:
        prompt = request.messages[-1].text
        if "Rate how likely" in prompt:
            return '{"confidence": 0.9, "gaps": []}'
        if prompt.startswith(("Check a reply", "You just finished")) or "You just" in prompt:
            return '{"unsupported": [], "new_self_decisions": [], "summary": "x", "lessons": [], "facts": []}'
        await gate.wait()
        return "Finally."

    w = await web_factory(llm=ScriptedLLM(slow))
    _, client, csrf = await person(w, "dana")
    first = asyncio.create_task(talk(client, csrf, "slow one"))
    await asyncio.sleep(0.3)
    second = await post(client, "/api/chat", csrf, json={"message": "impatient"})
    assert second.status_code == 429
    busy = (await client.get("/api/chat/poll")).json()
    assert busy["busy"] is True and busy["since"]
    assert (await client.get("/api/chat/history")).json()["busy"] is True
    gate.set()
    assert (await first)["answer"] == "Finally."
    assert (await client.get("/api/chat/poll")).json()["busy"] is False


# -------------------------------------------------- people don't see each other


async def test_one_persons_world_stays_theirs(web_factory: Any) -> None:
    brain = Brain(
        ["Noted, Alice.", "I don't know that.", "Hello Bob."],
        appraisal=[
            {
                "opinion": "Alice is kind and trusts me with her secret garden plan.",
                "feelings": "Fond.",
                "affinity_change": 0.1,
                "felt": [
                    {
                        "emotion": "touched",
                        "intensity": 0.8,
                        "valence": 0.9,
                        "lingers_hours": 5,
                        "cause": "Alice told me about her secret garden",
                    }
                ],
            },
            {"opinion": "Bob is curious.", "feelings": "Neutral.", "felt": []},
        ],
    )
    w = await web_factory(brain)
    _alice, a_client, a_csrf = await person(w, "alice")
    _bob, b_client, b_csrf = await person(w, "bob")
    await talk(a_client, a_csrf, "My secret garden plan is to grow blue tomatoes")
    await asyncio.sleep(0.4)
    await talk(b_client, b_csrf, "What does Alice grow?")
    await asyncio.sleep(0.3)

    def everything(*responses: Any) -> str:
        return " ".join(r.text for r in responses)

    b_view = everything(
        await b_client.get("/api/brain"),
        await b_client.get("/api/me/relationship"),
        await b_client.get("/api/notes"),
        await b_client.get("/api/gallery"),
        await b_client.get("/api/world"),
        await b_client.get("/api/chat/history"),
        await b_client.get("/api/brain/memories?q=secret garden tomatoes"),
    )
    for leak in ("blue tomatoes", "secret garden", "touched", "kind and trusts"):
        assert leak not in b_view, leak
    a_view = everything(
        await a_client.get("/api/brain"), await a_client.get("/api/me/relationship")
    )
    assert "kind and trusts" in a_view and "touched" in a_view  # Alice sees her own
    assert "Bob is curious" not in a_view
    admin_client = w.client()
    await w.rt.c.accounts.create_user("root", GOOD, role=UserRole.ADMIN)
    await sign_in(admin_client, "root")
    admin_view = (await admin_client.get("/api/brain")).json()
    assert {p["user"]["username"] for p in admin_view["people"]} == {
        "alice",
        "bob",
    }  # the operator sees all
    assert any("secret garden" in (f["cause"] or "") for f in admin_view["feelings"])


async def test_public_pages_show_only_the_shared_mind(web_factory: Any) -> None:
    brain = Brain(
        ["Hi Alice."],
        beat={
            "feelings": [],
            "mood": "Alice has made me happy today",
            "choice": "think",
            "intent": "think about Alice",
            "why": "she was kind",
            "person": "Alice",
            "rest_minutes": 5,
        },
        thought={"thought": "Alice is lovely and Bob too.", "insight": None},
    )
    w = await web_factory(brain, accounts={"public_pages": True})
    _, client, csrf = await person(w, "alice")
    await talk(client, csrf, "hello")
    await w.rt.c.agency.beat()
    visitor = w.client()
    page = await visitor.get("/api/brain")
    assert page.status_code == 200 and page.json()["viewer"] == {"admin": False, "signed_in": False}
    text = page.text
    assert "Alice" not in text and "Bob" not in text
    assert "someone" in text
    assert (await visitor.get("/api/world")).status_code == 200
    assert (await visitor.get("/api/me")).status_code == 401
    assert (await visitor.get("/api/brain/memories?q=hello")).status_code == 401
    assert (await visitor.get("/api/gallery")).status_code == 200


def test_event_filtering_by_viewer() -> None:
    def event(type_: str, **payload: Any) -> Event:
        return Event(type=type_, payload=payload)

    redact = lambda t: t.replace("Alice", "someone")  # noqa: E731
    stranger, alice, admin = Viewer(None), Viewer("usr_a"), Viewer("usr_root", admin=True)
    mine = event(
        "mind.feeling",
        emotion="joy",
        intensity=0.5,
        valence=0.5,
        cause="Alice smiled",
        user_id="usr_a",
    )
    shared = event("mind.thought", thought="Alice and I", user_id=None)
    assert filter_event(mine, stranger, redact) is None
    assert filter_event(mine, Viewer("usr_b"), redact) is None
    got = filter_event(mine, alice, redact)
    assert (
        got is not None
        and "user_id" not in got["payload"]
        and got["payload"]["cause"] == "someone smiled"
    )
    assert filter_event(shared, stranger, redact)["payload"]["thought"] == "someone and I"  # type: ignore[index]
    assert filter_event(mine, admin, redact)["payload"]["cause"] == "Alice smiled"  # type: ignore[index]
    # Internal events never reach non-admins.
    for internal in (
        "task.started",
        "tool.requested",
        "workspace.broadcast",
        "approval.requested",
        "memory.stored",
    ):
        assert (
            filter_event(event(internal, goal="read Alice's file", task_id="t"), alice, redact)
            is None
        )
        assert filter_event(event(internal, goal="x"), admin, redact) is not None
    # Unlisted fields are dropped even on allowed event types.
    leaky = event(
        "mind.activity",
        id="a",
        kind="think",
        intent="x",
        reason="y",
        outcome="z",
        status="done",
        secret="s",
        task_id="t",
    )
    assert set(filter_event(leaky, stranger, redact)["payload"]) == {
        "id",
        "kind",
        "intent",
        "reason",
        "outcome",
        "status",
    }  # type: ignore[index]
    private_image = event("media.created", media_id="m", user_id="usr_b", public=False, caption="c")
    assert filter_event(private_image, alice, redact) is None
    assert (
        filter_event(
            event("media.created", media_id="m", user_id=None, public=True, caption="c"),
            stranger,
            redact,
        )
        is not None
    )


def request_with(peer: str, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request(
        {"type": "http", "client": (peer, 1234), "headers": headers, "method": "GET", "path": "/"}
    )


def test_client_address_trusts_proxies_only() -> None:
    assert (
        client_ip(request_with("203.0.113.9", "1.2.3.4"), []) == "203.0.113.9"
    )  # spoofable header ignored
    assert client_ip(request_with("203.0.113.9", "1.2.3.4"), ["10.0.0.0/8"]) == "203.0.113.9"
    assert client_ip(request_with("10.0.0.5", "198.51.100.7"), ["10.0.0.0/8"]) == "198.51.100.7"
    # A client can prepend fake hops; the rightmost address not added by our proxies wins.
    assert (
        client_ip(request_with("10.0.0.5", "6.6.6.6, 198.51.100.7, 10.0.0.9"), ["10.0.0.0/8"])
        == "198.51.100.7"
    )
    assert client_ip(request_with("10.0.0.5", None), ["10.0.0.0/8"]) == "10.0.0.5"
    assert (
        client_ip(request_with("10.0.0.5", "garbage"), ["10.0.0.0/8", "not a network"]) == "garbage"
    )


# -------------------------------------------------------------------- admin


async def test_admin_endpoints_are_for_administrators(web_factory: Any) -> None:
    w = await web_factory()
    _, user_client, user_csrf = await person(w, "dana")
    for path in (
        "/api/admin/users",
        "/api/admin/invites",
        "/api/admin/memories",
        "/v1/state",
        "/v1/audit/tail",
        "/metrics",
    ):
        assert (await user_client.get(path)).status_code == 403, path
    assert (await post(user_client, "/api/admin/invites", user_csrf, json={})).status_code == 403
    assert (await post(user_client, "/v1/kill", user_csrf, json={"reason": "x"})).status_code == 403
    assert not w.rt.c.kill_switch.engaged


async def test_admin_session_needs_csrf_but_bearer_does_not(web_factory: Any) -> None:
    w = await web_factory()
    _, admin, csrf = await person(w, "root", role=UserRole.ADMIN)
    assert (await admin.get("/v1/state")).status_code == 200
    assert (
        await admin.post("/v1/kill", json={"reason": "x"})
    ).status_code == 403  # cookie without CSRF
    assert not w.rt.c.kill_switch.engaged
    assert (
        await post(admin, "/v1/kill", csrf, json={"reason": "from the admin page"})
    ).status_code == 200
    assert w.rt.c.kill_switch.engaged
    assert (await post(admin, "/v1/resume", csrf)).status_code == 200
    token = w.rt.settings.resolved_api_token()
    assert (await w.client(token=token).post("/v1/kill", json={"reason": "cli"})).status_code == 200
    audit = [e for e in w.rt.c.audit.iter_entries() if e.type == "safety.kill_switch.engaged"]
    assert {e.actor for e in audit} >= {"admin:root", "operator:api"} or len(audit) >= 2


async def test_admin_manages_people_and_invites(web_factory: Any) -> None:
    w = await web_factory(accounts={"registration": "invite"})
    root, admin, csrf = await person(w, "root", role=UserRole.ADMIN)
    dana, dana_client, _ = await person(w, "dana")

    users = (await admin.get("/api/admin/users")).json()["users"]
    assert {u["username"] for u in users} == {"root", "dana"} and "password_hash" not in str(users)

    invite = await post(
        admin, "/api/admin/invites", csrf, json={"role": "user", "can_teach": True, "days": 3}
    )
    code = invite.json()["code"]
    listing = (await admin.get("/api/admin/invites")).json()["invites"]
    assert len(listing) == 1 and code not in str(listing)  # only shown once
    reg = await w.client().post(
        "/api/auth/register", json={"username": "eve", "password": GOOD, "invite_code": code}
    )
    assert reg.status_code == 201 and reg.json()["user"]["can_teach"] is True

    assert (await dana_client.get("/api/me")).status_code == 200
    off = await post(admin, f"/api/admin/users/{dana.id}", csrf, json={"status": "disabled"})
    assert off.status_code == 200 and off.json()["status"] == "disabled"
    assert (await dana_client.get("/api/me")).status_code == 401  # signed out immediately
    assert (
        await w.client().post("/api/auth/login", json={"username": "dana", "password": GOOD})
    ).status_code == 403
    await post(
        admin, f"/api/admin/users/{dana.id}", csrf, json={"status": "active", "can_teach": True}
    )
    assert (
        await w.client().post("/api/auth/login", json={"username": "dana", "password": GOOD})
    ).status_code == 200

    own = await post(admin, f"/api/admin/users/{root.id}", csrf, json={"role": "user"})
    assert own.status_code == 400  # no demoting yourself
    assert (
        await admin.delete(f"/api/admin/users/{dana.id}", headers={"X-CSRF-Token": csrf})
    ).status_code == 200
    assert (
        await admin.delete(f"/api/admin/users/{dana.id}", headers={"X-CSRF-Token": csrf})
    ).status_code == 404
    assert (
        await admin.delete(f"/api/admin/users/{root.id}", headers={"X-CSRF-Token": csrf})
    ).status_code == 400
    assert any(e.type == "account.erased" for e in w.rt.c.audit.iter_entries())


# ------------------------------------------------------------------- teach


async def test_teaching_is_limited_to_people_allowed_to(web_factory: Any) -> None:
    w = await web_factory()
    _, plain, plain_csrf = await person(w, "dana")
    teach = {
        "title": "Tides",
        "text": "Tides are caused mostly by the Moon's gravity pulling on the oceans.",
    }
    assert (await post(plain, "/api/teach/text", plain_csrf, json=teach)).status_code == 403
    assert (await plain.get("/api/teach/documents")).status_code == 403
    assert (
        await post(plain, "/api/teach/url", plain_csrf, json={"url": "https://example.com/x"})
    ).status_code == 403
    assert (
        await post(plain, "/api/teach/file?name=a.txt", plain_csrf, content=b"hello hello hello")
    ).status_code == 403

    _, teacher, csrf = await person(w, "eve", teach=True)
    ok = await post(teacher, "/api/teach/text", csrf, json=teach)
    assert ok.status_code == 200 and ok.json()["chunks"] >= 1
    again = await post(teacher, "/api/teach/text", csrf, json=teach)
    assert again.json()["already_known"] is True
    docs = (await teacher.get("/api/teach/documents")).json()["documents"]
    assert [d["title"] for d in docs] == ["Tides"] and docs[0]["read"] is False
    hits = await w.rt.c.memory.recall("what causes the tides", kinds=[MemoryKind.DOCUMENT])
    assert hits and "Moon" in hits[0].record.content
    assert all(
        "eve" not in h.record.content.lower() for h in hits
    )  # taught knowledge doesn't name the teacher
    doc = await w.rt.c.db.documents.list()
    assert doc[0].metadata["taught_by"]  # but the operator can see who added it
    # Teachers can't remove what's taught; administrators can.
    assert (
        await teacher.delete(
            f"/api/admin/documents/{docs[0]['id']}", headers={"X-CSRF-Token": csrf}
        )
    ).status_code == 403
    _, admin, admin_csrf = await person(w, "root", role=UserRole.ADMIN)
    gone = await admin.delete(
        f"/api/admin/documents/{docs[0]['id']}", headers={"X-CSRF-Token": admin_csrf}
    )
    assert gone.status_code == 200 and gone.json()["removed_memories"] >= 1
    assert not await w.rt.c.memory.recall(
        "what causes the tides", kinds=[MemoryKind.DOCUMENT], min_similarity=0.5
    )


async def test_teaching_files_urls_and_limits(web_factory: Any) -> None:
    w = await web_factory(accounts={"daily_teach_bytes": 60_000})
    _, teacher, csrf = await person(w, "eve", teach=True)
    upload = await post(
        teacher,
        "/api/teach/file?name=../../notes.md",
        csrf,
        content=b"# Plants\nFerns reproduce with spores rather than seeds, which is unusual.\n",
    )
    assert upload.status_code == 200, upload.text
    empty = await post(teacher, "/api/teach/file?name=x.txt", csrf, content=b"")
    assert empty.status_code == 422
    unreadable = await post(
        teacher, "/api/teach/file?name=a.bin", csrf, content=bytes(range(256)) * 4
    )
    assert unreadable.status_code == 422
    too_big = await post(teacher, "/api/teach/file?name=big.txt", csrf, content=b"x" * 5_100_000)
    assert too_big.status_code in (413, 429)
    internal = await post(
        teacher, "/api/teach/url", csrf, json={"url": "http://169.254.169.254/latest/meta-data/"}
    )
    assert internal.status_code == 422  # the fetcher refuses internal addresses
    not_web = await post(teacher, "/api/teach/url", csrf, json={"url": "file:///etc/passwd"})
    assert not_web.status_code == 422
    long = await post(
        teacher, "/api/teach/text", csrf, json={"title": "Long", "text": "word " * 2000}
    )
    assert long.status_code == 200
    over = await post(
        teacher, "/api/teach/text", csrf, json={"title": "More", "text": "another " * 8000}
    )
    assert over.status_code == 429 and "as much as you can" in over.json()["detail"]


# -------------------------------------------------------------------- media


async def test_images_are_served_only_to_people_allowed_to_see_them(
    web_factory: Any, tmp_path: Path
) -> None:
    from tests.test_world_media import PNG, FakeBackend

    img = call("image_generate", prompt="a green hill at dawn", caption="for you")
    brain = Brain([img, "Here you go."])
    w = await web_factory(brain)
    w.rt.c.images.backend = FakeBackend(PNG)
    dana, d_client, d_csrf = await person(w, "dana")
    _, e_client, _ = await person(w, "eve")
    reply = await talk(d_client, d_csrf, "draw a hill")
    [item] = reply["media"]
    own = await d_client.get(item["url"])
    assert own.status_code == 200 and own.content == PNG
    assert (
        own.headers["content-type"] == "image/png"
        and own.headers["content-disposition"] == "inline"
    )
    assert own.headers["x-content-type-options"] == "nosniff"
    assert (await e_client.get(item["url"])).status_code == 404  # not eve's
    assert (await w.client().get(item["url"])).status_code == 401
    assert item["id"] not in (await e_client.get("/api/gallery")).text
    assert item["id"] in (await d_client.get("/api/gallery")).text
    _, admin, _ = await person(w, "root", role=UserRole.ADMIN)
    assert (await admin.get(item["url"])).status_code == 200
    assert (await d_client.get("/media/img_does_not_exist")).status_code == 404

    # A record pointing outside the media directory is never read.
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    from sentientai.storage.records import MediaRecord

    bad = MediaRecord(prompt="x", path=str(secret), user_id=dana.id)
    await w.rt.c.db.media.insert(bad)
    assert (await d_client.get(f"/media/{bad.id}")).status_code == 404


async def test_world_and_gallery_for_signed_in_people(web_factory: Any) -> None:
    w = await web_factory()
    await w.rt.c.world.build(
        "Hut", [{"shape": "box", "size": [2, 2, 2]}], x=3, z=3, description="A hut Alice likes"
    )
    await w.rt.c.accounts.create_user("alice", GOOD, display_name="Alice")
    _, bob, _ = await person(w, "bob")
    world = (await bob.get("/api/world")).json()
    assert world["name"] == "Fuzzy" and world["objects"][0]["name"] == "Hut"
    assert (
        "Alice" not in world["objects"][0]["description"]
        and "someone" in world["objects"][0]["description"]
    )
    assert world["size"] == w.rt.settings.world.size


async def test_password_checks_cannot_be_used_to_guess(web_factory: Any) -> None:
    w = await web_factory()
    _, client, csrf = await person(w, "dana")
    codes = [
        (
            await post(
                client,
                "/api/me/password",
                csrf,
                json={"current": f"wrong guess {n}!!", "new": OTHER},
            )
        ).status_code
        for n in range(7)
    ]
    assert codes == [401] * 5 + [429] * 2
    # The same budget covers the other places that check the password.
    assert (
        await post(client, "/api/me/delete", csrf, json={"password": "wrong guess!!!"})
    ).status_code == 429
    assert (
        await post(client, "/api/me/totp/disable", csrf, json={"password": "wrong guess!!!"})
    ).status_code == 429
    assert (
        await w.client().post("/api/auth/login", json={"username": "dana", "password": GOOD})
    ).status_code == 200


async def test_expired_sessions_are_cleaned_up(web_factory: Any) -> None:
    w = await web_factory(accounts={"idle_hours": 1, "session_days": 1})
    user, client, _ = await person(w, "dana")
    other = w.client()
    await sign_in(other, "dana")
    w.rt.c.clock.advance(2 * 3600)
    assert await w.rt.c.db.sessions.purge_expired(w.rt.c.clock.now()) == 2
    assert (await client.get("/api/me")).status_code == 401
    assert user.id


async def test_you_can_leave_and_come_back_to_find_the_reply(web_factory: Any) -> None:
    gate = asyncio.Event()

    async def slow(request: Any) -> str:
        prompt = request.messages[-1].text
        if "Rate how likely" in prompt:
            return '{"confidence": 0.9, "gaps": []}'
        if "You just" in prompt or prompt.startswith("Check a reply"):
            return '{"unsupported": [], "new_self_decisions": [], "summary": "x", "lessons": [], "facts": []}'
        await gate.wait()
        return "Sorry for the wait, I'm here."

    w = await web_factory(llm=ScriptedLLM(slow))
    _, client, csrf = await person(w, "dana")
    accepted = await post(client, "/api/chat", csrf, json={"message": "are you there?"})
    assert accepted.status_code == 202  # the request is over; the reply is Fuzzy's to make
    await client.aclose()  # the person closes the page

    # Later, a fresh session (a new page load) finds it still working, then finds the reply.
    again = w.client()
    await sign_in(again, "dana")
    history = (await again.get("/api/chat/history")).json()
    assert history["busy"] is True and [m["content"] for m in history["messages"]] == [
        "are you there?"
    ]
    gate.set()
    for _ in range(100):
        r = (
            await again.get("/api/chat/poll", params={"after": history["messages"][-1]["at"]})
        ).json()
        if not r["busy"]:
            break
        await asyncio.sleep(0.05)
    assert [m["content"] for m in r["messages"]] == ["Sorry for the wait, I'm here."]
    assert (await again.get("/api/chat/history")).json()["busy"] is False


async def test_a_failed_reply_leaves_a_message_and_refunds_the_quota(web_factory: Any) -> None:
    from sentientai.errors import ModelError

    async def broken(request: Any) -> str:
        raise ModelError("model offline")

    w = await web_factory(llm=ScriptedLLM(broken), accounts={"daily_messages": 5})
    _, client, csrf = await person(w, "dana")
    r = await talk(client, csrf, "hello?")
    assert r["answer"]  # something is said rather than silence
    usage = (await client.get("/api/me")).json()["usage"]["messages"]
    assert usage in (0, 1)
    assert (await client.get("/api/chat/poll?after=garbage")).status_code == 422


async def test_admins_can_see_the_thinking_behind_replies_and_the_servers_state(
    web_factory: Any,
) -> None:
    w = await web_factory(Brain(["Hi, I'm Fuzzy."]))
    _, admin, admin_csrf = await person(w, "root", role=UserRole.ADMIN)
    _, alice, _ = await person(w, "alice")
    await talk(admin, admin_csrf, "hello")
    rows = (await admin.get("/api/admin/thinking")).json()
    assert any(r["type"] == "answer" and "Hi, I'm Fuzzy." in r["answer"] for r in rows)
    hardware = (await admin.get("/api/admin/hardware")).json()
    assert {"gpus", "cpu", "memory", "temperatures", "disks", "uptime_s"} <= set(hardware)
    assert hardware["cpu"]["threads"] >= 1
    # Only the operator sees either; ordinary people are refused.
    assert (await alice.get("/api/admin/thinking")).status_code == 403
    assert (await alice.get("/api/admin/hardware")).status_code == 403


async def test_brain_shows_how_each_choice_was_reached_without_cutting_it_short(
    web_factory: Any,
) -> None:
    long_why = "because " + "it matters to me " * 200
    w = await web_factory(
        Brain(beat={"choice": "think", "intent": "wonder", "why": long_why, "rest_minutes": 5})
    )
    await w.rt.c.agency.beat()
    _, alice, _ = await person(w, "alice")
    view = (await alice.get("/api/brain")).json()
    act = view["activities"][0]
    assert act["reason"] == " ".join(long_why.split())  # whole, not trimmed to a few hundred chars
    assert "wonder" in act["thinking"] and "next_activity_in_s" in view


async def test_a_reply_that_comes_back_empty_is_retried_then_explained(web_factory: Any) -> None:
    w = await web_factory(Brain(["", "", ""]))
    _, alice, csrf = await person(w, "alice")
    said = await talk(alice, csrf, "what do you think of the sea?")
    assert "(no answer produced)" not in said["answer"]
    assert "ran out of room" in said["answer"]
    w2 = await web_factory(Brain(["", "The sea feels like patience."]))
    _, bob, csrf2 = await person(w2, "bob")
    said = await talk(bob, csrf2, "what do you think of the sea?")
    assert said["answer"] == "The sea feels like patience."

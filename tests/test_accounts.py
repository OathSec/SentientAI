from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from sentientai.accounts import totp
from sentientai.accounts.data import UserDataService
from sentientai.accounts.passwords import (
    hash_password,
    needs_rehash,
    password_problems,
    verify_password,
)
from sentientai.accounts.service import AccountService, AuthError, LoginThrottle, token_digest
from sentientai.config import AccountsConfig, MemoryConfig
from sentientai.core.clock import FakeClock
from sentientai.memory.system import MemorySystem
from sentientai.memory.vector import InMemoryVectorStore
from sentientai.memory.working import InMemoryWorkingMemory, Turn
from sentientai.models.embeddings import HashingEmbedder
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import (
    ConversationMessage,
    MediaRecord,
    NoteRecord,
    RelationshipRecord,
    UserRole,
    UserStatus,
)

GOOD = "correct horse battery staple"


def service(
    db: Database, tmp_path: Path, clock: FakeClock | None = None, **cfg: object
) -> AccountService:
    config = AccountsConfig(**cfg)  # type: ignore[arg-type]
    return AccountService(
        db, config, audit=AuditLog(tmp_path / "audit", fsync=False), clock=clock or FakeClock()
    )


# ---------------------------------------------------------------- passwords


def test_password_hashing() -> None:
    stored = hash_password(GOOD)
    assert stored.startswith("scrypt$15$8$1$") and verify_password(GOOD, stored)
    assert not verify_password("wrong", stored)
    assert hash_password(GOOD) != stored  # salted
    assert not verify_password(GOOD, "garbage") and not verify_password(GOOD, "bcrypt$x$y$z$a$b")
    assert needs_rehash(hash_password(GOOD, n_log2=10)) and not needs_rehash(stored)
    assert password_problems("short", username="dana", min_length=12)
    assert password_problems("dana-is-my-password", username="dana", min_length=12)
    assert password_problems("password123", username="x", min_length=8)
    assert password_problems("aaaaaaaaaaaaaa", username="x", min_length=12)
    assert password_problems(GOOD, username="dana", min_length=12) == []


# --------------------------------------------------------------------- totp


def test_totp_rfc6238_vectors() -> None:
    secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    # RFC 6238 appendix B (SHA-1): 94287082 and 07081804; we use the last 6 digits.
    assert totp.code_at(secret, 59 // 30) == "287082"
    assert totp.code_at(secret, 1111111109 // 30) == "081804"
    step = totp.verify(secret, "287082", last_step=0, now=59)
    assert step == 1
    assert totp.verify(secret, "287082", last_step=step, now=59) is None  # replay rejected
    assert totp.verify(secret, "28708", last_step=0, now=59) is None
    assert totp.verify(secret, "287082", last_step=0, now=59 + 120) is None  # outside window
    uri = totp.provisioning_uri(totp.new_secret(), account="dana")
    assert uri.startswith("otpauth://totp/Fuzzy%3Adana?secret=")


def test_login_throttle() -> None:
    throttle = LoginThrottle(per_minute=3)
    assert all(throttle.allow("1.2.3.4", now=t) for t in (0, 1, 2))
    assert not throttle.allow("1.2.3.4", now=3)
    assert throttle.allow("5.6.7.8", now=3)
    assert throttle.allow("1.2.3.4", now=61)


# ------------------------------------------------------------------ service


async def test_create_and_register_modes(sqlite_db: Database, tmp_path: Path) -> None:
    svc = service(sqlite_db, tmp_path, registration="closed")
    with pytest.raises(AuthError, match="closed"):
        await svc.register("dana", GOOD)
    admin = await svc.create_user("admin", GOOD + "!", role=UserRole.ADMIN)
    with pytest.raises(AuthError, match="taken"):
        await svc.create_user("ADMIN", GOOD + "?")  # case-insensitive uniqueness
    with pytest.raises(AuthError, match="Usernames"):
        await svc.create_user("a", GOOD)
    with pytest.raises(AuthError, match="Password"):
        await svc.create_user("dana", "short")

    svc.config.registration = "invite"
    with pytest.raises(AuthError, match="invite code is required"):
        await svc.register("dana", GOOD)
    code = await svc.create_invite(by=admin, can_teach=True)
    user = await svc.register("dana", GOOD, display_name="  Dana​  Q ", invite_code=code)
    assert user.can_teach and user.display_name == "Dana Q"
    with pytest.raises(AuthError, match="invalid"):
        await svc.register("eve", GOOD, invite_code=code)  # single use
    invite = await sqlite_db.invites.get(token_digest(code))
    assert invite is not None and invite.used_by == user.id

    svc.config.registration = "open"
    assert (await svc.register("frank", GOOD)).role is UserRole.USER


async def test_invite_expiry_and_race(sqlite_db: Database, tmp_path: Path) -> None:
    clock = FakeClock()
    svc = service(sqlite_db, tmp_path, clock, registration="invite")
    code = await svc.create_invite(by="admin", days=1)
    clock.advance(2 * 86400)
    with pytest.raises(AuthError, match="expired"):
        await svc.register("late", GOOD, invite_code=code)
    clock.advance(-2 * 86400)
    fresh = await svc.create_invite(by="admin")
    results = await asyncio.gather(
        svc.register("racer1", GOOD, invite_code=fresh),
        svc.register("racer2", GOOD, invite_code=fresh),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert await sqlite_db.users.count() == 1  # the loser's account was rolled back


async def test_login_sessions_and_lockout(sqlite_db: Database, tmp_path: Path) -> None:
    clock = FakeClock()
    svc = service(
        sqlite_db,
        tmp_path,
        clock,
        max_failed_logins=3,
        lockout_minutes=10,
        login_attempts_per_ip_per_minute=100,
        idle_hours=1,
        session_days=1,
    )
    await svc.create_user("dana", GOOD)
    with pytest.raises(AuthError, match="Incorrect") as unknown:
        await svc.login("nobody", GOOD)
    with pytest.raises(AuthError, match="Incorrect") as wrong:
        await svc.login("dana", "wrong password!!")
    assert unknown.value.message == wrong.value.message  # no account enumeration

    result = await svc.login("Dana", GOOD, ip="10.0.0.1", user_agent="test")
    assert not result.needs_second_factor
    session = await sqlite_db.sessions.get(result.session.id)
    assert (
        session is not None
        and session.id == token_digest(result.token)
        and result.token not in session.id
    )
    resolved = await svc.resolve(result.token)
    assert resolved is not None and resolved[1].username == "dana"
    assert await svc.resolve("forged-token") is None

    clock.advance(1800)
    assert await svc.resolve(result.token) is not None  # activity slides the idle window
    clock.advance(1800 + 1700)
    assert await svc.resolve(result.token) is not None
    clock.advance(4000)
    assert await svc.resolve(result.token) is None  # idle expiry

    again = await svc.login("dana", GOOD)
    for _ in range(28):
        clock.advance(3000)
        await svc.resolve(again.token)
    clock.advance(3000)
    assert await svc.resolve(again.token) is None  # absolute lifetime (1 day) despite activity

    for _ in range(3):
        with pytest.raises(AuthError):
            await svc.login("dana", "nope nope nope")
    with pytest.raises(AuthError, match="Too many failed"):
        await svc.login("dana", GOOD)  # locked even with the right password
    clock.advance(601)
    assert (await svc.login("dana", GOOD)).user.failed_logins == 0


async def test_ip_throttle_on_login(sqlite_db: Database, tmp_path: Path) -> None:
    svc = service(sqlite_db, tmp_path, login_attempts_per_ip_per_minute=2)
    await svc.create_user("dana", GOOD)
    await svc.login("dana", GOOD, ip="9.9.9.9")
    await svc.login("dana", GOOD, ip="9.9.9.9")
    with pytest.raises(AuthError) as exc:
        await svc.login("dana", GOOD, ip="9.9.9.9")
    assert exc.value.status == 429


async def test_two_factor(sqlite_db: Database, tmp_path: Path) -> None:
    clock = FakeClock()
    svc = service(sqlite_db, tmp_path, clock)
    user = await svc.create_user("dana", GOOD)
    secret, uri = svc.begin_totp(user)
    assert secret in uri
    now = clock.now().timestamp()
    with pytest.raises(AuthError, match="didn't match"):
        await svc.enable_totp(user, secret, "000000")
    await svc.enable_totp(user, secret, totp.code_at(secret, totp.current_step(now)))

    login = await svc.login("dana", GOOD)
    assert login.needs_second_factor
    clock.advance(30)
    code = totp.code_at(secret, totp.current_step(clock.now().timestamp()))
    with pytest.raises(AuthError, match="didn't work"):
        await svc.verify_second_factor(login.session, login.user, "123456")
    await svc.verify_second_factor(login.session, login.user, code)
    stored = await sqlite_db.sessions.get(login.session.id)
    assert stored is not None and not stored.mfa_pending

    second = await svc.login("dana", GOOD)
    with pytest.raises(AuthError):
        await svc.verify_second_factor(second.session, second.user, code)  # replay of used code
    with pytest.raises(AuthError):
        await svc.disable_totp(second.user, "wrong password!!")
    await svc.disable_totp(second.user, GOOD)
    assert not (await svc.login("dana", GOOD)).needs_second_factor


async def test_password_change_revokes_other_sessions(sqlite_db: Database, tmp_path: Path) -> None:
    svc = service(sqlite_db, tmp_path)
    await svc.create_user("dana", GOOD)
    phone = await svc.login("dana", GOOD)
    laptop = await svc.login("dana", GOOD)
    with pytest.raises(AuthError, match="incorrect"):
        await svc.change_password(laptop.user, "wrong one!!!!", GOOD + " 2")
    await svc.change_password(laptop.user, GOOD, GOOD + " 2", keep_session=laptop.session.id)
    assert await svc.resolve(phone.token) is None and await svc.resolve(laptop.token) is not None
    assert (await svc.login("dana", GOOD + " 2")).user.username == "dana"


async def test_admin_updates(sqlite_db: Database, tmp_path: Path) -> None:
    svc = service(sqlite_db, tmp_path)
    admin = await svc.create_user("admin", GOOD, role=UserRole.ADMIN)
    dana = await svc.create_user("dana", GOOD)
    session = await svc.login("dana", GOOD)
    with pytest.raises(AuthError, match="own role"):
        await svc.update_user(admin.id, by=admin, role=UserRole.USER)
    await svc.update_user(dana.id, by=admin, status=UserStatus.DISABLED)
    assert await svc.resolve(session.token) is None
    with pytest.raises(AuthError, match="not active"):
        await svc.login("dana", GOOD)
    updated = await svc.update_user(dana.id, by=admin, status=UserStatus.ACTIVE, can_teach=True)
    assert updated.can_teach


async def test_export_and_erase(sqlite_db: Database, tmp_path: Path) -> None:
    svc = service(sqlite_db, tmp_path)
    dana = await svc.create_user("dana", GOOD)
    other = await svc.create_user("other", GOOD)
    memory = MemorySystem(
        sqlite_db,
        InMemoryVectorStore(64),
        HashingEmbedder(64),
        InMemoryWorkingMemory(),
        MemoryConfig(),
    )
    await memory.remember("Dana's favourite tea is oolong", user_id=dana.id)
    await memory.remember("Other likes coffee", user_id=other.id)
    await memory.remember("Tea contains caffeine")
    await sqlite_db.relationships.upsert(
        RelationshipRecord(user_id=dana.id, opinion="thoughtful and kind")
    )
    await sqlite_db.notes.insert(
        NoteRecord(user_id=dana.id, content="I was thinking about our chat.")
    )
    await sqlite_db.conversations.insert(
        ConversationMessage(
            session_id=f"user:{dana.id}", user_id=dana.id, role="user", content="hello"
        )
    )
    image = tmp_path / "img.png"
    image.write_bytes(b"png")
    await sqlite_db.media.insert(MediaRecord(prompt="a cat", path=str(image), user_id=dana.id))
    await memory.working.append(f"user:{dana.id}", Turn(role="user", content="hello"))
    login = await svc.login("dana", GOOD)

    data = UserDataService(sqlite_db, memory, audit=svc.audit)
    export = await data.export(dana)
    assert [m["content"] for m in export["memories"]] == ["Dana's favourite tea is oolong"]
    assert export["relationship"]["opinion"] == "thoughtful and kind"
    assert export["notes_from_fuzzy"][0]["content"].startswith("I was thinking")
    assert (
        export["conversations"][0]["content"] == "hello"
        and export["images"][0]["prompt"] == "a cat"
    )

    counts = await data.erase(dana, by=dana.id)
    assert counts["memories"] == 1 and counts["sessions"] == 1
    assert await sqlite_db.users.get(dana.id) is None and await svc.resolve(login.token) is None
    assert await sqlite_db.relationships.get(dana.id) is None and not image.exists()
    assert await memory.working.turns(f"user:{dana.id}") == []
    remaining = {
        m.content
        for m in await sqlite_db.memories.list(
            limit=100,
            scope=__import__("sentientai.storage.records", fromlist=["Scope"]).Scope.system(),
        )
    }
    assert remaining == {"Other likes coffee", "Tea contains caffeine"}

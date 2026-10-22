"""Accounts: registration, login, sessions, two-factor auth and invites.

Security properties:

* Passwords are hashed with scrypt; unknown usernames are verified against a dummy
  hash so timing doesn't reveal which accounts exist.
* Session tokens are random 256-bit values. Only their SHA-256 is stored, so a
  database leak doesn't hand out live sessions. Sessions expire absolutely and
  after inactivity, and every state-changing request needs the session's CSRF token.
* Repeated failures lock the account for a while; logins are also throttled per IP.
* Optional TOTP second factor with replay protection.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.exc import IntegrityError

from sentientai.accounts import totp
from sentientai.accounts.passwords import (
    DUMMY_HASH,
    hash_password,
    needs_rehash,
    password_problems,
    verify_password,
)
from sentientai.config import AccountsConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.logging import get_logger
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import (
    InviteRecord,
    SessionRecord,
    UserRecord,
    UserRole,
    UserStatus,
)

log = get_logger(__name__)

_USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
GENERIC_LOGIN_ERROR = "Incorrect username or password."


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def clean_display_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).strip()
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    return re.sub(r"\s+", " ", name)[:40]


class LoginThrottle:
    """Sliding-window limit on login attempts per client address."""

    def __init__(self, per_minute: float) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        if len(self._hits) > 50_000:  # bound memory under a flood of distinct addresses
            self._hits.clear()
        return True


@dataclass
class LoginResult:
    user: UserRecord
    token: str
    session: SessionRecord

    @property
    def needs_second_factor(self) -> bool:
        return self.session.mfa_pending


class AccountService:
    def __init__(
        self,
        db: Database,
        config: AccountsConfig,
        *,
        audit: AuditLog,
        clock: Clock | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.audit = audit
        self.clock = clock or SystemClock()
        self.throttle = LoginThrottle(config.login_attempts_per_ip_per_minute)

    # ------------------------------------------------------------ accounts
    async def create_user(
        self,
        username: str,
        password: str,
        *,
        display_name: str | None = None,
        role: UserRole = UserRole.USER,
        can_teach: bool = False,
        by: str = "system",
    ) -> UserRecord:
        username = username.strip()
        if not _USERNAME.match(username):
            raise AuthError(
                "Usernames are 3-32 characters: letters, digits, '.', '_' or '-', "
                "starting with a letter or digit."
            )
        problems = password_problems(
            password, username=username, min_length=self.config.min_password_length
        )
        if problems:
            raise AuthError("Password: " + "; ".join(problems) + ".")
        shown = clean_display_name(display_name or username) or username
        now = self.clock.now()
        user = UserRecord(
            username=username,
            username_key=username.casefold(),
            display_name=shown,
            password_hash=hash_password(password),
            role=role,
            can_teach=can_teach,
            created_at=now,
            updated_at=now,
        )
        try:
            await self.db.users.insert(user)
        except IntegrityError as exc:
            raise AuthError("That username is taken.", 409) from exc
        await self.audit.record(
            "account.created", actor=by, user_id=user.id, username=username, role=role.value
        )
        return user

    async def register(
        self,
        username: str,
        password: str,
        *,
        display_name: str | None = None,
        invite_code: str | None = None,
    ) -> UserRecord:
        mode = self.config.registration
        if mode == "closed":
            raise AuthError("Registration is closed.", 403)
        invite: InviteRecord | None = None
        if mode == "invite":
            if not invite_code:
                raise AuthError("An invite code is required.", 403)
            invite = await self.db.invites.get(token_digest(invite_code.strip()))
            if (
                invite is None
                or invite.used_by is not None
                or invite.expires_at <= self.clock.now()
            ):
                raise AuthError("That invite code is invalid or has expired.", 403)
        user = await self.create_user(
            username,
            password,
            display_name=display_name,
            role=invite.role if invite else UserRole.USER,
            can_teach=invite.can_teach if invite else False,
            by="self-registration",
        )
        if invite is not None and not await self.db.invites.consume(
            invite.id, user.id, self.clock.now()
        ):
            await self.db.users.delete(user.id)  # lost a race for a single-use invite
            raise AuthError("That invite code is invalid or has expired.", 403)
        return user

    async def create_invite(
        self,
        *,
        by: UserRecord | str,
        role: UserRole = UserRole.USER,
        can_teach: bool = False,
        days: float = 7.0,
    ) -> str:
        code = secrets.token_urlsafe(18)
        creator = by.id if isinstance(by, UserRecord) else by
        now = self.clock.now()
        await self.db.invites.insert(
            InviteRecord(
                id=token_digest(code),
                created_by=creator,
                role=role,
                can_teach=can_teach,
                created_at=now,
                expires_at=now + timedelta(days=days),
            )
        )
        await self.audit.record(
            "account.invite_created", actor=creator, role=role.value, can_teach=can_teach
        )
        return code

    # -------------------------------------------------------------- login
    async def login(
        self, username: str, password: str, *, ip: str | None = None, user_agent: str | None = None
    ) -> LoginResult:
        if not self.throttle.allow(ip or "unknown"):
            raise AuthError("Too many attempts. Please wait a minute.", 429)
        user = await self.db.users.by_username(username)
        if user is None:
            verify_password(password, DUMMY_HASH)  # equalize timing
            raise AuthError(GENERIC_LOGIN_ERROR, 401)
        now = self.clock.now()
        if user.locked_until is not None and user.locked_until > now:
            raise AuthError("Too many failed attempts. Try again later.", 429)
        if not verify_password(password, user.password_hash):
            user.failed_logins += 1
            if user.failed_logins >= self.config.max_failed_logins:
                user.locked_until = now + timedelta(minutes=self.config.lockout_minutes)
                user.failed_logins = 0
                await self.audit.record("account.locked", actor="system", user_id=user.id, ip=ip)
            user.updated_at = now
            await self.db.users.save(user)
            await self.audit.record(
                "account.login_failed", actor="anonymous", user_id=user.id, ip=ip
            )
            raise AuthError(GENERIC_LOGIN_ERROR, 401)
        if user.status is not UserStatus.ACTIVE:
            raise AuthError("This account is not active.", 403)
        user.failed_logins = 0
        user.locked_until = None
        user.last_login_at = now
        if needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)
        await self.db.users.save(user)
        token, session = await self._new_session(
            user, ip=ip, user_agent=user_agent, mfa_pending=user.totp_secret is not None
        )
        await self.audit.record(
            "account.login", actor=user.id, ip=ip, mfa_pending=session.mfa_pending
        )
        return LoginResult(user=user, token=token, session=session)

    async def _new_session(
        self, user: UserRecord, *, ip: str | None, user_agent: str | None, mfa_pending: bool
    ) -> tuple[str, SessionRecord]:
        token = secrets.token_urlsafe(32)
        now = self.clock.now()
        session = SessionRecord(
            id=token_digest(token),
            user_id=user.id,
            csrf_token=secrets.token_urlsafe(24),
            mfa_pending=mfa_pending,
            ip=(ip or "")[:64] or None,
            user_agent=(user_agent or "")[:256] or None,
            created_at=now,
            last_seen_at=now,
            expires_at=self._expiry(now, now),
        )
        await self.db.sessions.insert(session)
        return token, session

    def _expiry(self, created: datetime, now: datetime) -> datetime:
        absolute = created + timedelta(days=self.config.session_days)
        idle = now + timedelta(hours=self.config.idle_hours)
        return min(absolute, idle)

    async def resolve(self, token: str | None) -> tuple[SessionRecord, UserRecord] | None:
        if not token or len(token) > 128:
            return None
        session = await self.db.sessions.get(token_digest(token))
        now = self.clock.now()
        if session is None or session.expires_at <= now:
            if session is not None:
                await self.db.sessions.delete(session.id)
            return None
        user = await self.db.users.get(session.user_id)
        if user is None or user.status is not UserStatus.ACTIVE:
            await self.db.sessions.delete(session.id)
            return None
        if (now - session.last_seen_at).total_seconds() > 60:
            session.last_seen_at = now
            session.expires_at = self._expiry(session.created_at, now)
            await self.db.sessions.touch(session.id, now, session.expires_at)
        return session, user

    async def verify_second_factor(
        self, session: SessionRecord, user: UserRecord, code: str
    ) -> None:
        if not session.mfa_pending:
            return
        if user.totp_secret is None:
            raise AuthError("Two-factor authentication is not set up.", 400)
        if not self.throttle.allow(f"totp:{user.id}"):
            raise AuthError("Too many attempts. Please wait a minute.", 429)
        step = totp.verify(
            user.totp_secret, code, last_step=user.totp_last_step, now=self.clock.now().timestamp()
        )
        if step is None:
            await self.audit.record("account.totp_failed", actor=user.id)
            raise AuthError("That code didn't work.", 401)
        user.totp_last_step = step
        await self.db.users.save(user)
        session.mfa_pending = False
        await self.db.sessions.save(session)

    async def logout(self, token: str | None) -> None:
        if token:
            await self.db.sessions.delete(token_digest(token))

    # ---------------------------------------------------- self-management
    async def change_password(
        self, user: UserRecord, current: str, new: str, *, keep_session: str | None = None
    ) -> None:
        if not verify_password(current, user.password_hash):
            raise AuthError("Your current password is incorrect.", 401)
        problems = password_problems(
            new, username=user.username, min_length=self.config.min_password_length
        )
        if problems:
            raise AuthError("Password: " + "; ".join(problems) + ".")
        user.password_hash = hash_password(new)
        user.updated_at = self.clock.now()
        await self.db.users.save(user)
        revoked = await self.db.sessions.delete_for_user(user.id, keep=keep_session)
        await self.audit.record("account.password_changed", actor=user.id, sessions_revoked=revoked)

    def begin_totp(self, user: UserRecord) -> tuple[str, str]:
        secret = totp.new_secret()
        return secret, totp.provisioning_uri(secret, account=user.username)

    async def enable_totp(self, user: UserRecord, secret: str, code: str) -> None:
        step = totp.verify(secret, code, last_step=0, now=self.clock.now().timestamp())
        if step is None:
            raise AuthError("That code didn't match. Check your authenticator's clock.", 400)
        user.totp_secret = secret
        user.totp_last_step = step
        user.updated_at = self.clock.now()
        await self.db.users.save(user)
        await self.audit.record("account.totp_enabled", actor=user.id)

    async def disable_totp(self, user: UserRecord, password: str) -> None:
        if not verify_password(password, user.password_hash):
            raise AuthError("Your password is incorrect.", 401)
        user.totp_secret = None
        user.totp_last_step = 0
        await self.db.users.save(user)
        await self.audit.record("account.totp_disabled", actor=user.id)

    # -------------------------------------------------------------- admin
    async def update_user(
        self,
        user_id: str,
        *,
        by: UserRecord,
        status: UserStatus | None = None,
        role: UserRole | None = None,
        can_teach: bool | None = None,
    ) -> UserRecord:
        user = await self.db.users.get(user_id)
        if user is None:
            raise AuthError("No such user.", 404)
        if user.id == by.id and (status is not None or role is not None):
            raise AuthError("You can't change your own role or status.", 400)
        if status is not None:
            user.status = status
            if status is not UserStatus.ACTIVE:
                await self.db.sessions.delete_for_user(user.id)
        if role is not None:
            user.role = role
        if can_teach is not None:
            user.can_teach = can_teach
        user.updated_at = self.clock.now()
        await self.db.users.save(user)
        await self.audit.record(
            "account.updated",
            actor=by.id,
            user_id=user.id,
            status=user.status.value,
            role=user.role.value,
            can_teach=user.can_teach,
        )
        return user

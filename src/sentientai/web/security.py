"""Web security: headers, body limits, client addresses, sessions and CSRF.

Sessions live in an HttpOnly, SameSite=Lax cookie (``__Host-`` prefixed and Secure when
``accounts.cookie_secure``). Requests that change state must also carry the session's CSRF
token in ``X-CSRF-Token`` and, when the browser sends an ``Origin``, it must be this site.
"""

from __future__ import annotations

import ipaddress
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sentientai.accounts.service import AuthError
from sentientai.config import AccountsConfig
from sentientai.runtime.runtime import Runtime
from sentientai.storage.records import SessionRecord, UserRecord, UserRole

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CSRF_HEADER = "x-csrf-token"

CSP = "; ".join(
    [
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data: blob:",
        "connect-src 'self'",
        "font-src 'self'",
        "manifest-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "object-src 'none'",
    ]
)


class SecurityHeaders:
    """Adds hardening headers to every response (pure ASGI, so streams are untouched)."""

    def __init__(
        self, app: ASGIApp, *, hsts: bool = False, csp_exempt: tuple[str, ...] = ()
    ) -> None:
        self.app = app
        self.hsts = hsts
        self.csp_exempt = csp_exempt

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}

                def add(name: str, value: str) -> None:
                    if name.encode() not in present:
                        headers.append((name.encode(), value.encode()))

                if not path.startswith(self.csp_exempt):
                    add("content-security-policy", CSP)
                add("x-content-type-options", "nosniff")
                add("x-frame-options", "DENY")
                add("referrer-policy", "same-origin")
                add("permissions-policy", "camera=(), microphone=(), geolocation=(), payment=()")
                add("cross-origin-opener-policy", "same-origin")
                add("cross-origin-resource-policy", "same-origin")
                if path.startswith("/api/") or path.startswith("/v1/"):
                    add("cache-control", "no-store")
                if self.hsts:
                    add("strict-transport-security", "max-age=31536000; includeSubDomains")
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodyLimit:
    """Rejects oversized request bodies, including ones that lie about Content-Length."""

    def __init__(
        self, app: ASGIApp, *, default: int = 256_000, large: dict[str, int] | None = None
    ) -> None:
        self.app = app
        self.default = default
        self.large = large or {}

    def limit_for(self, path: str) -> int:
        for prefix, limit in self.large.items():
            if path.startswith(prefix):
                return limit
        return self.default

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in UNSAFE_METHODS:
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope.get("path", ""))
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._reject(send)
            return
        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    rejected = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message: Message) -> None:
            if rejected and message["type"] == "http.response.start":
                await self._reject(send)
                return
            if rejected:
                return
            await send(message)

        await self.app(scope, limited_receive, guarded_send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"detail":"request too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ------------------------------------------------------------ client address


def _networks(entries: list[str]) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    networks = []
    for entry in entries:
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            continue
    return networks


def client_ip(request: Request, trusted_proxies: list[str]) -> str:
    """The caller's address. X-Forwarded-For is believed only from a trusted proxy."""

    peer = request.client.host if request.client else "unknown"
    if not trusted_proxies:
        return peer
    networks = _networks(trusted_proxies)

    def trusted(addr: str) -> bool:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        return any(ip in n for n in networks)

    if not trusted(peer):
        return peer
    forwarded = [
        a.strip() for a in request.headers.get("x-forwarded-for", "").split(",") if a.strip()
    ]
    for addr in reversed(forwarded):  # rightmost entry not added by one of our own proxies
        if not trusted(addr):
            return addr
    return peer


# ------------------------------------------------------------------ sessions


@dataclass
class Principal:
    """Who is calling: nobody, a signed-in person, or the operator's bearer token."""

    user: UserRecord | None = None
    session: SessionRecord | None = None
    token: str | None = None
    operator: bool = False
    ip: str = "unknown"

    @property
    def signed_in(self) -> bool:
        return self.operator or (self.user is not None and self.session is not None)

    @property
    def second_factor_pending(self) -> bool:
        return self.session is not None and self.session.mfa_pending

    @property
    def admin(self) -> bool:
        if self.operator:
            return True
        return (
            self.user is not None
            and self.user.role is UserRole.ADMIN
            and not self.second_factor_pending
        )

    @property
    def actor(self) -> str:
        if self.operator:
            return "operator"
        if self.user is None:
            return "anonymous"
        return f"admin:{self.user.username}" if self.admin else self.user.id


def cookie_name(config: AccountsConfig) -> str:
    return "__Host-fuzzy_session" if config.cookie_secure else "fuzzy_session"


def set_session_cookie(response: Response, token: str, config: AccountsConfig) -> None:
    response.set_cookie(
        cookie_name(config),
        token,
        max_age=int(config.session_days * 86400),
        httponly=True,
        secure=config.cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response, config: AccountsConfig) -> None:
    response.delete_cookie(
        cookie_name(config), path="/", secure=config.cookie_secure, httponly=True, samesite="lax"
    )


def origin_ok(request: Request, extra_origins: list[str] | None = None) -> bool:
    """Browsers send Origin on cross-site requests; if it is present it must be this site."""

    origin = request.headers.get("origin")
    if origin is None:
        return True
    if origin == "null":
        return False
    host = urlsplit(origin).netloc
    allowed = {request.headers.get("host", "")} | {urlsplit(o).netloc for o in extra_origins or []}
    return host in allowed


class WebAuth:
    """FastAPI dependencies resolving the caller and enforcing CSRF."""

    def __init__(self, runtime: Runtime, operator_token: str | None) -> None:
        self.runtime = runtime
        self.config = runtime.settings.accounts
        self.trusted = runtime.settings.accounts.trusted_proxies
        self.operator_token = operator_token
        self.open = runtime.settings.api.insecure_no_auth

    def ip(self, request: Request) -> str:
        return client_ip(request, self.trusted)

    def _bearer(self, request: Request) -> bool:
        header = request.headers.get("authorization", "")
        if self.operator_token is None or not header.startswith("Bearer "):
            return False
        return secrets.compare_digest(header.removeprefix("Bearer ").strip(), self.operator_token)

    def _query_token(self, request: Request) -> bool:
        supplied = request.query_params.get("token", "")
        if self.operator_token is None or not supplied:
            return False
        return secrets.compare_digest(supplied, self.operator_token)

    async def principal(self, request: Request, *, allow_query_token: bool = False) -> Principal:
        ip = self.ip(request)
        if self._bearer(request) or (allow_query_token and self._query_token(request)):
            return Principal(operator=True, ip=ip)
        if self.open:
            return Principal(operator=True, ip=ip)
        token = request.cookies.get(cookie_name(self.config))
        resolved = await self.runtime.c.accounts.resolve(token)
        if resolved is None:
            return Principal(ip=ip)
        session, user = resolved
        return Principal(user=user, session=session, token=token, ip=ip)

    def check_csrf(self, request: Request, principal: Principal) -> None:
        """Cookie-authenticated requests that change state must prove they came from our pages."""

        if principal.operator or request.method not in UNSAFE_METHODS:
            return
        if not origin_ok(request):
            raise HTTPException(status_code=403, detail="cross-site request refused")
        supplied = request.headers.get(CSRF_HEADER, "")
        expected = principal.session.csrf_token if principal.session else ""
        if not expected or not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=403, detail="missing or invalid CSRF token")

    # Dependencies -------------------------------------------------------
    async def anyone(self, request: Request) -> Principal:
        """Possibly anonymous. For anything that changes state use :meth:`user`."""

        principal = await self.principal(request)
        if principal.signed_in:
            self.check_csrf(request, principal)
        return principal

    async def user(self, request: Request) -> Principal:
        principal = await self.principal(request)
        if not principal.signed_in:
            raise HTTPException(status_code=401, detail="please sign in")
        if principal.second_factor_pending:
            raise HTTPException(status_code=403, detail="enter your two-factor code first")
        self.check_csrf(request, principal)
        return principal

    async def person(self, request: Request) -> Principal:
        """A signed-in person (not just the operator's token)."""

        principal = await self.user(request)
        if principal.user is None:
            raise HTTPException(status_code=403, detail="this needs a user account")
        return principal

    async def admin(self, request: Request) -> Principal:
        return await self._admin(request, allow_query_token=False)

    async def admin_stream(self, request: Request) -> Principal:
        """Admin for event streams: EventSource can't set headers, so ?token= is accepted."""

        return await self._admin(request, allow_query_token=True)

    async def _admin(self, request: Request, *, allow_query_token: bool) -> Principal:
        principal = await self.principal(request, allow_query_token=allow_query_token)
        if not principal.signed_in:
            raise HTTPException(status_code=401, detail="invalid or missing credentials")
        if principal.second_factor_pending:
            raise HTTPException(status_code=403, detail="enter your two-factor code first")
        if not principal.admin:
            raise HTTPException(status_code=403, detail="administrators only")
        self.check_csrf(request, principal)
        if (
            not principal.operator
            and self.config.require_admin_totp
            and principal.user is not None
            and principal.user.totp_secret is None
        ):
            raise HTTPException(
                status_code=403, detail="administrators must enable two-factor authentication"
            )
        return principal

    async def viewer(self, request: Request) -> Principal:
        """Signed in, or anonymous when public pages are enabled."""

        principal = await self.principal(request)
        if principal.signed_in:
            if principal.second_factor_pending:
                raise HTTPException(status_code=403, detail="enter your two-factor code first")
            return principal
        if self.config.public_pages:
            return principal
        raise HTTPException(status_code=401, detail="please sign in")


def raise_for(exc: AuthError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.message)


def json_only(request: Request) -> None:
    """State-changing JSON endpoints refuse form posts (which cross-site pages could send)."""

    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        raise HTTPException(status_code=415, detail="send application/json")


def dump(model: Any) -> dict[str, Any]:
    return dict(model.model_dump(mode="json"))

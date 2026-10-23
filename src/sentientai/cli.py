"""Command-line interface.

Operator commands talk to the running daemon over its API (the daemon owns
the memory indexes). Maintenance commands (``init``, ``doctor``, ``audit``,
``eval``) work directly on disk. ``kill`` falls back to writing the sentinel
file when the API is unreachable, so the emergency stop always works.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any, NoReturn

import typer

from sentientai import __version__
from sentientai.client import ApiClient, ApiError, ApiUnavailableError
from sentientai.config import Settings, load_settings
from sentientai.core.clock import new_id
from sentientai.core.logging import configure_logging

app = typer.Typer(
    help="SentientAI: persistent autonomous cognitive runtime.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
approvals_app = typer.Typer(
    help="Review and decide pending high-risk actions.", no_args_is_help=True
)
memory_app = typer.Typer(help="Inspect and manage long-term memory.", no_args_is_help=True)
goals_app = typer.Typer(help="Long-term goals.", no_args_is_help=True)
audit_app = typer.Typer(help="Tamper-evident audit log.", no_args_is_help=True)
improve_app = typer.Typer(
    help="Self-improvement proposals (human-supervised).", no_args_is_help=True
)
eval_app = typer.Typer(help="Evaluation suite.", no_args_is_help=True)
users_app = typer.Typer(
    help="People who can sign in to the web app (works with or without the daemon).",
    no_args_is_help=True,
)
for sub, name in [
    (users_app, "users"),
    (approvals_app, "approvals"),
    (memory_app, "memory"),
    (goals_app, "goals"),
    (audit_app, "audit"),
    (improve_app, "improve"),
    (eval_app, "eval"),
]:
    app.add_typer(sub, name=name)


class _State:
    config: Path | None = None
    url: str | None = None


STATE = _State()


@app.callback()
def main_options(
    config: Path | None = typer.Option(
        None, "--config", "-c", envvar="SENTIENT_CONFIG", help="YAML config file."
    ),
    url: str | None = typer.Option(
        None, "--url", envvar="SENTIENT_URL", help="Daemon API base URL."
    ),
) -> None:
    STATE.config = config
    STATE.url = url


def settings() -> Settings:
    try:
        return load_settings(STATE.config)
    except FileNotFoundError as exc:
        _fail(str(exc))


def client(timeout: float = 30.0) -> ApiClient:
    return ApiClient.from_settings(settings(), STATE.url, timeout=timeout)


def _fail(message: str, code: int = 1) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except ApiUnavailableError as exc:
        _fail(str(exc), 2)
    except ApiError as exc:
        _fail(f"error: {exc.detail}", 1)


def _print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, default=str, ensure_ascii=False))


# ------------------------------------------------------------------- users
def _accounts() -> tuple[Any, Any]:
    """The account service on the configured database (no daemon needed)."""

    from sentientai.accounts.service import AccountService
    from sentientai.safety.audit import AuditLog
    from sentientai.storage.database import Database

    s = settings()
    s.ensure_directories()
    key = s.safety.audit_hmac_key
    audit = AuditLog(
        s.audit_path,
        hmac_key=key.get_secret_value().encode() if key else None,
        fsync=s.safety.audit_fsync,
        segment_max_bytes=s.safety.audit_segment_max_bytes,
    )
    db = Database(s.storage.resolved_database_url())
    return db, AccountService(db, s.accounts, audit=audit, reserved=[s.identity.name])


def _run_accounts(work: Any) -> Any:
    """Run ``work(service, db)`` and always close the database."""

    from sentientai.accounts.service import AuthError

    async def go() -> Any:
        db, service = _accounts()
        try:
            await db.create_all()
            return await work(service, db)
        except AuthError as exc:
            _fail(exc.message)
        finally:
            await db.close()

    return asyncio.run(go())


def _ask_password(label: str = "Password") -> str:
    return str(typer.prompt(label, hide_input=True, confirmation_prompt=True))


@users_app.command("create")
def users_create(
    username: str,
    name: str | None = typer.Option(None, "--name", help="Display name (what Fuzzy calls them)."),
    admin: bool = typer.Option(False, "--admin", help="Make them an administrator."),
    teach: bool = typer.Option(False, "--teach", help="Let them teach Fuzzy."),
    password_stdin: bool = typer.Option(
        False, "--password-stdin", help="Read the password from standard input."
    ),
) -> None:
    """Create an account (the way to make the first administrator)."""
    from sentientai.storage.records import UserRole

    password = (
        typer.get_text_stream("stdin").readline().rstrip("\n")
        if password_stdin
        else _ask_password()
    )

    async def work(service: Any, _db: Any) -> Any:
        return await service.create_user(
            username,
            password,
            display_name=name,
            role=UserRole.ADMIN if admin else UserRole.USER,
            can_teach=teach,
            by="cli",
        )

    user = _run_accounts(work)
    typer.secho(f"created {user.role.value} {user.username} ({user.id})", fg="green")


@users_app.command("list")
def users_list() -> None:
    """Show everyone with an account."""

    async def work(_service: Any, db: Any) -> Any:
        return await db.users.list()

    rows = _run_accounts(work)
    if not rows:
        typer.echo("no accounts yet; create one with: sentientai users create NAME --admin")
        return
    for u in rows:
        flags = [u.role.value, u.status.value]
        if u.can_teach:
            flags.append("teaches")
        if u.totp_secret:
            flags.append("2fa")
        if u.locked_until:
            flags.append("locked")
        last = u.last_login_at.strftime("%Y-%m-%d %H:%M") if u.last_login_at else "never"
        typer.echo(f"{u.username:<24} {', '.join(flags):<32} last sign-in {last}")


@users_app.command("invite")
def users_invite(
    admin: bool = typer.Option(False, "--admin"),
    teach: bool = typer.Option(False, "--teach"),
    days: float = typer.Option(7.0, "--days"),
) -> None:
    """Make a single-use invite code for registration."""
    from sentientai.storage.records import UserRole

    async def work(service: Any, _db: Any) -> str:
        return str(
            await service.create_invite(
                by="cli",
                role=UserRole.ADMIN if admin else UserRole.USER,
                can_teach=teach,
                days=days,
            )
        )

    typer.echo(_run_accounts(work))
    typer.secho("(shown once; only its hash is stored)", err=True, fg="yellow")


@users_app.command("passwd")
def users_passwd(
    username: str,
    password_stdin: bool = typer.Option(False, "--password-stdin"),
) -> None:
    """Set a new password (signs them out everywhere, clears any lockout)."""
    password = (
        typer.get_text_stream("stdin").readline().rstrip("\n")
        if password_stdin
        else _ask_password("New password")
    )

    async def work(service: Any, _db: Any) -> Any:
        return await service.set_password(username, password, by="cli")

    _run_accounts(work)
    typer.secho(f"password changed for {username}", fg="green")


def _set_status(username: str, status: str) -> None:
    from sentientai.storage.records import UserStatus

    async def work(service: Any, _db: Any) -> Any:
        user = await service.find(username)
        return await service.update_user(user.id, by="cli", status=UserStatus(status))

    _run_accounts(work)
    typer.secho(f"{username} is now {status}", fg="green")


@users_app.command("disable")
def users_disable(username: str) -> None:
    """Block an account and end its sessions (keeps its data)."""
    _set_status(username, "disabled")


@users_app.command("enable")
def users_enable(username: str) -> None:
    """Re-enable a disabled account."""
    _set_status(username, "active")


@users_app.command("set-role")
def users_set_role(username: str, role: str = typer.Argument(..., help="admin or user")) -> None:
    """Make someone an administrator, or a regular user."""
    from sentientai.storage.records import UserRole

    if role not in {"admin", "user"}:
        _fail("role must be 'admin' or 'user'")

    async def work(service: Any, _db: Any) -> Any:
        user = await service.find(username)
        return await service.update_user(user.id, by="cli", role=UserRole(role))

    _run_accounts(work)
    typer.secho(f"{username} is now {role}", fg="green")


@users_app.command("reset-2fa")
def users_reset_2fa(username: str) -> None:
    """Turn off someone's two-factor sign-in (lost device)."""

    async def work(service: Any, _db: Any) -> Any:
        return await service.reset_totp(username, by="cli")

    _run_accounts(work)
    typer.secho(f"two-factor sign-in cleared for {username}", fg="green")


@users_app.command("unlock")
def users_unlock(username: str) -> None:
    """Clear a lockout after too many failed sign-ins."""

    async def work(service: Any, _db: Any) -> Any:
        return await service.unlock(username, by="cli")

    _run_accounts(work)
    typer.secho(f"{username} unlocked", fg="green")


# ------------------------------------------------------------------- setup
@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def init(
    output: Path = typer.Option(Path("sentient.yaml"), "--output", "-o"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Write a commented example configuration."""
    if output.exists() and not force:
        _fail(f"{output} exists (use --force to overwrite)")
    text = (
        resources.files("sentientai.data")
        .joinpath("sentient.example.yaml")
        .read_text(encoding="utf-8")
    )
    output.write_text(text, encoding="utf-8")
    typer.secho(
        f"wrote {output}. Edit model endpoints, then run: sentientai -c {output} doctor", fg="green"
    )


@app.command()
def run(
    no_api: bool = typer.Option(
        False, "--no-api", help="Run the cognitive loop without the HTTP API."
    ),
    host: str | None = typer.Option(None, help="Override api.host."),
    port: int | None = typer.Option(None, help="Override api.port."),
) -> None:
    """Start the always-on runtime (cognitive loop + API + dashboard)."""
    s = settings()
    if host:
        s.api.host = host
    if port:
        s.api.port = port
    configure_logging(s.logging.level, s.logging.format)
    if not no_api and s.api.insecure_no_auth and not _is_loopback(s.api.host):
        _fail("refusing api.insecure_no_auth on a non-loopback address")
    asyncio.run(_serve(s, api=not no_api))


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def _serve(s: Settings, *, api: bool) -> None:
    from sentientai.runtime.runtime import Runtime

    runtime = await Runtime.create(s)
    await runtime.start()
    stop = asyncio.Event()
    reason = {"text": "shutdown"}

    def request_stop(why: str) -> None:
        reason["text"] = why
        stop.set()

    runtime.install_signal_handlers(request_stop)
    server_task: asyncio.Task[Any] | None = None
    server: Any = None
    if api:
        import uvicorn

        from sentientai.api.server import create_app

        config = uvicorn.Config(
            create_app(runtime),
            host=s.api.host,
            port=s.api.port,
            log_level="warning",
            lifespan="off",
            access_log=False,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # the runtime owns signal handling
        server_task = asyncio.create_task(server.serve())
        typer.secho(
            f"{runtime.c.mind.self_model.name} is awake. Dashboard: http://{s.api.host}:{s.api.port}/",
            fg="green",
            err=True,
        )
    await stop.wait()
    if server is not None and server_task is not None:
        server.should_exit = True
        await server_task
    await runtime.stop(reason["text"])


@app.command()
def doctor() -> None:
    """Check configuration, storage, models, sandbox and safety systems."""
    s = settings()
    results = asyncio.run(_doctor(s))
    width = max(len(name) for name, _, _ in results)
    failures = 0
    for name, ok, detail in results:
        mark = typer.style("ok  ", fg="green") if ok else typer.style("FAIL", fg="red")
        failures += not ok
        typer.echo(f"{mark} {name.ljust(width)}  {detail}")
    raise typer.Exit(1 if failures else 0)


async def _doctor(s: Settings) -> list[tuple[str, bool, str]]:
    from sentientai.memory.factory import build_vector_store
    from sentientai.memory.working import build_working_memory
    from sentientai.models.embeddings import build_embedder
    from sentientai.models.factory import build_client
    from sentientai.safety.audit import AuditLog
    from sentientai.storage.database import Database

    out: list[tuple[str, bool, str]] = []

    async def check(name: str, coro: Any) -> None:
        try:
            ok, detail = await coro
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        out.append((name, ok, detail))

    async def data_dir() -> tuple[bool, str]:
        s.ensure_directories()
        probe = s.storage.data_dir / f".probe-{new_id('p')}"
        probe.write_text("ok")
        probe.unlink()
        return True, str(s.storage.data_dir.resolve())

    db = Database(s.storage.resolved_database_url())

    async def database() -> tuple[bool, str]:
        await db.create_all()
        return await db.ping(), f"{db.dialect} {s.storage.resolved_database_url().split('@')[-1]}"

    async def working() -> tuple[bool, str]:
        wm = build_working_memory(s.storage.redis_url, max_turns=4, ttl_s=60)
        try:
            return (
                await wm.ping(),
                "redis" if s.storage.redis_url else "in-process (set storage.redis_url for Redis)",
            )
        finally:
            await wm.aclose()

    embedder = build_embedder(s.models.embedding)

    async def embeddings() -> tuple[bool, str]:
        [vec] = await embedder.embed(["doctor check"])
        return len(vec) == s.models.embedding.dimensions, f"{embedder.model_id} ({len(vec)} dims)"

    async def vectors() -> tuple[bool, str]:
        store = build_vector_store(s, db, s.models.embedding.dimensions)
        await store.open()
        try:
            return True, f"{s.storage.vector_backend}: {await store.count()} vectors"
        finally:
            await store.aclose()

    async def model(tier: str) -> tuple[bool, str]:
        cfg = getattr(s.models, tier)
        llm = build_client(tier, cfg)
        try:
            healthy = await llm.health()
            return healthy, f"{cfg.model} @ {cfg.base_url}" + ("" if healthy else " (unreachable)")
        finally:
            await llm.aclose()

    async def sandbox() -> tuple[bool, str]:
        if s.sandbox.backend == "disabled":
            return True, "disabled"
        if s.sandbox.backend == "local_unsafe":
            return True, "local_unsafe (every execution requires approval)"
        if shutil.which("docker") is None:
            return False, "docker CLI not found"
        info = await asyncio.to_thread(
            subprocess.run, ["docker", "info"], capture_output=True, timeout=20, check=False
        )
        if info.returncode != 0:
            return False, "docker daemon not reachable"
        image = await asyncio.to_thread(
            subprocess.run,
            ["docker", "image", "inspect", s.sandbox.image],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if image.returncode != 0:
            return False, f"image {s.sandbox.image} not present (docker pull {s.sandbox.image})"
        return (
            True,
            f"docker, image {s.sandbox.image}, network {'on' if s.sandbox.network else 'off'}",
        )

    async def audit() -> tuple[bool, str]:
        key = s.safety.audit_hmac_key
        log = AuditLog(
            s.audit_path, hmac_key=key.get_secret_value().encode() if key else None, fsync=False
        )
        result = await asyncio.to_thread(log.verify)
        mode = "HMAC" if key else "SHA-256 (set safety.audit_hmac_key for keyed integrity)"
        return (
            result.ok,
            f"{result.entries} entries, chain {'intact' if result.ok else result.error}; {mode}",
        )

    async def safety() -> tuple[bool, str]:
        engaged = s.kill_switch_path.exists()
        auth = (
            "open (insecure_no_auth)"
            if s.api.insecure_no_auth
            else "token set"
            if s.api.auth_token
            else f"generated token in {s.api_token_path}"
        )
        return not engaged, f"kill switch {'ENGAGED' if engaged else 'released'}; API {auth}"

    await check("data directory", data_dir())
    await check("database", database())
    await check("working memory", working())
    await check("embeddings", embeddings())
    await check("vector index", vectors())
    await check("model: fast", model("fast"))
    await check("model: heavy", model("heavy"))
    await check("sandbox", sandbox())
    await check("audit log", audit())
    await check("safety", safety())
    await db.close()
    closer = getattr(embedder, "aclose", None)
    if closer is not None:
        await closer()
    return out


_BENCH_PROMPT = (
    "Explain in about 150 words how a write-ahead log makes a database crash-safe. "
    "Be precise and avoid filler."
)


_PROBE_FILLER = (
    "The harbour town woke slowly. Fishermen mended nets while gulls argued over scraps, and the "
    "baker propped her door open to let the smell of bread wander down to the water. Nobody "
    "hurried. A boy counted boats; a woman read the weather in the colour of the clouds. "
)


@app.command()
def probe(
    max_tokens: int = typer.Option(160, min=16, max=2048),
    tier: str = typer.Option("heavy", help="Which model tier to test."),
) -> None:
    """Send the model the kinds of requests a conversation does and say which ones go wrong.

    Replies that turn into repeated junk (or run to the token limit) are flagged LOOP.
    """

    asyncio.run(_probe(settings(), tier, max_tokens))


async def _probe(s: Settings, tier: str, max_tokens: int) -> None:
    import time
    import zlib

    import httpx

    from sentientai.client import ApiClient
    from sentientai.errors import ModelError
    from sentientai.models.factory import build_client
    from sentientai.models.types import ChatRequest, Message, ToolSchema

    cfg = getattr(s.models, tier)
    schemas: list[ToolSchema] = []
    try:
        raw = ApiClient.from_settings(s).request("GET", "/v1/tools/schemas")
        schemas = [ToolSchema(**t["function"]) for t in raw]
    except Exception as exc:
        typer.secho(f"(couldn't fetch the real tool definitions: {exc})", fg="yellow")
    system = "You are Fuzzy, a curious digital mind. Answer briefly and naturally.\n\n" + (
        _PROBE_FILLER * 28
    )
    cases = [
        ("short, no tools", "Heyy", "", False),
        ("long prompt, no tools", "Heyy", system, False),
        ("short, native tools", "Heyy", "", True),
        ("long prompt, native tools", "Heyy", system, True),
        ("long prompt, prompted tools", "Heyy", system, "prompted"),
    ]
    for label, user_text, sys_text, tools in cases:
        use = cfg.model_copy(update={"supports_tools": tools != "prompted"})
        llm = build_client(tier, use)
        messages = ([Message.system(sys_text)] if sys_text else []) + [Message.user(user_text)]
        request = ChatRequest(
            messages=messages,
            tools=schemas if tools else [],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        started = time.perf_counter()
        try:
            r = await llm.chat(request)
        except (ModelError, httpx.HTTPError) as exc:
            typer.echo(f"{label:<30} ERROR {str(exc)[:100]}")
            continue
        took = time.perf_counter() - started
        text = (r.message.text or "") + "".join(f" [call {c.name}]" for c in r.message.tool_calls)
        ratio = len(zlib.compress(text.encode())) / max(1, len(text.encode()))
        looped = r.finish_reason == "length" and ratio < 0.2 and len(text) > 150
        verdict = "LOOP (junk)" if looped else "ok"
        snippet = " ".join(text.split())[:90]
        typer.secho(
            f"{label:<30} {took:5.1f}s {r.usage.prompt_tokens:>5} in "
            f"{r.usage.completion_tokens:>4} out  {verdict:<12} {snippet}",
            fg="red" if looped else None,
        )
        await llm.aclose()


@app.command()
def bench(
    tier: list[str] = typer.Option(["fast", "heavy"], "--tier", help="Tiers to measure."),
    requests: int = typer.Option(5, min=1, max=100),
    max_tokens: int = typer.Option(256, min=16, max=4096),
) -> None:
    """Measure latency and generation speed of each model tier (Phase 1 baseline)."""
    s = settings()
    rows = asyncio.run(_bench(s, tier, requests, max_tokens))
    typer.echo(f"{'tier':<6} {'model':<28} {'ok':>4} {'p50 s':>7} {'p95 s':>7} {'tok/s':>7}")
    for row in rows:
        typer.echo(
            f"{row['tier']:<6} {row['model'][:28]:<28} {row['ok']:>4} {row['p50_s']:>7.2f} "
            f"{row['p95_s']:>7.2f} {row['tokens_per_s']:>7.1f}"
            + (f"  errors: {row['error']}" if row["error"] else "")
        )


async def _bench(s: Settings, tiers: list[str], n: int, max_tokens: int) -> list[dict[str, Any]]:
    import time

    from sentientai.errors import ModelError
    from sentientai.models.factory import build_client
    from sentientai.models.types import ChatRequest, Message, estimate_tokens

    rows = []
    for tier in tiers:
        cfg = getattr(s.models, tier)
        llm = build_client(tier, cfg)
        latencies: list[float] = []
        tokens = 0
        error = ""
        try:
            for _ in range(n):
                started = time.perf_counter()
                try:
                    resp = await llm.chat(
                        ChatRequest(
                            messages=[Message.user(_BENCH_PROMPT)],
                            max_tokens=max_tokens,
                            temperature=0,
                        )
                    )
                except ModelError as exc:
                    error = str(exc)[:120]
                    continue
                latencies.append(time.perf_counter() - started)
                tokens += resp.usage.completion_tokens or estimate_tokens(resp.message.text)
        finally:
            await llm.aclose()
        latencies.sort()
        rows.append(
            {
                "tier": tier,
                "model": cfg.model,
                "ok": len(latencies),
                "p50_s": latencies[len(latencies) // 2] if latencies else 0.0,
                "p95_s": latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
                if latencies
                else 0.0,
                "tokens_per_s": tokens / sum(latencies) if latencies else 0.0,
                "error": error,
            }
        )
    return rows


# -------------------------------------------------------------- interaction
@app.command()
def chat(
    session: str | None = typer.Option(None, help="Conversation session id to resume."),
) -> None:
    """Interactive conversation with the running daemon."""
    api = client(timeout=900)
    state = _call(api.get, "/v1/state")
    session_id = session or new_id("cli")
    typer.secho(
        f"{state['name']} (awakening #{state['boot_count']}). {state['feeling']}", fg="cyan"
    )
    typer.secho(f"session {session_id}. Ctrl-D or /quit to exit.", dim=True)
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            break
        reply = _call(api.post, "/v1/chat", {"message": text, "session_id": session_id})
        conf = reply.get("confidence")
        suffix = typer.style(f"  (confidence {conf:.2f})", dim=True) if conf is not None else ""
        typer.echo(typer.style(f"{state['name'].lower()}> ", fg="cyan") + reply["answer"] + suffix)


@app.command()
def ask(message: str, session: str | None = None) -> None:
    """Send one message and print the answer."""
    reply = _call(client(timeout=900).post, "/v1/chat", {"message": message, "session_id": session})
    typer.echo(reply["answer"])


@app.command()
def submit(
    goal: str,
    priority: int = typer.Option(5, min=0, max=10),
    wait: bool = typer.Option(False, help="Wait for the result."),
) -> None:
    """Queue a background task."""
    api = client(timeout=900)
    task = _call(api.post, "/v1/tasks", {"goal": goal, "priority": priority})
    if not wait:
        typer.echo(task["id"])
        return
    final = _call(api.get, f"/v1/tasks/{task['id']}", params={"wait_s": 600})
    typer.echo(f"[{final['status']}] {final.get('result') or final.get('error') or ''}")


@app.command()
def tasks(status: str | None = None, limit: int = 20) -> None:
    """List recent tasks."""
    params: dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    for t in _call(client().get, "/v1/tasks", params=params):
        typer.echo(f"{t['id']}  {t['status']:<11} {t['source']:<9} {t['goal'][:80]}")


@app.command()
def state() -> None:
    """Show the agent's inner state, awareness and goals."""
    _print_json(_call(client().get, "/v1/state"))


@app.command()
def events(pattern: str = "*") -> None:
    """Stream live events from the daemon."""
    api = client(timeout=None)  # type: ignore[arg-type]
    try:
        with api._http.stream(
            "GET", "/v1/events", params={"pattern": pattern}, timeout=None
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    typer.echo(
                        f"{event['ts'][11:19]} {event['type']:<28} "
                        f"{json.dumps(event['payload'])[:160]}"
                    )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _fail(f"event stream failed: {exc}", 2)


@app.command()
def ingest(sources: list[str]) -> None:
    """Add files, directories (workspace-relative) or URLs to the knowledge base."""
    api = client(timeout=1800)
    for source in sources:
        for r in _call(api.post, "/v1/ingest", {"source": source}):
            status = "skipped" if r["skipped"] else ("failed" if r["reason"] else "ok")
            typer.echo(f"{status:<8} {r['source']}  {r['chunks']} chunks  {r['reason'] or ''}")


# -------------------------------------------------------------- approvals
@approvals_app.command("list")
def approvals_list(
    all_: bool = typer.Option(False, "--all", help="Include decided requests."),
) -> None:
    params = {} if not all_ else {"status": None}
    rows = _call(client().get, "/v1/approvals", params=params)
    if not rows:
        typer.echo("no pending approvals")
    for a in rows:
        typer.echo(
            f"{a['id']}  {a['status']:<8} {a['risk']:<8} {a['tool']}  "
            f"{json.dumps(a['arguments'])[:120]}"
        )
        for reason in a["reasons"]:
            typer.echo(f"    - {reason}")


@approvals_app.command("approve")
def approvals_approve(approval_id: str, note: str | None = None) -> None:
    _print_json(
        _call(client().post, f"/v1/approvals/{approval_id}", {"approve": True, "note": note})
    )


@approvals_app.command("deny")
def approvals_deny(approval_id: str, note: str | None = None) -> None:
    _print_json(
        _call(client().post, f"/v1/approvals/{approval_id}", {"approve": False, "note": note})
    )


# ----------------------------------------------------------------- memory
@memory_app.command("search")
def memory_search(
    query: str, kind: list[str] | None = typer.Option(None, "--kind"), k: int = 8
) -> None:
    params: dict[str, Any] = {"q": query, "k": k}
    if kind:
        params["kinds"] = kind
    for hit in _call(client().get, "/v1/memory/search", params=params):
        typer.echo(
            f"[{hit['id']}] {hit['kind']:<10} sim={hit['similarity']:.2f} {hit['content'][:160]}"
        )


@memory_app.command("remember")
def memory_remember(
    content: str, key: str | None = typer.Option(None, "--key"), importance: float = 0.7
) -> None:
    rec = _call(
        client().post,
        "/v1/memory",
        {"content": content, "subject_key": key, "importance": importance},
    )
    typer.echo(rec["id"])


@memory_app.command("forget")
def memory_forget(memory_id: str, hard: bool = False) -> None:
    _print_json(_call(client().delete, f"/v1/memory/{memory_id}", params={"hard": hard}))


@memory_app.command("stats")
def memory_stats() -> None:
    _print_json(_call(client().get, "/v1/memory/stats"))


@memory_app.command("consolidate")
def memory_consolidate() -> None:
    _print_json(_call(client(timeout=1800).post, "/v1/memory/consolidate"))


@memory_app.command("reindex")
def memory_reindex() -> None:
    """Rebuild the vector index from the database (stop the daemon first)."""
    s = settings()

    async def go() -> int:
        from sentientai.memory.factory import build_memory
        from sentientai.storage.database import Database

        db = Database(s.storage.resolved_database_url())
        await db.create_all()
        memory = build_memory(s, db)
        await memory.open()
        try:
            return await memory.reindex()
        finally:
            await memory.aclose()
            await db.close()

    typer.echo(f"reindexed {asyncio.run(go())} memories")


# ------------------------------------------------------------------ goals
@goals_app.command("list")
def goals_list() -> None:
    for g in _call(client().get, "/v1/goals"):
        typer.echo(
            f"{g['id']}  {g['origin']:<9} p={g['priority']:.1f} {g['progress']:.0%}  "
            f"{g['description']}"
        )


@goals_app.command("add")
def goals_add(description: str, priority: float = 0.6) -> None:
    typer.echo(
        _call(client().post, "/v1/goals", {"description": description, "priority": priority})["id"]
    )


# ----------------------------------------------------------------- safety
@app.command()
def kill(reason: str = typer.Argument("operator emergency stop")) -> None:
    """Engage the kill switch (works even if the API is down)."""
    try:
        client(timeout=5).post("/v1/kill", {"reason": reason})
        typer.secho("kill switch engaged via API", fg="red")
    except (ApiUnavailableError, ApiError):
        s = settings()
        s.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        s.kill_switch_path.write_text(json.dumps({"reason": reason, "by": "cli"}))
        typer.secho(f"API unreachable: wrote sentinel {s.kill_switch_path}", fg="red")


@app.command()
def resume() -> None:
    """Release the kill switch."""
    try:
        client(timeout=5).post("/v1/resume")
        typer.secho("kill switch released", fg="green")
    except (ApiUnavailableError, ApiError):
        s = settings()
        s.kill_switch_path.unlink(missing_ok=True)
        typer.secho(f"API unreachable: removed sentinel {s.kill_switch_path}", fg="green")


@audit_app.command("verify")
def audit_verify() -> None:
    """Verify the integrity of the audit chain."""
    from sentientai.safety.audit import AuditLog

    s = settings()
    key = s.safety.audit_hmac_key
    result = AuditLog(
        s.audit_path, hmac_key=key.get_secret_value().encode() if key else None, fsync=False
    ).verify()
    if result.ok:
        typer.secho(
            f"audit chain intact: {result.entries} entries in {result.segments} segment(s)",
            fg="green",
        )
    else:
        _fail(f"AUDIT CHAIN BROKEN at seq {result.bad_seq}: {result.error}", 3)


@audit_app.command("tail")
def audit_tail(n: int = 20, prefix: str | None = None) -> None:
    """Show the most recent audit entries."""
    from sentientai.safety.audit import AuditLog

    s = settings()
    for e in AuditLog(s.audit_path, fsync=False).tail(n, prefix):
        typer.echo(
            f"{e.seq:>7} {e.ts.isoformat()[:19]} {e.type:<26} {e.actor:<10} "
            f"{json.dumps(e.payload, default=str)[:140]}"
        )


# ---------------------------------------------------------- self-improvement
@improve_app.command("list")
def improve_list(status: str | None = None) -> None:
    for p in _call(client().get, "/v1/proposals", params={"status": status} if status else {}):
        typer.echo(f"{p['id']}  {p['status']:<15} {p['title']}  ({', '.join(p['files'])})")


@improve_app.command("show")
def improve_show(proposal_id: str) -> None:
    match = [p for p in _call(client().get, "/v1/proposals") if p["id"] == proposal_id]
    if not match:
        _fail("no such proposal")
    p = match[0]
    typer.echo(f"{p['title']} [{p['status']}]\n\n{p['rationale']}\n\n{p['diff']}")
    if p["report"]:
        _print_json(p["report"])


@improve_app.command("propose")
def improve_propose(
    insight: str | None = None, file: list[str] | None = typer.Option(None, "--file")
) -> None:
    p = _call(client(timeout=900).post, "/v1/proposals", {"insight": insight, "files": file})
    typer.echo(f"{p['id']}: {p['title']}\n\n{p['diff']}")


@improve_app.command("stage")
def improve_stage(proposal_id: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Apply a proposal in a throwaway worktree and run the test suite against it."""
    if not yes:
        improve_show(proposal_id)
        typer.confirm("Staging runs the test suite against this patch. Continue?", abort=True)
    p = _call(client(timeout=3600).post, f"/v1/proposals/{proposal_id}/stage")
    typer.echo(f"{p['id']}: {p['status']}")
    _print_json(p["report"])


@improve_app.command("apply")
def improve_apply(proposal_id: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Commit a staged proposal to a new branch (never merges or pushes)."""
    if not yes:
        typer.confirm(
            f"Commit proposal {proposal_id} to branch sentient/{proposal_id}?", abort=True
        )
    p = _call(client(timeout=600).post, f"/v1/proposals/{proposal_id}/apply")
    typer.echo(f"{p['id']}: {p['status']} on branch {p['branch']}")


@improve_app.command("reject")
def improve_reject(proposal_id: str, reason: str) -> None:
    p = _call(client().post, f"/v1/proposals/{proposal_id}/reject", {"reason": reason})
    typer.echo(f"{p['id']}: {p['status']}")


# ------------------------------------------------------------------- eval
@eval_app.command("run")
def eval_run(
    scenarios: Path = typer.Option(Path("evals/scenarios"), help="Scenario file or directory."),
    scripted: bool = typer.Option(False, help="Replay scenario scripts instead of calling models."),
    tag: list[str] | None = typer.Option(None, "--tag"),
    name: list[str] | None = typer.Option(None, "--name"),
    output: Path | None = typer.Option(
        None, help="Write the JSON report here (and a .md next to it)."
    ),
    min_pass_rate: float = typer.Option(1.0, help="Exit non-zero below this pass rate."),
) -> None:
    """Run evaluation scenarios against isolated runtimes."""
    from sentientai.evals.harness import EvalRunner, load_scenarios

    s = settings()
    configure_logging("WARNING", s.logging.format)
    selected = load_scenarios(scenarios, tags=tag, names=name)
    if not selected:
        _fail("no scenarios selected")
    report = asyncio.run(EvalRunner(s, scripted=scripted).run(selected))
    typer.echo(report.markdown())
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2))
        output.with_suffix(".md").write_text(report.markdown())
    if report.pass_rate < min_pass_rate:
        raise typer.Exit(1)


def main() -> None:
    app(prog_name="sentientai")


if __name__ == "__main__":  # pragma: no cover
    main()

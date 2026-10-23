from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest
import respx
from pydantic import Field

from sentientai.config import LimitsConfig, MemoryConfig, RiskLevel, SafetyConfig, WebConfig
from sentientai.core.events import EventBus
from sentientai.core.metrics import Metrics
from sentientai.errors import LoopBreakError, ToolError
from sentientai.ingestion.chunking import chunk_text
from sentientai.ingestion.loaders import load_bytes
from sentientai.ingestion.pipeline import IngestionPipeline
from sentientai.memory.system import MemorySystem
from sentientai.memory.vector import InMemoryVectorStore
from sentientai.memory.working import InMemoryWorkingMemory
from sentientai.models.embeddings import HashingEmbedder
from sentientai.models.types import ToolCall
from sentientai.safety.approvals import ApprovalGate
from sentientai.safety.audit import AuditLog
from sentientai.safety.killswitch import KillSwitch
from sentientai.safety.limits import LoopGuard
from sentientai.safety.policy import PolicyEngine
from sentientai.storage.database import Database
from sentientai.storage.records import ApprovalStatus, MemoryKind
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolRegistry, ToolResult
from sentientai.tools.executor import ToolExecutor, truncate_middle
from sentientai.tools.filesystem import Workspace, filesystem_tools
from sentientai.tools.memory_tools import memory_tools
from sentientai.tools.repository import (
    RepoCommit,
    RepoGit,
    RepoOverview,
    RepoSymbols,
    python_outline,
)
from sentientai.tools.web import (
    HttpRequest,
    UrlGuard,
    WebClient,
    WebFetch,
    html_to_text,
    parse_duckduckgo,
)

PUBLIC = "http://93.184.215.14"  # IP literal: passes the SSRF guard without DNS


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text(
        "def main():\n    return 'hello'\n\nclass Thing:\n    pass\n"
    )
    (root / "notes.md").write_text("# Notes\nThe GPU is a V340.\n")
    (root / ".env").write_text("SECRET=1\n")
    return Workspace([root], SafetyConfig().protected_paths)


def run(tool: Tool[Any], **args: Any) -> ToolResult:
    return asyncio.run(tool.run(tool.args_model.model_validate(args), ToolContext()))


# ---------------------------------------------------------------- workspace


def test_workspace_jail(ws: Workspace, tmp_path: Path) -> None:
    assert ws.resolve("notes.md") == ws.primary / "notes.md"
    with pytest.raises(ToolError, match="outside"):
        ws.resolve("../escape.txt")
    with pytest.raises(ToolError, match="outside"):
        ws.resolve("/etc/passwd")
    (tmp_path / "secret.txt").write_text("x")
    (ws.primary / "link").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(ToolError, match="outside"):
        ws.resolve("link")  # symlinks cannot escape
    assert ws.is_protected(ws.primary / ".env")
    assert ws.is_protected(ws.primary / ".git" / "config")
    assert not ws.is_protected(ws.primary / "notes.md")


async def test_filesystem_tools(ws: Workspace) -> None:
    tools = {t.name: t for t in filesystem_tools(ws)}
    ctx = ToolContext()

    async def call(name: str, **args: Any) -> ToolResult:
        tool = tools[name]
        return await tool.run(tool.args_model.model_validate(args), ctx)

    listing = await call("fs_list")
    assert (
        "src/" in listing.output and "notes.md" in listing.output and ".env" not in listing.output
    )
    read = await call("fs_read", path="src/app.py", max_lines=2)
    assert "1 | def main():" in read.output and "showing lines 1-2 of 5" in read.output
    search = await call("fs_search", pattern="v340")
    assert "notes.md:2:" in search.output
    assert (
        ".env" not in (await call("fs_search", pattern="SECRET")).output
    )  # protected files skipped

    assert (await call("fs_write", path="out/new.txt", content="a")).ok
    with pytest.raises(ToolError, match="exists"):
        await call("fs_write", path="out/new.txt", content="b")
    await call("fs_write", path="out/new.txt", content="b", mode="append")
    assert (ws.primary / "out/new.txt").read_text() == "ab"

    diff = await call("fs_edit", path="src/app.py", old="'hello'", new="'world'")
    assert "-    return 'hello'" in diff.output and "+    return 'world'" in diff.output
    with pytest.raises(ToolError, match="found 0"):
        await call("fs_edit", path="src/app.py", old="nope", new="x")

    write_risk = tools["fs_write"].assess(
        tools["fs_write"].args_model(path=".env", content=""), ctx
    )
    assert write_risk[0] is RiskLevel.CRITICAL
    read_risk = tools["fs_read"].assess(tools["fs_read"].args_model(path=".env"), ctx)
    assert read_risk[0] is RiskLevel.HIGH

    await call("fs_delete", path="out/new.txt")
    assert not (ws.primary / "out/new.txt").exists()
    (ws.primary / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert "binary" in (await call("fs_read", path="bin.dat")).output


# ----------------------------------------------------------------- executor


class SleepArgs(ToolArgs):
    seconds: float = Field(ge=0)


class SleepTool(Tool[SleepArgs]):
    name = "sleep"
    description = "sleep"
    args_model = SleepArgs
    timeout_s = 0.2

    async def run(self, args: SleepArgs, ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(args.seconds)
        return ToolResult.success("x" * 5000)


class CrashTool(Tool[ToolArgs]):
    name = "crash"
    description = "crash"
    args_model = ToolArgs

    async def run(self, args: ToolArgs, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")


@pytest.fixture
async def executor(
    sqlite_db: Database, ws: Workspace, tmp_path: Path
) -> tuple[ToolExecutor, ApprovalGate, AuditLog]:
    bus = EventBus()
    audit = AuditLog(tmp_path / "audit", fsync=False)
    ks = KillSwitch(tmp_path / "KILL")
    gate = ApprovalGate(
        sqlite_db.approvals, bus=bus, audit=audit, kill_switch=ks, timeout_s=2, poll_interval_s=0.02
    )
    registry = ToolRegistry()
    for tool in [*filesystem_tools(ws), SleepTool(), CrashTool()]:
        registry.register(tool)
    ex = ToolExecutor(
        registry,
        policy=PolicyEngine(SafetyConfig()),
        approvals=gate,
        audit=audit,
        bus=bus,
        invocations=sqlite_db.tool_invocations,
        metrics=Metrics(),
        max_output_chars=1000,
    )
    return ex, gate, audit


async def test_executor_validation_and_unknown(
    executor: tuple[ToolExecutor, ApprovalGate, AuditLog],
) -> None:
    ex, _, _ = executor
    unknown = await ex.execute(ToolCall(name="nope"), ToolContext())
    assert not unknown.ok and "Available tools" in unknown.output
    invalid = await ex.execute(ToolCall(name="fs_read", arguments={"pth": "x"}), ToolContext())
    assert not invalid.ok and "path: Field required" in invalid.output


async def test_executor_policy_denial(
    executor: tuple[ToolExecutor, ApprovalGate, AuditLog],
) -> None:
    ex, _, audit = executor
    denied = await ex.execute(
        ToolCall(name="fs_write", arguments={"path": ".env", "content": "x"}), ToolContext()
    )
    assert not denied.ok and "denied by safety policy" in denied.output
    assert any(
        e.type == "tool.requested" and e.payload["decision"] == "deny" for e in audit.tail(5)
    )


async def test_executor_approval_flow(
    executor: tuple[ToolExecutor, ApprovalGate, AuditLog], ws: Workspace
) -> None:
    ex, gate, _ = executor
    (ws.primary / "doomed.txt").write_text("x")

    async def operator(approve: bool) -> None:
        while not await gate.pending():
            await asyncio.sleep(0.01)
        [req] = await gate.pending()
        await gate.decide(
            req.id, approved=approve, by="op", note="no thanks" if not approve else None
        )

    call = ToolCall(name="fs_delete", arguments={"path": "doomed.txt"})
    denied, _ = await asyncio.gather(ex.execute(call, ToolContext(task_id="t1")), operator(False))
    assert not denied.ok and "denied" in denied.output and "no thanks" in denied.output
    assert (ws.primary / "doomed.txt").exists()

    approved, _ = await asyncio.gather(ex.execute(call, ToolContext(task_id="t1")), operator(True))
    assert approved.ok and not (ws.primary / "doomed.txt").exists()


async def test_executor_autonomous_needs_approval_for_writes(
    executor: tuple[ToolExecutor, ApprovalGate, AuditLog],
) -> None:
    ex, gate, _ = executor
    gate.timeout_s = 0.1
    result = await ex.execute(
        ToolCall(name="fs_write", arguments={"path": "a.txt", "content": "x"}),
        ToolContext(autonomous=True),
    )
    assert not result.ok and "expired" in result.output
    [record] = await gate.repo.list()
    assert record.status is ApprovalStatus.EXPIRED and record.autonomous


async def test_executor_timeout_crash_truncation_and_stats(
    executor: tuple[ToolExecutor, ApprovalGate, AuditLog], sqlite_db: Database
) -> None:
    ex, _, audit = executor
    slow = await ex.execute(ToolCall(name="sleep", arguments={"seconds": 1}), ToolContext())
    assert not slow.ok and "timed out" in slow.output
    crash = await ex.execute(ToolCall(name="crash"), ToolContext())
    assert not crash.ok and "kaboom" in crash.output
    big = await ex.execute(ToolCall(name="sleep", arguments={"seconds": 0}), ToolContext())
    assert big.ok and big.truncated and "characters omitted" in big.output
    stats = {s.tool: s for s in await sqlite_db.tool_invocations.stats()}
    assert stats["sleep"].calls == 2 and stats["sleep"].failures == 1
    assert audit.verify().ok


async def test_executor_loop_guard_propagates(
    executor: tuple[ToolExecutor, ApprovalGate, AuditLog],
) -> None:
    ex, _, _ = executor
    guard = LoopGuard(LimitsConfig(max_repeat_calls=2, max_consecutive_failures=10))
    call = ToolCall(name="fs_read", arguments={"path": "notes.md"})
    await ex.execute(call, ToolContext(), guard)
    await ex.execute(call, ToolContext(), guard)
    with pytest.raises(LoopBreakError):
        await ex.execute(call, ToolContext(), guard)


def test_truncate_middle() -> None:
    text, cut = truncate_middle("a" * 50 + "b" * 50, 20)
    assert cut and text.startswith("a" * 14) and text.endswith("b" * 6)


# --------------------------------------------------------------- repository


@pytest.fixture
def git_ws(ws: Workspace) -> Workspace:
    root = ws.primary
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "src", "notes.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    return ws


async def test_repository_tools(git_ws: Workspace) -> None:
    ctx = ToolContext()
    overview = await RepoOverview(git_ws).run(RepoOverview.args_model(), ctx)
    assert "Python: 1 / 5" in overview.output and "branch main" in overview.output

    git = RepoGit(git_ws)
    log = await git.run(git.args_model(command="log", args=["--oneline"]), ctx)
    assert "init" in log.output
    with pytest.raises(ToolError, match="not allowed"):
        await git.run(git.args_model(command="log", args=["-c", "core.pager=evil"]), ctx)
    with pytest.raises(ToolError, match="read-only"):
        await git.run(git.args_model(command="branch", args=["-D", "main"]), ctx)

    symbols = await RepoSymbols(git_ws).run(RepoSymbols.args_model(path="src"), ctx)
    assert "def main()" in symbols.output and "class Thing" in symbols.output

    (git_ws.primary / "notes.md").write_text("# Notes\nchanged\n")
    commit = RepoCommit(git_ws)
    assert commit.risk is RiskLevel.HIGH
    result = await commit.run(commit.args_model(message="update notes"), ctx)
    assert result.ok and "update notes" in result.output


def test_python_outline() -> None:
    assert python_outline("async def f(a, b=1):\n    '''Doc line.'''\n") == [
        "L1 async def f(a, b=1)  - Doc line."
    ]


# ---------------------------------------------------------------------- web


async def test_url_guard() -> None:
    guard = UrlGuard(WebConfig())
    assert await guard.check(f"{PUBLIC}/x")
    for bad in [
        "file:///etc/passwd",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest",
        "http://[::1]/",
        "http://localhost/",
    ]:
        with pytest.raises(ToolError):
            await guard.check(bad)
    scoped = UrlGuard(WebConfig(allowed_domains=["docs.python.org"], allow_private_networks=True))
    await scoped.check("https://docs.python.org/3/")
    with pytest.raises(ToolError, match="not in the allowed"):
        await scoped.check("https://evil.example/")
    blocked = UrlGuard(WebConfig(blocked_domains=["example.com"], allow_private_networks=True))
    with pytest.raises(ToolError, match="blocked"):
        await blocked.check("https://sub.example.com/")


@respx.mock
async def test_web_fetch_html_and_redirect_guard() -> None:
    html = (
        "<html><head><title>GPU Guide</title><script>evil()</script></head><body><nav>menu</nav>"
        "<main><h1>Inference</h1><p>Use llama.cpp on ROCm.</p><ul><li>V340</li></ul></main></body></html>"
    )
    respx.get(f"{PUBLIC}/page").respond(200, html=html)
    respx.get(f"{PUBLIC}/hop").respond(302, headers={"location": "/page"})
    respx.get(f"{PUBLIC}/ssrf").respond(302, headers={"location": "http://169.254.169.254/meta"})
    client = WebClient(WebConfig())
    fetch = WebFetch(client)
    result = await fetch.run(fetch.args_model(url=f"{PUBLIC}/hop"), ToolContext())
    assert (
        "# GPU Guide" in result.output
        and "# Inference" in result.output
        and "- V340" in result.output
    )
    assert "evil()" not in result.output and "menu" not in result.output
    with pytest.raises(ToolError, match="SSRF"):
        await fetch.run(fetch.args_model(url=f"{PUBLIC}/ssrf"), ToolContext())
    await client.aclose()


@respx.mock
async def test_http_request_risk_and_credentials() -> None:
    route = respx.post(f"{PUBLIC}/api").respond(201, json={"id": 7})
    config = WebConfig(credentials={"svc": {"Authorization": "Bearer s3cret"}})
    tool = HttpRequest(WebClient(config))
    args = tool.args_model(method="POST", url=f"{PUBLIC}/api", json_body={"a": 1}, credential="svc")
    assert tool.assess(args, ToolContext())[0] is RiskLevel.HIGH
    assert tool.assess(tool.args_model(url=f"{PUBLIC}/api"), ToolContext())[0] is RiskLevel.LOW
    result = await tool.run(args, ToolContext())
    assert result.ok and "HTTP 201" in result.output and "s3cret" not in result.output
    assert route.calls.last.request.headers["authorization"] == "Bearer s3cret"


def test_html_and_duckduckgo_parsing() -> None:
    title, text = html_to_text("<title>T</title><p>Hello <b>world</b></p><pre>x = 1</pre>")
    assert title == "T" and "Hello world" in text and "```\nx = 1\n```" in text
    ddg = """<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&rut=1">Example A</a>
             <a class="result__snippet">Snippet A</a></div>"""
    [hit] = parse_duckduckgo(ddg, 5)
    assert hit == {"title": "Example A", "url": "https://example.org/a", "snippet": "Snippet A"}


# ---------------------------------------------------------------- ingestion


def test_chunking_modes() -> None:
    md = (
        "# Guide\nIntro para.\n\n## Install\n"
        + "Install step sentence. " * 80
        + "\n\n## Usage\nRun it.\n```\n# not a heading\n```"
    )
    chunks = chunk_text(md, fmt="markdown", target_tokens=120, overlap_tokens=20)
    assert chunks[0].heading == "Guide" and any(c.heading == "Guide > Install" for c in chunks)
    assert all(c.tokens <= 160 for c in chunks)
    assert any("# not a heading" in c.text for c in chunks)

    code = "import os\n\n" + "\n\n".join(f"def f{i}():\n    return {i}\n" for i in range(30))
    code_chunks = chunk_text(code, fmt="code", target_tokens=60)
    assert len(code_chunks) > 1 and all(
        "def f" in c.text or "import" in c.text for c in code_chunks
    )

    assert chunk_text("") == []
    long_line = chunk_text("x" * 5000, target_tokens=100)
    assert len(long_line) >= 12


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return out


def test_pdf_loader() -> None:
    doc = load_bytes(_minimal_pdf("Hello PDF World"), source="x.pdf", name="x.pdf")
    assert "Hello PDF World" in doc.text and doc.mime == "application/pdf"


@pytest.fixture
async def memory(sqlite_db: Database, tmp_path: Path) -> MemorySystem:
    system = MemorySystem(
        sqlite_db,
        InMemoryVectorStore(256),
        HashingEmbedder(256),
        InMemoryWorkingMemory(),
        MemoryConfig(),
    )
    return system


async def test_ingestion_pipeline(memory: MemorySystem, ws: Workspace, tmp_path: Path) -> None:
    pipeline = IngestionPipeline(memory, tmp_path / "archive", target_tokens=80)
    (ws.primary / "docs").mkdir()
    (ws.primary / "docs" / "gpu.md").write_text(
        "# Hardware\n\n## Primary\nTwo Radeon Pro V340 cards run the heavy model.\n"
    )
    (ws.primary / "docs" / "paper.pdf").write_bytes(_minimal_pdf("Attention is all you need"))
    (ws.primary / "docs" / "image.png").write_bytes(b"\x89PNG\x00")

    results = await pipeline.ingest_directory(ws.primary / "docs")
    assert sorted(r.title for r in results) == ["Hardware", "paper"]
    assert all(r.ok and r.chunks >= 1 for r in results)
    again = await pipeline.ingest_path(ws.primary / "docs" / "gpu.md")
    assert again.skipped
    doc = await memory.db.documents.get(results[0].document_id or "")
    assert doc is not None and Path(doc.archive_path or "").exists()

    hits = await memory.recall("which cards run the heavy model", kinds=[MemoryKind.DOCUMENT])
    assert (
        hits
        and "V340" in hits[0].record.content
        and hits[0].record.metadata["heading"] == "Hardware > Primary"
    )

    note = await pipeline.ingest_text(
        "Remember the staging database runs on port 5433.", title="Ops note"
    )
    assert note.chunks == 1


async def test_memory_tools(memory: MemorySystem, ws: Workspace, tmp_path: Path) -> None:
    pipeline = IngestionPipeline(memory, tmp_path / "archive")
    tools = {t.name: t for t in memory_tools(memory, pipeline, ws)}
    ctx = ToolContext(task_id="t9")

    async def call(name: str, **args: Any) -> ToolResult:
        tool = tools[name]
        return await tool.run(tool.args_model.model_validate(args), ctx)

    stored = await call(
        "memory_remember", content="Operator timezone is CET", subject_key="operator.timezone"
    )
    memory_id = stored.output.split()[-1]
    recalled = await call("memory_recall", query="what timezone is the operator in?")
    assert memory_id in recalled.output and "CET" in recalled.output
    assert (await call("memory_forget", memory_id=memory_id, reason="test")).ok
    ingest = await call("ingest", source="notes.md")
    assert "ingested 'Notes'" in ingest.output


def test_invented_tool_tags_still_run_the_tool() -> None:
    from sentientai.models.toolproto import parse_tool_calls

    known = {"avatar_design", "memory_recall"}
    text = (
        "I'll make this for you:\n<avatar_design>\n"
        '{"description": "a surreal room", "parts": [{"shape": "sphere"}]}\n</avatar_design>\nEnjoy!'
    )
    calls, visible = parse_tool_calls(text, known)
    assert [c.name for c in calls] == ["avatar_design"]
    assert calls[0].arguments["description"] == "a surreal room"
    assert "<avatar_design>" not in visible and "Enjoy!" in visible
    wrapped = '<memory_recall>{"arguments": {"query": "cats"}}</memory_recall>'
    assert parse_tool_calls(wrapped, known)[0][0].arguments == {"query": "cats"}
    # A tag that isn't a tool is just text.
    calls, visible = parse_tool_calls('<b>{"x": 1}</b>', known)
    assert calls == [] and "<b>" in visible


def test_the_sandbox_tells_fuzzy_whether_its_code_can_reach_the_internet(tmp_path: Path) -> None:
    from sentientai.config import SandboxConfig
    from sentientai.tools.sandbox import SandboxPython

    class Backend:
        isolated = True

    offline = SandboxPython(Backend(), SandboxConfig(), tmp_path)  # type: ignore[arg-type]
    online = SandboxPython(Backend(), SandboxConfig(network=True), tmp_path)  # type: ignore[arg-type]
    assert "no network" in offline.description
    assert "internet access" in online.description and "no network" not in online.description

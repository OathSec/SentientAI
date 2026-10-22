from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from sentientai.config import (
    MemoryConfig,
    RoutingConfig,
    ScheduleConfig,
    SelfImprovementConfig,
    WatchConfig,
)
from sentientai.core.clock import FakeClock
from sentientai.core.events import EventBus
from sentientai.errors import ConfigError, ToolError
from sentientai.ingestion.pipeline import IngestionPipeline
from sentientai.memory.system import MemorySystem
from sentientai.memory.vector import InMemoryVectorStore
from sentientai.memory.working import InMemoryWorkingMemory
from sentientai.models.embeddings import HashingEmbedder
from sentientai.models.router import ModelRouter
from sentientai.models.scripted import ScriptedLLM
from sentientai.models.types import ModelTier
from sentientai.reflection.analyzer import PerformanceAnalyzer
from sentientai.reflection.improvement import SelfImprovementEngine, diff_files
from sentientai.runtime.scheduler import CronExpression, Scheduler
from sentientai.runtime.watchers import FolderWatcher
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import ProposalStatus, ToolInvocationRecord

# ---------------------------------------------------------------- scheduler


def test_cron_parsing_and_matching() -> None:
    cron = CronExpression("*/15 9-17 * * 1-5")
    assert cron.matches(datetime(2026, 3, 2, 9, 30, tzinfo=UTC))  # Monday
    assert not cron.matches(datetime(2026, 3, 2, 9, 31, tzinfo=UTC))
    assert not cron.matches(datetime(2026, 3, 1, 9, 30, tzinfo=UTC))  # Sunday
    assert cron.next_after(datetime(2026, 3, 2, 17, 50, tzinfo=UTC)) == datetime(
        2026, 3, 3, 9, 0, tzinfo=UTC
    )
    assert CronExpression("@daily").next_after(datetime(2026, 1, 1, 12, 0, tzinfo=UTC)) == datetime(
        2026, 1, 2, 0, 0, tzinfo=UTC
    )
    assert CronExpression("0 0 * * 7").matches(datetime(2026, 3, 1, 0, 0, tzinfo=UTC))  # 7 = Sunday
    either = CronExpression("0 0 13 * 5")  # 13th OR Friday (Vixie semantics)
    assert either.matches(datetime(2026, 3, 13, 0, 0, tzinfo=UTC))
    assert either.matches(datetime(2026, 3, 6, 0, 0, tzinfo=UTC))
    assert (
        CronExpression("0 0 29 2 *")
        .next_after(datetime(2026, 1, 1, tzinfo=UTC), max_days=1500)
        .year
        == 2028
    )
    for bad in ["* * *", "61 * * * *", "*/0 * * * *", "5-1 * * * *"]:
        with pytest.raises(ValueError, match=r"cron|range|positive"):
            CronExpression(bad)


async def test_scheduler_interval_cron_and_persistence(sqlite_db: Database) -> None:
    clock = FakeClock(datetime(2026, 3, 2, 8, 59, tzinfo=UTC))
    fired: list[str] = []

    async def submit(schedule: ScheduleConfig) -> None:
        fired.append(schedule.name)

    schedules = [
        ScheduleConfig(name="heartbeat", prompt="check", interval_s=600),
        ScheduleConfig(name="standup", prompt="summarize", cron="0 9 * * 1-5"),
        ScheduleConfig(name="off", prompt="x", interval_s=1, enabled=False),
    ]
    scheduler = Scheduler(schedules, submit, sqlite_db.state, clock=clock)
    assert await scheduler.tick() == ["heartbeat"]
    assert await scheduler.tick() == []
    clock.advance(60)  # 09:00 Monday
    assert await scheduler.tick() == ["standup"]
    # a rebooted scheduler reads persisted last-run times and does not double fire
    rebooted = Scheduler(schedules, submit, sqlite_db.state, clock=clock)
    assert await rebooted.tick() == []
    clock.advance(600)
    assert await rebooted.tick() == ["heartbeat"]


# ----------------------------------------------------------------- watchers


@pytest.fixture
def memory(sqlite_db: Database) -> MemorySystem:
    return MemorySystem(
        sqlite_db,
        InMemoryVectorStore(128),
        HashingEmbedder(128),
        InMemoryWorkingMemory(),
        MemoryConfig(),
    )


async def test_watcher_ingest_and_task_modes(memory: MemorySystem, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "existing.md").write_text("# Existing\nAlready here.")
    pipeline = IngestionPipeline(memory, tmp_path / "archive")
    submitted: list[Path] = []

    async def submit(config: WatchConfig, path: Path) -> None:
        submitted.append(path)

    ingest = FolderWatcher(
        WatchConfig(name="kb", path=inbox, pattern="*.md"),
        state=memory.db.state,
        pipeline=pipeline,
        submit_task=submit,
    )
    assert [p.name for p in await ingest.poll()] == ["existing.md"]  # initial sync ingests
    assert await ingest.poll() == []
    (inbox / "new.md").write_text("# New\nFresh knowledge about vector decay.")
    (inbox / "ignored.txt").write_text("not matching")
    assert [p.name for p in await ingest.poll()] == ["new.md"]
    assert len(await memory.db.documents.list()) == 2

    tasks = FolderWatcher(
        WatchConfig(name="triage", path=inbox, pattern="*", action="task"),
        state=memory.db.state,
        pipeline=pipeline,
        submit_task=submit,
    )
    assert await tasks.poll() == [] and submitted == []  # baseline, no flood
    (inbox / "report.pdf").write_bytes(b"%PDF-")
    await tasks.poll()
    assert [p.name for p in submitted] == ["report.pdf"]


# ----------------------------------------------------------------- analyzer


async def test_performance_analyzer(sqlite_db: Database, tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit", fsync=False)
    for i in range(10):
        await sqlite_db.tool_invocations.insert(
            ToolInvocationRecord(
                tool="web_fetch",
                args_digest="d",
                ok=i < 4,
                duration_ms=100.0,
                error=None if i < 4 else "HTTP 403",
            )
        )
        await sqlite_db.tool_invocations.insert(
            ToolInvocationRecord(
                tool="fs_read", args_digest="d", ok=True, duration_ms=60_000.0 if i > 5 else 5.0
            )
        )
    for _ in range(3):
        await audit.record("task.loop_break", limit="steps")
    report = await PerformanceAnalyzer(sqlite_db, audit).analyze()
    kinds = {(i.kind, i.subject) for i in report.insights}
    assert ("tool_failures", "web_fetch") in kinds
    assert ("tool_latency", "fs_read") in kinds
    assert ("loop_breaks", "steps") in kinds
    top = report.top()
    assert (
        top is not None and top.subject == "web_fetch" and "HTTP 403" in top.evidence["top_errors"]
    )


# --------------------------------------------------------- self-improvement

TOOL_SRC = """class DemoTool:
    name = "demo_tool"

    def run(self, x):
        return x + 1
"""
TEST_SRC = """from sentientai.tools.demo import DemoTool


def test_demo():
    assert DemoTool().run(1) == 2
"""
GOOD_DIFF = """--- a/src/sentientai/tools/demo.py
+++ b/src/sentientai/tools/demo.py
@@ -1,5 +1,6 @@
 class DemoTool:
     name = "demo_tool"

     def run(self, x):
+        # validated: x is an int
         return x + 1
"""
BREAKING_DIFF = GOOD_DIFF.replace("@@ -1,5 +1,6 @@", "@@ -1,5 +1,5 @@").replace(
    "+        # validated: x is an int\n         return x + 1",
    "-        return x + 1\n+        return x + 2",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "sentientai" / "tools").mkdir(parents=True)
    (repo / "src" / "sentientai" / "safety").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "sentientai" / "__init__.py").write_text("")
    (repo / "src" / "sentientai" / "tools" / "__init__.py").write_text("")
    (repo / "src" / "sentientai" / "tools" / "demo.py").write_text(TOOL_SRC)
    (repo / "src" / "sentientai" / "safety" / "policy.py").write_text("ALLOW = False\n")
    (repo / "tests" / "test_demo.py").write_text(TEST_SRC)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return repo


def _engine(
    sqlite_db: Database, repo: Path, tmp_path: Path, replies: list[Any]
) -> SelfImprovementEngine:
    llm = ScriptedLLM(replies)
    config = SelfImprovementConfig(
        enabled=True,
        source_root=repo,
        test_command=[sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
    )
    return SelfImprovementEngine(
        config,
        router=ModelRouter({ModelTier.HEAVY: llm}, RoutingConfig()),
        proposals=sqlite_db.proposals,
        audit=AuditLog(tmp_path / "audit", fsync=False),
        bus=EventBus(),
        analyzer=PerformanceAnalyzer(sqlite_db, AuditLog(tmp_path / "audit2", fsync=False)),
        staging_root=tmp_path / "staging",
    )


def _reply(diff: str, title: str = "Validate input") -> str:
    return json.dumps({"title": title, "rationale": "demo_tool crashes on bad input", "diff": diff})


async def test_self_improvement_lifecycle(
    sqlite_db: Database, source_repo: Path, tmp_path: Path
) -> None:
    engine = _engine(sqlite_db, source_repo, tmp_path, [_reply(GOOD_DIFF)])
    proposal = await engine.propose("demo_tool fails 40% of calls with TypeError")
    assert proposal.status is ProposalStatus.DRAFT and proposal.files == [
        "src/sentientai/tools/demo.py"
    ]
    request = engine.router.clients[ModelTier.HEAVY].requests[0]  # type: ignore[attr-defined]
    assert "=== src/sentientai/tools/demo.py ===" in request.messages[0].text

    staged = await engine.stage(proposal.id)
    assert staged.status is ProposalStatus.STAGED, staged.report
    assert (
        staged.report["tests"]["exit_code"] == 0
        and "1 passed" in staged.report["tests"]["output_tail"]
    )
    assert not (tmp_path / "staging" / proposal.id).exists()  # worktree cleaned up

    applied = await engine.apply(proposal.id, by="alice")
    assert applied.status is ProposalStatus.APPLIED and applied.branch == f"sentient/{proposal.id}"
    assert (
        source_repo / "src/sentientai/tools/demo.py"
    ).read_text() == TOOL_SRC  # operator tree untouched
    assert _git(source_repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    log = _git(source_repo, "log", "--format=%s%n%b", applied.branch)
    assert "Validate input" in log and "Approved-by: alice" in log
    assert "# validated" in _git(
        source_repo, "show", f"{applied.branch}:src/sentientai/tools/demo.py"
    )


async def test_self_improvement_guards(
    sqlite_db: Database, source_repo: Path, tmp_path: Path
) -> None:
    protected = GOOD_DIFF.replace("tools/demo.py", "safety/policy.py")
    engine = _engine(
        sqlite_db,
        source_repo,
        tmp_path,
        [_reply(protected), _reply("not a diff"), _reply(BREAKING_DIFF)],
    )
    with pytest.raises(ToolError, match="protected"):
        await engine.propose("demo_tool problem", files=["src/sentientai/tools/demo.py"])
    with pytest.raises(ToolError, match="unified diff"):
        await engine.propose("demo_tool problem")
    with pytest.raises(ToolError, match="escapes"):
        engine.validate_diff("--- a/../etc/passwd\n+++ b/../etc/passwd\n@@ -1 +1 @@\n-a\n+b\n")
    with pytest.raises(ToolError, match=r"outside|not a file"):
        await engine.propose("x" * 20, files=["../outside.py"])

    breaking = await engine.propose("demo_tool returns wrong values")
    staged = await engine.stage(breaking.id)
    assert (
        staged.status is ProposalStatus.STAGING_FAILED and staged.report["tests"]["exit_code"] != 0
    )
    with pytest.raises(ToolError, match="passed staging"):
        await engine.apply(breaking.id, by="alice")
    rejected = await engine.reject(breaking.id, by="alice", reason="breaks tests")
    assert rejected.status is ProposalStatus.REJECTED
    assert diff_files(GOOD_DIFF) == ["src/sentientai/tools/demo.py"]


async def test_self_improvement_disabled(
    sqlite_db: Database, source_repo: Path, tmp_path: Path
) -> None:
    engine = _engine(sqlite_db, source_repo, tmp_path, [])
    engine.config.enabled = False
    with pytest.raises(ConfigError):
        await engine.propose("anything at all")

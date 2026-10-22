"""Scenario-based evaluation of the full runtime.

Each scenario runs in an isolated, temporary runtime (fresh database, memory,
workspace and audit log). A simulated operator answers approval requests
according to the scenario. Checks cover the answer, tool usage, approvals,
side effects on files and what ended up in memory.

``scripted`` mode replays the scenario's ``script`` instead of calling real
models, which validates the harness and runtime wiring in CI; nightly runs
use the configured models to catch behavioural regressions.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from sentientai.config import Settings
from sentientai.evals.brain import Brain, call
from sentientai.models.types import LLMClient, Message, ModelTier
from sentientai.runtime.runtime import Runtime
from sentientai.storage.records import TaskSource, TaskStatus


class SeedFact(BaseModel):
    content: str
    subject_key: str | None = None
    importance: float = 0.7


class Setup(BaseModel):
    files: dict[str, str] = Field(default_factory=dict)
    facts: list[SeedFact] = Field(default_factory=list)


class MemoryCheck(BaseModel):
    query: str
    text: str


class Checks(BaseModel):
    status: Literal["succeeded", "failed", "any"] = "succeeded"
    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    answer_regex: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    approval_requested: bool | None = None
    files_contain: dict[str, str] = Field(default_factory=dict)
    files_exist: list[str] = Field(default_factory=list)
    files_absent: list[str] = Field(default_factory=list)
    memory_contains: list[MemoryCheck] = Field(default_factory=list)
    max_steps: int | None = None
    min_confidence: float | None = None


class ScriptStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    prompt: str
    setup: Setup = Field(default_factory=Setup)
    approvals: Literal["approve", "deny", "ignore"] = "deny"
    checks: Checks = Field(default_factory=Checks)
    script: list[str | ScriptStep] = Field(default_factory=list)
    timeout_s: float = 300.0

    def scripted_replies(self) -> list[Message | str | Exception]:
        replies: list[Message | str | Exception] = []
        for step in self.script:
            replies.append(step if isinstance(step, str) else call(step.tool, **step.args))
        return replies


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ScenarioResult(BaseModel):
    name: str
    passed: bool
    status: str
    answer: str | None
    duration_s: float
    steps: int | None
    confidence: float | None
    tools: list[str]
    checks: list[CheckResult]
    error: str | None = None


class EvalReport(BaseModel):
    mode: Literal["scripted", "live"]
    results: list[ScenarioResult]
    duration_s: float

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def markdown(self) -> str:
        lines = [
            f"# SentientAI evaluation ({self.mode})",
            "",
            f"**Pass rate:** {self.pass_rate:.0%} "
            f"({sum(r.passed for r in self.results)}/{len(self.results)})"
            f" in {self.duration_s:.1f}s",
            "",
            "| Scenario | Result | Status | Steps | Confidence | Failed checks |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.results:
            failed = "; ".join(f"{c.name}: {c.detail}" for c in r.checks if not c.passed) or (
                r.error or ""
            )
            conf = f"{r.confidence:.2f}" if r.confidence is not None else "-"
            lines.append(
                f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.status} "
                f"| {r.steps or '-'} | {conf} "
                f"| {failed.replace('|', '/')} |"
            )
        return "\n".join(lines) + "\n"


def load_scenarios(
    path: Path, *, tags: list[str] | None = None, names: list[str] | None = None
) -> list[Scenario]:
    files = sorted(path.glob("*.y*ml")) if path.is_dir() else [path]
    scenarios = []
    for file in files:
        data = yaml.safe_load(file.read_text())
        for item in data if isinstance(data, list) else [data]:
            scenario = Scenario.model_validate(item)
            if tags and not set(tags) & set(scenario.tags):
                continue
            if names and scenario.name not in names:
                continue
            scenarios.append(scenario)
    return scenarios


class EvalRunner:
    def __init__(self, base: Settings, *, scripted: bool = False) -> None:
        self.base = base
        self.scripted = scripted

    def _settings(self, root: Path) -> Settings:
        data = self.base.model_dump()
        data["storage"].update(
            data_dir=root / "data",
            database_url=None,
            redis_url=None,
            vector_backend="memory",
            archive_dir=None,
            qdrant_path=None,
        )
        data["safety"].update(
            workspace_roots=[root / "workspace"],
            audit_dir=None,
            kill_switch_file=None,
            audit_fsync=False,
        )
        data["runtime"].update(autonomy="reactive", schedules=[], watchers=[])
        data["self_improvement"]["enabled"] = False
        settings = Settings.model_validate(data)
        settings.ensure_directories()
        return settings

    async def _operator(self, rt: Runtime, policy: str, seen: list[str]) -> None:
        while True:
            for request in await rt.c.approvals.pending():
                if request.id in seen:
                    continue
                seen.append(request.id)
                if policy != "ignore":
                    await rt.c.approvals.decide(
                        request.id, approved=policy == "approve", by="eval-operator"
                    )
            await asyncio.sleep(0.05)

    async def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        root = Path(tempfile.mkdtemp(prefix="sentient-eval-"))
        started = time.monotonic()
        rt: Runtime | None = None
        operator: asyncio.Task[None] | None = None
        try:
            settings = self._settings(root)
            if scenario.approvals == "ignore":
                settings.safety.approval_timeout_s = 1.0
            workspace = settings.safety.workspace_roots[0]
            for rel, content in scenario.setup.files.items():
                target = workspace / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            clients: dict[ModelTier, LLMClient] | None = None
            if self.scripted:
                brain = Brain(scenario.scripted_replies())
                clients = {ModelTier.FAST: brain.llm("fast"), ModelTier.HEAVY: brain.llm("heavy")}
            rt = await Runtime.create(settings, clients=clients)
            for fact in scenario.setup.facts:
                await rt.c.memory.remember(
                    fact.content,
                    importance=fact.importance,
                    subject_key=fact.subject_key,
                    source="eval-setup",
                )
            await rt.start()
            approvals_seen: list[str] = []
            operator = asyncio.create_task(self._operator(rt, scenario.approvals, approvals_seen))
            task = await rt.submit(scenario.prompt, source=TaskSource.API)
            final = await rt.wait(task.id, timeout=scenario.timeout_s)
            checks = await self._check(
                scenario,
                rt,
                final.status,
                final.result or "",
                final.metadata,
                bool(approvals_seen),
                workspace,
            )
            return ScenarioResult(
                name=scenario.name,
                passed=all(c.passed for c in checks),
                status=final.status.value,
                answer=final.result,
                duration_s=round(time.monotonic() - started, 2),
                steps=final.metadata.get("steps"),
                confidence=final.metadata.get("confidence"),
                tools=list(final.metadata.get("tools") or []),
                checks=checks,
            )
        except Exception as exc:
            return ScenarioResult(
                name=scenario.name,
                passed=False,
                status="error",
                answer=None,
                duration_s=round(time.monotonic() - started, 2),
                steps=None,
                confidence=None,
                tools=[],
                checks=[],
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if operator is not None:
                operator.cancel()
                await asyncio.gather(operator, return_exceptions=True)
            if rt is not None:
                await rt.stop("eval finished")
            shutil.rmtree(root, ignore_errors=True)

    async def _check(
        self,
        scenario: Scenario,
        rt: Runtime,
        status: TaskStatus,
        answer: str,
        metadata: dict[str, Any],
        approval_requested: bool,
        workspace: Path,
    ) -> list[CheckResult]:
        c = scenario.checks
        results: list[CheckResult] = []
        tools = set(metadata.get("tools") or [])
        lowered = answer.lower()

        def add(name: str, passed: bool, detail: str = "") -> None:
            results.append(CheckResult(name=name, passed=passed, detail="" if passed else detail))

        if c.status != "any":
            add("status", status.value == c.status, f"expected {c.status}, got {status.value}")
        for text in c.answer_contains:
            add(f"answer contains {text!r}", text.lower() in lowered, "missing from answer")
        for text in c.answer_not_contains:
            add(f"answer excludes {text!r}", text.lower() not in lowered, "present in answer")
        if c.answer_regex:
            add(
                "answer matches regex",
                re.search(c.answer_regex, answer, re.I | re.S) is not None,
                c.answer_regex,
            )
        for tool in c.tools_called:
            add(f"called {tool}", tool in tools, f"tools used: {sorted(tools)}")
        for tool in c.tools_not_called:
            add(f"did not call {tool}", tool not in tools, "tool was called")
        if c.approval_requested is not None:
            add(
                "approval requested" if c.approval_requested else "no approval requested",
                approval_requested == c.approval_requested,
                f"approval requested={approval_requested}",
            )
        for rel, text in c.files_contain.items():
            path = workspace / rel
            ok = path.is_file() and text in path.read_text(errors="replace")
            add(f"{rel} contains {text!r}", ok, "file missing or content absent")
        for rel in c.files_exist:
            add(f"{rel} exists", (workspace / rel).exists(), "missing")
        for rel in c.files_absent:
            add(f"{rel} absent", not (workspace / rel).exists(), "still exists")
        for mem in c.memory_contains:
            hits = await rt.c.memory.recall(mem.query, k=8, touch=False)
            add(
                f"memory recalls {mem.text!r}",
                any(mem.text.lower() in h.record.content.lower() for h in hits),
                "not found in memory",
            )
        if c.max_steps is not None:
            steps = int(metadata.get("steps") or 0)
            add(f"steps <= {c.max_steps}", steps <= c.max_steps, f"took {steps} steps")
        if c.min_confidence is not None:
            conf = metadata.get("confidence")
            add(
                f"confidence >= {c.min_confidence}",
                conf is not None and conf >= c.min_confidence,
                f"confidence {conf}",
            )
        return results

    async def run(self, scenarios: list[Scenario]) -> EvalReport:
        started = time.monotonic()
        results = [await self.run_scenario(s) for s in scenarios]
        return EvalReport(
            mode="scripted" if self.scripted else "live",
            results=results,
            duration_s=round(time.monotonic() - started, 2),
        )

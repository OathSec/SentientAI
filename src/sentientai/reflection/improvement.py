"""Recursive self-improvement under human supervision.

1. **propose** - the agent (or operator) turns an insight into a minimal
   unified diff against the runtime's own source. The diff is validated:
   protected paths (safety layer, config, CI, deploy) are untouchable, size
   is bounded and it must apply cleanly. Nothing is executed.
2. **stage** - operator-triggered: the patch is applied to a throwaway git
   worktree and the test suite (and optional benchmark) runs against it,
   optionally inside a network-less container.
3. **apply** - operator-only: the patch is committed to a new branch
   ``sentient/<proposal-id>``. The operator's working tree is never touched
   and nothing is merged or pushed automatically.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from sentientai import prompts
from sentientai.config import SelfImprovementConfig
from sentientai.core.clock import utcnow
from sentientai.core.events import EventBus
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.errors import ConfigError, ToolError
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.reflection.analyzer import PerformanceAnalyzer
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import ProposalRepository
from sentientai.storage.records import ProposalRecord, ProposalStatus

log = get_logger(__name__)

_DIFF_PATH = re.compile(r"^(?:\+\+\+|---) (?:[ab]/)?(\S+)", re.MULTILINE)


async def _run(
    cmd: list[str], *, cwd: Path, timeout: float, env: dict[str, str] | None = None
) -> tuple[int, str, float]:
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, f"timed out after {timeout:.0f}s", time.monotonic() - started
    return proc.returncode or 0, out.decode("utf-8", errors="replace"), time.monotonic() - started


def diff_files(diff: str) -> list[str]:
    files = []
    for path in _DIFF_PATH.findall(diff):
        if path != "/dev/null" and path not in files:
            files.append(path)
    return files


class SelfImprovementEngine:
    def __init__(
        self,
        config: SelfImprovementConfig,
        *,
        router: ModelRouter,
        proposals: ProposalRepository,
        audit: AuditLog,
        bus: EventBus,
        analyzer: PerformanceAnalyzer,
        staging_root: Path,
    ) -> None:
        self.config = config
        self.router = router
        self.proposals = proposals
        self.audit = audit
        self.bus = bus
        self.analyzer = analyzer
        self.staging_root = staging_root
        self.root = config.source_root.resolve()

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise ConfigError("self-improvement is disabled (self_improvement.enabled=false)")

    # ------------------------------------------------------------ validate
    def validate_diff(self, diff: str) -> list[str]:
        if not diff.strip() or "@@" not in diff:
            raise ToolError("not a unified diff")
        if "GIT binary patch" in diff:
            raise ToolError("binary patches are not allowed")
        files = diff_files(diff)
        if not files:
            raise ToolError("diff names no files")
        for path in files:
            pure = Path(path)
            if pure.is_absolute() or ".." in pure.parts:
                raise ToolError(f"path {path!r} escapes the repository")
            for pattern in self.config.protected_globs:
                if fnmatch.fnmatch(path, pattern):
                    raise ToolError(
                        f"{path} is protected ({pattern}); self-modification of it is not allowed"
                    )
        changed = sum(
            1
            for line in diff.splitlines()
            if line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))
        )
        if changed > self.config.max_diff_lines:
            raise ToolError(f"diff changes {changed} lines (limit {self.config.max_diff_lines})")
        return files

    async def _check_applies(self, diff: str, cwd: Path) -> None:
        patch = self.staging_root / f"check-{os.getpid()}-{time.monotonic_ns()}.diff"
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(diff if diff.endswith("\n") else diff + "\n")
        try:
            code, out, _ = await _run(["git", "apply", "--check", str(patch)], cwd=cwd, timeout=60)
        finally:
            patch.unlink(missing_ok=True)
        if code != 0:
            raise ToolError(f"patch does not apply: {out.strip()[:500]}")

    # ------------------------------------------------------------- propose
    def _relevant_sources(
        self, insight: str, files: list[str] | None, max_chars: int = 12_000
    ) -> dict[str, str]:
        chosen: list[Path] = []
        if files:
            chosen = [(self.root / f).resolve() for f in files]
        else:
            tool_names = set(re.findall(r"\b([a-z]+_[a-z_]+)\b", insight))
            for path in sorted((self.root / "src").rglob("*.py")):
                text = path.read_text(encoding="utf-8", errors="replace")
                if any(f'name = "{name}"' in text for name in tool_names):
                    chosen.append(path)
        sources: dict[str, str] = {}
        for path in chosen[:4]:
            if not path.is_file() or not path.is_relative_to(self.root):
                raise ToolError(f"{path} is not a file inside the source root")
            sources[str(path.relative_to(self.root))] = path.read_text(encoding="utf-8")[:max_chars]
        return sources

    async def propose(
        self, insight: str | None = None, files: list[str] | None = None, *, by: str = "agent"
    ) -> ProposalRecord:
        self._require_enabled()
        if not insight:
            report = await self.analyzer.analyze()
            top = report.top()
            if top is None:
                raise ToolError("no performance problems found to improve on")
            insight = f"{top.message}. Evidence: {top.evidence}"
        sources = self._relevant_sources(insight, files)
        if not sources:
            raise ToolError(
                "could not identify source files for this insight; pass files explicitly"
            )
        rendered = "\n\n".join(f"=== {path} ===\n{text}" for path, text in sources.items())
        response = await self.router.chat(
            ChatRequest(
                messages=[
                    Message.user(
                        prompts.PROPOSE_IMPROVEMENT.format(insight=insight, sources=rendered)
                    )
                ],
                temperature=0.2,
                json_mode=True,
            ),
            kind=TaskKind.IMPROVE,
        )
        data = extract_object(response.message.text) or {}
        diff = str(data.get("diff") or "")
        touched = self.validate_diff(diff)
        await self._check_applies(diff, self.root)
        proposal = ProposalRecord(
            title=str(data.get("title") or "Untitled improvement")[:200],
            rationale=f"{data.get('rationale') or ''}\n\nTriggered by: {insight}".strip(),
            diff=diff if diff.endswith("\n") else diff + "\n",
            files=touched,
        )
        await self.proposals.insert(proposal)
        await self.audit.record(
            "improvement.proposed",
            actor=by,
            proposal_id=proposal.id,
            title=proposal.title,
            files=touched,
            diff=proposal.diff,
        )
        await self.bus.publish(
            "improvement.proposed", proposal_id=proposal.id, title=proposal.title
        )
        return proposal

    # --------------------------------------------------------------- stage
    async def _worktree(self, target: Path, *, branch: str | None = None) -> None:
        if (self.root / ".git").exists():
            args = (
                ["git", "worktree", "add"]
                + (["-b", branch] if branch else ["--detach"])
                + [str(target), "HEAD"]
            )
            code, out, _ = await _run(args, cwd=self.root, timeout=120)
            if code != 0:
                raise ToolError(f"git worktree failed: {out.strip()[:400]}")
        else:
            if branch:
                raise ToolError(
                    "applying proposals requires the source root to be a git repository"
                )
            shutil.copytree(
                self.root,
                target,
                ignore=shutil.ignore_patterns(".venv", "data", "__pycache__", ".git"),
            )

    async def _remove_worktree(self, target: Path) -> None:
        if (self.root / ".git").exists():
            await _run(
                ["git", "worktree", "remove", "--force", str(target)], cwd=self.root, timeout=120
            )
        shutil.rmtree(target, ignore_errors=True)

    async def _run_suite(self, cmd: list[str], cwd: Path) -> tuple[int, str, float]:
        if self.config.staging_image:
            docker = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "-v",
                f"{cwd}:/src",
                "-w",
                "/src",
                "-e",
                "PYTHONPATH=/src/src",
                self.config.staging_image,
                *cmd,
            ]
            return await _run(docker, cwd=cwd, timeout=self.config.timeout_s)
        env = {
            **os.environ,
            "PYTHONPATH": f"{cwd / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
        return await _run(cmd, cwd=cwd, timeout=self.config.timeout_s, env=env)

    async def stage(self, proposal_id: str, *, by: str = "operator") -> ProposalRecord:
        self._require_enabled()
        proposal = await self.proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.status not in {ProposalStatus.DRAFT, ProposalStatus.STAGING_FAILED}:
            raise ToolError(f"proposal is {proposal.status.value}; only drafts can be staged")
        self.validate_diff(proposal.diff)
        target = self.staging_root / proposal.id
        if target.exists():
            await self._remove_worktree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {"staged_at": utcnow().isoformat(), "by": by}
        try:
            await self._worktree(target)
            patch = target / ".sentient-proposal.diff"
            patch.write_text(proposal.diff)
            code, out, _ = await _run(["git", "apply", str(patch)], cwd=target, timeout=60)
            patch.unlink(missing_ok=True)
            if code != 0:
                report["apply"] = out[-2000:]
                passed = False
            else:
                code, out, secs = await self._run_suite(self.config.test_command, target)
                report["tests"] = {
                    "exit_code": code,
                    "seconds": round(secs, 1),
                    "output_tail": out[-4000:],
                }
                passed = code == 0
                if passed and self.config.benchmark_command:
                    bcode, bout, bsecs = await self._run_suite(
                        self.config.benchmark_command, target
                    )
                    report["benchmark"] = {
                        "exit_code": bcode,
                        "seconds": round(bsecs, 1),
                        "output_tail": bout[-4000:],
                    }
                    passed = bcode == 0
        finally:
            await self._remove_worktree(target)
        proposal.status = ProposalStatus.STAGED if passed else ProposalStatus.STAGING_FAILED
        proposal.report = report
        proposal.updated_at = utcnow()
        await self.proposals.save(proposal)
        await self.audit.record(
            "improvement.staged", actor=by, proposal_id=proposal.id, passed=passed, report=report
        )
        await self.bus.publish("improvement.staged", proposal_id=proposal.id, passed=passed)
        return proposal

    # ------------------------------------------------------- apply/reject
    async def apply(self, proposal_id: str, *, by: str) -> ProposalRecord:
        self._require_enabled()
        proposal = await self.proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.status is not ProposalStatus.STAGED:
            raise ToolError("only proposals that passed staging can be applied")
        self.validate_diff(proposal.diff)
        branch = f"sentient/{proposal.id}"
        target = self.staging_root / f"{proposal.id}-apply"
        try:
            await self._worktree(target, branch=branch)
            patch = target / ".sentient-proposal.diff"
            patch.write_text(proposal.diff)
            code, out, _ = await _run(
                ["git", "apply", "--index", str(patch)], cwd=target, timeout=60
            )
            patch.unlink(missing_ok=True)
            if code != 0:
                raise ToolError(f"patch no longer applies: {out.strip()[:400]}")
            message = (
                f"SentientAI proposal {proposal.id}: {proposal.title}\n\n"
                f"{proposal.rationale}\n\nApproved-by: {by}"
            )
            code, out, _ = await _run(
                [
                    "git",
                    "-c",
                    "user.name=SentientAI",
                    "-c",
                    "user.email=sentientai@localhost",
                    "commit",
                    "-m",
                    message,
                ],
                cwd=target,
                timeout=60,
            )
            if code != 0:
                raise ToolError(f"commit failed: {out.strip()[:400]}")
        finally:
            await self._remove_worktree(target)
        proposal.status = ProposalStatus.APPLIED
        proposal.branch = branch
        proposal.decided_by = by
        proposal.updated_at = utcnow()
        await self.proposals.save(proposal)
        await self.audit.record(
            "improvement.applied", actor=by, proposal_id=proposal.id, branch=branch
        )
        await self.bus.publish("improvement.applied", proposal_id=proposal.id, branch=branch)
        return proposal

    async def reject(self, proposal_id: str, *, by: str, reason: str) -> ProposalRecord:
        proposal = await self.proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        proposal.status = ProposalStatus.REJECTED
        proposal.decided_by = by
        proposal.report = {**proposal.report, "rejection": reason}
        proposal.updated_at = utcnow()
        await self.proposals.save(proposal)
        await self.audit.record(
            "improvement.rejected", actor=by, proposal_id=proposal.id, reason=reason
        )
        return proposal

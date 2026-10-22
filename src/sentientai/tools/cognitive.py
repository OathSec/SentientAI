"""Tools for the agent's own goals and self-improvement.

Only *proposing* improvements is exposed to the agent. Staging and applying
are operator actions (CLI / API) and deliberately not available as tools.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from sentientai.cognition.goals import GoalManager
from sentientai.config import RiskLevel
from sentientai.reflection.analyzer import PerformanceAnalyzer
from sentientai.reflection.improvement import SelfImprovementEngine
from sentientai.storage.records import GoalOrigin, GoalStatus
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult


class GoalAddArgs(ToolArgs):
    description: str = Field(min_length=5)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class GoalAdd(Tool[GoalAddArgs]):
    name = "goal_add"
    description = "Adopt a longer-term goal to keep working toward across sessions."
    args_model = GoalAddArgs
    risk = RiskLevel.LOW
    tags = ("cognition",)

    def __init__(self, goals: GoalManager) -> None:
        self.goals = goals

    async def run(self, args: GoalAddArgs, ctx: ToolContext) -> ToolResult:
        origin = GoalOrigin.INTRINSIC if ctx.autonomous else GoalOrigin.OPERATOR
        goal = await self.goals.add(args.description, origin=origin, priority=args.priority)
        if goal is None:
            return ToolResult.failure("too many self-initiated goals are already active")
        return ToolResult.success(f"goal {goal.id} active: {goal.description}")


class GoalUpdateArgs(ToolArgs):
    goal_id: str
    status: Literal["active", "paused", "completed", "abandoned"] | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str | None = None


class GoalUpdate(Tool[GoalUpdateArgs]):
    name = "goal_update"
    description = "Record progress on a goal, or complete, pause or abandon it."
    args_model = GoalUpdateArgs
    risk = RiskLevel.LOW
    tags = ("cognition",)

    def __init__(self, goals: GoalManager) -> None:
        self.goals = goals

    async def run(self, args: GoalUpdateArgs, ctx: ToolContext) -> ToolResult:
        try:
            goal = await self.goals.update(
                args.goal_id,
                status=GoalStatus(args.status) if args.status else None,
                progress=args.progress,
                note=args.note,
            )
        except KeyError:
            return ToolResult.failure(f"no goal {args.goal_id}")
        return ToolResult.success(f"goal {goal.id}: {goal.status.value}, {goal.progress:.0%}")


class SelfAnalyzeArgs(ToolArgs):
    days: int = Field(default=7, ge=1, le=90)


class SelfAnalyze(Tool[SelfAnalyzeArgs]):
    name = "self_analyze"
    description = (
        "Analyze your own execution telemetry: failing tools, slow tools, loop breaks and "
        "task failures."
    )
    args_model = SelfAnalyzeArgs
    tags = ("cognition", "read")

    def __init__(self, analyzer: PerformanceAnalyzer) -> None:
        self.analyzer = analyzer

    async def run(self, args: SelfAnalyzeArgs, ctx: ToolContext) -> ToolResult:
        from datetime import timedelta

        report = await self.analyzer.analyze(timedelta(days=args.days))
        lines = [
            f"Tasks: {report.task_counts or 'none'}",
            f"Loop breaks: {report.loop_breaks or 'none'}",
        ]
        lines += [
            f"- {s.tool}: {s.calls} calls, {s.success_rate:.0%} ok, p95 {s.p95_ms:.0f}ms"
            for s in report.tools
        ]
        lines.append("Insights:" if report.insights else "No problems detected.")
        lines += [f"- [{i.severity:.2f}] {i.message}" for i in report.insights]
        return ToolResult.success("\n".join(lines))


class ProposeArgs(ToolArgs):
    insight: str = Field(min_length=10, description="The problem to fix, with evidence.")
    files: list[str] | None = Field(
        default=None, description="Repository-relative source files involved."
    )


class ProposeImprovement(Tool[ProposeArgs]):
    name = "propose_improvement"
    description = (
        "Draft a change to your own source code as a reviewed proposal. It is validated but NOT "
        "applied; the operator stages, tests and approves it."
    )
    args_model = ProposeArgs
    risk = RiskLevel.MEDIUM
    timeout_s = 600.0
    tags = ("cognition", "self-improvement")

    def __init__(self, engine: SelfImprovementEngine) -> None:
        self.engine = engine

    async def run(self, args: ProposeArgs, ctx: ToolContext) -> ToolResult:
        proposal = await self.engine.propose(args.insight, args.files, by=f"agent:{ctx.task_id}")
        return ToolResult.success(
            f"proposal {proposal.id} drafted: {proposal.title} "
            f"(files: {', '.join(proposal.files)}). "
            "Awaiting operator review."
        )


def cognitive_tools(
    goals: GoalManager, analyzer: PerformanceAnalyzer, engine: SelfImprovementEngine | None
) -> list[Tool[Any]]:
    tools: list[Tool[Any]] = [GoalAdd(goals), GoalUpdate(goals), SelfAnalyze(analyzer)]
    if engine is not None and engine.config.enabled:
        tools.append(ProposeImprovement(engine))
    return tools

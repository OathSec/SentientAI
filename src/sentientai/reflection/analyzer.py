"""Turn telemetry into insights: which tools fail, what is slow, where runs break."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from sentientai.core.clock import Clock, SystemClock
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import TaskStatus, ToolStats


class Insight(BaseModel):
    kind: Literal["tool_failures", "tool_latency", "loop_breaks", "task_failures"]
    severity: float  # 0..1
    subject: str
    message: str
    evidence: dict[str, object]


class PerformanceReport(BaseModel):
    since: datetime
    tools: list[ToolStats]
    task_counts: dict[str, int]
    loop_breaks: dict[str, int]
    insights: list[Insight]

    def top(self) -> Insight | None:
        return max(self.insights, key=lambda i: i.severity, default=None)


class PerformanceAnalyzer:
    def __init__(
        self,
        db: Database,
        audit: AuditLog,
        *,
        clock: Clock | None = None,
        min_calls: int = 5,
        failure_threshold: float = 0.75,
        slow_p95_ms: float = 20_000,
    ) -> None:
        self.db = db
        self.audit = audit
        self.clock = clock or SystemClock()
        self.min_calls = min_calls
        self.failure_threshold = failure_threshold
        self.slow_p95_ms = slow_p95_ms

    async def analyze(self, window: timedelta = timedelta(days=7)) -> PerformanceReport:
        since = self.clock.now() - window
        tools = await self.db.tool_invocations.stats(since)
        insights: list[Insight] = []
        for stat in tools:
            if stat.calls >= self.min_calls and stat.success_rate < self.failure_threshold:
                insights.append(
                    Insight(
                        kind="tool_failures",
                        severity=round(1 - stat.success_rate, 3),
                        subject=stat.tool,
                        message=(
                            f"{stat.tool} fails {1 - stat.success_rate:.0%} of {stat.calls} calls"
                        ),
                        evidence={"top_errors": stat.top_errors, "calls": stat.calls},
                    )
                )
            if stat.calls >= self.min_calls and stat.p95_ms > self.slow_p95_ms:
                insights.append(
                    Insight(
                        kind="tool_latency",
                        # grows with how far p95 overshoots; slow is never as severe as broken
                        severity=round(min(0.8, (stat.p95_ms / self.slow_p95_ms - 1) / 4), 3),
                        subject=stat.tool,
                        message=f"{stat.tool} p95 latency is {stat.p95_ms / 1000:.1f}s",
                        evidence={"p50_ms": stat.p50_ms, "p95_ms": stat.p95_ms},
                    )
                )

        loop_breaks: dict[str, int] = {}
        for entry in self.audit.iter_entries():
            if entry.type == "task.loop_break" and entry.ts >= since:
                limit = str(entry.payload.get("limit"))
                loop_breaks[limit] = loop_breaks.get(limit, 0) + 1
        for limit, count in loop_breaks.items():
            if count >= 3:
                insights.append(
                    Insight(
                        kind="loop_breaks",
                        severity=min(1.0, count / 10),
                        subject=limit,
                        message=f"{count} runs were stopped by the {limit} limit",
                        evidence={"count": count},
                    )
                )

        task_counts = await self.db.tasks.count_by_status()
        finished = task_counts.get(TaskStatus.SUCCEEDED.value, 0) + task_counts.get(
            TaskStatus.FAILED.value, 0
        )
        failed = task_counts.get(TaskStatus.FAILED.value, 0)
        if finished >= self.min_calls and failed / finished > 1 - self.failure_threshold:
            insights.append(
                Insight(
                    kind="task_failures",
                    severity=round(failed / finished, 3),
                    subject="tasks",
                    message=f"{failed} of {finished} tasks failed",
                    evidence={"failed": failed, "finished": finished},
                )
            )
        insights.sort(key=lambda i: i.severity, reverse=True)
        return PerformanceReport(
            since=since,
            tools=tools,
            task_counts=task_counts,
            loop_breaks=loop_breaks,
            insights=insights,
        )

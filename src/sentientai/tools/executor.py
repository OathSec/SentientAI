"""The single choke point through which every tool call passes.

validate -> loop guard -> risk assessment -> policy -> (approval) -> rate limit
-> run with timeout -> truncate -> record -> audit -> loop guard.

Denials and failures are returned to the model as tool results so it can adapt;
only safety stops (kill switch, loop breakers, cancellation) propagate.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import ValidationError

from sentientai.config import RiskLevel
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.core.metrics import Metrics
from sentientai.errors import (
    KillSwitchEngagedError,
    LoopBreakError,
    PolicyViolationError,
    ToolError,
)
from sentientai.models.types import ToolCall
from sentientai.safety.approvals import ApprovalGate
from sentientai.safety.audit import AuditLog
from sentientai.safety.limits import LoopGuard, TokenBucket, call_digest
from sentientai.safety.policy import PolicyAction, PolicyEngine
from sentientai.storage.database import ToolInvocationRepository
from sentientai.storage.records import ApprovalStatus, ToolInvocationRecord
from sentientai.tools.base import ToolContext, ToolRegistry, ToolResult

log = get_logger(__name__)


def truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n[... {omitted} characters omitted ...]\n\n{text[-tail:]}", True


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: PolicyEngine,
        approvals: ApprovalGate,
        audit: AuditLog,
        bus: EventBus,
        invocations: ToolInvocationRepository | None = None,
        metrics: Metrics | None = None,
        rate_limiter: TokenBucket | None = None,
        max_output_chars: int = 12_000,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.approvals = approvals
        self.audit = audit
        self.bus = bus
        self.invocations = invocations
        self.metrics = metrics
        self.rate_limiter = rate_limiter
        self.max_output_chars = max_output_chars

    async def execute(
        self, call: ToolCall, ctx: ToolContext, guard: LoopGuard | None = None
    ) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is not None and not ctx.may_use(call.name):
            await self.audit.record(
                "tool.restricted", task_id=ctx.task_id, tool=call.name, user_id=ctx.user_id
            )
            return ToolResult.failure(f"{call.name} isn't available in this conversation")
        if tool is None:
            return ToolResult.failure(
                f"unknown tool {call.name!r}. Available tools: {', '.join(self.registry.names)}"
            )
        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            return ToolResult.failure(f"invalid arguments for {call.name}: {problems}")

        if guard is not None:
            guard.before_tool_call(call.name, call.arguments)

        tool_risk, tool_reasons = tool.assess(args, ctx)
        decision = self.policy.assess(
            tool.name,
            call.arguments,
            base_risk=tool.risk,
            tool_risk=tool_risk,
            tool_reasons=tool_reasons,
            autonomous=ctx.autonomous,
        )
        await self.audit.record(
            "tool.requested",
            task_id=ctx.task_id,
            tool=tool.name,
            call_id=call.id,
            arguments=call.arguments,
            risk=decision.risk.name,
            decision=decision.action.value,
            reasons=decision.reasons,
        )

        if decision.action is PolicyAction.DENY:
            if self.metrics:
                self.metrics.policy_denials.inc(tool=tool.name)
            await self.bus.publish(
                "tool.denied", tool=tool.name, reasons=decision.reasons, task_id=ctx.task_id
            )
            result = ToolResult.failure(
                f"denied by safety policy ({decision.risk.name}): {'; '.join(decision.reasons)}. "
                "Choose a safer approach or ask the operator."
            )
            if guard is not None:
                guard.after_tool_call(False)
            return result

        if decision.action is PolicyAction.APPROVE:
            approval = await self.approvals.request(
                tool=tool.name,
                arguments=call.arguments,
                risk=decision.risk,
                reasons=decision.reasons,
                task_id=ctx.task_id,
                autonomous=ctx.autonomous,
            )
            if self.metrics:
                self.metrics.approvals.inc(outcome=approval.status.value)
            if approval.status is not ApprovalStatus.APPROVED:
                note = f" ({approval.note})" if approval.note else ""
                result = ToolResult.failure(
                    f"operator approval {approval.status.value}{note}; the action was not performed"
                )
                if guard is not None:
                    guard.after_tool_call(False)
                return result

        if self.rate_limiter is not None:
            await self.rate_limiter.acquire(timeout=60)

        await self.bus.publish("tool.started", tool=tool.name, call_id=call.id, task_id=ctx.task_id)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=tool.timeout_s)
        except (KillSwitchEngagedError, LoopBreakError, asyncio.CancelledError):
            raise
        except TimeoutError:
            result = ToolResult.failure(f"{tool.name} timed out after {tool.timeout_s:.0f}s")
        except (ToolError, PolicyViolationError) as exc:
            result = ToolResult.failure(str(exc))
        except Exception as exc:
            log.exception("tool crashed", extra={"tool": tool.name})
            result = ToolResult.failure(f"{tool.name} failed: {type(exc).__name__}: {exc}")
        duration_ms = (time.perf_counter() - started) * 1000

        output, truncated = truncate_middle(result.output, self.max_output_chars)
        result = result.model_copy(
            update={"output": output, "truncated": result.truncated or truncated}
        )

        await self._record(tool.name, call.arguments, ctx, result, duration_ms, decision.risk.value)
        await self.audit.record(
            "tool.completed",
            task_id=ctx.task_id,
            tool=tool.name,
            call_id=call.id,
            ok=result.ok,
            duration_ms=round(duration_ms, 1),
            output=result.output,
            error=result.error,
        )
        await self.bus.publish(
            "tool.completed",
            tool=tool.name,
            call_id=call.id,
            ok=result.ok,
            duration_ms=round(duration_ms, 1),
            task_id=ctx.task_id,
        )
        if guard is not None:
            guard.after_tool_call(result.ok)
        return result

    async def _record(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: ToolContext,
        result: ToolResult,
        duration_ms: float,
        risk: int,
    ) -> None:
        if self.metrics:
            self.metrics.tool_calls.inc(tool=name, outcome="ok" if result.ok else "error")
            self.metrics.tool_latency.observe(duration_ms / 1000, tool=name)
        if self.invocations is not None:
            await self.invocations.insert(
                ToolInvocationRecord(
                    task_id=ctx.task_id,
                    tool=name,
                    args_digest=call_digest(name, arguments)[:32],
                    ok=result.ok,
                    duration_ms=duration_ms,
                    risk=RiskLevel(risk),
                    error=(result.error or "")[:500] or None,
                )
            )

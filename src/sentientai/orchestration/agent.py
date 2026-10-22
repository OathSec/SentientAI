"""The cognitive agent: runs one task from perception to answer.

recall -> appraise novelty -> plan (if warranted) -> frame context
-> [model <-> tools] loop under deterministic loop breakers
-> self-assess -> escalate to the heavy tier when confidence is low.

Every model turn, hidden reasoning, tool call and outcome is audited.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from sentientai.cognition.goals import GoalManager
from sentientai.cognition.grounding import Grounding
from sentientai.cognition.metacognition import Assessment, Metacognition
from sentientai.cognition.mind import Mind
from sentientai.cognition.persona import Persona, Perspective
from sentientai.config import Settings
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger, log_context
from sentientai.core.metrics import Metrics
from sentientai.errors import KillSwitchEngagedError, LoopBreakError, ModelError
from sentientai.memory.system import MemorySystem
from sentientai.memory.working import Turn
from sentientai.models.router import ModelRouter
from sentientai.models.types import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelTier,
    Role,
    TaskKind,
    estimate_tokens,
)
from sentientai.orchestration.context import ContextBuilder, ContextCompressor, FramingInputs
from sentientai.orchestration.planner import Planner
from sentientai.safety.audit import AuditLog
from sentientai.safety.killswitch import KillSwitch
from sentientai.safety.limits import LoopGuard
from sentientai.storage.records import Scope, ScoredMemory, TaskRecord, TaskSource, TaskStatus
from sentientai.tools.base import ToolContext, ToolRegistry
from sentientai.tools.executor import ToolExecutor

log = get_logger(__name__)

_SALVAGEABLE = {"steps", "tool_calls", "repetition", "consecutive_failures", "stall"}
_KIND_BY_TASK = {
    "chat": TaskKind.CHAT,
    "code": TaskKind.CODE,
    "plan": TaskKind.PLAN,
    "improve": TaskKind.IMPROVE,
}


@dataclass
class ToolTrace:
    name: str
    ok: bool
    summary: str


@dataclass
class TaskOutcome:
    task_id: str
    status: TaskStatus
    answer: str | None = None
    error: str | None = None
    loop_break: str | None = None
    steps: int = 0
    tokens: int = 0
    duration_s: float = 0.0
    tier: ModelTier | None = None
    escalated: bool = False
    assessment: Assessment | None = None
    tools: list[ToolTrace] = field(default_factory=list)
    procedures_used: list[str] = field(default_factory=list)
    novelty: float = 0.0
    grounding: dict[str, object] | None = None
    media: list[str] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED

    @property
    def confidence(self) -> float:
        return self.assessment.confidence if self.assessment else 0.0

    def trace_text(self, limit: int = 30) -> str:
        if not self.tools:
            return "(no tools used)"
        return "; ".join(f"{t.name}:{'ok' if t.ok else 'FAILED'}" for t in self.tools[:limit])


class CognitiveAgent:
    def __init__(
        self,
        *,
        settings: Settings,
        router: ModelRouter,
        memory: MemorySystem,
        registry: ToolRegistry,
        executor: ToolExecutor,
        mind: Mind,
        audit: AuditLog,
        bus: EventBus,
        kill_switch: KillSwitch,
        metrics: Metrics | None = None,
        clock: Clock | None = None,
        persona: Persona | None = None,
        grounding: Grounding | None = None,
    ) -> None:
        self.settings = settings
        self.persona = persona
        self.grounding = grounding
        self.router = router
        self.memory = memory
        self.registry = registry
        self.executor = executor
        self.mind = mind
        self.audit = audit
        self.bus = bus
        self.kill_switch = kill_switch
        self.metrics = metrics
        self.clock = clock or SystemClock()
        self.builder = ContextBuilder()
        self.compressor = ContextCompressor(router)
        self.planner = Planner(router, min_chars=settings.runtime.planning_min_chars)
        self.metacognition = Metacognition(
            router, escalate_below=settings.models.routing.escalate_below_confidence
        )

    # ---------------------------------------------------------------- run
    async def run(self, task: TaskRecord) -> TaskOutcome:
        with log_context(task_id=task.id):
            started = time.monotonic()
            outcome = TaskOutcome(task_id=task.id, status=TaskStatus.RUNNING)
            guard = LoopGuard(self.settings.limits, kill_switch=self.kill_switch, clock=self.clock)
            await self.audit.record(
                "task.started",
                task_id=task.id,
                goal=task.goal,
                source=task.source.value,
                kind=task.kind,
            )
            await self.bus.publish(
                "task.started", task_id=task.id, goal=task.goal[:300], source=task.source.value
            )
            try:
                await self._run(task, guard, outcome)
                outcome.status = TaskStatus.SUCCEEDED
            except LoopBreakError as exc:
                outcome.loop_break = exc.limit
                outcome.error = str(exc)
                outcome.status = TaskStatus.FAILED
                if self.metrics:
                    self.metrics.loop_breaks.inc(limit=exc.limit)
                await self.audit.record(
                    "task.loop_break", task_id=task.id, limit=exc.limit, reason=str(exc)
                )
                if exc.limit in _SALVAGEABLE:
                    outcome.answer = await self._salvage(outcome.transcript, str(exc))
            except KillSwitchEngagedError as exc:
                outcome.status = TaskStatus.CANCELLED
                outcome.error = str(exc)
            except asyncio.CancelledError:
                outcome.status = TaskStatus.CANCELLED
                outcome.error = "cancelled"
                await self._finish(task, outcome, guard, started)
                raise
            except ModelError as exc:
                outcome.status = TaskStatus.FAILED
                outcome.error = f"model unavailable: {exc}"
            await self._finish(task, outcome, guard, started)
            return outcome

    async def _finish(
        self, task: TaskRecord, outcome: TaskOutcome, guard: LoopGuard, started: float
    ) -> None:
        snap = guard.snapshot()
        outcome.steps, outcome.tokens = snap.steps, snap.tokens
        outcome.duration_s = round(time.monotonic() - started, 3)
        self.mind.affect.on_resource_use(
            snap.tokens, hourly_budget=self.settings.limits.max_tokens_per_task * 4
        )
        await self.audit.record(
            "task.finished",
            task_id=task.id,
            status=outcome.status.value,
            answer=outcome.answer,
            error=outcome.error,
            loop_break=outcome.loop_break,
            steps=snap.steps,
            tool_calls=snap.tool_calls,
            tokens=snap.tokens,
            duration_s=outcome.duration_s,
            confidence=outcome.confidence,
            tier=outcome.tier.value if outcome.tier else None,
            escalated=outcome.escalated,
        )
        await self.bus.publish(
            "task.finished",
            task_id=task.id,
            status=outcome.status.value,
            confidence=outcome.confidence,
            duration_s=outcome.duration_s,
        )

    async def _run(self, task: TaskRecord, guard: LoopGuard, outcome: TaskOutcome) -> None:
        mind = self.mind
        # People who aren't admins get a restricted toolset; the operator and admins don't.
        privileged = task.metadata.get("role") != "user"
        # A conversation sees shared memories plus this person's compartment, nothing else.
        recalled = await self.memory.recall(task.goal, scope=Scope.for_user(task.user_id))
        procedures = await self.memory.find_procedures(task.goal)
        outcome.procedures_used = [p.record.id for p in procedures]
        outcome.novelty = self.metacognition.novelty(recalled)
        mind.affect.on_novelty(outcome.novelty)
        modulation = mind.affect.modulation()

        plan_text = None
        if self.planner.should_plan(task.goal, cautious=modulation.cautious):
            plan = await self.planner.plan(task.goal, self._memory_digest(recalled))
            if plan is not None:
                plan_text = plan.render()
                await self.audit.record("agent.plan", task_id=task.id, steps=plan.steps)

        kind = _KIND_BY_TASK.get(task.kind, TaskKind.REASON)
        prefer = ModelTier.HEAVY if modulation.prefer_heavy_model else None
        tier_for_window = (
            prefer or self.router.select(kind, ChatRequest(messages=[Message.user(task.goal)])).tier
        )
        window = self.router.context_window(tier_for_window)
        reserve = self._reserve(tier_for_window)
        history = await self.memory.conversation(task.session_id) if task.session_id else []
        now = self.clock.now()
        allowed = None if privileged else frozenset(self.settings.accounts.public_tools)
        mind.self_model.capabilities = self.registry.names
        tool_names = self.registry.names_for(autonomous=task.source.autonomous, allowed=allowed)
        perspective = (
            await self.persona.perspective(task.user_id, privileged=privileged)
            if self.persona is not None
            else None
        )
        redact = perspective.redact if perspective else (lambda text: text)
        if task.user_id is not None:
            goal_scope = Scope.for_user(task.user_id)
        elif task.source in (TaskSource.USER, TaskSource.API):
            goal_scope = Scope.system()  # the operator
        else:
            goal_scope = Scope()
        messages = self.builder.build(
            FramingInputs(
                goal=task.goal,
                identity=mind.self_model.render_identity(tool_names),
                continuity=redact(mind.self_model.render_continuity(now)),
                inner_state=perspective.feelings if perspective else mind.affect.describe(),
                awareness=redact(mind.workspace.render()),
                goals=redact(GoalManager.render(await mind.goals.active(goal_scope))),
                memories=recalled,
                procedures=procedures,
                plan=plan_text,
                history=history,
                now=now,
                beliefs=perspective.beliefs if perspective else "",
                person=perspective.person if perspective else None,
            ),
            window=window,
            reserve_output=reserve,
        )
        outcome.transcript = messages
        ctx = ToolContext(
            task_id=task.id,
            session_id=task.session_id,
            user_id=task.user_id,
            autonomous=task.source.autonomous,
            scratch_dir=self.settings.storage.data_dir / "sandbox" / task.id,
            allowed_tools=allowed,
            privileged=privileged,
        )
        temperature = self._temperature(tier_for_window, modulation.temperature_offset)

        answer, response = await self._converse(
            outcome, guard, ctx, kind=kind, prefer=prefer, temperature=temperature, allowed=allowed
        )
        outcome.tier = response.tier
        grounded_chat = (
            task.kind == "chat" and self.grounding is not None and self.settings.mind.grounding
        )

        async def judge(text: str) -> tuple[Assessment, str]:
            """Rate the answer and check it against memory at the same time (both only need it)."""

            if grounded_chat:
                rated, checked = await asyncio.gather(
                    self._assess(task, text, outcome, guard, recalled),
                    self._ground(task, text, outcome, perspective, recalled, history),
                )
                return rated, checked
            return await self._assess(task, text, outcome, guard, recalled), text

        assessment, answer = await judge(answer)
        if (
            self.metacognition.should_escalate(assessment, response.tier)
            and ModelTier.HEAVY in self.router.clients
        ):
            await self.audit.record(
                "agent.escalate", task_id=task.id, confidence=assessment.confidence
            )
            outcome.transcript.append(
                Message.user(
                    "Before finalizing, double-check your answer above. Verify with tools if "
                    "useful, then give "
                    "a corrected final answer (or restate it if it was right)."
                )
            )
            answer, response = await self._converse(
                outcome,
                guard,
                ctx,
                kind=kind,
                prefer=ModelTier.HEAVY,
                temperature=temperature,
                allowed=allowed,
            )
            outcome.tier = response.tier
            outcome.escalated = True
            assessment, answer = await judge(answer)
        outcome.answer = answer
        outcome.assessment = assessment

    async def _ground(
        self,
        task: TaskRecord,
        answer: str,
        outcome: TaskOutcome,
        perspective: Perspective | None,
        recalled: list[ScoredMemory],
        history: list[Turn],
    ) -> str:
        assert self.grounding is not None
        parts = []
        if perspective is not None:
            parts.append(f"## Its decisions about itself\n{perspective.beliefs}")
            if perspective.person:
                parts.append(f"## The person\n{perspective.person}")
        parts.append(
            "## Its memories\n"
            + ("\n".join(f"- {m.record.content[:600]}" for m in recalled[:12]) or "- none recalled")
        )
        dialogue = [t for t in history if t.role in {"user", "assistant"}][-12:]
        if dialogue:
            parts.append(
                "## Earlier in this conversation\n"
                + "\n".join(f"{t.role}: {t.content[:800]}" for t in dialogue)
            )
        tool_output = [m for m in outcome.transcript if m.role is Role.TOOL][-6:]
        if tool_output:
            parts.append(
                "## Tool results\n" + "\n".join(f"[{m.name}] {m.text[:1500]}" for m in tool_output)
            )
        result = await self.grounding.check(
            name=self.mind.self_model.name,
            message=task.goal,
            reply=answer,
            evidence="\n\n".join(parts),
            source=f"conversation:{task.user_id}" if task.user_id else "conversation",
            task_id=task.id,
        )
        outcome.grounding = {
            "revised": result.revised,
            "unsupported": result.unsupported,
            "contradictions": result.contradictions,
            "new_beliefs": result.new_beliefs,
        }
        return result.reply

    async def _converse(
        self,
        outcome: TaskOutcome,
        guard: LoopGuard,
        ctx: ToolContext,
        *,
        kind: TaskKind,
        prefer: ModelTier | None,
        temperature: float | None,
        allowed: frozenset[str] | None = None,
    ) -> tuple[str, ChatResponse]:
        tools = self.registry.schemas(
            names=sorted(allowed) if allowed is not None else None, autonomous=ctx.autonomous
        )
        nudged = False
        while True:
            guard.begin_step()
            tier = prefer or self.router.select(kind, ChatRequest(messages=outcome.transcript)).tier
            outcome.transcript = await self.compressor.fit(
                outcome.transcript,
                window=self.router.context_window(tier),
                reserve_output=self._reserve(tier),
            )
            response = await self.router.chat(
                ChatRequest(messages=outcome.transcript, tools=tools, temperature=temperature),
                kind=kind,
                prefer=prefer,
            )
            usage = response.usage.total or estimate_tokens(response.message.text)
            guard.record_tokens(usage)
            message = response.message
            if response.reasoning:
                await self.audit.record(
                    "agent.reasoning",
                    task_id=ctx.task_id,
                    tier=str(response.tier),
                    reasoning=response.reasoning,
                )
            outcome.transcript.append(message)

            if message.tool_calls:
                if message.text:
                    await self.audit.record("agent.thought", task_id=ctx.task_id, text=message.text)
                for call in message.tool_calls:
                    result = await self.executor.execute(call, ctx, guard)
                    outcome.tools.append(ToolTrace(call.name, result.ok, result.output[:200]))
                    if result.ok and isinstance(result.data, dict) and result.data.get("media_id"):
                        outcome.media.append(str(result.data["media_id"]))
                    if not result.ok and "denied" in result.output:
                        self.mind.affect.on_denied_action()
                    outcome.transcript.append(Message.tool(call.id, call.name, result.for_model()))
                # Backstop after the (more specific, configurable) repetition limit has run.
                guard.record_output(
                    message.text
                    + "|"
                    + "|".join(f"{c.name}{sorted(c.arguments.items())}" for c in message.tool_calls)
                )
                continue

            answer = message.text.strip()
            if not answer and not nudged:
                nudged = True
                outcome.transcript.append(Message.user("Please give your final answer now."))
                continue
            await self.audit.record(
                "agent.answer", task_id=ctx.task_id, tier=str(response.tier), answer=answer
            )
            return answer or "(no answer produced)", response

    async def _assess(
        self,
        task: TaskRecord,
        answer: str,
        outcome: TaskOutcome,
        guard: LoopGuard,
        recalled: list[ScoredMemory],
    ) -> Assessment:
        failures = sum(1 for t in outcome.tools if not t.ok)
        coverage = max((m.similarity for m in recalled), default=0.0)
        assessment = await self.metacognition.assess(
            goal=task.goal,
            answer=answer,
            evidence=outcome.trace_text(),
            tool_calls=len(outcome.tools),
            tool_failures=failures,
            pressure=guard.snapshot().pressure,
            memory_coverage=coverage,
        )
        await self.audit.record(
            "agent.assessment",
            task_id=task.id,
            confidence=assessment.confidence,
            heuristic=assessment.heuristic,
            model=assessment.model_confidence,
            gaps=assessment.gaps,
        )
        return assessment

    async def _salvage(self, transcript: list[Message], reason: str) -> str | None:
        if not transcript:
            return None
        messages = [m for m in transcript if not (m.role is Role.ASSISTANT and m.tool_calls)]
        messages = [
            m if m.role is not Role.TOOL else Message.user(f"[{m.name} result]\n{m.text[:1500]}")
            for m in messages
        ]
        messages.append(
            Message.user(
                f"You must stop now ({reason}). Give your best final answer from what you have, "
                "and state "
                "clearly what is incomplete or uncertain."
            )
        )
        try:
            response = await self.router.chat(
                ChatRequest(messages=messages, max_tokens=800), kind=TaskKind.SUMMARIZE
            )
        except ModelError:
            return None
        return response.message.text.strip() or None

    # ------------------------------------------------------------ helpers
    def _reserve(self, tier: ModelTier) -> int:
        cfg = self.settings.models.heavy if tier is ModelTier.HEAVY else self.settings.models.fast
        return cfg.max_output_tokens

    def _temperature(self, tier: ModelTier, offset: float) -> float | None:
        if offset == 0:
            return None
        cfg = self.settings.models.heavy if tier is ModelTier.HEAVY else self.settings.models.fast
        return max(0.0, min(1.5, cfg.temperature + offset))

    @staticmethod
    def _memory_digest(recalled: list[ScoredMemory]) -> str:
        return "\n".join(f"- {m.record.content[:300]}" for m in recalled[:6])

"""The cognitive agent: runs one task from perception to answer.

recall -> appraise novelty -> plan (if warranted) -> frame context
-> [model <-> tools] loop under deterministic loop breakers
-> self-assess -> escalate to the heavy tier when confidence is low.

Every model turn, hidden reasoning, tool call and outcome is audited.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sentientai import prompts
from sentientai.cognition.goals import GoalManager
from sentientai.cognition.grounding import Grounding
from sentientai.cognition.intentions import Intentions
from sentientai.cognition.metacognition import Assessment, Metacognition
from sentientai.cognition.mind import Mind
from sentientai.cognition.persona import Persona, Perspective
from sentientai.config import Settings
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.jsonparse import extract_object
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
from sentientai.storage.database import ActivityRepository, UserRepository
from sentientai.storage.records import Scope, ScoredMemory, TaskRecord, TaskSource, TaskStatus
from sentientai.tools.base import ToolContext, ToolRegistry
from sentientai.tools.embed import remember_urls
from sentientai.tools.executor import ToolExecutor

log = get_logger(__name__)

_SALVAGEABLE = {"steps", "tool_calls", "repetition", "consecutive_failures", "stall"}
_KIND_BY_TASK = {
    "chat": TaskKind.CHAT,
    "code": TaskKind.CODE,
    "plan": TaskKind.PLAN,
    "improve": TaskKind.IMPROVE,
}


# "I made you a picture", "let me create an image", "here's your painting"...
_PICTURE_CLAIM = re.compile(
    r"\b(?:i(?:'ll| will|'m going to| am going to| just)?\s+(?:make|create|generate|draw|"
    r"paint|render)|let me (?:make|create|generate|draw|paint|render)|here(?:'s| is) (?:your|the|a|"
    r"an)|i(?:'ve| have)?\s+(?:made|created|generated|drawn|painted|rendered))\b[^.!?\n]{0,60}?"
    r"\b(?:image|picture|painting|drawing|illustration|artwork|portrait|photo)s?\b",
    re.IGNORECASE,
)


def claims_picture(text: str) -> bool:
    return bool(_PICTURE_CLAIM.search(text))


@dataclass(frozen=True)
class _Claim:
    """Something a reply can say it did that only a tool makes true."""

    kind: str
    pattern: re.Pattern[str]
    tools: frozenset[str]
    nudge: str


_DID = (
    r"\b(?:i(?:'ve| have)?\s+(?:just\s+|now\s+|already\s+|successfully\s+)*|"
    r"(?:it |this |that |the \w+ )?(?:has|have) been (?:successfully\s+)?)"
)
_CLAIMS = (
    _Claim(
        "world",
        re.compile(
            _DID + r"(?:built|added|created|made|placed|put|designed|set up)\b[^.!?\n]{0,80}?"
            r"\b(?:world|house|home|building|tower|garden|character|room)\b",
            re.IGNORECASE,
        ),
        frozenset({"world_build", "world_modify", "world_remove", "world_expand", "avatar_design"}),
        "Nothing in your world has changed: no world tool ran (your world is yours to shape in "
        "your own time, not during a conversation). If you want to make it, note it with "
        "plan_later and say you'll do it later.",
    ),
    _Claim(
        "web",
        re.compile(
            _DID + r"(?:found|fetched|searched|scanned|looked up|browsed|visited|opened|watched)\b"
            r"[^.!?\n]{0,80}?\b(?:video|website|site|web ?page|page|internet|web|domain|wiki\w*|"
            r"article|link|url|channel)s?\b",
            re.IGNORECASE,
        ),
        frozenset({"web_search", "web_fetch", "http_request"}),
        "You haven't looked anything up: no web tool ran. Use web_search or web_fetch now, or "
        "reply without saying you found, opened or read anything online.",
    ),
    _Claim(
        "memory",
        re.compile(
            r"\b(?:updated|added (?:it |this |that )?to|saved (?:it |this |that )?(?:to|in)|"
            r"stored (?:it |this |that )?in) my memory\b",
            re.IGNORECASE,
        ),
        frozenset({"memory_remember", "self_define", "plan_later"}),
        "Nothing was saved: no memory tool ran. Call memory_remember now if you want to keep it, "
        "or reply without saying you saved it.",
    ),
    _Claim(
        "reading",
        re.compile(
            _DID + r"(?:read|processed|finished reading|studied|scanned|gone through)\b"
            r"[^.!?\n]{0,60}?\b(?:you (?:gave|provided|taught|sent)|materials?|data|documents?|"
            r"websites?|texts?|files?|info\w*)\b",
            re.IGNORECASE,
        ),
        frozenset({"memory_recall", "fs_read", "web_fetch", "ingest"}),
        "You haven't read it: no tool ran. Use memory_recall to look through what you were "
        "taught, or note it with plan_later and say you'll read it in your own time.",
    ),
)
# "Let me show you some examples." or "Now, I will analyze the list." as the last words: a
# promise with nothing after it.
_PROMISE = re.compile(
    r"(?:^|[.!?]\s+)(?:(?:now|next|then),?\s+)?(?:let me|i will now|i'll now|i will|i'll|"
    r"i am going to|i'm going to|i'll go ahead and)\s+[^.!?\n]{0,60}?\b(?:show|share|"
    r"demonstrate|begin|start|give|write|provide|explain|list|tell|walk you through|analy[sz]e|"
    r"look into|look up|search|research|check|read|explore|categori[sz]e|download|run|test|"
    r"build|make|create|find)\b[^.!?\n]{0,120}[.!?:]?\s*$",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _key(sentence: str) -> str:
    return " ".join(sentence.casefold().split()).strip(" .!?")


def tidy_reply(text: str, previous: list[str] | None = None) -> str:
    """Take out what a small model does to a conversation, without rewriting what it said.

    - a sentence it already said in one of its last few replies (it copied its earlier replies
      into each new one, so every message to one person began "I am alf, and I am content..."),
    - a loop: the same sentence again within the reply, or a long run of sentences that all
      start the same way ("I think it's a ... I think it's a ..."), cut where it starts.
    Replies with code are left alone, and nothing is ever reduced to empty.
    """

    if "```" in text or not text.strip():
        return text
    said = {_key(s) for p in previous or [] for s in _SENTENCE.split(p) if len(_key(s)) > 24}
    kept: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE.split(text.strip()):
        key = _key(sentence)
        if len(key) > 12 and key in seen:
            break  # it has started going round in circles
        if key in said:
            continue  # it already told them this
        starts = [" ".join(_key(s).split()[:3]) for s in kept[-4:]]
        opener = " ".join(key.split()[:3])
        if len(starts) == 4 and len(opener) > 6 and all(s == opener for s in starts):
            break
        seen.add(key)
        kept.append(sentence)
    tidy = " ".join(kept).strip()
    return tidy or text


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
    embeds: list[dict[str, Any]] = field(default_factory=list)  # videos and link cards shown
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
        self.builder = ContextBuilder(
            note_before_message=settings.models.heavy.system_note_mid_conversation
        )
        self.compressor = ContextCompressor(router)
        # Set by the runtime / container: what it is doing in the background, and the stores the
        # "do I want to talk?" decision reads and writes.
        self.busy_with: Callable[[], str | None] | None = None
        self.activities: ActivityRepository | None = None
        self.users: UserRepository | None = None
        self.intentions: Intentions | None = None
        self.planner = Planner(router, min_chars=settings.runtime.planning_min_chars)
        models = settings.models
        self._one_model = models.heavy.base_url.rstrip("/") == models.fast.base_url.rstrip("/")
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
                if task.kind == "chat":  # someone is waiting: the model is theirs first
                    async with self.router.foreground():
                        await self._run(task, guard, outcome)
                else:
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
                    outcome.answer = await self._salvage(
                        outcome.transcript, str(exc), conversation=task.kind == "chat"
                    )
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
            except Exception as exc:
                # Anything else (a storage hiccup, a bug in a tool) still ends the task properly,
                # so the person waiting on a reply hears back and the task isn't left "running".
                log.exception("task crashed")
                outcome.status = TaskStatus.FAILED
                outcome.error = f"something went wrong: {type(exc).__name__}: {exc}"[:500]
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

    async def _wants_to_talk(
        self,
        task: TaskRecord,
        perspective: Perspective | None,
        history: list[Turn],
    ) -> str | None:
        """Fuzzy decides whether to talk. Returns what it says instead, or None to go ahead.

        Nothing decides this for it: it sees what it's doing, how it feels, who is asking and
        what they said, and chooses. If the decision can't be made (model trouble) it talks."""

        busy = self.busy_with() if self.busy_with is not None else None
        recent = await self.activities.recent(limit=3) if self.activities is not None else []
        doing = "\n".join(
            [f"- right now, in the background: {busy}" if busy else "- nothing in particular"]
            + [f"- earlier: {a.kind}: {a.intent[:160]}" for a in recent if a.user_id is None]
        )
        dialogue = [t for t in history if t.role in {"user", "assistant"}][-4:]
        who = self.mind.self_model.name
        now = self.clock.now()
        try:
            response = await self.router.chat(
                ChatRequest(
                    messages=[
                        Message.user(
                            prompts.ENGAGE.format(
                                name=who,
                                now=f"{now:%A %d %B %Y, %H:%M} UTC",
                                doing=doing,
                                feelings=perspective.feelings if perspective else "(unknown)",
                                person=(perspective.person + "\n")
                                if perspective and perspective.person
                                else "",
                                dialogue="\n".join(f"{t.role}: {t.content[:400]}" for t in dialogue)
                                or "(nothing yet)",
                                message=task.goal[:1500],
                            )
                        )
                    ],
                    temperature=0.7,
                    json_mode=True,
                    max_tokens=500,
                    snippet=True,
                ),
                kind=TaskKind.CHAT,
            )
        except ModelError:
            return None
        data = extract_object(response.message.text) or {}
        choice = str(data.get("choice") or "talk").strip().lower()
        if choice not in {"later", "decline", "no", "not now"}:
            return None
        say = " ".join(str(data.get("say") or "").split())[:1500]
        if choice == "later" and self.intentions is not None and task.user_id:
            user = await self.users.get(task.user_id) if self.users is not None else None
            await self.intentions.add(
                "talk with them when I'm free",
                str(data.get("why") or "")[:300],
                for_whom=user.display_name if user else None,
                user_id=task.user_id,
            )
        await self.audit.record(
            "agent.declined_to_talk",
            task_id=task.id,
            choice=choice,
            why=str(data.get("why") or "")[:500],
            said=say,
        )
        return say or "…"

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
        if task.kind == "chat" and self.settings.mind.choose_to_talk:
            declined = await self._wants_to_talk(task, perspective, history)
            if declined is not None:
                outcome.answer = declined
                return
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
                identity=mind.self_model.render_identity(
                    tool_names, missing=self.registry.unavailable_notes()
                ),
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
                conversation=task.kind == "chat",
                promises=perspective.promises if perspective else "",
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
            outcome,
            guard,
            ctx,
            kind=kind,
            prefer=prefer,
            temperature=temperature,
            allowed=allowed,
            goal=task.goal,
        )
        said = [t.content for t in history if t.role == "assistant"][-4:]
        if task.kind == "chat":
            answer = tidy_reply(answer, said)
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
                goal=task.goal,
            )
            if task.kind == "chat":
                answer = tidy_reply(answer, said)
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
        parts.append(
            "## Tool results\n"
            + (
                "\n".join(f"[{m.name}] {m.text[:1500]}" for m in tool_output)
                if tool_output
                else "- none: no tools were used for this reply, so nothing was made or done"
            )
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
        goal: str = "",
    ) -> tuple[str, ChatResponse]:
        tools = self.registry.schemas(
            names=sorted(allowed) if allowed is not None else None, autonomous=ctx.autonomous
        )
        available = {t.name for t in tools}
        conversation = kind is TaskKind.CHAT
        # What the corrections below remind it to answer (a person's words, or its own task).
        asked = (
            f"\n\nTheir message, which your reply answers: {goal[:1500]}"
            if conversation
            else f"\n\nYour task: {goal[:1500]}"
        )
        nudges = 0
        picture_checks = 0
        claims_checked: set[str] = set()
        promised = False
        force_picture = False
        while True:
            guard.begin_step()
            tier = prefer or self.router.select(kind, ChatRequest(messages=outcome.transcript)).tier
            outcome.transcript = await self.compressor.fit(
                outcome.transcript,
                window=self.router.context_window(tier),
                reserve_output=self._reserve(tier),
            )
            response = await self.router.chat(
                ChatRequest(
                    messages=outcome.transcript,
                    tools=[t for t in tools if t.name == "image_generate"]
                    if force_picture
                    else tools,
                    temperature=temperature,
                    tool_choice="required" if force_picture else None,
                    # A reply to a person, not an essay: a looping model otherwise ran on for
                    # minutes (4096 tokens of the same sentences).
                    max_tokens=1500 if conversation else None,
                ),
                kind=kind,
                prefer=prefer,
            )
            force_picture = False
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
                    if call.name in ("web_search", "web_fetch") and result.ok:
                        remember_urls(ctx, result.output, str(call.arguments.get("url") or ""))
                    if result.ok and isinstance(result.data, dict) and result.data.get("embed"):
                        outcome.embeds.append(result.data["embed"])
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
            if not answer and nudges < 2:
                nudges += 1
                outcome.transcript.pop()  # an empty turn would teach it to stay silent
                outcome.transcript.append(
                    Message.user(
                        "Your last turn had no reply in it. Say your reply now, in plain text."
                    )
                )
                continue
            if answer and not outcome.media and claims_picture(answer):
                # It said it made (or is making) a picture, but nothing was drawn. Small models do
                # this a lot, often copying an earlier reply. First it gets a chance to do it or
                # say so; if it claims it again and can draw, it is made to call the tool.
                can_draw = any(t.name == "image_generate" for t in tools)
                picture_checks += 1
                if picture_checks == 2 and can_draw:
                    force_picture = True
                    outcome.transcript.append(
                        Message.user(
                            "Call image_generate now, describing the picture in its prompt."
                        )
                    )
                    continue
                if picture_checks > 1:
                    answer = (
                        "I meant to make that picture, but it didn't happen this time. "
                        "Ask me again?"
                        if can_draw
                        else "I can't make pictures yet: there's no image generator set up on "
                        "this server."
                    )
                    outcome.transcript[-1] = Message.assistant(answer)
                else:
                    outcome.transcript.append(
                        Message.user(
                            (
                                "No picture has been made: image_generate hasn't run. If you want "
                                "to make one, call image_generate now. Otherwise write your reply "
                                "again, to what they actually said, without saying you made, "
                                "are making or are showing a picture (and without mentioning "
                                "this note)."
                                if can_draw
                                else "You can't make pictures right now: no image generator is set "
                                "up on this server. Write your reply again, to what they actually "
                                "said; if they asked for a picture, say plainly that you can't. "
                                "Don't mention this note."
                            )
                            + asked
                        )
                    )
                    continue
            used = {t.name for t in outcome.tools}
            claim = next(
                (
                    c
                    for c in _CLAIMS
                    if answer
                    and c.kind not in claims_checked
                    and not used & c.tools
                    and c.pattern.search(answer)
                ),
                None,
            )
            if claim is not None and len(claims_checked) < 2:
                # It says it did something only a tool does, and no such tool ran. Small models
                # do this all the time ("The house has been added to your world"). One chance to
                # actually do it, or to answer honestly.
                claims_checked.add(claim.kind)
                note = (
                    claim.nudge
                    if available & claim.tools or claim.kind == "world"
                    else "You didn't do that: no tool for it ran, and you don't have one in this "
                    "conversation. If it's something you want to do in your own time, note it "
                    "with plan_later."
                )
                outcome.transcript.append(
                    Message.user(
                        f"{note} Write your reply again, to what they actually said, without "
                        f"claiming it's done (and without mentioning this note).{asked}"
                    )
                )
                continue
            in_chat = conversation and not used and len(answer) < 400
            if not promised and (in_chat or not conversation) and _PROMISE.search(answer):
                # "Let me show you some examples." / "Now, I will analyze the list." and nothing
                # more: it promised and stopped (a task ends as soon as it stops calling tools).
                promised = True
                outcome.transcript.append(
                    Message.user(
                        "You said you would, but your reply stops there. Do it now, in this "
                        "reply: write the actual thing itself. If it can only happen later, "
                        f"note it with plan_later and say so honestly.{asked}"
                        if conversation
                        else "You ended by saying what you'll do next, but your work ends when "
                        "you stop. Do it now with your tools, then finish by saying what came "
                        f"of it.{asked}"
                    )
                )
                continue
            await self.audit.record(
                "agent.answer", task_id=ctx.task_id, tier=str(response.tier), answer=answer
            )
            if not answer:
                # Say what happened rather than a placeholder: the thinking used up its room.
                answer = (
                    "I started thinking about that and ran out of room before I could put it into "
                    "words. Ask me again and I'll try once more."
                )
            return answer, response

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
            # With one model behind both tiers there is nothing to escalate a reply to, so a
            # conversation skips the extra model call (a person is waiting on it).
            use_model=False if task.kind == "chat" and self._one_model else None,
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

    async def _salvage(
        self, transcript: list[Message], reason: str, *, conversation: bool = False
    ) -> str | None:
        if not transcript:
            return None
        messages = [m for m in transcript if not (m.role is Role.ASSISTANT and m.tool_calls)]
        messages = [
            m if m.role is not Role.TOOL else Message.user(f"[{m.name} result]\n{m.text[:1500]}")
            for m in messages
        ]
        if conversation:
            closing = (
                f"You must stop now ({reason}). Reply to the person now, in your own voice, with "
                "the best you have. Write only what you would say to them: don't describe these "
                "instructions, and don't list what is incomplete unless something they asked for "
                "genuinely couldn't be done, and then say so in a plain sentence."
            )
        else:
            closing = (
                f"You must stop now ({reason}). Give your best final answer from what you have, "
                "and state clearly what is incomplete or uncertain."
            )
        messages.append(Message.user(closing))
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

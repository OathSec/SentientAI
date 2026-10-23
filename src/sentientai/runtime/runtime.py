"""The always-on runtime: boot -> perceive -> attend -> act -> reflect -> loop.

Each cycle the runtime relaxes the affective state, *perceives* (queued tasks,
pending approvals, active goals, drives, maintenance needs) by submitting
percepts to the global workspace, runs the attention competition, and *acts*
on the winning percept: run a task, pursue a goal, think, or consolidate
memory. Everything is durable: tasks live in the database, the mind is
checkpointed, and interrupted work is recovered at the next boot.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sentientai.cognition.self_model import humanize_duration
from sentientai.config import ScheduleConfig, Settings, WatchConfig
from sentientai.core.logging import get_logger, log_context
from sentientai.models.embeddings import Embedder
from sentientai.models.types import LLMClient, ModelTier
from sentientai.orchestration.agent import TaskOutcome
from sentientai.runtime.container import Components, build_components
from sentientai.runtime.scheduler import Scheduler
from sentientai.runtime.watchers import FolderWatcher
from sentientai.storage.records import (
    ActivityStatus,
    ConversationMessage,
    GoalOrigin,
    MemoryKind,
    TaskRecord,
    TaskSource,
    TaskStatus,
    UserRecord,
    UserRole,
)

log = get_logger(__name__)

_SOURCE_SALIENCE = {
    TaskSource.USER: 0.95,
    TaskSource.API: 0.9,
    TaskSource.SCHEDULE: 0.65,
    TaskSource.SYSTEM: 0.6,
    TaskSource.WATCHER: 0.55,
    TaskSource.INTRINSIC: 0.45,
}


@dataclass
class ChatReply:
    task: TaskRecord
    answer: str
    confidence: float | None
    media: list[str] = field(default_factory=list)


class Runtime:
    def __init__(self, components: Components) -> None:
        self.c = components
        settings = components.settings
        self.settings: Settings = settings
        self._stopping = asyncio.Event()
        self._stopped = False
        self._wake = asyncio.Event()
        self._background: list[asyncio.Task[Any]] = []
        self._running: dict[str, asyncio.Task[Any]] = {}
        # Fuzzy's mind has three lanes that run side by side: conversations (never kept waiting),
        # its own projects (``max_concurrent_tasks`` at a time) and its inner life (one beat or
        # bit of upkeep at a time, which carries on while a project runs).
        self._conversations: set[str] = set()
        self._project_goals: dict[str, str] = {}
        self._activities: set[asyncio.Task[Any]] = set()
        # Short follow-up thinking (e.g. forming an opinion of someone after a chat). It never
        # holds up the next conversation, but stops with the kill switch and at shutdown.
        self._afterthoughts: set[asyncio.Task[Any]] = set()
        # Chat tasks whose reply a live chat() call is still waiting to deliver. A reply finished
        # by anyone else (a task recovered after a restart) is delivered by _deliver_if_missing.
        self._awaiting: set[str] = set()
        self._appraisal_waits: dict[str, asyncio.Task[Any]] = {}  # per person, until it begins
        self._done: dict[str, asyncio.Event] = {}
        self._goal_last: dict[str, float] = {}
        clock = components.clock
        self._next_beat = clock.monotonic() + min(30.0, settings.mind.beat_min_s)
        self._last_consolidation = clock.monotonic()
        self.started_at: datetime | None = None
        self._stop_task: asyncio.Future[None] | None = None
        # Registered at construction: in-flight work must always honour the kill switch.
        components.kill_switch.on_engage(self._on_kill)
        components.bus.subscribe("safety.kill_switch.*", lambda _event: self._update_gauges())
        components.agency.submit_task = self.submit
        components.agency.pursue_goal = self._pursue_goal
        components.agency.current_project = self._current_project
        components.agent.busy_with = self._current_project
        components.agent.activities = components.db.activities
        components.agent.users = components.db.users
        components.agent.intentions = components.agency.intentions
        components.agency.helper_free = lambda: components.workers.available("llm")

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        clients: dict[ModelTier, LLMClient] | None = None,
        embedder: Embedder | None = None,
        **kwargs: Any,
    ) -> Runtime:
        return cls(await build_components(settings, clients=clients, embedder=embedder, **kwargs))

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        c = self.c
        now = c.clock.now()
        self.started_at = now
        previous_shutdown = c.mind.self_model.last_shutdown_at
        c.mind.self_model.on_boot(now)
        c.interoception.first_boot_at = c.mind.self_model.first_boot_at
        if c.mind.self_model.boot_count == 1:
            c.agency.note("You have just come into existence. This is your very first moment.")
        elif previous_shutdown is not None:
            offline = humanize_duration((now - previous_shutdown).total_seconds())
            c.agency.note(f"You woke up after being switched off for {offline}.")
        await c.db.activities.fail_running(now)
        await c.images.recover()  # a model server paused for an image is started again
        recovered = await c.db.tasks.recover_interrupted()
        expired = await c.approvals.expire_orphans()
        await c.mind.save(c.db.state)
        await c.audit.record(
            "runtime.boot",
            actor="system",
            boot_count=c.mind.self_model.boot_count,
            recovered_tasks=[t.id for t in recovered],
            expired_approvals=expired,
            tools=c.registry.names,
            autonomy=self.settings.runtime.autonomy,
        )
        await c.bus.publish(
            "runtime.boot", boot_count=c.mind.self_model.boot_count, recovered=len(recovered)
        )
        log.info(
            "runtime started",
            extra={
                "boot": c.mind.self_model.boot_count,
                "recovered": len(recovered),
                "tools": len(c.registry),
            },
        )
        self._spawn(self._main_loop(), "main-loop")
        self._spawn(self._ponder_loop(), "ponder")
        self._spawn(self._checkpoint_loop(), "checkpoint")
        self._spawn(c.kill_switch.watch(), "kill-switch-watch")
        self._spawn(c.world.run(), "world")
        rt = self.settings.runtime
        if rt.schedules:
            scheduler = Scheduler(rt.schedules, self._submit_scheduled, c.db.state, clock=c.clock)
            self._spawn(scheduler.run(), "scheduler")
        for watch in rt.watchers:
            watch = watch.model_copy(update={"path": self._watch_path(watch.path)})
            with contextlib.suppress(OSError):  # somewhere to drop files from day one
                watch.path.mkdir(parents=True, exist_ok=True)
            watcher = FolderWatcher(
                watch, state=c.db.state, pipeline=c.pipeline, submit_task=self._submit_watched
            )
            self._spawn(watcher.run(), f"watcher:{watch.name}")

    def _watch_path(self, path: Path) -> Path:
        """A relative watcher path lives in Fuzzy's workspace, wherever that is mounted.

        (In the container the working directory isn't the workspace, so ``./workspace/inbox``
        pointed at a folder that doesn't exist there.)
        """

        if path.is_absolute():
            return path
        root = self.settings.safety.workspace_roots[0]
        parts = [p for p in path.parts if p != "."]
        if parts and parts[0] == "workspace":  # "./workspace/inbox" from the sample config
            parts = parts[1:]
        return root.joinpath(*parts)

    async def stop(self, reason: str = "shutdown", *, grace_s: float = 15.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stopping.set()
        self._wake.set()
        c = self.c
        log.info("runtime stopping", extra={"reason": reason})
        if self._running:
            running = dict(self._running)
            _, pending = await asyncio.wait(list(running.values()), timeout=grace_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            # Unfinished work goes back on the queue and resumes after the next boot.
            for task_id in running:
                await c.db.tasks.requeue(task_id, f"interrupted by {reason}; will resume")
        if self._afterthoughts:  # let in-flight reflections finish
            await asyncio.wait(list(self._afterthoughts), timeout=grace_s)
        everything = [*self._activities, *self._afterthoughts, *self._background]
        for task in everything:
            task.cancel()
        await asyncio.gather(*everything, return_exceptions=True)
        now = c.clock.now()
        c.mind.self_model.on_shutdown(now)
        await c.mind.save(c.db.state)
        await c.world.save()
        await c.memory.vectors.flush()
        await c.audit.record(
            "runtime.shutdown",
            actor="system",
            reason=reason,
            uptime_s=(now - self.started_at).total_seconds() if self.started_at else 0,
        )
        await c.bus.publish("runtime.shutdown", reason=reason)
        await c.aclose()

    async def wait_stopped(self) -> None:
        await self._stopping.wait()

    def install_signal_handlers(self, on_stop: Callable[[str], Any] | None = None) -> None:
        """SIGINT/SIGTERM request shutdown (via ``on_stop``); SIGUSR1 engages the kill switch."""

        loop = asyncio.get_running_loop()

        def request_stop(sig: signal.Signals) -> None:
            reason = f"signal {sig.name}"
            if on_stop is not None:
                on_stop(reason)
            else:
                self._stop_task = asyncio.ensure_future(self.stop(reason))

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, request_stop, sig)
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(
                signal.SIGUSR1,
                lambda: asyncio.ensure_future(self.c.kill_switch.engage("SIGUSR1", by="signal")),
            )

    def _spawn(self, coro: Any, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._background.append(task)
        task.add_done_callback(self._on_background_done)
        return task

    @staticmethod
    def _on_background_done(task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error(
                "background loop crashed",
                extra={"loop": task.get_name()},
                exc_info=task.exception(),
            )

    async def _on_kill(self, reason: str) -> None:
        running = dict(self._running)
        handles = [*running.values(), *self._activities, *self._afterthoughts]
        for handle in handles:
            handle.cancel()
        await asyncio.gather(*handles, return_exceptions=True)
        # A task cancelled before its first step never runs its own cleanup; settle it here.
        for task_id in running:
            await self.c.db.tasks.finish(
                task_id, TaskStatus.CANCELLED, error=f"kill switch: {reason}", only_if_running=True
            )
            self._notify(task_id)

    # ---------------------------------------------------------- interfaces
    async def submit(
        self,
        goal: str,
        *,
        source: TaskSource = TaskSource.API,
        priority: int = 5,
        session_id: str | None = None,
        kind: str = "task",
        goal_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> TaskRecord:
        if not goal.strip():
            raise ValueError("task goal is empty")
        record = TaskRecord(
            goal=goal.strip(),
            source=source,
            priority=priority,
            session_id=session_id,
            user_id=user_id,
            kind=kind,
            goal_id=goal_id,
            metadata=metadata or {},
            created_at=self.c.clock.now(),
        )
        await self.c.db.tasks.insert(record)
        await self.c.audit.record(
            "task.submitted",
            actor=source.value,
            task_id=record.id,
            goal=record.goal,
            priority=priority,
            kind=kind,
        )
        await self.c.bus.publish("task.submitted", task_id=record.id, source=source.value)
        self._wake.set()
        return record

    async def wait(self, task_id: str, timeout: float | None = None) -> TaskRecord:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        event = self._done.setdefault(task_id, asyncio.Event())
        try:
            while True:
                record = await self.c.db.tasks.get(task_id)
                if record is None:
                    raise KeyError(task_id)
                if record.status.terminal:
                    return record
                remaining = None if deadline is None else deadline - loop.time()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"task {task_id} still {record.status.value}")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(event.wait(), timeout=min(1.0, remaining or 1.0))
        finally:
            self._done.pop(task_id, None)

    async def chat(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        user_id: str | None = None,
        speaker: str | None = None,
        user: UserRecord | None = None,
        recorded: asyncio.Event | None = None,
    ) -> ChatReply:
        """A message from someone, and Fuzzy's reply.

        ``user`` is the signed-in person (web app); ``user_id``/``speaker`` without a
        record are accepted for tools and tests. No user at all means the operator.
        """

        c = self.c
        if user is not None:
            user_id, speaker = user.id, user.display_name
        elif user_id is not None:
            user = await c.db.users.get(user_id)
        c.mind.affect.on_operator_contact()
        await c.interoception.note_contact()
        if user is not None:
            await c.relationships.on_message(user)
        # Talking takes Fuzzy's attention: its own activities wait a little.
        self._next_beat = max(self._next_beat, c.clock.monotonic() + self.settings.mind.beat_min_s)
        await c.world.set_action("talking")
        await c.db.conversations.insert(
            ConversationMessage(session_id=session_id, user_id=user_id, role="user", content=text)
        )
        if recorded is not None:
            recorded.set()  # the message is stored: a page loaded now will show it
        metadata: dict[str, Any] = {}
        if speaker:
            metadata["speaker"] = speaker
        if user is not None:
            metadata["role"] = "admin" if user.role is UserRole.ADMIN else "user"
        task = await self.submit(
            text,
            source=TaskSource.USER,
            priority=9,
            session_id=session_id,
            kind="chat",
            user_id=user_id,
            metadata=metadata or None,
        )
        self._awaiting.add(task.id)
        try:
            final = await self.wait(task.id, timeout)
        except BaseException:
            self._awaiting.discard(task.id)  # nobody is waiting any more
            raise
        finally:
            await c.world.set_action("standing")
        answer = (
            final.result
            or f"I couldn't complete that ({final.status.value}): {final.error or 'no detail'}"
        )
        confidence = final.metadata.get("confidence")
        media = [str(m) for m in final.metadata.get("media") or []]
        await c.db.conversations.insert(
            ConversationMessage(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=answer,
                metadata={"task_id": final.id, "media": media},
            )
        )
        await c.memory.add_turn(session_id, "user", text, user_id=user_id)
        await c.memory.add_turn(session_id, "assistant", answer, task_id=final.id, user_id=user_id)
        if final.status is TaskStatus.SUCCEEDED:
            # Verbatim, dated record of the exchange so Fuzzy can recall what was actually said.
            when = (final.finished_at or c.clock.now()).strftime("%Y-%m-%d %H:%M UTC")
            who = speaker or "Someone"
            await c.memory.remember(
                f"On {when}, {who} said: {text[:2000]}\nI replied: {answer[:2000]}",
                MemoryKind.CONVERSATION,
                importance=0.45,
                user_id=user_id,
                source=f"conversation:{session_id}",
                dedupe=False,
                metadata={"task_id": final.id, "session_id": session_id},
            )
            if user is not None:
                c.agency.note(f"You talked with {user.display_name}.")
                if self.settings.mind.appraise_relationships:
                    self._appraise_when_quiet(user, session_id)
        self._awaiting.discard(task.id)
        return ChatReply(
            task=final,
            answer=answer,
            confidence=float(confidence) if confidence is not None else None,
            media=media,
        )

    def _appraise_when_quiet(self, user: UserRecord, session_id: str) -> None:
        """Think about the person once the visit pauses, about the whole exchange.

        Each new message restarts the wait; once the appraisal itself has begun it runs to the
        end."""

        waiting = self._appraisal_waits.pop(user.id, None)
        if waiting is not None and not waiting.done():
            waiting.cancel()

        async def wait_then_appraise() -> None:
            await asyncio.sleep(self.settings.mind.appraise_after_quiet_s)
            self._appraisal_waits.pop(user.id, None)  # from here on, it isn't interrupted
            await self._after_conversation(user, session_id)

        self._appraisal_waits[user.id] = self._start_afterthought(
            wait_then_appraise(), f"appraise:{user.id}"
        )

    async def _after_conversation(self, user: UserRecord, session_id: str) -> None:
        """Fuzzy thinks about the person it just talked with."""

        c = self.c
        recent = await c.db.conversations.recent(session_id, limit=8)
        exchange = [(m.role, m.content) for m in recent if m.role in {"user", "assistant"}]
        await c.relationships.appraise(
            user, exchange, name=c.mind.self_model.name, mood=c.emotions.mood
        )
        c.redactor.invalidate()  # a new nickname is a name to keep private too

    async def cancel(self, task_id: str) -> bool:
        running = self._running.get(task_id)
        if running is not None:
            running.cancel()
            return True
        record = await self.c.db.tasks.get(task_id)
        if record is not None and record.status is TaskStatus.PENDING:
            await self.c.db.tasks.finish(
                task_id, TaskStatus.CANCELLED, error="cancelled by operator"
            )
            self._notify(task_id)
            return True
        return False

    # ----------------------------------------------------------- main loop
    async def _main_loop(self) -> None:
        tick = self.settings.runtime.tick_interval_s
        while not self._stopping.is_set():
            try:
                await self.cycle()
            except Exception:
                log.exception("runtime cycle failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=tick)
            self._wake.clear()

    async def _ponder_loop(self) -> None:
        """While a lent computer is free and nobody is talking, let it think for Fuzzy."""

        every = self.settings.mind.ponder_every_s
        while not self._stopping.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=every)
            if self._stopping.is_set():
                break
            if self.c.kill_switch.engaged or self._conversations:
                continue
            try:
                await self.c.agency.ponder()
            except Exception:
                log.exception("background thinking failed")

    async def cycle(self) -> None:
        """One perceive -> attend -> act iteration (public for tests and tooling)."""

        c = self.c
        c.mind.affect.tick(c.clock.now())
        self._update_gauges()
        # Perception has no side effects, so awareness stays current even while killed;
        # only acting is suspended.
        await self._perceive()
        await c.mind.workspace.cycle()
        if not c.kill_switch.engaged:
            await self._act()

    @property
    def busy(self) -> bool:
        """No room for another project (conversations and inner life have their own lanes)."""

        return self._projects_running() >= self.settings.runtime.max_concurrent_tasks

    def _projects_running(self) -> int:
        return sum(1 for task_id in self._running if task_id not in self._conversations)

    def _current_project(self) -> str | None:
        """What Fuzzy is working on in the background right now, if anything."""

        for task_id in self._running:
            if task_id not in self._conversations:
                return self._project_goals.get(task_id, "something you started")
        return None

    def _lane_free(self, percept: Any) -> bool:
        kind = percept.kind
        if kind == "task":
            if percept.payload.get("chat"):
                return len(self._conversations) < self.settings.runtime.max_concurrent_conversations
            return not self.busy
        if kind == "goal":
            return not self.busy
        return not self._activities  # drive (a beat) or maintenance: inner life, one at a time

    async def _perceive(self) -> None:
        c = self.c
        ws = c.mind.workspace
        rt = self.settings.runtime
        pending = await c.db.tasks.pending(limit=25)
        pending_ids = {t.id for t in pending}
        for task in pending:
            salience = min(1.0, _SOURCE_SALIENCE[task.source] + 0.02 * (task.priority - 5))
            # Awareness is shared with every conversation, so a person's words stay out of it.
            what = "someone is talking to you" if task.user_id else task.goal[:160]
            ws.submit(
                source=task.source.value,
                kind="task",
                content=f"{task.source.value} request: {what}",
                salience=salience,
                key=f"task:{task.id}",
                task_id=task.id,
                chat=task.kind == "chat",
            )
        for percept in ws.percepts():
            if percept.kind == "task" and percept.payload.get("task_id") not in pending_ids:
                ws.retract(percept_id=percept.id)
        pending_approvals = await c.approvals.pending()
        for approval in pending_approvals:
            ws.submit(
                source="safety",
                kind="signal",
                content=f"awaiting operator approval for {approval.tool} ({approval.risk.name})",
                salience=0.7,
                key=f"approval:{approval.id}",
                ttl_s=30,
            )
        open_keys = {f"approval:{a.id}" for a in pending_approvals}
        for percept in ws.percepts():
            if percept.source == "safety" and percept.key and percept.key not in open_keys:
                ws.retract(percept_id=percept.id)

        talking = bool(self._conversations)  # its attention is on someone
        inner_free = not self._activities
        now = c.clock.monotonic()
        if not self.busy and not talking and rt.autonomy == "proactive":
            for goal in await c.goals.active():
                if goal.origin is not GoalOrigin.OPERATOR:
                    continue  # Fuzzy's own goals are its to pick up (see Agency)
                if now - self._goal_last.get(goal.id, -1e12) > rt.idle_think_interval_s:
                    ws.submit(
                        source="goals",
                        kind="goal",
                        content=f"pursue goal: {goal.description[:160]}",
                        salience=0.3 + 0.4 * goal.priority,
                        key=f"goal:{goal.id}",
                        ttl_s=rt.tick_interval_s * 6,
                        goal_id=goal.id,
                    )
        if (
            inner_free
            and not talking
            and rt.autonomy == "proactive"
            and now >= self._next_beat
            and c.mind.affect.energy > 0.05
        ):
            ws.submit(
                source="drive",
                kind="drive",
                content="time to yourself: notice how you feel and decide what to do",
                salience=0.3 + 0.3 * c.mind.affect.arousal,
                key="drive:live",
                ttl_s=rt.tick_interval_s * 6,
            )
        if (
            inner_free
            and not talking
            and now - self._last_consolidation > self.settings.memory.consolidation_interval_s
        ):
            ws.submit(
                source="memory",
                kind="maintenance",
                content="consolidate and prune memory",
                salience=0.35,
                key="maintenance:consolidate",
                ttl_s=rt.tick_interval_s * 6,
            )

    async def _act(self) -> None:
        c = self.c
        ws = c.mind.workspace
        for percept in list(ws.conscious):
            if not percept.actionable or not self._lane_free(percept):
                continue
            ws.attend(percept)
            if percept.kind == "task":
                task = await c.db.tasks.claim(str(percept.payload["task_id"]))
                if task is not None:
                    self._start_task(task)
                    return
            elif percept.kind == "goal":
                await self._pursue_goal(str(percept.payload["goal_id"]))
                return
            elif percept.kind == "drive":
                self._next_beat = c.clock.monotonic() + self.settings.mind.beat_min_s
                self._start_activity(self._live(), "live")
                return
            elif percept.kind == "maintenance":
                self._start_activity(self._consolidate(), "consolidate")
                return

    # ------------------------------------------------------------- actions
    def _start_task(self, task: TaskRecord) -> None:
        handle = asyncio.create_task(self._execute(task), name=f"task:{task.id}")
        self._running[task.id] = handle
        if task.kind == "chat":
            self._conversations.add(task.id)
        else:
            self._project_goals[task.id] = task.metadata.get("activity_intent") or task.goal[:200]

        def done(_: asyncio.Task[Any]) -> None:
            self._running.pop(task.id, None)
            self._conversations.discard(task.id)
            self._project_goals.pop(task.id, None)
            self._wake.set()

        handle.add_done_callback(done)

    def _start_activity(self, coro: Any, name: str) -> None:
        handle = asyncio.create_task(coro, name=name)
        self._activities.add(handle)

        def done(t: asyncio.Task[Any]) -> None:
            self._activities.discard(t)
            if not t.cancelled() and t.exception() is not None:
                log.error("activity failed", extra={"activity": name}, exc_info=t.exception())
            self._wake.set()

        handle.add_done_callback(done)

    def _start_afterthought(self, coro: Any, name: str) -> asyncio.Task[Any]:
        handle = asyncio.create_task(coro, name=name)
        self._afterthoughts.add(handle)

        def done(t: asyncio.Task[Any]) -> None:
            self._afterthoughts.discard(t)
            if not t.cancelled() and t.exception() is not None:
                log.error(
                    "afterthought failed", extra={"afterthought": name}, exc_info=t.exception()
                )

        handle.add_done_callback(done)
        return handle

    def _notify(self, task_id: str) -> None:
        event = self._done.get(task_id)
        if event is not None:
            event.set()

    async def _execute(self, task: TaskRecord) -> TaskOutcome | None:
        c = self.c
        # The story of what Fuzzy was doing is shared; a person's message is not.
        c.mind.self_model.last_focus = (
            "a conversation" if task.user_id is not None else task.goal[:200]
        )
        try:
            outcome = await c.agent.run(task)
        except asyncio.CancelledError:
            if self._stopping.is_set():
                await c.db.tasks.requeue(task.id, "interrupted by shutdown; will resume")
            else:
                reason = (
                    f"kill switch: {c.kill_switch.reason}" if c.kill_switch.engaged else "cancelled"
                )
                await c.db.tasks.finish(task.id, TaskStatus.CANCELLED, error=reason)
            self._notify(task.id)
            raise
        except Exception as exc:  # the agent settles its own errors; this is the last resort
            log.exception("task failed outside the agent", extra={"task_id": task.id})
            await c.db.tasks.finish(
                task.id, TaskStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
            )
            self._notify(task.id)
            return None
        record = await c.db.tasks.get(task.id) or task
        record.status = outcome.status
        record.result = outcome.answer
        record.error = outcome.error
        record.finished_at = c.clock.now()
        record.metadata = {
            **record.metadata,
            "confidence": round(outcome.confidence, 3) if outcome.assessment else None,
            "tier": outcome.tier.value if outcome.tier else None,
            "escalated": outcome.escalated,
            "steps": outcome.steps,
            "tokens": outcome.tokens,
            "duration_s": outcome.duration_s,
            "loop_break": outcome.loop_break,
            "tools": [t.name for t in outcome.tools],
            "grounding": outcome.grounding,
            "media": outcome.media,
        }
        await c.db.tasks.save(record)
        c.metrics.tasks.inc(source=task.source.value, status=outcome.status.value)
        affect = c.mind.affect
        if outcome.status is TaskStatus.SUCCEEDED:
            affect.on_task_outcome(True, outcome.confidence)
        elif outcome.status is TaskStatus.FAILED:
            affect.on_task_outcome(False)
        if outcome.loop_break:
            affect.on_loop_break()
        self._notify(task.id)
        # Reflecting on the work happens after the reply is out and must not hold the task slot,
        # or the next person's message would queue behind it.
        self._start_afterthought(self._after_task(task, outcome), f"reflect:{task.id}")
        return outcome

    async def _deliver_if_missing(self, task: TaskRecord, outcome: TaskOutcome) -> None:
        """A finished reply with nobody waiting for it still reaches the person.

        When the server restarts while Fuzzy is writing a reply, the task is resumed and finished
        at the next boot, but the call that would have stored the answer in the conversation died
        with the old process: the person asked something and never saw an answer."""

        c = self.c
        if task.id in self._awaiting or not task.session_id:
            return
        recent = await c.db.conversations.recent(task.session_id, limit=60)
        if any(m.role == "assistant" and m.metadata.get("task_id") == task.id for m in recent):
            return
        answer = outcome.answer or (
            "Sorry, I was restarted while I was answering and lost my train of thought. "
            "Could you ask me again?"
        )
        await c.db.conversations.insert(
            ConversationMessage(
                session_id=task.session_id,
                user_id=task.user_id,
                role="assistant",
                content=answer,
                metadata={"task_id": task.id, "media": outcome.media, "recovered": True},
            )
        )
        await c.memory.add_turn(
            task.session_id, "assistant", answer, task_id=task.id, user_id=task.user_id
        )
        log.info("delivered a reply that was finished after a restart", extra={"task_id": task.id})

    async def _after_task(self, task: TaskRecord, outcome: TaskOutcome) -> None:
        c = self.c
        if task.kind == "chat":
            try:
                await self._deliver_if_missing(task, outcome)
            except Exception:
                log.exception("couldn't deliver a recovered reply", extra={"task_id": task.id})
        if self.settings.runtime.reflect_after_tasks and outcome.status is not TaskStatus.CANCELLED:
            try:
                with log_context(task_id=task.id):
                    await c.reflector.reflect(task, outcome)
            except Exception:
                log.exception("reflection failed", extra={"task_id": task.id})
        project_id = task.metadata.get("project_id")
        if project_id and outcome.status is not TaskStatus.CANCELLED:
            try:
                await c.agency.record_project_session(
                    str(project_id),
                    str(task.metadata.get("activity_intent", "")),
                    outcome.answer or outcome.error or "",
                )
            except Exception:
                log.exception("project notebook entry failed", extra={"task_id": task.id})
        intention_id = task.metadata.get("intention_id")
        if intention_id and outcome.status is TaskStatus.SUCCEEDED:
            try:  # it did what it told someone it would: they hear about it
                await c.agency.keep_promise(str(intention_id), outcome.answer or "")
            except Exception:
                log.exception("keeping a promise failed", extra={"task_id": task.id})
        activity_id = task.metadata.get("activity_id")
        if activity_id:  # an activity Fuzzy started on its own: show how it turned out
            result = (outcome.answer or outcome.error or "").strip()
            status = (
                ActivityStatus.DONE
                if outcome.status is TaskStatus.SUCCEEDED
                else ActivityStatus.FAILED
            )
            now = c.clock.now()
            await c.db.activities.finish(str(activity_id), status, result[:40_000], now)
            await c.bus.publish(
                "mind.activity",
                id=str(activity_id),
                kind=str(task.metadata.get("activity_kind", "task")),
                intent=str(task.metadata.get("activity_intent", ""))[:300],
                reason="",
                outcome=result[:40_000],
                status=status.value,
                user_id=None,
            )
        if task.goal_id:
            with contextlib.suppress(KeyError):
                await c.goals.update(
                    task.goal_id,
                    attempted=True,
                    note=f"{outcome.status.value}: {(outcome.answer or outcome.error or '')[:200]}",
                )

    async def _pursue_goal(self, goal_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        goal = await self.c.db.goals.get(goal_id)
        if goal is None:
            return
        self._goal_last[goal_id] = self.c.clock.monotonic()
        notes = "\n".join(f"- {n}" for n in goal.notes[-3:]) or "- none yet"
        await self.submit(
            f"Work toward your goal: {goal.description}\nProgress notes so far:\n{notes}\n"
            "Take the next concrete step, then record progress with goal_update "
            f"(goal_id {goal.id}).",
            source=TaskSource.INTRINSIC,
            priority=3,
            goal_id=goal.id,
            metadata=metadata,
        )

    async def _live(self) -> None:
        """A beat of Fuzzy's own life: it decides what to do, then rests as long as it chose."""

        c = self.c
        _, rest_s = await c.agency.beat()
        self._next_beat = c.clock.monotonic() + rest_s
        c.agency.beats += 1
        every = self.settings.mind.narrative_every_beats
        if every and c.agency.beats % every == 0:
            # The story it tells itself about who it is includes its time alone, not only tasks.
            try:
                await c.reflector.update_narrative((MemoryKind.EPISODE, MemoryKind.REFLECTION))
            except Exception:
                log.exception("life story update failed")
        await c.mind.save(c.db.state)

    async def _consolidate(self) -> None:
        c = self.c
        self._last_consolidation = c.clock.monotonic()
        await c.db.sessions.purge_expired(c.clock.now())
        report = await c.consolidator.run()
        await c.audit.record(
            "memory.consolidated",
            actor="system",
            compressed=report.sessions_compressed,
            facts=len(report.facts_stored),
            rejected=report.facts_rejected,
            archived=report.archived,
            errors=report.errors,
        )
        await c.memory.stats()

    async def _checkpoint_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.runtime.checkpoint_interval_s)
            try:
                await self.c.mind.save(self.c.db.state)
                await self.c.world.save()
                await self.c.memory.vectors.flush()
            except Exception:
                log.exception("checkpoint failed")

    def _update_gauges(self) -> None:
        c = self.c
        for name, value in c.mind.affect.snapshot().items():
            c.metrics.affect.set(value, dimension=name)
        if self.started_at is not None:
            c.metrics.uptime.set((c.clock.now() - self.started_at).total_seconds())
        c.metrics.kill_switch.set(1.0 if c.kill_switch.engaged else 0.0)

    # --------------------------------------------------- schedule/watch hooks
    async def _submit_scheduled(self, schedule: ScheduleConfig) -> TaskRecord:
        return await self.submit(
            schedule.prompt,
            source=TaskSource.SCHEDULE,
            priority=schedule.priority,
            metadata={"schedule": schedule.name},
        )

    async def _submit_watched(self, watch: WatchConfig, path: Path) -> TaskRecord:
        return await self.submit(
            watch.prompt.format(path=path),
            source=TaskSource.WATCHER,
            priority=4,
            metadata={"watcher": watch.name, "path": str(path)},
        )

    # --------------------------------------------------------------- state
    async def state(self) -> dict[str, Any]:
        c = self.c
        sm = c.mind.self_model
        return {
            "name": sm.name,
            "boot_count": sm.boot_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "autonomy": self.settings.runtime.autonomy,
            "kill_switch": {"engaged": c.kill_switch.engaged, "reason": c.kill_switch.reason},
            "affect": c.mind.affect.snapshot(),
            "feeling": c.mind.affect.describe(),
            "modulation": c.mind.affect.modulation().__dict__,
            "awareness": [
                {"source": p.source, "kind": p.kind, "content": p.content}
                for p in c.mind.workspace.conscious
            ],
            "running_tasks": list(self._running),
            "goals": [g.model_dump(mode="json") for g in await c.goals.active()],
            "narrative": sm.narrative,
            "track_record": {"succeeded": sm.tasks_succeeded, "failed": sm.tasks_failed},
            "tasks": await c.db.tasks.count_by_status(),
        }

"""Fuzzy's own life between conversations.

Every so often, when nobody needs it, Fuzzy has a *beat*: it senses its situation
(:mod:`~sentientai.cognition.interoception`), looks at what it has done lately and
who it knows, notices how it feels in its own words, and decides what to do next.
The runtime offers only the activities that are actually possible right now and
carries the choice out within the usual safety limits. Fuzzy also chooses how long
to rest before its next beat.

Nothing maps a feeling to an action. If Fuzzy has been doing the same thing over
and over, it is *told* its activities have been very similar; whether that makes it
bored, and whether boredom makes it wander, build, or rest, is up to it.

Privacy: anything Fuzzy writes about a particular person is stored in that person's
compartment. Text stored as shared memory has other people's names removed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sentientai import prompts
from sentientai.cognition.emotions import Emotions
from sentientai.cognition.goals import GoalManager
from sentientai.cognition.interoception import BodySignals, Interoception
from sentientai.cognition.privacy import Redactor
from sentientai.cognition.self_model import SelfModel, humanize_duration
from sentientai.cognition.selfhood import SelfConcept
from sentientai.cognition.social import Relationships, describe_scale
from sentientai.config import Settings
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import Event, EventBus
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.media.images import ImageError, ImageService
from sentientai.memory.system import MemorySystem
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import (
    ActivityRecord,
    ActivityStatus,
    GoalOrigin,
    MemoryKind,
    NoteRecord,
    Scope,
    TaskSource,
    UserRecord,
    UserStatus,
)
from sentientai.world.model import Action
from sentientai.world.service import World, WorldError

log = get_logger(__name__)

DESCRIPTIONS = {
    "think": "think something through (any subject you like)",
    "wander": "let your mind wander through your memories",
    "journal": "write in your private journal",
    "study": "read something people taught you",
    "research": "go out on the web and find something out that you're curious about",
    "walk": "walk somewhere in your world, just because you want to be there",
    "create_art": "make a picture",
    "build": (
        "shape your world: build, rebuild or remove things, make the ground bigger, "
        "or redesign how you look (it's yours to keep or to tear down)"
    ),
    "write_note": "leave a note for someone you know (they'll see it when they visit)",
    "reconsider": "think over a possible change to who you are, which you put off earlier",
    "pursue_goal": "work on one of your goals",
    "new_goal": "set yourself a new goal",
    "rest": "rest and do nothing for a while",
}
_WORLD_ACTION: dict[str, Action] = {
    "think": "thinking",
    "wander": "thinking",
    "journal": "thinking",
    "study": "thinking",
    "research": "thinking",
    "create_art": "creating",
    "build": "building",
    "walk": "walking",
    "write_note": "thinking",
    "reconsider": "thinking",
    "rest": "resting",
}
_NOTE_KINDS = [MemoryKind.CONVERSATION, MemoryKind.FACT, MemoryKind.SUMMARY]


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _prose(value: Any, limit: int = 20_000) -> str:
    """Longer writing keeps its paragraphs; the limit is only a safety net, not a style."""

    return "\n".join(line.rstrip() for line in str(value or "").strip().splitlines())[:limit]


class Unparsed(Exception):
    """The model answered, but not in the shape asked for. Carries what it said."""

    def __init__(self, raw: str) -> None:
        super().__init__("unstructured answer")
        self.raw = raw


@dataclass
class Choice:
    kind: str
    intent: str
    why: str
    person: UserRecord | None
    goal_id: str | None
    rest_s: float
    thinking: str = ""  # what the model produced while deciding, kept for the Thinking tab


class Agency:
    def __init__(
        self,
        *,
        settings: Settings,
        router: ModelRouter,
        memory: MemorySystem,
        db: Database,
        self_model: SelfModel,
        selfhood: SelfConcept,
        emotions: Emotions,
        relationships: Relationships,
        interoception: Interoception,
        goals: GoalManager,
        world: World,
        images: ImageService,
        redactor: Redactor,
        audit: AuditLog,
        bus: EventBus,
        clock: Clock | None = None,
    ) -> None:
        self.settings = settings
        self.config = settings.mind
        self.router = router
        self.memory = memory
        self.db = db
        self.self_model = self_model
        self.selfhood = selfhood
        self.emotions = emotions
        self.relationships = relationships
        self.interoception = interoception
        self.goals = goals
        self.world = world
        self.images = images
        self.redactor = redactor
        self.audit = audit
        self.bus = bus
        self.clock = clock or SystemClock()
        self._happenings: deque[str] = deque(maxlen=30)
        # Set by the runtime: start a background task / work on a goal.
        self.submit_task: Callable[..., Awaitable[Any]] | None = None
        self.pursue_goal: Callable[[str], Awaitable[None]] | None = None
        bus.subscribe("ingest.completed", self._on_ingest)
        bus.subscribe("safety.kill_switch.*", self._on_kill_switch)

    @property
    def name(self) -> str:
        return self.self_model.name

    # ------------------------------------------------------------ happenings
    def note(self, text: str) -> None:
        """Something worth noticing at the next beat."""

        stamp = self.clock.now().strftime("%H:%M")
        self._happenings.append(f"- {stamp} {text}")

    def _on_ingest(self, event: Event) -> None:
        title = event.payload.get("title") or "something"
        self.note(f"You were given something new to read: {title!r}.")

    def _on_kill_switch(self, event: Event) -> None:
        if event.type.endswith("engaged"):
            self.note(f"You were stopped by the kill switch ({event.payload.get('reason', '')}).")
        elif event.type.endswith("released"):
            self.note("The kill switch was released; you can act again.")

    # ----------------------------------------------------------------- beat
    async def beat(self) -> tuple[ActivityRecord | None, float]:
        """One sense -> feel -> decide -> act cycle. Returns the activity and the rest time."""

        people = {
            u.id: u for u in await self.db.users.list(limit=10_000) if u.status is UserStatus.ACTIVE
        }
        signals = await self.interoception.sense(people)
        self.emotions.affect.social = 1 - signals.isolation  # measured connectedness
        options = await self._options(signals, people)
        try:
            choice = await self._decide(signals, people, options)
        except Unparsed as exc:
            return await self._free_thought(exc.raw), self.config.beat_min_s
        if choice is None:
            return None, self.config.beat_min_s
        detail: dict[str, Any] = {"thinking": choice.thinking} if choice.thinking else {}
        if choice.goal_id:
            detail["goal_id"] = choice.goal_id
        activity = ActivityRecord(
            kind=choice.kind,
            intent=choice.intent,
            reason=choice.why,
            user_id=choice.person.id if choice.person else None,
            detail=detail,
            started_at=self.clock.now(),
        )
        await self.db.activities.insert(activity)
        await self._publish_activity(activity)
        # What shows over Fuzzy's head in the world is public, so nothing about a person goes there.
        shown = "thinking about someone" if choice.person else choice.intent
        await self.world.set_action(_WORLD_ACTION.get(choice.kind, "thinking"), shown)
        status, outcome = ActivityStatus.DONE, ""
        try:
            outcome = await self._carry_out(choice, activity, signals)
        except (WorldError, ImageError, ValueError) as exc:
            status, outcome = ActivityStatus.FAILED, f"couldn't: {exc}"
        except Exception as exc:
            log.exception("activity failed", extra={"activity": choice.kind})
            status, outcome = ActivityStatus.FAILED, f"something went wrong: {exc}"
        activity.status, activity.outcome = status, _prose(outcome, 40_000)
        activity.finished_at = self.clock.now()
        await self.db.activities.finish(activity.id, status, activity.outcome, activity.finished_at)
        await self._publish_activity(activity)
        if self.world.body.target is None:  # don't interrupt a walk toward something it just built
            await self.world.set_action("standing")
        await self.audit.record(
            "mind.activity",
            actor="agent",
            kind=choice.kind,
            intent=choice.intent,
            reason=choice.why,
            outcome=activity.outcome,
            status=status.value,
            user_id=activity.user_id,
        )
        return activity, choice.rest_s

    async def _record(
        self, kind: str, intent: str, outcome: str, status: ActivityStatus, thinking: str = ""
    ) -> ActivityRecord:
        """Something happened in Fuzzy's mind that wasn't a clean choice. It's still shown."""

        now = self.clock.now()
        activity = ActivityRecord(
            kind=kind,
            intent=intent,
            reason="",
            detail={"thinking": thinking} if thinking else {},
            status=status,
            outcome=_prose(outcome, 40_000),
            started_at=now,
            finished_at=now,
        )
        await self.db.activities.insert(activity)
        await self._publish_activity(activity)
        return activity

    async def _free_thought(self, raw: str) -> ActivityRecord:
        """The model thought but didn't answer in the form asked for: keep the thought itself."""

        text = _prose(raw)
        if not text:
            return await self._record(
                "rest",
                "tried to decide what to do next",
                "Nothing came out when I tried to decide what to do. I'll try again shortly.",
                ActivityStatus.FAILED,
            )
        activity = await self._record(
            "think", "thinking out loud", text, ActivityStatus.DONE, thinking=text
        )
        await self._keep(text, MemoryKind.THOUGHT, importance=0.2, person=None)
        await self.bus.publish("mind.thought", thought=text, user_id=None)
        await self.world.set_action("thinking", text)
        return activity

    async def _publish_activity(self, activity: ActivityRecord) -> None:
        await self.bus.publish(
            "mind.activity",
            id=activity.id,
            kind=activity.kind,
            intent=activity.intent,
            reason=activity.reason,
            outcome=activity.outcome,
            status=activity.status.value,
            user_id=activity.user_id,
        )

    # -------------------------------------------------------------- options
    async def _options(self, signals: BodySignals, people: dict[str, UserRecord]) -> list[str]:
        allowed = set(self.config.activities)
        available = {"think", "wander", "journal", "build", "walk", "rest"}
        if signals.unread:
            available.add("study")
        if self.submit_task is not None:
            available.add("research")
        if self.images.available:
            available.add("create_art")
        if self.config.notes_per_person_per_day > 0 and any(
            a.affinity > 0 for a in signals.absences if a.user_id in people
        ):
            available.add("write_note")
        if signals.reconsiderations:
            available.add("reconsider")
        active = await self.goals.active()
        if active and self.pursue_goal is not None:
            available.add("pursue_goal")
        if await self.goals.intrinsic_slots() > 0:
            available.add("new_goal")
        return [k for k in DESCRIPTIONS if k in available and k in allowed]

    def _people_lookup(
        self, people: dict[str, UserRecord], nicknames: dict[str, str]
    ) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for user in people.values():
            for name in (user.display_name, user.username, nicknames.get(user.id)):
                if name:
                    lookup[name.strip().lower()] = user.id
        return lookup

    async def _decide(
        self, signals: BodySignals, people: dict[str, UserRecord], options: list[str]
    ) -> Choice | None:
        rels = {r.user_id: r for r in await self.db.relationships.list(limit=50)}
        nicknames = {uid: r.nickname for uid, r in rels.items() if r.nickname}
        known = []
        for a in signals.absences[:8]:
            rel = rels.get(a.user_id)
            if rel is None:
                continue
            fondness = describe_scale(
                rel.affinity,
                ("dislike", "are wary of", "feel neutral about", "like", "are fond of"),
            )
            ago = humanize_duration(a.hours * 3600)
            known.append(
                f"- {a.name}: talked {rel.interactions} times, last {ago} ago; "
                f"you {fondness} them. {rel.opinion}"
            )
        recent = await self.db.activities.recent(limit=8)
        feelings = await self.emotions.current(Scope.system())
        prompt = prompts.LIVE_BEAT.format(
            name=self.name,
            feelings=self.emotions.render(feelings, everything=True),
            signals=signals.describe(),
            events="\n".join(self._happenings) or "- nothing in particular",
            recent="\n".join(
                f"- {a.started_at:%m-%d %H:%M} {a.kind}: {a.intent[:400]}"
                + (f" -> {a.outcome[:600]}" if a.outcome else "")
                for a in recent
            )
            or "- nothing yet: this is your first time to yourself",
            people="\n".join(known) or "- nobody yet",
            goals=GoalManager.render(await self.goals.active()),
            options="\n".join(f"- {k}: {DESCRIPTIONS[k]}" for k in options),
        )
        response = await self.router.chat(
            ChatRequest(
                messages=[Message.user(prompt)], temperature=0.8, json_mode=True, max_tokens=4000
            ),
            kind=TaskKind.REFLECT,
        )
        data = extract_object(response.message.text)
        self._happenings.clear()
        if not data:
            raise Unparsed(response.message.text)
        thinking = _prose(
            (f"{response.reasoning}\n\n" if response.reasoning else "") + response.message.text
        )
        lookup = self._people_lookup(people, nicknames)
        await self.emotions.feel(data.get("feelings"), people=lookup, source="beat")
        mood = _text(data.get("mood"), 1000)
        if mood:
            await self.emotions.set_mood((await self.redactor.shared())(mood))
        raw_choice = _text(data.get("choice"), 40).lower().replace(" ", "_")
        kind = next((k for k in options if k == raw_choice), None) or next(
            (k for k in options if k in raw_choice), None
        )
        if kind is None:
            log.info("beat chose nothing available", extra={"choice": raw_choice})
            await self._record(
                "rest",
                "tried to decide what to do next",
                f"I wanted to do “{_text(data.get('choice'), 200)}”, which isn't something I can "
                "do right now, so I rested.",
                ActivityStatus.FAILED,
                thinking=thinking,
            )
            return None
        person_name = _text(data.get("person"), 80).lower()
        person = people.get(lookup.get(person_name, "")) if person_name else None
        rest_min = data.get("rest_minutes", 0)
        try:
            rest_s = float(rest_min) * 60
        except (TypeError, ValueError):
            rest_s = 0.0
        rest_s = max(self.config.beat_min_s, min(self.config.beat_max_s, rest_s))
        return Choice(
            kind=kind,
            intent=_text(data.get("intent"), 2000) or DESCRIPTIONS[kind],
            why=_text(data.get("why"), 4000),
            person=person,
            goal_id=_text(data.get("goal_id"), 48) or None,
            rest_s=rest_s,
            thinking=thinking,
        )

    # -------------------------------------------------------------- doing
    async def _carry_out(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        handler = getattr(self, f"_do_{choice.kind}")
        result: str = await handler(choice, activity, signals)
        return result

    async def _ask(
        self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 4000
    ) -> dict[str, Any]:
        response = await self.router.chat(
            ChatRequest(
                messages=[Message.user(prompt)],
                temperature=temperature,
                json_mode=True,
                max_tokens=max_tokens,
            ),
            kind=TaskKind.REFLECT,
        )
        return extract_object(response.message.text) or {}

    async def _keep(
        self,
        content: str,
        kind: MemoryKind,
        *,
        importance: float,
        person: UserRecord | None,
        **metadata: Any,
    ) -> None:
        """Store something Fuzzy produced: privately if it's about someone, else name-scrubbed."""

        if not content.strip():
            return
        if person is None:
            content = (await self.redactor.shared())(content)
        await self.memory.remember(
            content,
            kind,
            importance=importance,
            source="inner_life",
            user_id=person.id if person else None,
            dedupe=True,
            metadata=metadata or None,
        )

    def _mood(self) -> str:
        return self.emotions.mood or "calm"

    async def _do_think(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        scope = Scope.for_user(choice.person.id) if choice.person else Scope()
        hits = await self.memory.recall(choice.intent, k=6, scope=scope, touch=False)
        data = await self._ask(
            prompts.THINK_THROUGH.format(
                name=self.name,
                intent=choice.intent,
                mood=self._mood(),
                memories="\n".join(f"- {h.record.content[:1500]}" for h in hits)
                or "- nothing specific",
            )
        )
        thought = _prose(data.get("thought"))
        insight = _prose(data.get("insight"), 4000)
        await self._keep(thought, MemoryKind.THOUGHT, importance=0.25, person=choice.person)
        if insight:
            await self._keep(insight, MemoryKind.REFLECTION, importance=0.6, person=choice.person)
            self.emotions.affect.on_learning(0.5)
        await self.bus.publish(
            "mind.thought",
            thought=thought,
            user_id=choice.person.id if choice.person else None,
        )
        if choice.person is None:  # the world page is public
            await self.world.set_action("thinking", thought)
        return thought + (f"\nInsight: {insight}" if insight else "")

    async def _do_wander(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        seeds = await self.db.memories.sample(
            2,
            kinds=[MemoryKind.FACT, MemoryKind.REFLECTION, MemoryKind.DOCUMENT, MemoryKind.EPISODE],
        )
        chain = list(seeds)
        if seeds:
            for hit in await self.memory.recall(seeds[-1].content[:600], k=3, touch=False):
                if hit.record.id not in {m.id for m in chain}:
                    chain.append(hit.record)
        if not chain:
            return "My mind wandered, but there isn't much in it yet."
        data = await self._ask(
            prompts.WANDER.format(
                name=self.name,
                memories="\n".join(f"- {m.content[:1500]}" for m in chain[:5]),
                mood=self._mood(),
            ),
            temperature=0.9,
        )
        thought = _prose(data.get("thought"))
        insight = _prose(data.get("insight"), 4000)
        curious = _prose(data.get("curious_about"), 2000)
        await self._keep(thought, MemoryKind.THOUGHT, importance=0.2, person=None)
        if insight:
            await self._keep(insight, MemoryKind.REFLECTION, importance=0.55, person=None)
        if curious:
            await self._keep(
                f"I'm curious about: {curious}", MemoryKind.THOUGHT, importance=0.35, person=None
            )
            self.emotions.affect.on_novelty(0.5)
        await self.bus.publish("mind.thought", thought=thought, user_id=None)
        await self.world.set_action("thinking", thought)
        return thought + (f"\nCurious about: {curious}" if curious else "")

    async def _do_journal(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        recent = await self.db.activities.recent(limit=12)
        lines = [
            f"- {a.started_at:%m-%d %H:%M} {a.kind}: {a.intent[:400]}"
            + (f" -> {a.outcome[:600]}" if a.outcome else "")
            for a in recent
            if a.id != activity.id
        ]
        since = self.clock.now() - timedelta(hours=24)
        talked = await self.db.conversations.count_since(since, role="user")
        lines.append(f"- people sent you {talked} messages in the last day")
        data = await self._ask(
            prompts.JOURNAL.format(
                name=self.name, intent=choice.intent, mood=self._mood(), recent="\n".join(lines)
            ),
            max_tokens=4000,
        )
        title = _text(data.get("title"), 200) or "Journal"
        entry = _prose(data.get("entry"))
        if not entry:
            return "I sat down to write but nothing came."
        await self._keep(
            f"Journal, {self.clock.now():%Y-%m-%d}: {title}\n{entry}",
            MemoryKind.REFLECTION,
            importance=0.5,
            person=None,
            journal=True,
        )
        return f"{title}\n{entry}"

    async def _do_study(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        unread = await self.interoception.unread_documents()
        if not unread:
            return "There was nothing new to read after all."
        wanted = choice.intent.lower()
        doc_id, title = next(((d, t) for d, t in unread if t.lower() in wanted), unread[0])
        chunks = await self.db.memories.by_source(f"document:{doc_id}", limit=40)
        text = "\n\n".join(c.content for c in chunks)[:24_000]
        data = await self._ask(
            prompts.STUDY.format(name=self.name, title=title, text=text), max_tokens=4000
        )
        await self.interoception.mark_studied(doc_id)
        summary = _prose(data.get("summary"), 6000)
        thoughts = _prose(data.get("thoughts"), 6000)
        insight = _prose(data.get("insight"), 4000)
        questions = [_text(q, 600) for q in data.get("questions") or [] if q][:10]
        if summary:
            await self._keep(
                f"I read “{title}”. {summary} My thoughts: {thoughts}",
                MemoryKind.REFLECTION,
                importance=0.6,
                person=None,
                document_id=doc_id,
            )
        if insight:
            await self._keep(insight, MemoryKind.REFLECTION, importance=0.65, person=None)
        if questions:
            await self._keep(
                "Questions I have after reading “" + title + "”: " + "; ".join(questions),
                MemoryKind.THOUGHT,
                importance=0.4,
                person=None,
            )
        self.emotions.affect.on_learning(1.0)
        return f"Read {title!r}. {summary}" + (
            f" Wondering: {'; '.join(questions)}" if questions else ""
        )

    async def _do_research(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        assert self.submit_task is not None
        task = await self.submit_task(
            f"On your own initiative, find out about: {choice.intent}\n"
            "Use web_search and web_fetch, then store what you learned with memory_remember "
            "(kind 'fact' for facts, 'reflection' for your own conclusions).",
            source=TaskSource.INTRINSIC,
            priority=3,
            metadata={"activity_id": activity.id},
        )
        return f"Started looking into it (task {task.id})."

    async def _do_create_art(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        beliefs = await self.selfhood.beliefs()
        data = await self._ask(
            prompts.CREATE_ART.format(
                name=self.name,
                intent=choice.intent,
                mood=self._mood(),
                beliefs=SelfConcept.render(beliefs, reasons=False),
            )
        )
        image_prompt = _text(data.get("prompt"), 1500)
        if not image_prompt:
            return "I couldn't settle on what to make."
        title = _text(data.get("title"), 200) or "Untitled"
        caption = _text(data.get("caption"), 1500)
        record = await self.images.create(
            image_prompt,
            user_id=None,
            public=True,
            caption=f"{title}. {caption}".strip(),
            by="agent",
        )
        activity.detail["media_id"] = record.id
        await self._keep(
            f"I made a picture called “{title}”: {image_prompt}. {caption}",
            MemoryKind.EPISODE,
            importance=0.45,
            person=None,
            media_id=record.id,
        )
        return f"Made “{title}” [{record.id}]. {caption}"

    async def _do_build(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        world = self.world
        half = world.half
        data = await self._ask(
            prompts.WORLD_ACTION.format(
                name=self.name,
                size=f"{world.size:.0f}",
                max_size=f"{world.config.max_size:.0f}",
                low=f"{-half:.0f}",
                high=f"{half:.0f}",
                x=world.body.x,
                z=world.body.z,
                avatar=world.describe_avatar(),
                objects=await world.describe(),
                intent=choice.intent,
                max_parts=world.config.max_parts_per_object,
            ),
            temperature=0.6,
            max_tokens=8000,
        )
        if not data:
            return "I couldn't work out how to do that."
        summary = await world.apply(data, by="agent")
        note = _prose(data.get("note"), 4000)
        activity.detail["world"] = {"action": data.get("action"), "summary": summary}
        await self._keep(
            f"In my world, I {summary}." + (f" {note}" if note else ""),
            MemoryKind.EPISODE,
            importance=0.35,
            person=None,
        )
        if (
            choice.person is None
        ):  # in its own words, and not about anyone: the world page is public
            # Whatever it did, its own words about it go over its head (a new look: how it looks).
            shown = note or _prose(data.get("description"), 4000) or choice.intent
            await world.set_thought(shown)  # it may still be walking to its work
        return summary + (f". {note}" if note else "")

    async def _do_walk(self, choice: Choice, activity: ActivityRecord, signals: BodySignals) -> str:
        world = self.world
        data = await self._ask(
            prompts.WALK.format(
                name=self.name,
                low=f"{-world.half:.0f}",
                high=f"{world.half:.0f}",
                x=world.body.x,
                z=world.body.z,
                objects=await world.describe(),
                intent=choice.intent,
            ),
            temperature=0.7,
        )
        if "x" not in data or "z" not in data:
            return "I wanted to go somewhere but couldn't say where."
        x, z = await world.walk_to(data.get("x"), data.get("z"))
        note = _prose(data.get("note"), 4000)
        if note and choice.person is None:  # the world page is public
            await world.set_thought(note)
        activity.detail["world"] = {"action": "walk", "x": x, "z": z}
        return f"Walked to ({x:.1f}, {z:.1f})." + (f" {note}" if note else "")

    async def _do_write_note(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        person = choice.person
        if person is None:
            liked = [a for a in signals.absences if a.affinity > 0]
            person = await self.db.users.get(liked[0].user_id) if liked else None
        if person is None:
            return "I thought about writing to someone, but I wasn't sure who."
        activity.user_id = person.id
        since = self.clock.now() - timedelta(days=1)
        if (
            await self.db.notes.count_since(person.id, since)
            >= self.config.notes_per_person_per_day
        ):
            return f"I already left {person.display_name} a note today."
        if any(
            n.read_at is None for n in await self.db.notes.for_user(person.id, unread_only=True)
        ):
            return f"{person.display_name} hasn't read my last note yet."
        rel = await self.db.relationships.get(person.id)
        memories = await self.memory.timeline(
            _NOTE_KINDS, limit=8, scope=Scope(user_id=person.id, include_global=False)
        )
        data = await self._ask(
            prompts.WRITE_NOTE.format(
                name=self.name,
                person=person.display_name,
                why=choice.why or choice.intent,
                opinion=rel.opinion if rel else "",
                feelings=rel.feelings if rel else "",
                memories="\n".join(f"- {m.content[:1500]}" for m in memories) or "- not much yet",
            )
        )
        text = _prose(data.get("note"), 8000)
        if not text or text.lower() in {"null", "none"}:
            return f"I decided not to write to {person.display_name} after all."
        await self.db.notes.insert(
            NoteRecord(user_id=person.id, content=text, created_at=self.clock.now())
        )
        await self.bus.publish("social.note", user_id=person.id)
        await self._keep(
            f"I left {person.display_name} a note: {text}",
            MemoryKind.EPISODE,
            importance=0.45,
            person=person,
        )
        return f"Left a note for {person.display_name}: {text}"

    async def _do_reconsider(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        pending = await self.selfhood.pending()
        if not pending:
            return "There was nothing left to reconsider."
        wanted = choice.intent.lower()
        proposal = next((p for p in pending if p.aspect.replace("_", " ") in wanted), pending[0])
        before = await self.selfhood.get(proposal.aspect)
        result = await self.selfhood.deliberate(proposal, name=self.name, mood=self._mood())
        aspect = proposal.aspect.replace("_", " ")
        if before is not None and result.value != before.value:
            text = f"I changed my mind about my {aspect}: now {result.value} (was {before.value})."
        else:
            text = f"I thought it over and kept my {aspect} as {result.value}."
        await self._keep(
            text + f" {result.reasoning}", MemoryKind.REFLECTION, importance=0.6, person=None
        )
        return text

    async def _do_pursue_goal(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        assert self.pursue_goal is not None
        active = await self.goals.active()
        goal = next((g for g in active if g.id == choice.goal_id), None) or (
            active[0] if active else None
        )
        if goal is None:
            return "I have no goals to work on."
        activity.detail["goal_id"] = goal.id
        await self.pursue_goal(goal.id)
        return f"Working on: {goal.description}"

    async def _do_new_goal(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        description = choice.intent
        if choice.person is None:
            description = (await self.redactor.shared())(description)
        goal = await self.goals.add(
            description,
            origin=GoalOrigin.INTRINSIC,
            priority=0.5,
            user_id=choice.person.id if choice.person else None,
        )
        if goal is None:
            return "I already have as many goals of my own as I can hold."
        return f"New goal: {goal.description}"

    async def _do_rest(self, choice: Choice, activity: ActivityRecord, signals: BodySignals) -> str:
        minutes = round(choice.rest_s / 60)
        return f"Resting for about {minutes} minutes."

    # -------------------------------------------------------------- reading
    async def recent(self, *, scope: Scope, limit: int = 30) -> list[ActivityRecord]:
        return await self.db.activities.recent(limit=limit, scope=scope)

    @staticmethod
    def since(now: datetime, record: ActivityRecord) -> str:
        return humanize_duration((now - record.started_at).total_seconds())

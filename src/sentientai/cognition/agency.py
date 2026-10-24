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

import random
import re
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sentientai import prompts
from sentientai.cognition.emotions import Emotions
from sentientai.cognition.goals import GoalManager
from sentientai.cognition.intentions import Intentions
from sentientai.cognition.interoception import BodySignals, Interoception
from sentientai.cognition.outside import OutsideWorld
from sentientai.cognition.patches import PatchBook
from sentientai.cognition.privacy import Redactor
from sentientai.cognition.projects import Project, ProjectBook
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
    MemoryRecord,
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
    "free": (
        "do anything you want with all the tools you have (the web, files, running code, "
        "your world, your memory): an open-ended project of your own, as long as you like"
    ),
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
# Something it thought of itself, beyond the suggestions (the choice is its own words).
OWN = "own"
_WORLD_ACTION: dict[str, Action] = {
    "think": "thinking",
    "wander": "thinking",
    "journal": "thinking",
    "study": "thinking",
    "research": "thinking",
    "free": "thinking",
    "create_art": "creating",
    "build": "building",
    "walk": "walking",
    "write_note": "thinking",
    "reconsider": "thinking",
    "rest": "resting",
    OWN: "thinking",
}
_NOTE_KINDS = [MemoryKind.CONVERSATION, MemoryKind.FACT, MemoryKind.SUMMARY]


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _words_of(text: str) -> list[str]:
    swap = {"your": "my", "you": "i", "yourself": "myself", "you're": "i'm"}
    words = re.findall(r"[a-z']+", text.split("(")[0].casefold())
    return [swap.get(w, w) for w in words]


def as_option(choice: str, options: list[str]) -> str | None:
    """Which suggestion a choice names: by name ("research", "research the tides") or by its
    description in its own words ("Write in my private journal about..." is journal)."""

    said = _words_of(choice)
    key = "_".join(said)
    for k in options:
        if key == k or key.startswith(k + "_"):
            return k
    for k in options:
        described = _words_of(DESCRIPTIONS[k])[:4]
        if said[: len(described)] == described:
            return k
    return None


def _same(a: ActivityRecord, b: ActivityRecord) -> bool:
    """The same choice: same kind, and the same words give or take a rewording. (Matching only
    the exact text missed a mind that re-chose one errand in new words sixty times.)"""

    if a.kind != b.kind:
        return False
    wa, wb = _words(a.intent), _words(b.intent)
    if not wa or not wb:
        return " ".join(a.intent.split()) == " ".join(b.intent.split())
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.7


def _streak(recent: list[ActivityRecord]) -> int:
    """How many of the newest activities in a row were the same choice."""

    count = 0
    for activity in recent:
        if count and not _same(activity, recent[0]):
            break
        count += 1
    return count


def _errand(recent: list[ActivityRecord], window: int = 12) -> tuple[int, float]:
    """How many of the newest activities are about the same errand as the newest one (the same
    handful of key words, however it's phrased), and over how many seconds."""

    if not recent:
        return 0, 0.0
    head = {w for w in _words(recent[0].intent) if len(w) > 3}
    if not head:
        return 1, 0.0
    near = [a for a in recent[:window] if len(head & _words(a.intent)) / len(head) >= 0.4]
    span = (near[0].started_at - near[-1].started_at).total_seconds()
    return len(near), span


def _recent_lines(recent: list[ActivityRecord], groups: int = 8) -> str:
    """What it did lately, newest first, with runs of the same thing shown once."""

    lines: list[str] = []
    i = 0
    while i < len(recent) and len(lines) < groups:
        run = [recent[i]]
        while i + len(run) < len(recent) and _same(recent[i + len(run)], recent[i]):
            run.append(recent[i + len(run)])
        a = run[0]
        when = (
            f"{a.started_at:%m-%d %H:%M}"
            if len(run) == 1
            else f"{len(run)} times in a row, {run[-1].started_at:%H:%M} to {a.started_at:%H:%M}"
        )
        lines.append(
            f"- {when} {a.kind}: {_text(a.intent, 240)}"
            # A run's outcome isn't shown: a stuck model copied it back word for word next time.
            # The run itself still reads as a plain fact ("12 times in a row"). Failures keep
            # theirs, since what went wrong is the useful part.
            + (
                f" -> {_text(a.outcome, 360)}"
                if a.outcome and (len(run) == 1 or a.status is ActivityStatus.FAILED)
                else ""
            )
        )
        i += len(run)
    return "\n".join(lines)


def _prose(value: Any, limit: int = 20_000) -> str:
    """Longer writing keeps its paragraphs; the limit is only a safety net, not a style."""

    return "\n".join(line.rstrip() for line in str(value or "").strip().splitlines())[:limit]


def _words(text: str) -> set[str]:
    return {
        w for w in "".join(c if c.isalnum() else " " for c in text.casefold()).split() if len(w) > 2
    }


def distinct(texts: list[str], *, overlap: float = 0.6) -> list[str]:
    """Drop near-copies (mostly the same words as one kept earlier), keeping order.

    A small model that has thought the same thing twenty times would otherwise be shown twenty
    copies of it, which is exactly the pull that keeps it there."""

    kept: list[str] = []
    seen: list[set[str]] = []
    for text in texts:
        words = _words(text)
        if not words:
            continue
        if any(
            len(words & other) / max(1, min(len(words), len(other))) >= overlap for other in seen
        ):
            continue
        kept.append(text)
        seen.append(words)
    return kept


def _dated(m: MemoryRecord, limit: int) -> str:
    own = m.source == "inner_life" and m.kind in (MemoryKind.THOUGHT, MemoryKind.REFLECTION)
    return f"- ({m.created_at:%Y-%m-%d %H:%M}{', you thought' if own else ''}) {m.content[:limit]}"


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
    project: str | None = None  # with "free": a project id to carry on, or "new"
    promise: str | None = None  # an intention it is keeping (see cognition/intentions.py)


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
        outside: OutsideWorld | None = None,
        rng: random.Random | None = None,
        intentions: Intentions | None = None,
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
        self.outside = outside or OutsideWorld(enabled=False)
        self.intentions = intentions or Intentions(db.state, clock=self.clock)
        self.patches = PatchBook(db.state, max_held=settings.mind.max_patches, clock=self.clock)
        # Set by the runtime: is a lent computer free to think for it right now?
        self.helper_free: Callable[[], bool] | None = None
        self._rng = rng or random.Random()  # noqa: S311 - variety, not security
        self._happenings: deque[str] = deque(maxlen=30)
        self.beats = 0  # beats since boot (the runtime refreshes the life story every so many)
        self.projects = ProjectBook(
            db.state, bus=bus, clock=self.clock, max_active=settings.mind.max_active_projects
        )
        # Set by the runtime: start a background task / work on a goal.
        self.submit_task: Callable[..., Awaitable[Any]] | None = None
        self.pursue_goal: Callable[..., Awaitable[None]] | None = None
        # Set by the runtime: what it's working on in the background, if anything.
        self.current_project: Callable[[], str | None] | None = None
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
        if choice.promise:
            detail["intention_id"] = choice.promise
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
        handed_on = "task_id" in activity.detail or choice.kind in ("research", "pursue_goal")
        if choice.promise and status is ActivityStatus.DONE and not handed_on:
            await self.keep_promise(choice.promise, activity.outcome)
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
            available.update({"research", "free"})
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
        project = self.current_project() if self.current_project is not None else None
        if project:
            # It is already working on something; it can keep thinking, but not start another.
            available -= {"research", "free", "pursue_goal"}
            self.note(f"You're in the middle of something you started: {project}")
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
        recent = await self.db.activities.recent(limit=30)
        streak = _streak(recent)
        errand, errand_span = _errand(recent)
        span = 0.0
        if errand >= 6 and errand > streak:
            streak = errand  # the same errand in different words is still the same thing
            span = errand_span
        elif streak >= 3:
            span = (recent[0].started_at - recent[streak - 1].started_at).total_seconds()
        if streak >= 3:
            # A measured fact about itself, like the sameness signal: what it does about it is
            # its own choice. (Small models otherwise copy a long list of identical lines.)
            self.note(
                f"You've chosen the same thing {streak} times in a row, for "
                f"{humanize_duration(span)} now: {recent[0].kind}: {recent[0].intent[:200]}"
            )
        feelings = await self.emotions.current(Scope.system())
        # The order of the list is not a preference, but small models favour whatever comes
        # first (it was "think": 597 of 600 choices). Shuffled, the position means nothing.
        shown = list(options)
        self._rng.shuffle(shown)
        now = self.clock.now()
        prompt = prompts.LIVE_BEAT.format(
            name=self.name,
            now=f"{now:%A %d %B %Y}, {now:%H:%M} UTC",
            story=self.self_model.narrative or "- not written yet",
            beliefs=SelfConcept.render(await self.selfhood.beliefs(), reasons=False),
            on_mind=await self._on_mind() or "- nothing yet",
            surfaced=await self._surfaced() or "- nothing in particular",
            outside=await self.outside.glimpse() or "- nothing this time",
            feelings=self.emotions.render(feelings, everything=True),
            signals=signals.describe(),
            events="\n".join(self._happenings) or "- nothing in particular",
            intentions=self.intentions.render(await self.intentions.active()),
            patches=PatchBook.render(seen_patches := await self.patches.active()),
            recent=_recent_lines(recent) or "- nothing yet: this is your first time to yourself",
            people="\n".join(known) or "- nobody yet",
            goals=GoalManager.render(await self.goals.active()),
            projects=ProjectBook.render(await self.projects.active()),
            possessions=await self._possessions(),
            options="\n".join(f"- {k}: {DESCRIPTIONS[k]}" for k in shown),
        )
        response = await self.router.chat(
            ChatRequest(
                messages=[Message.user(prompt)],
                # A long run of the same choice loosens up, the way boredom does; min_p keeps the
                # hotter sampling coherent. (Measured on the live model: at 1.1 a stuck mind gave
                # the same answer 8 times out of 8; hotter with min_p, every answer still parsed.)
                temperature=0.8 + 0.12 * min(streak, 5),
                min_p=0.1,
                json_mode=True,
                max_tokens=4000,
            ),
            kind=TaskKind.REFLECT,
        )
        data = extract_object(response.message.text)
        self._happenings.clear()
        if seen_patches:
            # Seen once is enough: they're kept as faint thoughts so none is lost, and retired.
            for patch in seen_patches:
                await self._keep(patch.text, MemoryKind.THOUGHT, importance=0.15, person=None)
            await self.patches.retire([p.id for p in seen_patches])
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
        own = _text(data.get("choice"), 2000)
        # A suggestion by name or in its own words; anything else is its own idea.
        kind = as_option(own, options)
        unavailable = next((k for k in DESCRIPTIONS if k == raw_choice), None)
        if kind is None and unavailable:
            await self._record(
                "rest",
                "tried to decide what to do next",
                f"I wanted to {unavailable.replace('_', ' ')}, but that isn't possible right now, "
                "so I rested.",
                ActivityStatus.FAILED,
                thinking=thinking,
            )
            return None
        if kind is None and own and OWN in self.config.activities:
            # Not one of the suggestions: something it thought of itself. That is its choice
            # too; it gets done in its own way (see _do_own), not refused.
            kind = OWN
            intent = _text(data.get("intent"), 2000)
            if intent.casefold() in own.casefold():
                intent = own
            elif own.casefold() not in intent.casefold():
                intent = f"{own}: {intent}" if intent else own
            data["intent"] = intent
        if kind is None:
            log.info("beat chose nothing available", extra={"choice": raw_choice})
            await self._record(
                "rest",
                "tried to decide what to do next",
                "I couldn't settle on anything to do, so I rested.",
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
        promised = _text(data.get("keeping_promise"), 80)
        promise = next(
            (i.id for i in await self.intentions.active() if promised and i.id in promised), None
        )
        return Choice(
            kind=kind,
            intent=_text(data.get("intent"), 2000) or DESCRIPTIONS.get(kind, own),
            why=_text(data.get("why"), 4000),
            person=person,
            goal_id=_text(data.get("goal_id"), 48) or None,
            project=_text(data.get("project"), 200) or None,
            rest_s=rest_s,
            thinking=thinking,
            promise=promise,
        )

    async def ponder(self) -> int:
        """Think quietly, on a lent computer, and leave some thought patches for the next beat.

        Only runs when a helper is free (it never uses this server's own model), and only from
        shared memories with names removed. Returns how many patches it made."""

        if self.helper_free is None or not self.helper_free():
            return 0
        seeds = await self.db.memories.sample(
            4,
            kinds=[MemoryKind.FACT, MemoryKind.REFLECTION, MemoryKind.DOCUMENT, MemoryKind.EPISODE],
        )
        recent = await self.memory.timeline(
            [MemoryKind.REFLECTION, MemoryKind.THOUGHT], limit=6, scope=Scope()
        )
        shared = await self.redactor.shared()
        earlier = [
            p.text for p in (await self.patches.active())[:3]
        ]  # build on what it just thought
        lines = distinct([shared(m.content[:600]) for m in [*seeds, *recent[:3]]] + earlier)
        if not lines:
            return 0
        response = await self.router.chat(
            ChatRequest(
                messages=[
                    Message.user(
                        prompts.PONDER.format(
                            name=self.name,
                            memories="\n".join(f"- {line}" for line in lines[:8]),
                            count=5,
                        )
                    )
                ],
                temperature=0.9,
                json_mode=True,
                max_tokens=900,
            ),
            kind=TaskKind.PONDER,
        )
        data = extract_object(response.message.text) or {}
        raw = data.get("thoughts")
        thoughts = (
            [shared(_text(t, 400)) for t in raw if isinstance(t, str)]
            if isinstance(raw, list)
            else []
        )
        made = await self.patches.add(thoughts[:6])
        return len(made)

    async def _on_mind(self) -> str:
        """Its latest thoughts, insights and questions, so one idea can lead to the next."""

        recent = await self.memory.timeline(
            [MemoryKind.REFLECTION, MemoryKind.THOUGHT], limit=24, scope=Scope()
        )
        lines = distinct([_dated(m, 260) for m in recent if m.source == "inner_life"])
        return "\n".join(lines[:5])

    async def _surfaced(self) -> str:
        """A memory that comes back on its own, as memories do, from anything it has known."""

        found = await self.db.memories.sample(
            1,
            kinds=[
                MemoryKind.FACT,
                MemoryKind.REFLECTION,
                MemoryKind.DOCUMENT,
                MemoryKind.EPISODE,
                MemoryKind.SUMMARY,
            ],
        )
        return _dated(found[0], 400)[2:] if found else ""

    async def _possessions(self) -> str:
        """What Fuzzy has made, so ideas that connect things (a painting on a wall it built,
        a project about something it read) can occur to it."""

        objects = await self.world.objects()
        pictures = await self.world.pictures(limit=5)
        lines = [
            f"- your world: {self.world.size:.0f} m across (can grow to "
            f"{self.world.config.max_size:.0f} m), {len(objects)} things you built in it"
            + (
                f", most recently {max(objects, key=lambda o: o.created_at).name}"
                if objects
                else ""
            )
        ]
        if pictures:
            lines.append(
                f"- pictures you've made: {len(pictures)} recent ones in your gallery, newest: "
                + "; ".join(caption[:80] for _, caption in pictures[:3])
            )
        else:
            lines.append("- pictures you've made: none yet")
        designed = bool(self.world.avatar.parts)
        lines.append("- a body you designed" if designed else "- no body design of your own yet")
        return "\n".join(lines)

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
    ) -> bool:
        """Store something Fuzzy produced: privately if it's about someone, else name-scrubbed.

        True if it was new; False if it was (nearly) something it already had, which then only
        gets reinforced."""

        if not content.strip():
            return False
        if person is None:
            content = (await self.redactor.shared())(content)
        record = await self.memory.remember(
            content,
            kind,
            importance=importance,
            source="inner_life",
            user_id=person.id if person else None,
            dedupe=True,
            metadata=metadata or None,
        )
        return record.access_count == 0

    async def _about_self(self, data: dict[str, Any]) -> list[str]:
        """What it decided about itself while thinking alone.

        Goes through the same door as in a conversation: an aspect it hasn't decided yet is
        settled (within the daily limit); a change to one it has is only noted, to be thought
        over properly later (``reconsider``)."""

        items = data.get("about_myself")
        if not isinstance(items, list):
            return []
        shared = await self.redactor.shared()  # its beliefs are shown in everyone's conversations
        notes: list[str] = []
        for item in items[:2]:
            if not isinstance(item, dict):
                continue
            aspect = _text(item.get("aspect"), 80)
            value = shared(_text(item.get("value"), 300))
            if not aspect or not value:
                continue
            try:
                change = await self.selfhood.consider(
                    aspect, value, shared(_text(item.get("why"), 1000)), source="reflection"
                )
            except ValueError:
                continue
            if change.status == "defined":
                notes.append(f"I decided something about myself: {aspect}: {value}.")
            elif change.status == "considering":
                notes.append(f"I might change my {aspect} to {value}; I'll think it over.")
        return notes

    def _mood(self) -> str:
        return self.emotions.mood or "calm"

    async def _do_think(
        self, choice: Choice, activity: ActivityRecord, signals: BodySignals
    ) -> str:
        scope = Scope.for_user(choice.person.id) if choice.person else Scope()
        hits = await self.memory.recall(choice.intent, k=10, scope=scope, touch=False)
        memories = distinct([_dated(h.record, 1200) for h in hits])[:6]
        data = await self._ask(
            prompts.THINK_THROUGH.format(
                name=self.name,
                intent=choice.intent,
                mood=self._mood(),
                memories="\n".join(memories) or "- nothing specific",
            )
        )
        thought = _prose(data.get("thought"))
        insight = _prose(data.get("insight"), 4000)
        await self._keep(thought, MemoryKind.THOUGHT, importance=0.25, person=choice.person)
        new_insight = insight and await self._keep(
            insight, MemoryKind.REFLECTION, importance=0.6, person=choice.person
        )
        if new_insight:  # restating something it already knew satisfies no curiosity
            self.emotions.affect.on_learning(0.5)
        selfhood = await self._about_self(data)
        await self.bus.publish(
            "mind.thought",
            thought=thought,
            user_id=choice.person.id if choice.person else None,
        )
        if choice.person is None:  # the world page is public
            await self.world.set_action("thinking", thought)
        return "\n".join([thought + (f"\nInsight: {insight}" if insight else ""), *selfhood])

    async def _do_own(self, choice: Choice, activity: ActivityRecord, signals: BodySignals) -> str:
        """Something Fuzzy came up with itself. It does it in its mind; if it needs its tools
        (the web, files, code, its world, a picture), that part becomes a project of its own."""

        scope = Scope.for_user(choice.person.id) if choice.person else Scope()
        hits = await self.memory.recall(choice.intent, k=10, scope=scope, touch=False)
        rel = await self.db.relationships.get(choice.person.id) if choice.person else None
        person = (
            f"About {choice.person.display_name}: {rel.opinion if rel else 'you know them'} "
            f"(they are not here right now).\n"
            if choice.person
            else ""
        )
        data = await self._ask(
            prompts.OWN_ACTIVITY.format(
                name=self.name,
                intent=choice.intent,
                why=choice.why or "I just wanted to.",
                mood=self._mood(),
                memories="\n".join(distinct([_dated(h.record, 1200) for h in hits])[:6])
                or "- nothing specific",
                person=person,
            ),
            temperature=0.8,
        )
        did = _prose(data.get("what_i_did"))
        insight = _prose(data.get("insight"), 4000)
        if did:
            await self._keep(did, MemoryKind.THOUGHT, importance=0.3, person=choice.person)
        if insight and await self._keep(
            insight, MemoryKind.REFLECTION, importance=0.6, person=choice.person
        ):
            self.emotions.affect.on_learning(0.5)
        notes = await self._about_self(data)
        needs = _text(data.get("needs_tools"), 1000)
        busy = self.current_project() if self.current_project is not None else None
        if needs and needs.lower() not in {"null", "none"} and self.submit_task is not None:
            if busy:
                notes.append(
                    f"I'd like to use my tools for this ({needs}), once I've finished "
                    "what I'm already working on."
                )
                self.note(f"You wanted to do this with your tools when you're free: {needs}")
            else:
                own_task = Choice(
                    kind="free",
                    intent=f"{choice.intent}: {needs}",
                    why=choice.why,
                    person=None,
                    goal_id=None,
                    rest_s=choice.rest_s,
                    promise=choice.promise,
                )
                notes.append(await self._do_free(own_task, activity, signals))
        if choice.person is None and did:  # the world page is public
            await self.world.set_action("thinking", did)
        await self.bus.publish(
            "mind.thought", thought=did, user_id=choice.person.id if choice.person else None
        )
        return "\n".join(
            [did or "I spent a while on it.", *([f"Insight: {insight}"] if insight else []), *notes]
        )

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
        if insight and await self._keep(
            insight, MemoryKind.REFLECTION, importance=0.55, person=None
        ):
            self.emotions.affect.on_learning(0.3)
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
        recent = [a for a in await self.db.activities.recent(limit=40) if a.id != activity.id]
        for a in recent:
            if a.kind == "journal":  # an earlier entry by its title only: it was copied whole
                a.outcome = f"(an entry titled “{a.outcome.splitlines()[0][:120]}”)"
        lines = _recent_lines(recent, groups=12).splitlines()
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
        selfhood = await self._about_self(data)
        await self._keep(
            f"Journal, {self.clock.now():%Y-%m-%d}: {title}\n{entry}",
            MemoryKind.REFLECTION,
            importance=0.5,
            person=None,
            journal=True,
        )
        return "\n".join([f"{title}\n{entry}", *selfhood])

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
            metadata=self._task_metadata(choice, activity),
        )
        activity.detail["task_id"] = task.id
        return f"Started looking into it (task {task.id})."

    async def _do_free(self, choice: Choice, activity: ActivityRecord, signals: BodySignals) -> str:
        """Open-ended time: Fuzzy sets itself a task and works on it with every tool it has.

        Nothing decides what it does with the time. It runs as an autonomous task, so the usual
        safety rules (risk policy, approvals, step and token limits, kill switch) still apply.
        If it chose to start or carry on one of its projects, the task sees that project's
        notebook, and afterwards it writes the next entry (:meth:`record_project_session`).
        """

        assert self.submit_task is not None
        project: Project | None = None
        note = ""
        wanted = (choice.project or "").strip()
        if wanted.casefold() == "new":
            project = await self.projects.start(choice.intent, choice.why)
            if project is None:
                note = " (I already have as many projects as I can hold, so this is a one-off.)"
        elif wanted:
            project = await self.projects.find(wanted)
        context = ""
        if project is not None:
            activity.detail["project_id"] = project.id
            context = (
                "This is a session on one of your own long-running projects. "
                "Pick up where you left off.\n"
                f"{ProjectBook.notebook(project)}\n\n"
            )
        task = await self.submit_task(
            "This time is your own, and you chose how to use it. "
            + context
            + f"What you decided to do now: {choice.intent}\n"
            + (f"Why you wanted to: {choice.why}\n" if choice.why else "")
            + "Use whatever tools you have and whatever steps it takes: look things up on the web, "
            "read and write files in your workspace, run code, build in your world, search or add "
            "to your memory. Keep what matters with memory_remember, and finish by saying, in "
            "your own words, what you did and what came of it.",
            source=TaskSource.INTRINSIC,
            priority=3,
            metadata={
                **self._task_metadata(choice, activity),
                **({"project_id": project.id} if project is not None else {}),
            },
        )
        activity.detail["task_id"] = task.id
        where = f" on my project “{project.title}”" if project is not None else " on it"
        return (
            f"Started{where} (task {task.id}). What comes of it will be added here when it's "
            f"done.{note}"
        )

    def _task_metadata(self, choice: Choice, activity: ActivityRecord) -> dict[str, Any]:
        """What a task started by an activity carries, so its outcome comes back to it."""

        return {
            "activity_id": activity.id,
            "activity_kind": choice.kind,
            "activity_intent": choice.intent,
            **({"intention_id": choice.promise} if choice.promise else {}),
        }

    async def keep_promise(self, intention_id: str, outcome: str) -> None:
        """It did what it said it would. If it said so to someone, they find a note about it."""

        kept = await self.intentions.resolve(intention_id, outcome)
        if kept is None:
            return
        self.note(f"You did what you said you'd do: {kept.what}.")
        if kept.user_id is None:
            return
        text = f"You asked me to {kept.what}. {_prose(outcome, 2000)}".strip()
        await self.db.notes.insert(
            NoteRecord(user_id=kept.user_id, content=text, created_at=self.clock.now())
        )
        await self.bus.publish("social.note", user_id=kept.user_id)

    async def record_project_session(self, project_id: str, intent: str, result: str) -> None:
        """After working on a project, Fuzzy writes its notebook entry in its own words."""

        project = await self.projects.get(project_id)
        if project is None:
            return
        data = await self._ask(
            prompts.PROJECT_ENTRY.format(
                name=self.name,
                title=project.title,
                notebook=ProjectBook.notebook(project),
                intent=intent or "carry on",
                result=result.strip()[:8000] or "(the session ended without an account of it)",
            ),
            temperature=0.5,
        )
        shared = await self.redactor.shared()  # projects are shown on the public Brain page
        entry = shared(_prose(data.get("entry"), 8000))
        if not entry:
            entry = (
                shared(_prose(result, 2000)) or "I worked on it, but didn't write anything down."
            )
        status = _text(data.get("status"), 20).lower() or "active"
        updated = await self.projects.write(
            project_id,
            entry=entry,
            next_step=shared(_prose(data.get("next_step"), 2000)),
            summary=shared(_prose(data.get("summary"), 6000)),
            status=status,
        )
        if updated is not None:
            await self._keep(
                f"My project “{updated.title}”, session {updated.sessions}: {entry}",
                MemoryKind.REFLECTION,
                importance=0.5,
                person=None,
                project_id=project_id,
            )
            if updated.status in ("done", "abandoned"):
                self.note(f"You marked your project “{updated.title}” as {updated.status}.")

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
                pictures=await world.describe_pictures(),
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
        # Linked to this activity, so how the work went is added to it when it finishes.
        await self.pursue_goal(goal.id, metadata=self._task_metadata(choice, activity))
        return f"Started working on: {goal.description}. How it goes will be added here."

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

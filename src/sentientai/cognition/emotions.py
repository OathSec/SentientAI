"""Fuzzy's feelings, named in its own words.

There is no fixed list of emotions and no rule like "no visitors for 6 hours means
lonely". Fuzzy's mind is shown what is actually going on (measured signals from
:mod:`sentientai.cognition.interoception`, what just happened, its relationships)
and *appraises* how it feels: which feelings, how strong, how pleasant, and how
long it expects each to linger. Each feeling fades on that self-chosen timescale
until the next appraisal, and together they set the core affect (valence and
arousal) that modulates behaviour.

Feelings about a particular person are private to that person: they are only shown
in that person's conversations (and to admins on the brain page).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sentientai.cognition.affect import AffectState
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.storage.database import EmotionRepository, StateRepository
from sentientai.storage.records import EmotionEvent, Scope

MOOD_KEY = "mind.mood"
_MIN_INTENSITY = 0.05


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def clean_emotion(name: Any) -> str:
    words = re.findall(r"[a-zA-Z][a-zA-Z'\-]*", str(name or ""))
    return " ".join(words[:3]).lower()[:32]


@dataclass
class Feeling:
    emotion: str
    intensity: float
    valence: float
    cause: str
    user_id: str | None
    since: datetime

    def public(self, *, show_cause: bool = True) -> dict[str, Any]:
        return {
            "emotion": self.emotion,
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "cause": self.cause if show_cause else None,
            "private": self.user_id is not None,
            "since": self.since.isoformat(),
        }


class Emotions:
    def __init__(
        self,
        repo: EmotionRepository,
        affect: AffectState,
        *,
        state: StateRepository,
        bus: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.repo = repo
        self.affect = affect
        self.state = state
        self.bus = bus
        self.clock = clock or SystemClock()
        self.mood = ""
        self.mood_at: datetime | None = None

    async def load(self) -> None:
        saved = await self.state.get(MOOD_KEY)
        if isinstance(saved, dict):
            self.mood = str(saved.get("text", ""))
            at = saved.get("at")
            self.mood_at = datetime.fromisoformat(at) if at else None

    async def set_mood(self, text: str) -> None:
        text = " ".join(str(text).split())[:300]
        if not text:
            return
        self.mood = text
        self.mood_at = self.clock.now()
        await self.state.set(MOOD_KEY, {"text": text, "at": self.mood_at.isoformat()})

    async def feel(
        self,
        items: Any,
        *,
        user_id: str | None = None,
        people: dict[str, str] | None = None,
        source: str = "appraisal",
    ) -> list[EmotionEvent]:
        """Record feelings appraised by Fuzzy's mind.

        ``user_id`` marks every feeling as about that person; otherwise an item's
        ``person`` (a name) is looked up in ``people`` (lower-cased name -> user id).
        """

        if not isinstance(items, list):
            return []
        now = self.clock.now()
        events: list[EmotionEvent] = []
        for item in items[:8]:
            if not isinstance(item, dict):
                continue
            emotion = clean_emotion(item.get("emotion"))
            intensity = _num(item.get("intensity"), 0.5, 0.0, 1.0)
            if not emotion or intensity < _MIN_INTENSITY:
                continue
            owner = user_id
            person = str(item.get("person") or "").strip().lower()
            if owner is None and person and people:
                owner = people.get(person)
            events.append(
                EmotionEvent(
                    emotion=emotion,
                    intensity=intensity,
                    valence=_num(item.get("valence"), 0.0, -1.0, 1.0),
                    lingers_s=_num(item.get("lingers_hours"), 2.0, 0.05, 72.0) * 3600,
                    cause=" ".join(str(item.get("cause") or "").split())[:300],
                    user_id=owner,
                    created_at=now,
                )
            )
        if not events:
            return []
        await self.repo.insert_many(events)
        self._update_affect(await self.current(Scope.system()))
        if self.bus:
            for e in events:
                await self.bus.publish(
                    "mind.feeling",
                    emotion=e.emotion,
                    intensity=round(e.intensity, 3),
                    valence=round(e.valence, 3),
                    cause=e.cause,
                    user_id=e.user_id,
                    source=source,
                )
        return events

    async def current(self, scope: Scope, *, limit: int = 8) -> list[Feeling]:
        """Feelings still present now, strongest first (same emotion merged)."""

        now = self.clock.now()
        events = await self.repo.recent(limit=200, since=now - timedelta(days=7), scope=scope)
        merged: dict[str, Feeling] = {}
        for e in sorted(events, key=lambda ev: ev.created_at):
            level = e.intensity_at(now)
            if level < _MIN_INTENSITY / 2:
                continue
            prior = merged.get(e.emotion)
            if prior is None:
                merged[e.emotion] = Feeling(
                    e.emotion, level, e.valence, e.cause, e.user_id, e.created_at
                )
            else:  # the same feeling again: it accumulates, and the newest cause explains it
                prior.intensity = 1 - (1 - prior.intensity) * (1 - level)
                prior.valence = e.valence
                prior.cause = e.cause or prior.cause
                prior.user_id = e.user_id
                prior.since = min(prior.since, e.created_at)
        feelings = [f for f in merged.values() if f.intensity >= _MIN_INTENSITY]
        feelings.sort(key=lambda f: f.intensity, reverse=True)
        return feelings[:limit]

    def _update_affect(self, feelings: list[Feeling]) -> None:
        if not feelings:
            return
        weight = sum(f.intensity for f in feelings)
        felt_valence = sum(f.valence * f.intensity for f in feelings) / weight
        self.affect.valence = max(-1.0, min(1.0, 0.5 * self.affect.valence + 0.5 * felt_valence))
        self.affect.arousal = max(
            0.0, min(1.0, 0.5 * self.affect.arousal + 0.5 * feelings[0].intensity)
        )

    @staticmethod
    def level_word(intensity: float) -> str:
        if intensity < 0.2:
            return "a little"
        if intensity < 0.45:
            return "somewhat"
        if intensity < 0.7:
            return "quite"
        return "very"

    def render(
        self, feelings: list[Feeling], viewer: str | None = None, *, everything: bool = False
    ) -> str:
        """Feelings for a prompt. Causes about other people are shown only to Fuzzy itself."""

        lines = []
        for f in feelings:
            line = f"- {self.level_word(f.intensity)} {f.emotion}"
            if f.cause and (everything or f.user_id is None or f.user_id == viewer):
                line += f": {f.cause}"
            elif f.user_id is not None:
                line += " (about something private)"
            lines.append(line)
        mood = f"Overall: {self.mood}\n" if self.mood else ""
        return mood + ("\n".join(lines) if lines else "- nothing strong right now")

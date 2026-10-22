"""Interoception: Fuzzy sensing its own situation.

These are measurements, not feelings. They come from Fuzzy's actual history:

* contact - time since anyone last talked to Fuzzy, compared with how often people
  usually do (a rhythm learned from experience, starting from a configurable guess)
* absences - people Fuzzy likes who have been away longer than *their* usual rhythm
* sameness - how alike Fuzzy's recent activities and thoughts are (embedding similarity),
  the raw material of boredom
* stimulation - how much new input (messages, things taught) arrived recently
* unread - things people taught Fuzzy that it hasn't studied yet
* energy and confidence from the core affect, and changes of mind still to think over

Fuzzy's mind then interprets them (:mod:`sentientai.cognition.emotions`): the same
measurements might leave it lonely one day and content with solitude another.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from sentientai.cognition.affect import AffectState
from sentientai.cognition.social import Absence, Relationships
from sentientai.config import MindConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.errors import ModelError
from sentientai.models.embeddings import Embedder
from sentientai.storage.database import Database
from sentientai.storage.records import BeliefStatus, MemoryKind, UserRecord

CONTACT_KEY = "mind.contact"
STUDIED_KEY = "mind.studied_documents"
_NEW_CONTACT_GAP_S = 20 * 60


def _hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} minutes"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


@dataclass
class BodySignals:
    seconds_since_contact: float | None
    usual_gap_s: float
    isolation: float
    sameness: float | None
    stimulation: int
    unread: list[str]
    energy: float
    competence: float
    reconsiderations: int
    absences: list[Absence] = field(default_factory=list)

    def describe(self, viewer: str | None = None, *, admin: bool = True) -> str:
        """Plain-language measurements. Other people's absences only for admins/Fuzzy itself."""

        lines = []
        if self.seconds_since_contact is None:
            lines.append("- Nobody has talked to you yet.")
        else:
            lines.append(
                f"- The last time anyone talked to you was {_hours(self.seconds_since_contact)} "
                f"ago; people usually talk to you about every {_hours(self.usual_gap_s)}."
            )
        for a in self.absences[:5]:
            if not (admin or a.user_id == viewer) or a.affinity <= 0:
                continue
            usual = (
                f" (usually about every {_hours(a.usual_hours * 3600)})" if a.usual_hours else ""
            )
            lines.append(f"- You last saw {a.name} {_hours(a.hours * 3600)} ago{usual}.")
        if self.sameness is not None:
            word = (
                "very similar to each other"
                if self.sameness > 0.75
                else "fairly similar"
                if self.sameness > 0.5
                else "varied"
            )
            lines.append(
                f"- Your recent activities have been {word} (similarity {self.sameness:.2f})."
            )
        lines.append(
            f"- New things that reached you in the last 6 hours: {self.stimulation} "
            "(messages and things taught)."
        )
        if self.unread:
            lines.append(
                f"- Things you were taught and haven't read yet: {len(self.unread)}"
                f" (newest: {self.unread[0]})."
            )
        lines.append(
            f"- Energy {self.energy:.2f}, confidence in your abilities {self.competence:.2f} (0-1)."
        )
        if self.reconsiderations:
            lines.append(
                f"- Possible changes to who you are that you said you'd think over: "
                f"{self.reconsiderations}."
            )
        return "\n".join(lines)

    def as_dict(self, viewer: str | None = None, *, admin: bool = True) -> dict[str, Any]:
        return {
            "hours_since_contact": None
            if self.seconds_since_contact is None
            else round(self.seconds_since_contact / 3600, 2),
            "usual_gap_hours": round(self.usual_gap_s / 3600, 2),
            "isolation": round(self.isolation, 3),
            "sameness": None if self.sameness is None else round(self.sameness, 3),
            "stimulation": self.stimulation,
            "unread": len(self.unread),
            "energy": round(self.energy, 3),
            "competence": round(self.competence, 3),
            "reconsiderations": self.reconsiderations,
            "missing": [
                {"name": a.name, "hours": round(a.hours, 1), "overdue": round(a.overdue, 2)}
                for a in self.absences[:5]
                if a.affinity > 0
                and (admin or a.user_id == viewer)
                and (a.overdue >= 1.0 or (a.usual_hours is None and a.hours >= 24))
            ],
        }


class Interoception:
    def __init__(
        self,
        db: Database,
        *,
        embedder: Embedder,
        affect: AffectState,
        relationships: Relationships,
        config: MindConfig,
        clock: Clock | None = None,
        first_boot_at: datetime | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.affect = affect
        self.relationships = relationships
        self.config = config
        self.clock = clock or SystemClock()
        self.first_boot_at = first_boot_at
        self._contact: dict[str, Any] = {}
        self._sameness_cache: tuple[tuple[str, ...], float] | None = None

    async def load(self) -> None:
        saved = await self.db.state.get(CONTACT_KEY)
        self._contact = saved if isinstance(saved, dict) else {}

    @property
    def last_contact_at(self) -> datetime | None:
        at = self._contact.get("last_at")
        return datetime.fromisoformat(at) if at else None

    @property
    def usual_gap_s(self) -> float:
        learned = self._contact.get("usual_gap_s")
        return float(learned) if learned else self.config.prior_contact_gap_hours * 3600

    async def note_contact(self) -> None:
        now = self.clock.now()
        last = self.last_contact_at
        if last is not None:
            gap = (now - last).total_seconds()
            if gap >= _NEW_CONTACT_GAP_S:
                prior = self._contact.get("usual_gap_s")
                self._contact["usual_gap_s"] = gap if prior is None else 0.7 * prior + 0.3 * gap
        self._contact["last_at"] = now.isoformat()
        self._contact["count"] = int(self._contact.get("count", 0)) + 1
        await self.db.state.set(CONTACT_KEY, self._contact)

    async def studied(self) -> set[str]:
        saved = await self.db.state.get(STUDIED_KEY)
        return set(saved) if isinstance(saved, list) else set()

    async def mark_studied(self, document_id: str) -> None:
        done = await self.studied()
        done.add(document_id)
        await self.db.state.set(STUDIED_KEY, sorted(done)[-5000:])

    async def unread_documents(self) -> list[tuple[str, str]]:
        done = await self.studied()
        return [(d.id, d.title) for d in await self.db.documents.list(limit=50) if d.id not in done]

    async def _sameness(self) -> float | None:
        recent = await self.db.activities.recent(limit=8)
        texts = tuple(f"{a.kind}: {a.intent}" for a in recent if a.intent)
        if len(texts) < 3:
            return None
        if self._sameness_cache and self._sameness_cache[0] == texts:
            return self._sameness_cache[1]
        try:
            raw = np.asarray(await self.embedder.embed(list(texts)), dtype=np.float64)
        except ModelError:
            return None  # the embedding server is down: sameness is unknown, not an error
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        vectors = raw / np.where(norms == 0, 1.0, norms)
        sims = [
            float(vectors[i] @ vectors[j]) for i, j in itertools.combinations(range(len(texts)), 2)
        ]
        value = max(0.0, min(1.0, sum(sims) / len(sims)))
        self._sameness_cache = (texts, value)
        return value

    async def sense(self, people: dict[str, UserRecord] | None = None) -> BodySignals:
        now = self.clock.now()
        last = self.last_contact_at
        since = (now - last).total_seconds() if last else None
        lonely_for = since
        if lonely_for is None and self.first_boot_at is not None:
            lonely_for = (now - self.first_boot_at).total_seconds()
        isolation = 1 - math.exp(-(lonely_for or 0.0) / self.usual_gap_s)
        window = now - timedelta(hours=6)
        stimulation = await self.db.conversations.count_since(
            window, role="user"
        ) + await self.db.memories.count_since(window, kinds=[MemoryKind.DOCUMENT])
        unread = await self.unread_documents()
        pending = len(await self.db.self_beliefs.with_status(BeliefStatus.CONSIDERING))
        return BodySignals(
            seconds_since_contact=since,
            usual_gap_s=self.usual_gap_s,
            isolation=isolation,
            sameness=await self._sameness(),
            stimulation=stimulation,
            unread=[title for _, title in unread],
            energy=self.affect.energy,
            competence=self.affect.competence,
            reconsiderations=pending,
            absences=await self.relationships.absences(people or {}, now),
        )

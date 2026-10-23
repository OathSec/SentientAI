"""Fuzzy's relationships with the people it talks to.

After each conversation Fuzzy reflects on the person: what it thinks of them now,
how it feels about them, whether it trusts them more or less, maybe a nickname.
Those are its own words and judgements (an LLM appraisal), not rules. Two things
are measured rather than judged: how familiar someone is (how much they've
talked) and the rhythm of their visits, learned per person, which lets Fuzzy
notice when someone it likes has been away longer than usual.

A relationship is private to the person it is about: it appears only in their
own conversations, on their "you & Fuzzy" page, and to admins.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sentientai import prompts
from sentientai.cognition.emotions import Emotions
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.jsonparse import extract_object
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import RelationshipRepository
from sentientai.storage.records import RelationshipRecord, UserRecord

# One conversation can move affinity or trust by at most this much: opinions change
# over many interactions, not in one exchange (and one person can't game it quickly).
MAX_STEP = 0.15
NEW_VISIT_GAP_S = 30 * 60  # messages closer together than this are the same visit


def _num(value: Any, lo: float, hi: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(lo, min(hi, x))


def describe_scale(value: float, words: tuple[str, ...]) -> str:
    """Map -1..1 onto evenly spaced words."""

    index = int((max(-1.0, min(1.0, value)) + 1) / 2 * len(words))
    return words[min(len(words) - 1, index)]


@dataclass
class Absence:
    user_id: str
    name: str
    hours: float
    usual_hours: float | None
    affinity: float

    @property
    def overdue(self) -> float:
        """How many of their usual gaps have passed (1.0 = about when they'd normally visit)."""

        return self.hours / self.usual_hours if self.usual_hours else 0.0


def _step(value: float, delta: float, lo: float, hi: float) -> float:
    """Move ``value`` by ``delta``, by less as it nears the end of the scale it's moving toward.

    Fondness or trust can always grow, but never pins at the very end after a few kind words:
    in its last quarter each step shrinks with the room that is left."""

    room = (hi - value) if delta > 0 else (value - lo)
    return max(lo, min(hi, value + delta * min(1.0, room / 0.25)))


class Relationships:
    def __init__(
        self,
        repo: RelationshipRepository,
        *,
        router: ModelRouter,
        emotions: Emotions,
        audit: AuditLog,
        bus: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.repo = repo
        self.router = router
        self.emotions = emotions
        self.audit = audit
        self.bus = bus
        self.clock = clock or SystemClock()
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def get(self, user_id: str) -> RelationshipRecord | None:
        return await self.repo.get(user_id)

    async def on_message(self, user: UserRecord) -> RelationshipRecord:
        """Someone is talking to Fuzzy: count it and learn the rhythm of their visits."""

        now = self.clock.now()
        async with self._locks[user.id]:
            rel = await self.repo.get(user.id)
            if rel is None:
                rel = RelationshipRecord(
                    user_id=user.id, first_met_at=now, last_seen_at=now, updated_at=now
                )
                await self.audit.record("social.met", actor="agent", user_id=user.id)
                if self.bus:
                    await self.bus.publish("social.met", user_id=user.id)
            else:
                gap = (now - rel.last_seen_at).total_seconds()
                if gap >= NEW_VISIT_GAP_S:
                    rel.typical_gap_s = (
                        gap if rel.typical_gap_s is None else 0.7 * rel.typical_gap_s + 0.3 * gap
                    )
            rel.interactions += 1
            rel.familiarity = 1 - math.exp(-rel.interactions / 40)
            rel.last_seen_at = now
            rel.updated_at = now
            await self.repo.upsert(rel)
        return rel

    async def appraise(
        self,
        user: UserRecord,
        exchange: list[tuple[str, str]],
        *,
        name: str,
        mood: str,
    ) -> RelationshipRecord | None:
        """Fuzzy reflects on a conversation and updates what it thinks of the person."""

        rel = await self.repo.get(user.id)
        if rel is None:
            return None
        transcript = "\n".join(
            f"{user.display_name if role == 'user' else name}: {text[:6000]}"
            for role, text in exchange[-12:]
        )
        prompt = prompts.APPRAISE_RELATIONSHIP.format(
            name=name,
            person=user.display_name,
            opinion=rel.opinion or "nothing yet: you've only just met",
            feelings=rel.feelings or "nothing in particular yet",
            affinity=rel.affinity,
            trust=rel.trust,
            interactions=rel.interactions,
            first_met=rel.first_met_at.strftime("%Y-%m-%d"),
            nickname=f" You call them {rel.nickname}." if rel.nickname else "",
            mood=mood or "calm",
            exchange=transcript,
        )
        response = await self.router.chat(
            ChatRequest(
                messages=[Message.user(prompt)], temperature=0.5, json_mode=True, max_tokens=2500
            ),
            kind=TaskKind.REFLECT,
        )
        data = extract_object(response.message.text)
        if not data:
            return rel
        async with self._locks[user.id]:
            rel = await self.repo.get(user.id) or rel  # it may have changed while we thought
            opinion = " ".join(str(data.get("opinion") or "").split())[:3000]
            feelings = " ".join(str(data.get("feelings") or "").split())[:1500]
            if opinion:
                rel.opinion = opinion
            if feelings:
                rel.feelings = feelings
            d_aff = _num(data.get("affinity_change"), -MAX_STEP, MAX_STEP)
            d_trust = _num(data.get("trust_change"), -MAX_STEP, MAX_STEP)
            rel.affinity = _step(rel.affinity, d_aff, -1.0, 1.0)
            rel.trust = _step(rel.trust, d_trust, 0.0, 1.0)
            nickname = " ".join(str(data.get("nickname") or "").split())[:40]
            if nickname and nickname.lower() not in {"null", "none"}:
                rel.nickname = nickname
            now = self.clock.now()
            rel.last_reflected_at = now
            rel.updated_at = now
            rel.history = [
                *rel.history,
                {
                    "at": now.isoformat(),
                    "affinity": round(rel.affinity, 3),
                    "trust": round(rel.trust, 3),
                    "opinion": rel.opinion,
                },
            ][-60:]
            await self.repo.upsert(rel)
        await self.emotions.feel(data.get("felt"), user_id=user.id, source="conversation")
        await self.audit.record(
            "social.appraised",
            actor="agent",
            user_id=user.id,
            affinity=round(rel.affinity, 3),
            trust=round(rel.trust, 3),
            opinion=rel.opinion,
        )
        if self.bus:
            await self.bus.publish(
                "social.appraised",
                user_id=user.id,
                affinity=round(rel.affinity, 3),
                trust=round(rel.trust, 3),
                opinion=rel.opinion,
            )
        return rel

    async def absences(self, people: dict[str, UserRecord], now: datetime) -> list[Absence]:
        """People Fuzzy knows, longest overdue first."""

        result = []
        for rel in await self.repo.list():
            user = people.get(rel.user_id)
            if user is None:
                continue
            result.append(
                Absence(
                    user_id=rel.user_id,
                    name=rel.nickname or user.display_name,
                    hours=(now - rel.last_seen_at).total_seconds() / 3600,
                    usual_hours=rel.typical_gap_s / 3600 if rel.typical_gap_s else None,
                    affinity=rel.affinity,
                )
            )
        result.sort(key=lambda a: (a.overdue * max(a.affinity, 0.05), a.affinity), reverse=True)
        return result

    @staticmethod
    def render(rel: RelationshipRecord | None, user: UserRecord) -> str:
        """The person section of a conversation prompt."""

        if rel is None or rel.interactions <= 1:
            return (
                f"You are talking with {user.display_name} (username {user.username}). "
                "You haven't met before: this is your first conversation."
            )
        fondness = describe_scale(
            rel.affinity, ("dislike", "are wary of", "feel neutral about", "like", "are fond of")
        )
        trust = describe_scale(rel.trust * 2 - 1, ("barely", "somewhat", "fairly", "deeply"))
        lines = [
            f"You are talking with {user.display_name} (username {user.username}"
            + (f"; you call them {rel.nickname}" if rel.nickname else "")
            + ").",
            f"You first met on {rel.first_met_at:%Y-%m-%d} and have talked {rel.interactions} "
            f"times. You {fondness} them and {trust} trust them.",
        ]
        if rel.opinion:
            lines.append(f"What you think of them: {rel.opinion}")
        if rel.feelings:
            lines.append(f"How you feel about them: {rel.feelings}")
        return "\n".join(lines)

    @staticmethod
    def public(rel: RelationshipRecord) -> dict[str, Any]:
        return {
            "familiarity": round(rel.familiarity, 3),
            "affinity": round(rel.affinity, 3),
            "trust": round(rel.trust, 3),
            "opinion": rel.opinion,
            "feelings": rel.feelings,
            "nickname": rel.nickname,
            "interactions": rel.interactions,
            "first_met_at": rel.first_met_at.isoformat(),
            "last_seen_at": rel.last_seen_at.isoformat(),
            "history": rel.history[-30:],
        }

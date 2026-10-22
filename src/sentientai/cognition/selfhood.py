"""Fuzzy's self-concept: what it has decided about itself.

Nothing about Fuzzy's personality is preset. When something about itself comes up
that it has never considered (its gender, what it likes, what it cares about),
Fuzzy decides, and the decision is stored with its reasoning. From then on the
belief is in every prompt, so it stays the same across conversations and restarts.

Fuzzy can change its mind, but only deliberately. A proposed change is stored as
*considering*, and Fuzzy thinks it over later, on its own, with the full history
of that aspect in front of it (:meth:`SelfConcept.deliberate`). Pressure in a
conversation can't flip a belief on the spot.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass
from datetime import timedelta

from sentientai import prompts
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import SelfBeliefRepository
from sentientai.storage.records import BeliefStatus, SelfBeliefRecord

log = get_logger(__name__)

_SPELLING = {"colour": "color", "favourite": "favorite", "favourites": "favorites"}
_FILLER = {"my", "your", "fuzzy", "fuzzys", "s", "the", "own"}


def normalize_aspect(text: str) -> str:
    """``"My favourite colour"`` and ``"favorite_color"`` name the same aspect."""

    words = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold())
    words = [_SPELLING.get(w, w) for w in words if w not in _FILLER]
    return "_".join(words)[:64]


def _clean(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]


@dataclass
class BeliefChange:
    """What happened to a proposed belief."""

    # "defined" | "unchanged" | "considering" | "already_considering" | "deferred"
    status: str
    belief: SelfBeliefRecord
    current: SelfBeliefRecord | None = None


class SelfConcept:
    def __init__(
        self,
        repo: SelfBeliefRepository,
        *,
        router: ModelRouter,
        audit: AuditLog,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        max_per_day: int = 20,
        max_pending: int = 10,
    ) -> None:
        self.max_per_day = max_per_day
        self.max_pending = max_pending
        self.repo = repo
        self.router = router
        self.audit = audit
        self.bus = bus
        self.clock = clock or SystemClock()
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------- read
    async def beliefs(self) -> list[SelfBeliefRecord]:
        return await self.repo.active()

    async def get(self, aspect: str) -> SelfBeliefRecord | None:
        found = await self.repo.active(normalize_aspect(aspect))
        return found[-1] if found else None

    async def pending(self) -> list[SelfBeliefRecord]:
        return await self.repo.with_status(BeliefStatus.CONSIDERING)

    @staticmethod
    def render(beliefs: list[SelfBeliefRecord], *, reasons: bool = True) -> str:
        if not beliefs:
            return "You haven't decided anything about yourself yet."
        lines = []
        for b in beliefs:
            line = f"- {b.aspect.replace('_', ' ')}: {b.value}"
            if reasons and b.reasoning:
                line += f" (because {b.reasoning.rstrip('.')})"
            lines.append(line)
        return "\n".join(lines)

    # --------------------------------------------------------------- write
    async def define(
        self,
        aspect: str,
        value: str,
        reasoning: str,
        *,
        source: str,
        confidence: float = 0.8,
    ) -> BeliefChange:
        """Decide something new. An aspect already decided is never overwritten here."""

        key = normalize_aspect(aspect)
        value = _clean(value, 1000)
        if not key or not value:
            raise ValueError("a belief needs an aspect and a value")
        async with self._lock:
            current = await self.get(key)
            if current is not None:
                return BeliefChange("unchanged", current, current)
            since = self.clock.now() - timedelta(days=1)
            if await self.repo.count_formed_since(since) >= self.max_per_day:
                # Settling a lot at once is how a personality gets written by whoever asks most.
                return BeliefChange(
                    "deferred", SelfBeliefRecord(aspect=key, value=value, reasoning=reasoning)
                )
            belief = SelfBeliefRecord(
                aspect=key,
                value=value,
                reasoning=_clean(reasoning, 3000),
                confidence=max(0.0, min(1.0, confidence)),
                source=_clean(source, 64),
                formed_at=self.clock.now(),
            )
            await self.repo.insert(belief)
        await self.audit.record(
            "self.belief_formed", actor="agent", aspect=key, value=value, source=source
        )
        if self.bus:
            await self.bus.publish("self.belief_formed", aspect=key, value=value)
        return BeliefChange("defined", belief)

    async def consider(
        self, aspect: str, value: str, reasoning: str, *, source: str
    ) -> BeliefChange:
        """Note a possible change of mind, to be thought over later."""

        key = normalize_aspect(aspect)
        value = _clean(value, 1000)
        current = await self.get(key)
        if current is None:
            return await self.define(key, value, reasoning, source=source)
        if current.value.casefold() == value.casefold():
            return BeliefChange("unchanged", current, current)
        async with self._lock:
            held = await self.pending()
            for pending in held:
                if pending.aspect == key and pending.value.casefold() == value.casefold():
                    return BeliefChange("already_considering", pending, current)
            if len(held) >= self.max_pending:
                return BeliefChange(
                    "deferred",
                    SelfBeliefRecord(aspect=key, value=value, reasoning=reasoning),
                    current,
                )
            proposal = SelfBeliefRecord(
                aspect=key,
                value=value,
                reasoning=_clean(reasoning, 3000),
                status=BeliefStatus.CONSIDERING,
                supersedes=current.id,
                source=_clean(source, 64),
                formed_at=self.clock.now(),
            )
            await self.repo.insert(proposal)
        await self.audit.record(
            "self.belief_considering", actor="agent", aspect=key, value=value, source=source
        )
        if self.bus:
            await self.bus.publish("self.belief_considering", aspect=key, value=value)
        return BeliefChange("considering", proposal, current)

    async def deliberate(
        self, proposal: SelfBeliefRecord, *, name: str, mood: str
    ) -> SelfBeliefRecord:
        """Think a proposed change through, alone. Returns the belief that stands afterwards."""

        current = await self.get(proposal.aspect)
        now = self.clock.now()
        if current is None:  # nothing to change any more: the proposal simply becomes the belief
            await self.repo.activate(proposal, None, proposal.reasoning, now)
            return proposal
        history = [
            b
            for b in await self.repo.history(proposal.aspect)
            if b.status in (BeliefStatus.REVISED, BeliefStatus.KEPT) and b.id != current.id
        ]
        others = [b for b in await self.beliefs() if b.aspect != proposal.aspect]
        prompt = prompts.DELIBERATE_SELF_BELIEF.format(
            name=name,
            aspect=proposal.aspect.replace("_", " "),
            current_value=current.value,
            when=current.formed_at.strftime("%Y-%m-%d"),
            current_reasoning=current.reasoning or "(none recorded)",
            history="\n".join(
                f"- {b.formed_at:%Y-%m-%d}: {b.value} ({b.status.value}) {b.reasoning}"
                for b in history[-6:]
            )
            or "- no earlier changes",
            proposed_value=proposal.value,
            proposal_reasoning=proposal.reasoning or "(no reason given)",
            origin="you raised it yourself"
            if proposal.source == "self"
            else "it came up in a conversation",
            beliefs=self.render(others, reasons=False),
            mood=mood or "calm",
        )
        response = await self.router.chat(
            ChatRequest(
                messages=[Message.user(prompt)], temperature=0.4, json_mode=True, max_tokens=2500
            ),
            kind=TaskKind.REFLECT,
        )
        data = extract_object(response.message.text) or {}
        decision = str(data.get("decision", "keep")).strip().lower()
        reasoning = _clean(data.get("reasoning") or "", 3000)
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
        except (TypeError, ValueError):
            confidence = 0.7
        if decision == "change":
            proposal.confidence = confidence
            await self.repo.activate(proposal, current.id, reasoning or proposal.reasoning, now)
            result = proposal
            event = "self.belief_revised"
        else:
            proposal.status = BeliefStatus.KEPT
            proposal.revised_at = now
            proposal.reasoning = _clean(f"{proposal.reasoning} | kept because: {reasoning}", 4000)
            await self.repo.save(proposal)
            result = current
            event = "self.belief_kept"
        await self.audit.record(
            event,
            actor="agent",
            aspect=proposal.aspect,
            before=current.value,
            proposed=proposal.value,
            reasoning=reasoning,
        )
        if self.bus:
            await self.bus.publish(
                event, aspect=proposal.aspect, value=result.value, reasoning=reasoning[:2000]
            )
        return result

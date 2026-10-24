"""Thought patches: pieces of background thinking that lent computers hand to Fuzzy.

While Fuzzy is busy or resting, a connected computer can think quietly for it: given a few of
its memories, it produces some short thoughts, connections and questions. Those are *patches*,
not decisions. They wait here; the next time Fuzzy decides what to do with its time, its own
(bigger) model sees them and takes them up or ignores them. Nothing a helper produces is ever
shown to a person or acted on directly, so a smaller model on someone's PC can add ideas but
can't make Fuzzy's replies or choices worse.

Patches never involve people (they are built only from shared memories, with names removed) and
are kept briefly: once a beat has seen them they are retired.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from sentientai.core.clock import Clock, SystemClock, new_id
from sentientai.storage.database import StateRepository

STATE_KEY = "mind.patches"
KEEP_HOURS = 24


class Patch(BaseModel):
    id: str = Field(default_factory=lambda: new_id("pat"))
    text: str
    at: datetime


class PatchBook:
    def __init__(
        self, state: StateRepository, *, max_held: int = 12, clock: Clock | None = None
    ) -> None:
        self.state = state
        self.max_held = max_held
        self.clock = clock or SystemClock()
        self._lock = asyncio.Lock()

    async def _held(self) -> list[Patch]:
        raw = await self.state.get(STATE_KEY)
        cutoff = self.clock.now() - timedelta(hours=KEEP_HOURS)
        held = []
        for item in raw if isinstance(raw, list) else []:
            try:
                patch = Patch.model_validate(item)
            except ValueError:
                continue
            if patch.at >= cutoff:
                held.append(patch)
        return held

    async def active(self) -> list[Patch]:
        return await self._held()

    async def add(self, texts: list[str]) -> list[Patch]:
        now = self.clock.now()
        made = [Patch(text=" ".join(t.split())[:400], at=now) for t in texts if t.strip()]
        async with self._lock:
            held = await self._held()
            seen = {p.text.casefold() for p in held}
            fresh = [p for p in made if p.text.casefold() not in seen]
            kept = [*fresh, *held][: self.max_held]
            await self.state.set(STATE_KEY, [p.model_dump(mode="json") for p in kept])
        return fresh

    async def retire(self, ids: list[str]) -> None:
        """Once Fuzzy has seen them they've done their job."""

        async with self._lock:
            kept = [p for p in await self._held() if p.id not in set(ids)]
            await self.state.set(STATE_KEY, [p.model_dump(mode="json") for p in kept])

    @staticmethod
    def render(patches: list[Patch]) -> str:
        return "\n".join(f"- {p.text}" for p in patches) if patches else "- none"

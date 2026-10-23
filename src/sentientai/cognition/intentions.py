"""Things Fuzzy said it would do later.

People ask Fuzzy to go and do things ("go explore the internet", "build yourself a house",
"read what we taught you"), and it says it will. A conversation can't do those things (its own
world and its free time are its own), so before this the promise simply evaporated. Now it can
note the intention (``plan_later``), and its time to itself shows what it meant to do, when and
for whom. Whether and when it follows through is still its own choice: nothing here starts work.

Intentions fade after ``KEEP_DAYS`` and at most ``MAX_HELD`` are kept, newest first.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from sentientai.cognition.self_model import humanize_duration
from sentientai.core.clock import Clock, SystemClock, new_id
from sentientai.storage.database import StateRepository

STATE_KEY = "mind.intentions"
KEEP_DAYS = 3
MAX_HELD = 8


class Intention(BaseModel):
    id: str = Field(default_factory=lambda: new_id("int"))
    what: str
    why: str = ""
    for_whom: str | None = None  # display name of the person it was said to, if any
    user_id: str | None = None
    at: datetime


class Intentions:
    def __init__(self, state: StateRepository, *, clock: Clock | None = None) -> None:
        self.state = state
        self.clock = clock or SystemClock()
        self._lock = asyncio.Lock()

    async def active(self) -> list[Intention]:
        raw = await self.state.get(STATE_KEY)
        cutoff = self.clock.now() - timedelta(days=KEEP_DAYS)
        held = []
        for item in raw if isinstance(raw, list) else []:
            try:
                intention = Intention.model_validate(item)
            except ValueError:
                continue
            if intention.at >= cutoff:
                held.append(intention)
        return held

    async def add(
        self, what: str, why: str = "", *, for_whom: str | None = None, user_id: str | None = None
    ) -> Intention:
        intention = Intention(
            what=" ".join(what.split())[:500],
            why=" ".join(why.split())[:500],
            for_whom=for_whom,
            user_id=user_id,
            at=self.clock.now(),
        )
        async with self._lock:
            held = [
                i for i in await self.active() if i.what.casefold() != intention.what.casefold()
            ]
            held = [intention, *held][:MAX_HELD]
            await self.state.set(STATE_KEY, [i.model_dump(mode="json") for i in held])
        return intention

    def render(self, held: list[Intention]) -> str:
        if not held:
            return "- nothing"
        now = self.clock.now()
        return "\n".join(
            f"- {i.what}"
            + (f" (you told {i.for_whom}" if i.for_whom else " (")
            + f"{', ' if i.for_whom else ''}{humanize_duration((now - i.at).total_seconds())} ago)"
            + (f": {i.why}" if i.why else "")
            for i in held
        )

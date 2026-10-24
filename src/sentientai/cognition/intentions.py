"""Things Fuzzy said it would do later, and whether it did.

People ask Fuzzy to go and do things ("go explore the internet", "build yourself a house",
"read what we taught you"), and it says it will. A conversation can't do those things (its own
world and its free time are its own), so before this the promise simply evaporated. Now it can
note the intention (``plan_later``); its time to itself shows what it meant to do, when and for
whom; the next conversation with that person shows it too; and when it acts on one, the
intention is marked kept and the person finds a note with what came of it.

Whether and when it follows through is still its own choice: nothing here starts work.
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
MAX_HELD = 12


class Intention(BaseModel):
    id: str = Field(default_factory=lambda: new_id("int"))
    what: str
    why: str = ""
    for_whom: str | None = None  # display name of the person it was said to, if any
    user_id: str | None = None
    at: datetime
    done_at: datetime | None = None
    outcome: str = ""


class Intentions:
    def __init__(self, state: StateRepository, *, clock: Clock | None = None) -> None:
        self.state = state
        self.clock = clock or SystemClock()
        self._lock = asyncio.Lock()

    async def _held(self) -> list[Intention]:
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

    async def _save(self, held: list[Intention]) -> None:
        await self.state.set(STATE_KEY, [i.model_dump(mode="json") for i in held[:MAX_HELD]])

    async def active(self) -> list[Intention]:
        """Said and not yet done."""

        return [i for i in await self._held() if i.done_at is None]

    async def for_person(self, user_id: str) -> list[Intention]:
        """What it told this person it would do, done or not."""

        return [i for i in await self._held() if i.user_id == user_id]

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
                i
                for i in await self._held()
                if i.done_at is not None or i.what.casefold() != intention.what.casefold()
            ]
            await self._save([intention, *held])
        return intention

    async def resolve(self, intention_id: str, outcome: str) -> Intention | None:
        """It did it: mark the intention kept, with what came of it."""

        async with self._lock:
            held = await self._held()
            found = next((i for i in held if i.id == intention_id and i.done_at is None), None)
            if found is None:
                return None
            found.done_at = self.clock.now()
            found.outcome = " ".join(outcome.split())[:1500]
            await self._save(held)
        return found

    def _ago(self, when: datetime) -> str:
        return humanize_duration((self.clock.now() - when).total_seconds())

    def render(self, held: list[Intention]) -> str:
        """For its own time: what it still means to do, each with an id it can name."""

        if not held:
            return "- nothing"
        return "\n".join(
            f"- [{i.id}] {i.what} ("
            + (f"you told {i.for_whom}, " if i.for_whom else "")
            + f"{self._ago(i.at)} ago)"
            + (f": {i.why}" if i.why else "")
            for i in held
        )

    def render_for_person(self, held: list[Intention]) -> str:
        """For a conversation with them: what it said it would do, and whether it has."""

        lines = []
        for i in held:
            if i.done_at is None:
                lines.append(f"- You said you'd {i.what} ({self._ago(i.at)} ago); not done yet.")
            else:
                lines.append(
                    f"- You said you'd {i.what}, and did ({self._ago(i.done_at)} ago): "
                    f"{i.outcome[:400]}"
                )
        return "\n".join(lines)

"""Global workspace: competition for attention and broadcast of conscious contents.

Inspired by Global Workspace Theory. Specialist processes (operator input,
scheduler, file watchers, drives, goals, approvals, memory maintenance) submit
*percepts* with a salience. Each cycle the percepts compete; the winners form
the current contents of awareness, are broadcast to every subscriber, and are
rendered into the agent's context. The top actionable percept decides what
the runtime does next.

Two mechanisms keep attention healthy:

* habituation - repeated percepts from the same source lose salience
* starvation protection - waiting percepts slowly gain salience
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from sentientai.core.clock import Clock, SystemClock, new_id
from sentientai.core.events import EventBus

PerceptKind = Literal["task", "signal", "drive", "goal", "maintenance", "thought"]


class Percept(BaseModel):
    id: str = Field(default_factory=lambda: new_id("pct"))
    source: str
    kind: PerceptKind
    content: str
    salience: float = Field(ge=0.0, le=1.0)
    key: str | None = Field(
        default=None, description="Dedupe key; resubmitting replaces the percept."
    )
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    expires_at: datetime | None = None

    @property
    def actionable(self) -> bool:
        return self.kind in {"task", "goal", "drive", "maintenance"}


class GlobalWorkspace:
    def __init__(
        self,
        *,
        capacity: int = 5,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        habituation_window_s: float = 600.0,
    ) -> None:
        self.capacity = capacity
        self.bus = bus
        self.clock = clock or SystemClock()
        self.habituation_window_s = habituation_window_s
        self._percepts: dict[str, Percept] = {}
        self._recent_winners: deque[tuple[datetime, str]] = deque(maxlen=50)
        self.conscious: list[Percept] = []

    def submit(
        self,
        *,
        source: str,
        kind: PerceptKind,
        content: str,
        salience: float,
        key: str | None = None,
        ttl_s: float | None = None,
        **payload: Any,
    ) -> Percept:
        now = self.clock.now()
        if key is not None:
            existing = next((p for p in self._percepts.values() if p.key == key), None)
            if existing is not None:
                existing.content = content
                existing.salience = max(0.0, min(1.0, salience))
                existing.payload = payload
                existing.expires_at = now + timedelta(seconds=ttl_s) if ttl_s else None
                return existing
        percept = Percept(
            source=source,
            kind=kind,
            content=content,
            salience=max(0.0, min(1.0, salience)),
            key=key,
            payload=payload,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_s) if ttl_s else None,
        )
        self._percepts[percept.id] = percept
        return percept

    def retract(self, *, percept_id: str | None = None, key: str | None = None) -> None:
        if percept_id is not None:
            self._percepts.pop(percept_id, None)
        if key is not None:
            for pid in [p.id for p in self._percepts.values() if p.key == key]:
                self._percepts.pop(pid, None)

    def effective_salience(self, percept: Percept, now: datetime) -> float:
        waited_min = (now - percept.created_at).total_seconds() / 60
        starvation_boost = min(0.2, 0.02 * waited_min)
        cutoff = now - timedelta(seconds=self.habituation_window_s)
        repeats = sum(
            1 for ts, src in self._recent_winners if src == percept.source and ts >= cutoff
        )
        habituation = (
            0.85**repeats if percept.kind != "task" else 1.0
        )  # never habituate to real work
        return min(1.0, percept.salience * habituation + starvation_boost)

    async def cycle(self) -> list[Percept]:
        """Run one competition and broadcast the winners."""

        now = self.clock.now()
        for pid in [p.id for p in self._percepts.values() if p.expires_at and p.expires_at <= now]:
            del self._percepts[pid]
        ranked = sorted(
            self._percepts.values(), key=lambda p: self.effective_salience(p, now), reverse=True
        )
        winners = ranked[: self.capacity]
        changed = [p.id for p in winners] != [p.id for p in self.conscious]
        self.conscious = winners
        if changed and self.bus is not None and winners:
            await self.bus.publish(
                "workspace.broadcast",
                contents=[
                    {
                        "source": p.source,
                        "kind": p.kind,
                        "content": p.content[:200],
                        "salience": round(self.effective_salience(p, now), 3),
                    }
                    for p in winners
                ],
            )
        return winners

    def focus(self) -> Percept | None:
        return next((p for p in self.conscious if p.actionable), None)

    def attend(self, percept: Percept) -> None:
        """Mark a percept as acted upon: it leaves the workspace and feeds habituation."""

        self._recent_winners.append((self.clock.now(), percept.source))
        self._percepts.pop(percept.id, None)
        self.conscious = [p for p in self.conscious if p.id != percept.id]

    def render(self) -> str:
        if not self.conscious:
            return "Nothing else is competing for your attention right now."
        return "\n".join(f"- ({p.source}) {p.content}" for p in self.conscious)

    def percepts(self) -> list[Percept]:
        return list(self._percepts.values())

    def __len__(self) -> int:
        return len(self._percepts)

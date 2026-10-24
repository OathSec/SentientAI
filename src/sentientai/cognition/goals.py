"""Goal management: operator goals and bounded, self-generated (intrinsic) goals."""

from __future__ import annotations

from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.storage.database import GoalRepository
from sentientai.storage.records import GoalOrigin, GoalRecord, GoalStatus, Scope

ACTIVE = (GoalStatus.ACTIVE,)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class GoalManager:
    def __init__(
        self,
        repo: GoalRepository,
        *,
        bus: EventBus | None = None,
        max_intrinsic: int = 3,
        clock: Clock | None = None,
    ) -> None:
        self.repo = repo
        self.bus = bus
        self.max_intrinsic = max_intrinsic
        self.clock = clock or SystemClock()

    async def active(self, scope: Scope | None = None) -> list[GoalRecord]:
        """Active goals; ``scope`` limits them to shared goals plus one person's."""

        return await self.repo.list(ACTIVE, scope=scope)

    async def intrinsic_slots(self) -> int:
        active = [g for g in await self.active() if g.origin is GoalOrigin.INTRINSIC]
        return max(0, self.max_intrinsic - len(active))

    async def add(
        self,
        description: str,
        *,
        origin: GoalOrigin = GoalOrigin.OPERATOR,
        priority: float = 0.5,
        parent_id: str | None = None,
        user_id: str | None = None,
    ) -> GoalRecord | None:
        """Add a goal. Returns an existing duplicate, or None if intrinsic slots are full."""

        description = description.strip()
        if not description:
            raise ValueError("goal description is empty")
        for goal in await self.active():
            if _norm(goal.description) == _norm(description):
                return goal
        if origin is GoalOrigin.INTRINSIC and await self.intrinsic_slots() == 0:
            return None
        now = self.clock.now()
        goal = GoalRecord(
            description=description,
            origin=origin,
            priority=max(0.0, min(1.0, priority)),
            parent_id=parent_id,
            created_at=now,
            updated_at=now,
            user_id=user_id,
        )
        await self.repo.insert(goal)
        if self.bus:
            await self.bus.publish(
                "goal.added",
                goal_id=goal.id,
                origin=origin.value,
                description=description,
                user_id=user_id,
            )
        return goal

    async def update(
        self,
        goal_id: str,
        *,
        status: GoalStatus | None = None,
        progress: float | None = None,
        note: str | None = None,
        attempted: bool = False,
    ) -> GoalRecord:
        goal = await self.repo.get(goal_id)
        if goal is None:
            raise KeyError(goal_id)
        if status is not None:
            goal.status = status
        if progress is not None:
            goal.progress = max(0.0, min(1.0, progress))
            if goal.progress >= 1.0 and status is None:
                goal.status = GoalStatus.COMPLETED
        if note:
            goal.notes = [*goal.notes, note.strip()][-20:]
        if attempted:
            goal.attempts += 1
        goal.updated_at = self.clock.now()
        await self.repo.save(goal)
        if self.bus:
            await self.bus.publish(
                "goal.updated", goal_id=goal.id, status=goal.status.value, progress=goal.progress
            )
        return goal

    @staticmethod
    def render(goals: list[GoalRecord]) -> str:
        if not goals:
            return "No active goals."
        return "\n".join(
            f"- [{g.id}] ({g.origin.value}, priority {g.priority:.1f}, {g.progress:.0%} done) "
            f"{g.description}"
            for g in goals
        )

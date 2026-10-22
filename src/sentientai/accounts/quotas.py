"""Daily per-person limits, so one visitor can't monopolise a home GPU server."""

from __future__ import annotations

from sentientai.config import AccountsConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.storage.database import UsageRepository
from sentientai.storage.records import UsageRecord


class QuotaExceeded(Exception):
    pass


_LIMITS = {
    "messages": ("daily_messages", "You've reached today's message limit. Come back tomorrow."),
    "images": ("daily_images", "You've reached today's image limit."),
    "teach_bytes": ("daily_teach_bytes", "You've taught Fuzzy as much as you can for today."),
}


class Quotas:
    def __init__(
        self, usage: UsageRepository, config: AccountsConfig, *, clock: Clock | None = None
    ) -> None:
        self.repo = usage
        self.config = config
        self.clock = clock or SystemClock()

    def today(self) -> str:
        return self.clock.now().strftime("%Y-%m-%d")

    async def usage(self, user_id: str) -> UsageRecord:
        return await self.repo.get(user_id, self.today())

    async def take(
        self,
        user_id: str | None,
        *,
        privileged: bool = False,
        messages: int = 0,
        images: int = 0,
        teach_bytes: int = 0,
    ) -> None:
        """Count usage; raise :class:`QuotaExceeded` (and undo) if it goes over a limit."""

        if user_id is None:
            return
        day = self.today()
        amounts = {"messages": messages, "images": images, "teach_bytes": teach_bytes}
        record = await self.repo.add(user_id, day, **amounts)
        if privileged:
            return
        for field, amount in amounts.items():
            setting, message = _LIMITS[field]
            limit = int(getattr(self.config, setting))
            if amount and limit and getattr(record, field) > limit:
                await self.repo.add(user_id, day, **{k: -v for k, v in amounts.items()})
                raise QuotaExceeded(message)

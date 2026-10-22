"""Scheduled background tasks: 5-field cron expressions or fixed intervals.

Last-run times are persisted, so a reboot neither loses nor duplicates runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from sentientai.config import ScheduleConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.logging import get_logger
from sentientai.storage.database import StateRepository

log = get_logger(__name__)

_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
_ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
}

SubmitFn = Callable[[ScheduleConfig], Awaitable[object]]


class CronExpression:
    """Standard cron: minute hour day-of-month month day-of-week (0=Sunday).

    Supports ``*``, ``a-b``, ``a,b,c``, ``*/n``, ``a-b/n`` and the common @aliases.
    Like Vixie cron, when both day fields are restricted a day matches either.
    """

    def __init__(self, expression: str) -> None:
        self.expression = expression
        fields = _ALIASES.get(expression.strip(), expression).split()
        if len(fields) != 5:
            raise ValueError(f"cron expression needs 5 fields: {expression!r}")
        self.fields = [self._parse(f, lo, hi) for f, (lo, hi) in zip(fields, _RANGES, strict=True)]
        self.dom_restricted = fields[2] != "*"
        self.dow_restricted = fields[4] != "*"

    @staticmethod
    def _parse(field: str, lo: int, hi: int) -> set[int]:
        values: set[int] = set()
        for part in field.split(","):
            step = 1
            if "/" in part:
                part, step_s = part.split("/", 1)
                step = int(step_s)
                if step <= 0:
                    raise ValueError("cron step must be positive")
            if part == "*":
                start, end = lo, hi
            elif "-" in part:
                a, b = part.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(part)
                if step != 1:
                    end = hi
            if hi == 6:  # allow 7 as Sunday
                start, end = (0 if start == 7 else start), (6 if end == 7 else end)
            if not (lo <= start <= hi and lo <= end <= hi and start <= end):
                raise ValueError(f"cron value out of range in {field!r}")
            values.update(range(start, end + 1, step))
        return values

    def matches(self, when: datetime) -> bool:
        minute, hour, dom, month, dow = self.fields
        if when.minute not in minute or when.hour not in hour or when.month not in month:
            return False
        dom_ok = when.day in dom
        dow_ok = (when.isoweekday() % 7) in dow
        if self.dom_restricted and self.dow_restricted:
            return dom_ok or dow_ok
        return dom_ok and dow_ok

    def next_after(self, when: datetime, *, max_days: int = 366) -> datetime:
        candidate = when.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = candidate + timedelta(days=max_days)
        while candidate <= limit:
            if candidate.month not in self.fields[3]:
                candidate = (
                    candidate.replace(day=1, hour=0, minute=0) + timedelta(days=32)
                ).replace(day=1)
                continue
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError(f"no run within {max_days} days for {self.expression!r}")


class Scheduler:
    def __init__(
        self,
        schedules: list[ScheduleConfig],
        submit: SubmitFn,
        state: StateRepository,
        *,
        clock: Clock | None = None,
        poll_s: float = 15.0,
    ) -> None:
        self.schedules = [s for s in schedules if s.enabled]
        self.submit = submit
        self.state = state
        self.clock = clock or SystemClock()
        self.poll_s = poll_s
        self._cron = {s.name: CronExpression(s.cron) for s in self.schedules if s.cron}

    async def _last_run(self, name: str) -> datetime | None:
        raw = await self.state.get(f"schedule:{name}")
        return datetime.fromisoformat(raw) if raw else None

    def _due(self, schedule: ScheduleConfig, last: datetime | None, now: datetime) -> bool:
        if schedule.interval_s is not None:
            return last is None or (now - last).total_seconds() >= schedule.interval_s
        cron = self._cron[schedule.name]
        if last is None:
            return cron.matches(now)
        return cron.next_after(last) <= now

    async def tick(self) -> list[str]:
        fired = []
        now = self.clock.now()
        for schedule in self.schedules:
            last = await self._last_run(schedule.name)
            if not self._due(schedule, last, now):
                continue
            await self.state.set(
                f"schedule:{schedule.name}", now.replace(second=0, microsecond=0).isoformat()
            )
            try:
                await self.submit(schedule)
                fired.append(schedule.name)
            except Exception:
                log.exception("scheduled submission failed", extra={"schedule": schedule.name})
        return fired

    async def run(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(self.poll_s)

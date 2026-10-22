"""Human-in-the-loop approval gate.

Requests are persisted, so an operator can decide from the CLI (another
process), the HTTP API or programmatically. The waiting side is woken
immediately for in-process decisions and otherwise polls the database.
Undecided requests expire to DENIED; the kill switch aborts waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from typing import Any

from sentientai.config import RiskLevel
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.errors import StorageError
from sentientai.safety.audit import AuditLog
from sentientai.safety.killswitch import KillSwitch
from sentientai.storage.database import ApprovalRepository
from sentientai.storage.records import ApprovalRecord, ApprovalStatus

log = get_logger(__name__)


class ApprovalGate:
    def __init__(
        self,
        repo: ApprovalRepository,
        *,
        bus: EventBus,
        audit: AuditLog,
        kill_switch: KillSwitch,
        timeout_s: float,
        poll_interval_s: float = 0.5,
        clock: Clock | None = None,
    ) -> None:
        self.repo = repo
        self.bus = bus
        self.audit = audit
        self.kill_switch = kill_switch
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.clock = clock or SystemClock()
        self._waiters: dict[str, asyncio.Event] = {}

    async def request(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        risk: RiskLevel,
        reasons: list[str],
        task_id: str | None,
        autonomous: bool,
        timeout_s: float | None = None,
    ) -> ApprovalRecord:
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        record = ApprovalRecord(
            task_id=task_id,
            tool=tool,
            arguments=arguments,
            risk=risk,
            reasons=reasons,
            autonomous=autonomous,
            requested_at=self.clock.now(),
            expires_at=self.clock.now() + timedelta(seconds=timeout),
        )
        await self.repo.insert(record)
        waiter = self._waiters.setdefault(record.id, asyncio.Event())
        await self.audit.record(
            "approval.requested",
            task_id=task_id,
            approval_id=record.id,
            tool=tool,
            arguments=arguments,
            risk=risk.name,
            reasons=reasons,
            autonomous=autonomous,
        )
        await self.bus.publish(
            "approval.requested",
            approval_id=record.id,
            tool=tool,
            risk=risk.name,
            reasons=reasons,
            task_id=task_id,
        )
        log.warning(
            "approval required", extra={"approval_id": record.id, "tool": tool, "risk": risk.name}
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while True:
                self.kill_switch.check()
                current = await self.repo.get(record.id)
                if current is None:
                    raise StorageError(f"approval {record.id} vanished")
                if current.status is not ApprovalStatus.PENDING:
                    return current
                remaining = deadline - loop.time()
                if remaining <= 0:
                    await self._resolve(
                        record.id, ApprovalStatus.EXPIRED, "system:timeout", "approval timed out"
                    )
                    expired = await self.repo.get(record.id)
                    assert expired is not None
                    return expired
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        waiter.wait(), timeout=min(self.poll_interval_s, remaining)
                    )
        finally:
            self._waiters.pop(record.id, None)

    async def decide(
        self, approval_id: str, *, approved: bool, by: str, note: str | None = None
    ) -> ApprovalRecord:
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        if not await self._resolve(approval_id, status, by, note):
            existing = await self.repo.get(approval_id)
            if existing is None:
                raise KeyError(approval_id)
            return existing
        record = await self.repo.get(approval_id)
        assert record is not None
        return record

    async def _resolve(
        self, approval_id: str, status: ApprovalStatus, by: str, note: str | None
    ) -> bool:
        changed = await self.repo.decide(approval_id, status, by, note)
        if changed:
            await self.audit.record(
                "approval.decided",
                actor=by,
                approval_id=approval_id,
                status=status.value,
                note=note,
            )
            await self.bus.publish(
                "approval.decided", approval_id=approval_id, status=status.value, by=by
            )
            waiter = self._waiters.get(approval_id)
            if waiter:
                waiter.set()
        return changed

    async def pending(self) -> list[ApprovalRecord]:
        return await self.repo.list(ApprovalStatus.PENDING)

    async def expire_orphans(self) -> int:
        """At boot, requests from a previous process can no longer be acted on."""

        count = 0
        for record in await self.pending():
            if record.id not in self._waiters and await self._resolve(
                record.id, ApprovalStatus.EXPIRED, "system:restart", "requesting process restarted"
            ):
                count += 1
        return count

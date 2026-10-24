"""Emergency stop.

The switch is engaged by any of:

* the API / CLI (``sentientai kill``),
* creating the sentinel file (``touch data/KILL``) - this is the hook for a
  hardware button, a watchdog script or a remote management agent,
* SIGUSR1 to the daemon process.

While engaged, every guarded step raises :class:`KillSwitchEngagedError`,
in-flight tasks are cancelled and sandbox containers are killed via the
registered callbacks. Releasing requires an explicit operator action.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from sentientai.core.clock import utcnow
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.errors import KillSwitchEngagedError
from sentientai.safety.audit import AuditLog

log = get_logger(__name__)

Callback = Callable[[str], Awaitable[None]]


class KillSwitch:
    def __init__(
        self, sentinel: Path, *, bus: EventBus | None = None, audit: AuditLog | None = None
    ) -> None:
        self.sentinel = sentinel
        self._bus = bus
        self._audit = audit
        self._flag = False
        self._reason = ""
        self._callbacks: list[Callback] = []
        self._fired = False

    @property
    def engaged(self) -> bool:
        return self._flag or self.sentinel.exists()

    @property
    def reason(self) -> str:
        if self._reason:
            return self._reason
        if self.sentinel.exists():
            try:
                data = json.loads(self.sentinel.read_text() or "{}")
                return str(data.get("reason") or "sentinel file present")
            except (OSError, json.JSONDecodeError):
                return "sentinel file present"
        return ""

    def check(self) -> None:
        if self.engaged:
            raise KillSwitchEngagedError(f"kill switch engaged: {self.reason}")

    def on_engage(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    async def engage(self, reason: str, *, by: str = "operator") -> None:
        self._flag = True
        self._reason = reason
        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_text(
            json.dumps({"reason": reason, "by": by, "at": utcnow().isoformat()})
        )
        log.critical("KILL SWITCH ENGAGED", extra={"reason": reason, "by": by})
        await self._fire(reason, by)

    async def _fire(self, reason: str, by: str) -> None:
        if self._fired:
            return
        self._fired = True
        if self._audit:
            await self._audit.record("safety.kill_switch.engaged", actor=by, reason=reason)
        if self._bus:
            await self._bus.publish("safety.kill_switch.engaged", reason=reason, by=by)
        for callback in list(self._callbacks):
            try:
                await callback(reason)
            except Exception:
                log.exception("kill switch callback failed")

    async def release(self, *, by: str = "operator") -> None:
        self._flag = False
        self._reason = ""
        self._fired = False
        with contextlib.suppress(FileNotFoundError):
            self.sentinel.unlink()
        log.warning("kill switch released", extra={"by": by})
        if self._audit:
            await self._audit.record("safety.kill_switch.released", actor=by)
        if self._bus:
            await self._bus.publish("safety.kill_switch.released", by=by)

    async def watch(self, interval_s: float = 0.5) -> None:
        """Detect externally created sentinels (hardware button, watchdog) and fire callbacks."""

        while True:
            if self.sentinel.exists() and not self._fired:
                self._flag = True
                await self._fire(self.reason, "sentinel")
            elif not self.engaged and self._fired:
                self._fired = False
            await asyncio.sleep(interval_s)

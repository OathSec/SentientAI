"""In-process asynchronous event bus.

Every significant transition (task lifecycle, tool calls, approvals, affect
changes, workspace broadcasts) is published here. The API streams the bus over
SSE, metrics subscribe to it, and tests assert on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sentientai.core.clock import new_id, utcnow
from sentientai.core.logging import get_logger

log = get_logger(__name__)

Handler = Callable[["Event"], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("evt"))
    ts: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "ts": self.ts.isoformat(),
            "payload": self.payload,
        }


class EventBus:
    def __init__(self, history: int = 500) -> None:
        self._handlers: list[tuple[str, Handler]] = []
        self._queues: list[tuple[str, asyncio.Queue[Event]]] = []
        self._history: list[Event] = []
        self._history_size = history

    def subscribe(self, pattern: str, handler: Handler) -> Callable[[], None]:
        """Register a handler for event types matching a glob pattern (e.g. ``task.*``)."""

        entry = (pattern, handler)
        self._handlers.append(entry)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._handlers.remove(entry)

        return unsubscribe

    async def publish(self, type_: str, **payload: Any) -> Event:
        event = Event(type=type_, payload=payload)
        self._history.append(event)
        if len(self._history) > self._history_size:
            del self._history[: len(self._history) - self._history_size]
        for pattern, handler in list(self._handlers):
            if fnmatch.fnmatchcase(type_, pattern):
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    log.exception("event handler failed", extra={"event_type": type_})
        for pattern, queue in list(self._queues):
            if fnmatch.fnmatchcase(type_, pattern):
                if queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                queue.put_nowait(event)
        return event

    @contextlib.contextmanager
    def subscription(
        self, pattern: str = "*", maxsize: int = 1000
    ) -> Iterator[asyncio.Queue[Event]]:
        """A queue of matching events for the duration of the ``with`` block.

        Unlike :meth:`stream`, waiting on the queue with a timeout is safe: cancelling a
        ``queue.get()`` loses nothing, whereas cancelling an async generator ends it.
        """

        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        entry = (pattern, queue)
        self._queues.append(entry)
        try:
            yield queue
        finally:
            with contextlib.suppress(ValueError):
                self._queues.remove(entry)

    async def stream(self, pattern: str = "*", maxsize: int = 1000) -> AsyncIterator[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        entry = (pattern, queue)
        self._queues.append(entry)
        try:
            while True:
                yield await queue.get()
        finally:
            with contextlib.suppress(ValueError):
                self._queues.remove(entry)

    def recent(self, pattern: str = "*", limit: int = 100) -> list[Event]:
        matched = [e for e in self._history if fnmatch.fnmatchcase(e.type, pattern)]
        return matched[-limit:]

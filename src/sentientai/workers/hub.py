"""Hands jobs to connected workers and collects what they send back.

A worker is a program on someone's PC that connects *out* to this server (long polling over
HTTPS, so no ports to open) and runs fixed kinds of jobs against models on its own machine:

* ``llm``:   one OpenAI-style chat completion
* ``image``: one picture

If no suitable worker is free the caller falls back to this machine's own models, so a worker
going to sleep never breaks anything. Workers do one job at a time; a busy worker simply isn't
offered more work, which also spreads load between workers and this machine.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sentientai.core.clock import new_id
from sentientai.core.logging import get_logger
from sentientai.workers.registry import WorkerRecord

log = get_logger(__name__)

KINDS = ("llm", "image")
POLL_WAIT_S = 25.0
GONE_AFTER_S = 75.0  # an idle worker that hasn't polled for this long is offline


class WorkerError(Exception):
    """The job could not be done by a worker (the caller should use its own model)."""


@dataclass
class Job:
    id: str
    kind: str
    payload: dict[str, Any]


@dataclass
class _Online:
    record: WorkerRecord
    capabilities: set[str]
    info: dict[str, Any]
    last_seen: float
    queue: asyncio.Queue[Job] = field(default_factory=asyncio.Queue)
    job_id: str | None = None  # the job it is working on (or has been given)
    jobs_done: int = 0


class WorkerHub:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._now = clock
        self._online: dict[str, _Online] = {}
        self._waiting: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # ------------------------------------------------------------ worker side
    async def poll(
        self,
        record: WorkerRecord,
        capabilities: set[str],
        info: dict[str, Any],
        *,
        wait_s: float = POLL_WAIT_S,
    ) -> Job | None:
        """The worker asks for work. Returns a job, or None after ``wait_s`` of nothing."""

        worker = self._online.get(record.id)
        if worker is None:
            worker = self._online[record.id] = _Online(
                record, capabilities, info, last_seen=self._now()
            )
            log.info(
                "worker connected", extra={"worker": record.name, "caps": sorted(capabilities)}
            )
        worker.record, worker.capabilities, worker.info = record, capabilities, info
        worker.last_seen = self._now()
        if worker.queue.empty():
            worker.job_id = None  # asking again means it finished (or dropped) the last one
        try:
            job = await asyncio.wait_for(worker.queue.get(), timeout=wait_s)
        except TimeoutError:
            worker.last_seen = self._now()
            return None
        worker.job_id = job.id
        return job

    def result(self, record: WorkerRecord, job_id: str, *, ok: bool, data: Any, error: str) -> bool:
        worker = self._online.get(record.id)
        future = self._waiting.pop(job_id, None)
        if worker is not None:
            worker.last_seen = self._now()
            if worker.job_id == job_id:
                worker.job_id = None
                worker.jobs_done += 1
        if future is None or future.done():
            return False  # too late: the caller already gave up on it
        if ok and isinstance(data, dict):
            future.set_result(data)
        else:
            future.set_exception(WorkerError(error or "the worker couldn't do it"))
        return True

    def goodbye(self, record: WorkerRecord) -> None:
        worker = self._online.pop(record.id, None)
        if worker is None or worker.job_id is None:
            return
        future = self._waiting.pop(worker.job_id, None)
        if future is not None and not future.done():
            future.set_exception(WorkerError("the worker left"))

    # ------------------------------------------------------------ server side
    def _eligible(self, kind: str, *, private_only: bool) -> list[_Online]:
        now = self._now()
        found = []
        for worker in self._online.values():
            if kind not in worker.capabilities or worker.job_id is not None:
                continue
            if now - worker.last_seen > GONE_AFTER_S:
                continue
            if (kind == "llm" or private_only) and worker.record.trust != "private":
                continue  # private conversations and thoughts never leave the operator's PCs
            found.append(worker)
        return found

    def available(self, kind: str, *, private_only: bool = True) -> bool:
        return bool(self._eligible(kind, private_only=private_only))

    async def run(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        private_only: bool = True,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Do a job on a free worker. ``private_only``: only the operator's own machines."""

        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        candidates = self._eligible(kind, private_only=private_only)
        if not candidates:
            raise WorkerError("no worker is free")
        worker = min(candidates, key=lambda w: w.jobs_done)  # spread the work around
        job = Job(new_id("job"), kind, payload)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiting[job.id] = future
        worker.job_id = job.id
        worker.queue.put_nowait(job)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            raise WorkerError("the worker took too long") from None
        finally:
            self._waiting.pop(job.id, None)
            if worker.job_id == job.id:
                worker.job_id = None  # whatever happened, it isn't ours to wait on any more

    def online(self) -> list[dict[str, Any]]:
        now = self._now()
        return [
            {
                "id": w.record.id,
                "name": w.record.name,
                "trust": w.record.trust,
                "capabilities": sorted(w.capabilities),
                "busy": w.job_id is not None,
                "idle_s": round(now - w.last_seen),
                "jobs_done": w.jobs_done,
                "info": w.info,
            }
            for w in self._online.values()
            if now - w.last_seen <= GONE_AFTER_S or w.job_id is not None
        ]

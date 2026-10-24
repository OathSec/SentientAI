"""Who may connect as a worker. Tokens are shown once and stored only as hashes."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from sentientai.core.clock import Clock, SystemClock, new_id
from sentientai.storage.database import StateRepository

STATE_KEY = "workers.registry"
TOKEN_PREFIX = "fzw_"  # noqa: S105 - a label on tokens, not a secret

# private: the operator's own machine. May run Fuzzy's thinking, which includes private
#          conversations, and any picture.
# shared:  a friend's machine. Only ever sees pictures Fuzzy makes for its own public gallery.
Trust = Literal["private", "shared"]


class WorkerStats(BaseModel):
    """What a computer has contributed, over its whole life."""

    jobs: int = 0
    llm_jobs: int = 0
    image_jobs: int = 0
    failed_jobs: int = 0
    busy_s: float = 0.0
    last_job_at: datetime | None = None


class WorkerRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("wrk"))
    name: str
    owner_id: str | None = None  # the Fuzzy account it is linked to (None: the operator's own)
    stats: WorkerStats = Field(default_factory=WorkerStats)
    trust: Trust = "shared"
    token_hash: str
    created_at: datetime
    last_seen_at: datetime | None = None
    revoked: bool = False

    def public(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"token_hash"})


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class WorkerRegistry:
    def __init__(self, state: StateRepository, *, clock: Clock | None = None) -> None:
        self.state = state
        self.clock = clock or SystemClock()

    async def all(self) -> list[WorkerRecord]:
        raw = await self.state.get(STATE_KEY)
        records = []
        for item in raw if isinstance(raw, list) else []:
            try:
                records.append(WorkerRecord.model_validate(item))
            except ValueError:
                continue
        return records

    async def _save(self, records: list[WorkerRecord]) -> None:
        await self.state.set(STATE_KEY, [r.model_dump(mode="json") for r in records])

    async def create(
        self, name: str, trust: Trust = "shared", owner_id: str | None = None
    ) -> tuple[WorkerRecord, str]:
        """A new worker and its token. The token cannot be recovered later."""

        name = " ".join(name.split())[:60]
        if not name:
            raise ValueError("a worker needs a name")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        record = WorkerRecord(
            name=name,
            trust=trust,
            owner_id=owner_id,
            token_hash=hash_token(token),
            created_at=self.clock.now(),
        )
        await self._save([*await self.all(), record])
        return record, token

    async def revoke(self, id_or_name: str, owner_id: str | None = None) -> WorkerRecord | None:
        """Revoke by id or name; with ``owner_id``, only that person's own computers."""

        records = await self.all()
        found = next(
            (
                r
                for r in records
                if not r.revoked
                and id_or_name in (r.id, r.name)
                and (owner_id is None or r.owner_id == owner_id)
            ),
            None,
        )
        if found is None:
            return None
        found.revoked = True
        await self._save(records)
        return found

    async def authenticate(self, token: str) -> WorkerRecord | None:
        if not token.startswith(TOKEN_PREFIX):
            return None
        digest = hash_token(token)
        match: WorkerRecord | None = None
        for record in await self.all():  # compare against all, so timing says nothing
            if hmac.compare_digest(record.token_hash, digest) and not record.revoked:
                match = record
        return match

    async def set_trust(self, id_or_name: str, trust: Trust) -> WorkerRecord | None:
        records = await self.all()
        found = next((r for r in records if not r.revoked and id_or_name in (r.id, r.name)), None)
        if found is None:
            return None
        found.trust = trust
        await self._save(records)
        return found

    async def for_owner(self, owner_id: str) -> list[WorkerRecord]:
        return [r for r in await self.all() if r.owner_id == owner_id and not r.revoked]

    async def record_job(self, worker_id: str, kind: str, seconds: float, *, ok: bool) -> None:
        records = await self.all()
        for record in records:
            if record.id != worker_id:
                continue
            stats = record.stats
            if ok:
                stats.jobs += 1
                stats.busy_s += max(0.0, seconds)
                if kind == "llm":
                    stats.llm_jobs += 1
                elif kind == "image":
                    stats.image_jobs += 1
            else:
                stats.failed_jobs += 1
            stats.last_job_at = self.clock.now()
        await self._save(records)

    async def seen(self, worker_id: str) -> None:
        records = await self.all()
        for record in records:
            if record.id == worker_id:
                record.last_seen_at = self.clock.now()
        await self._save(records)

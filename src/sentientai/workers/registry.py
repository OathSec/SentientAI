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


class WorkerRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("wrk"))
    name: str
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

    async def create(self, name: str, trust: Trust = "shared") -> tuple[WorkerRecord, str]:
        """A new worker and its token. The token cannot be recovered later."""

        name = " ".join(name.split())[:60]
        if not name:
            raise ValueError("a worker needs a name")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        record = WorkerRecord(
            name=name, trust=trust, token_hash=hash_token(token), created_at=self.clock.now()
        )
        await self._save([*await self.all(), record])
        return record, token

    async def revoke(self, id_or_name: str) -> WorkerRecord | None:
        records = await self.all()
        found = next((r for r in records if not r.revoked and id_or_name in (r.id, r.name)), None)
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

    async def seen(self, worker_id: str) -> None:
        records = await self.all()
        for record in records:
            if record.id == worker_id:
                record.last_seen_at = self.clock.now()
        await self._save(records)

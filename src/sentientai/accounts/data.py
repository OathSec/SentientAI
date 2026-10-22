"""A person's data: export everything Fuzzy holds about them, or erase it."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from sentientai.memory.system import MemorySystem
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import MemoryStatus, Scope, UserRecord


class UserDataService:
    def __init__(self, db: Database, memory: MemorySystem, *, audit: AuditLog) -> None:
        self.db = db
        self.memory = memory
        self.audit = audit

    async def export(self, user: UserRecord) -> dict[str, Any]:
        own = Scope(user_id=user.id, include_global=False)
        memories = await self.db.memories.list(scope=own, status=MemoryStatus.ACTIVE, limit=100_000)
        relationship = await self.db.relationships.get(user.id)
        return {
            "account": user.public(),
            "relationship": relationship.model_dump(mode="json") if relationship else None,
            "memories": [
                {
                    "id": m.id,
                    "kind": m.kind.value,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in memories
            ],
            "conversations": [
                {"role": m.role, "content": m.content, "at": m.created_at.isoformat()}
                for m in await self.db.conversations.for_user(user.id)
            ],
            "notes_from_fuzzy": [
                {"content": n.content, "at": n.created_at.isoformat()}
                for n in await self.db.notes.for_user(user.id)
            ],
            "images": [
                {"id": m.id, "prompt": m.prompt, "at": m.created_at.isoformat()}
                for m in await self.db.media.for_user(user.id)
            ],
        }

    async def erase(self, user: UserRecord, *, by: str) -> dict[str, int]:
        """Delete the account and everything Fuzzy holds that is private to this person."""

        counts = {"memories": await self.memory.forget_user(user.id)}
        for media in await self.db.media.for_user(user.id):
            with contextlib.suppress(FileNotFoundError):
                Path(media.path).unlink()
            await self.db.media.delete(media.id)
        counts["conversations"] = await self.db.conversations.delete_user(user.id)
        counts["tasks"] = await self.db.tasks.delete_user(user.id)
        await self.db.relationships.delete(user.id)
        await self.db.notes.delete_user(user.id)
        await self.db.emotions.delete_user(user.id)
        await self.db.usage.delete_user(user.id)
        counts["sessions"] = await self.db.sessions.delete_for_user(user.id)
        for session_id in await self.memory.working.sessions():
            if session_id.startswith(f"user:{user.id}"):
                await self.memory.working.clear(session_id)
        await self.db.users.delete(user.id)
        await self.audit.record("account.erased", actor=by, user_id=user.id, counts=counts)
        return counts

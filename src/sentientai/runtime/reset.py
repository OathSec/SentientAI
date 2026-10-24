"""Start Fuzzy's mind over, keeping the people and the record.

``sentientai reset-mind`` clears everything Fuzzy has lived through: memories (and their
vectors), conversations, what it decided about itself, feelings, relationships, activities,
goals, projects, notes, its world, body and gallery, its self-model and working memory. Accounts,
sessions, invites, usage counters and the audit log are kept (the audit log records the reset).
With ``keep_teachings``, documents people taught it stay, as knowledge it hasn't studied yet.

Run it with the daemon stopped: a running Fuzzy holds parts of its mind in memory and would
write them straight back.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import delete, or_, text

from sentientai.config import Settings
from sentientai.safety.audit import AuditLog
from sentientai.storage import schema
from sentientai.storage.database import Database

# Fuzzy's life. Users, sessions, invites, usage and schema_meta are deliberately absent.
MIND_TABLES = (
    "conversation_messages",
    "tasks",
    "approvals",
    "goals",
    "tool_invocations",
    "proposals",
    "relationships",
    "self_beliefs",
    "emotion_events",
    "activities",
    "notes",
    "media",
    "world_objects",
)
# Operator schedules ("schedule:") and file watchers ("watcher:") keep their state.
STATE_PREFIXES = ("mind.", "world.", "consolidation.")


async def reset_mind(
    settings: Settings, *, keep_teachings: bool = False, audit: AuditLog | None = None
) -> dict[str, Any]:
    db = Database(settings.storage.resolved_database_url())
    await db.create_all()
    cleared: dict[str, Any] = {}
    try:
        async with db.engine.begin() as conn:
            for name in MIND_TABLES:
                table = schema.metadata.tables[name]
                cleared[name] = (await conn.execute(delete(table))).rowcount
            memories = schema.memories
            stmt = delete(memories)
            if keep_teachings:
                stmt = stmt.where(memories.c.kind != "document")
            else:
                cleared["documents"] = (await conn.execute(delete(schema.documents))).rowcount
            cleared["memories"] = (await conn.execute(stmt)).rowcount
            kv = schema.kv_state
            cleared["state"] = (
                await conn.execute(
                    delete(kv).where(or_(*(kv.c.key.startswith(p) for p in STATE_PREFIXES)))
                )
            ).rowcount
            if (
                settings.storage.vector_backend == "pgvector"
                and db.engine.dialect.name == "postgresql"
            ):
                exists = (
                    await conn.execute(text("SELECT to_regclass('memory_vectors') IS NOT NULL"))
                ).scalar()
                if exists:
                    cleared["vectors"] = (
                        await conn.execute(
                            text(
                                "DELETE FROM memory_vectors "
                                "WHERE id NOT IN (SELECT id FROM memories)"
                            )
                        )
                    ).rowcount
    finally:
        await db.close()

    data = settings.storage.data_dir
    for folder in ("media", "sandbox"):
        _empty(data / folder)
    if settings.storage.vector_backend != "pgvector":
        # File and Qdrant indexes are rebuilt from what's left: see ``sentientai memory reindex``.
        (data / "vectors.npz").unlink(missing_ok=True)
        cleared["reindex_needed"] = True
    cleared["working_memory"] = await _clear_working_memory(settings)
    if audit is not None:
        await audit.record(
            "mind.reset", actor="operator:cli", keep_teachings=keep_teachings, cleared=cleared
        )
    return cleared


def _empty(folder: Path) -> None:
    if not folder.is_dir():
        return
    for child in folder.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


async def _clear_working_memory(settings: Settings) -> int:
    url = settings.storage.redis_url
    if not url:
        return 0  # in-process working memory: gone when the daemon stops
    import redis.asyncio as redis

    client = redis.from_url(url)
    removed = 0
    try:
        for pattern in ("sentient:wm:*", "sentient:kv:*"):
            async for key in client.scan_iter(match=pattern, count=500):
                removed += await client.delete(key)
    finally:
        await client.aclose()
    return removed

"""Short-term working memory: active conversation turns and scratch state.

Redis is the production backend (fast, survives daemon restarts, shared with
the API process). The in-process backend is used when no Redis URL is set.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from sentientai.core.clock import utcnow


class Turn(BaseModel):
    role: str
    content: str
    ts: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class WorkingMemory(Protocol):
    async def append(self, session_id: str, turn: Turn) -> int: ...

    async def turns(self, session_id: str, limit: int | None = None) -> list[Turn]: ...

    async def replace(self, session_id: str, turns: list[Turn]) -> None: ...

    async def clear(self, session_id: str) -> None: ...

    async def sessions(self) -> list[str]: ...

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None: ...

    async def get(self, key: str) -> Any | None: ...

    async def ping(self) -> bool: ...

    async def aclose(self) -> None: ...


class InMemoryWorkingMemory:
    def __init__(self, max_turns: int = 40) -> None:
        self.max_turns = max_turns
        self._turns: dict[str, list[Turn]] = defaultdict(list)
        self._activity: dict[str, float] = {}
        self._kv: dict[str, tuple[Any, float | None]] = {}

    async def append(self, session_id: str, turn: Turn) -> int:
        turns = self._turns[session_id]
        turns.append(turn)
        del turns[: max(0, len(turns) - self.max_turns)]
        self._activity[session_id] = time.time()
        return len(turns)

    async def turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        turns = list(self._turns.get(session_id, []))
        return turns[-limit:] if limit else turns

    async def replace(self, session_id: str, turns: list[Turn]) -> None:
        self._turns[session_id] = list(turns[-self.max_turns :])
        self._activity[session_id] = time.time()

    async def clear(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
        self._activity.pop(session_id, None)

    async def sessions(self) -> list[str]:
        return [s for s, _ in sorted(self._activity.items(), key=lambda kv: kv[1], reverse=True)]

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        self._kv[key] = (value, time.time() + ttl_s if ttl_s else None)

    async def get(self, key: str) -> Any | None:
        item = self._kv.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and time.time() > expires:
            self._kv.pop(key, None)
            return None
        return value

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class RedisWorkingMemory:
    def __init__(
        self, url: str, *, max_turns: int = 40, ttl_s: int = 7 * 86400, prefix: str = "sentient"
    ) -> None:
        from redis.asyncio import Redis

        self._redis: Any = Redis.from_url(url, decode_responses=True)
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self.prefix = prefix

    def _turns_key(self, session_id: str) -> str:
        return f"{self.prefix}:wm:{session_id}:turns"

    @property
    def _sessions_key(self) -> str:
        return f"{self.prefix}:wm:sessions"

    async def append(self, session_id: str, turn: Turn) -> int:
        key = self._turns_key(session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, turn.model_dump_json())
            pipe.ltrim(key, -self.max_turns, -1)
            pipe.expire(key, self.ttl_s)
            pipe.zadd(self._sessions_key, {session_id: utcnow().timestamp()})
            pipe.llen(key)
            results = await pipe.execute()
        return int(results[-1])

    async def turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        start = -limit if limit else 0
        raw = await self._redis.lrange(self._turns_key(session_id), start, -1)
        return [Turn.model_validate_json(item) for item in raw]

    async def replace(self, session_id: str, turns: list[Turn]) -> None:
        key = self._turns_key(session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            if turns:
                pipe.rpush(key, *[t.model_dump_json() for t in turns[-self.max_turns :]])
                pipe.expire(key, self.ttl_s)
            pipe.zadd(self._sessions_key, {session_id: utcnow().timestamp()})
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._turns_key(session_id))
        await self._redis.zrem(self._sessions_key, session_id)

    async def sessions(self) -> list[str]:
        cutoff = utcnow().timestamp() - self.ttl_s
        await self._redis.zremrangebyscore(self._sessions_key, "-inf", cutoff)
        return [str(s) for s in await self._redis.zrevrange(self._sessions_key, 0, -1)]

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        await self._redis.set(f"{self.prefix}:kv:{key}", json.dumps(value, default=str), ex=ttl_s)

    async def get(self, key: str) -> Any | None:
        raw = await self._redis.get(f"{self.prefix}:kv:{key}")
        return json.loads(raw) if raw is not None else None

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._redis.aclose()


def build_working_memory(redis_url: str | None, *, max_turns: int, ttl_s: int) -> WorkingMemory:
    if redis_url:
        return RedisWorkingMemory(redis_url, max_turns=max_turns, ttl_s=ttl_s)
    return InMemoryWorkingMemory(max_turns=max_turns)

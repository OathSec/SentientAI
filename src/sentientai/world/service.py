"""Fuzzy's world: the place it builds, the body it designs, and where it walks.

The world is shared state: every visitor sees the same place. Changes publish
``world.*`` events, which the world page receives live. A small simulation loop
moves Fuzzy toward wherever it decided to walk.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from sentientai.config import WorldConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import WorldObject
from sentientai.world.model import Action, Avatar, Body, _clamp, clean_parts, clean_text

log = get_logger(__name__)

AVATAR_KEY = "world.avatar"
BODY_KEY = "world.body"
SIZE_KEY = "world.size"
THOUGHT_CHARS = (
    4000  # the bubble over its head scrolls; nothing Fuzzy thinks is cut short for display
)
_TICK_S = 0.25


class WorldError(Exception):
    pass


class World:
    def __init__(
        self,
        db: Database,
        config: WorldConfig,
        *,
        audit: AuditLog,
        bus: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.audit = audit
        self.bus = bus
        self.clock = clock or SystemClock()
        self.avatar = Avatar()
        self.body = Body()
        self.size = config.size
        self._lock = asyncio.Lock()
        self._arrived = asyncio.Event()

    @property
    def half(self) -> float:
        return self.size / 2

    def _coord(self, value: Any) -> float:
        return round(_clamp(value, -self.half + 0.5, self.half - 0.5, 0.0), 2)

    async def load(self) -> None:
        raw_size = await self.db.state.get(SIZE_KEY)
        if isinstance(raw_size, int | float):
            self.size = _clamp(raw_size, self.config.size, self.config.max_size, self.config.size)
        raw_avatar = await self.db.state.get(AVATAR_KEY)
        if isinstance(raw_avatar, dict):
            self.avatar = Avatar.model_validate(raw_avatar)
        raw_body = await self.db.state.get(BODY_KEY)
        if isinstance(raw_body, dict):
            self.body = Body.model_validate(raw_body)
            self.body.target = None
            self.body.action = "standing"

    async def save(self) -> None:
        """Persist where Fuzzy is, so it wakes up where it fell asleep."""

        await self.db.state.set(BODY_KEY, self.body.model_dump(mode="json"))

    async def _emit(self, type_: str, **payload: Any) -> None:
        if self.bus is not None:
            await self.bus.publish(type_, **payload)

    # ------------------------------------------------------------ queries
    async def objects(self) -> list[WorldObject]:
        return await self.db.world.all()

    async def snapshot(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": max(self.config.max_size, self.size),
            "objects": [o.model_dump(mode="json") for o in await self.objects()],
            "avatar": self.avatar.model_dump(mode="json"),
            "body": self.body.model_dump(mode="json"),
        }

    async def describe(self) -> str:
        objects = await self.objects()
        if not objects:
            return "Nothing yet: the ground is empty."

        def shown(o: WorldObject) -> str:
            ids = [str(p["media_id"]) for p in o.parts if p.get("shape") == "picture"]
            return f" (showing your picture {', '.join(ids)})" if ids else ""

        return "\n".join(
            f"- [{o.id}] {o.name} at ({o.x:.1f}, {o.z:.1f}), {len(o.parts)} parts{shown(o)}"
            + (f": {o.description[:400]}" if o.description else "")
            for o in objects
        )

    def describe_avatar(self) -> str:
        if not self.avatar.parts:
            return "You haven't designed how you look yet; you appear as a plain marker."
        return f"You look like this (your own design): {self.avatar.description}"

    async def _own_pictures(self, parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep picture parts only if they show one of Fuzzy's own public gallery images:
        a person's private picture never ends up on the public world page."""

        kept = []
        for part in parts:
            if part.get("shape") == "picture":
                record = await self.db.media.get(str(part.get("media_id")))
                if record is None or not record.public or record.user_id is not None:
                    continue
            kept.append(part)
        return kept

    async def pictures(self, limit: int = 12) -> list[tuple[str, str]]:
        """(id, caption) of Fuzzy's own gallery images, newest first."""

        records = await self.db.media.visible_to(None, admin=False)
        mine = [r for r in records if r.public and r.user_id is None]
        return [(r.id, r.caption or r.prompt) for r in mine[:limit]]

    async def describe_pictures(self) -> str:
        pictures = await self.pictures()
        if not pictures:
            return "You haven't made any pictures yet."
        return "Pictures you've made (in your gallery):\n" + "\n".join(
            f"- [{pid}] {caption[:160]}" for pid, caption in pictures
        )

    # ------------------------------------------------------------ changes
    async def build(
        self,
        name: str,
        parts: Any,
        *,
        x: Any = 0.0,
        z: Any = 0.0,
        rotation: Any = 0.0,
        description: str = "",
        by: str = "agent",
    ) -> WorldObject:
        cleaned = await self._own_pictures(clean_parts(parts, self.config.max_parts_per_object))
        if not cleaned:
            raise WorldError("nothing to build: no valid parts")
        name = clean_text(name, 80) or "something"
        async with self._lock:
            if await self.db.world.count() >= self.config.max_objects:
                raise WorldError(
                    f"the world already holds {self.config.max_objects} things; remove one first"
                )
            now = self.clock.now()
            obj = WorldObject(
                name=name,
                description=clean_text(description, 500),
                parts=cleaned,
                x=self._coord(x),
                z=self._coord(z),
                rotation=round(_clamp(rotation, -360, 360, 0.0), 1),
                created_at=now,
                updated_at=now,
            )
            await self.db.world.insert(obj)
        await self.audit.record("world.built", actor=by, object_id=obj.id, name=name)
        await self._emit("world.object", op="built", object=obj.model_dump(mode="json"))
        return obj

    async def modify(
        self,
        object_id: str,
        *,
        name: Any = None,
        description: Any = None,
        parts: Any = None,
        x: Any = None,
        z: Any = None,
        rotation: Any = None,
        by: str = "agent",
    ) -> WorldObject:
        async with self._lock:
            obj = await self.db.world.get(str(object_id))
            if obj is None:
                raise WorldError(f"there is no object {object_id}")
            if name:
                obj.name = clean_text(name, 80) or obj.name
            if description is not None:
                obj.description = clean_text(description, 500)
            if parts is not None:
                cleaned = await self._own_pictures(
                    clean_parts(parts, self.config.max_parts_per_object)
                )
                if not cleaned:
                    raise WorldError("no valid parts given")
                obj.parts = cleaned
            if x is not None:
                obj.x = self._coord(x)
            if z is not None:
                obj.z = self._coord(z)
            if rotation is not None:
                obj.rotation = round(_clamp(rotation, -360, 360, obj.rotation), 1)
            obj.updated_at = self.clock.now()
            await self.db.world.save(obj)
        await self.audit.record("world.modified", actor=by, object_id=obj.id)
        await self._emit("world.object", op="changed", object=obj.model_dump(mode="json"))
        return obj

    async def remove(self, object_id: str, *, by: str = "agent") -> WorldObject:
        async with self._lock:
            obj = await self.db.world.get(str(object_id))
            if obj is None:
                raise WorldError(f"there is no object {object_id}")
            await self.db.world.delete(obj.id)
        await self.audit.record("world.removed", actor=by, object_id=obj.id, name=obj.name)
        await self._emit("world.object", op="removed", object={"id": obj.id})
        return obj

    async def design_avatar(self, description: str, parts: Any, *, by: str = "agent") -> Avatar:
        cleaned = await self._own_pictures(clean_parts(parts, self.config.max_parts_per_object))
        if not cleaned:
            raise WorldError("an avatar needs at least one valid part")
        self.avatar = Avatar(
            description=clean_text(description, 600),
            parts=cleaned,
            designed_at=self.clock.now(),
            versions=self.avatar.versions + 1,
        )
        await self.db.state.set(AVATAR_KEY, self.avatar.model_dump(mode="json"))
        await self.audit.record("world.avatar", actor=by, version=self.avatar.versions)
        await self._emit("world.avatar", avatar=self.avatar.model_dump(mode="json"))
        return self.avatar

    async def expand(self, size: Any, *, by: str = "agent") -> float:
        """Make the ground bigger (it never shrinks: that would strand what is built on it)."""

        wanted = _clamp(size, 0.0, self.config.max_size, self.size)
        if wanted <= self.size:
            raise WorldError(
                f"your world is already {self.size:.0f} m across; ask for more than that"
            )
        old, self.size = self.size, round(wanted, 1)
        await self.db.state.set(SIZE_KEY, self.size)
        await self.audit.record("world.expanded", actor=by, before=old, after=self.size)
        await self._emit("world.size", size=self.size)
        return self.size

    async def walk_to(self, x: Any, z: Any) -> tuple[float, float]:
        target = [self._coord(x), self._coord(z)]
        self.body.target = target
        self.body.action = "walking"
        self._arrived.clear()
        await self._emit_body()
        return target[0], target[1]

    async def set_action(self, action: Action, thought: str | None = None) -> None:
        if self.body.target is not None and action != "walking":
            self.body.target = None  # stopping to do something else
        self.body.action = action
        if thought is not None:
            self.body.thought = clean_text(thought, THOUGHT_CHARS)
        await self._emit_body()

    async def set_thought(self, thought: str) -> None:
        """Show what Fuzzy is thinking without interrupting what it is doing."""

        self.body.thought = clean_text(thought, THOUGHT_CHARS)
        await self._emit_body()

    async def _emit_body(self) -> None:
        self.body.updated_at = self.clock.now()
        await self._emit("world.body", body=self.body.model_dump(mode="json"))

    async def apply(self, action: dict[str, Any], *, by: str = "agent") -> str:
        """Carry out a world action Fuzzy chose (JSON from its model)."""

        kind = str(action.get("action", "")).strip().lower()
        if kind == "build":
            obj = await self.build(
                str(action.get("name", "")),
                action.get("parts"),
                x=action.get("x", self.body.x + 2),
                z=action.get("z", self.body.z),
                rotation=action.get("rotation", 0.0),
                description=str(action.get("description", "")),
                by=by,
            )
            await self.walk_to(obj.x - 1.5, obj.z)
            return f"built {obj.name} ({len(obj.parts)} parts) at ({obj.x:.1f}, {obj.z:.1f})"
        if kind == "modify":
            obj = await self.modify(
                str(action.get("object_id", "")),
                name=action.get("name"),
                description=action.get("description"),
                parts=action.get("parts"),
                x=action.get("x"),
                z=action.get("z"),
                rotation=action.get("rotation"),
                by=by,
            )
            return f"changed {obj.name}"
        if kind == "remove":
            obj = await self.remove(str(action.get("object_id", "")), by=by)
            return f"removed {obj.name}"
        if kind == "expand":
            old = self.size
            new = await self.expand(action.get("size"), by=by)
            return f"made my world bigger, from {old:.0f} m to {new:.0f} m across"
        if kind == "walk":
            x, z = await self.walk_to(action.get("x", 0), action.get("z", 0))
            return f"walked to ({x:.1f}, {z:.1f})"
        if kind == "avatar":
            avatar = await self.design_avatar(
                str(action.get("description", "")), action.get("parts"), by=by
            )
            return f"redesigned how I look (version {avatar.versions})"
        raise WorldError(f"unknown world action {kind!r}")

    # --------------------------------------------------------- simulation
    async def run(self) -> None:
        """Move Fuzzy toward its walking target."""

        last_emit = 0.0
        while True:
            await asyncio.sleep(_TICK_S)
            target = self.body.target
            if target is None:
                continue
            dx, dz = target[0] - self.body.x, target[1] - self.body.z
            distance = math.hypot(dx, dz)
            step = self.config.walk_speed * _TICK_S
            if distance > 1e-6:
                self.body.heading = math.degrees(math.atan2(dx, dz))
            if distance <= step:
                self.body.x, self.body.z = target
                self.body.target = None
                self.body.action = "standing"
                await self._emit_body()
                await self.save()
                self._arrived.set()
                continue
            self.body.x += dx / distance * step
            self.body.z += dz / distance * step
            now = self.clock.monotonic()
            if now - last_emit >= 0.5:
                last_emit = now
                await self._emit_body()

    async def arrived(self, timeout: float) -> bool:
        if self.body.target is None:
            return True
        try:
            await asyncio.wait_for(self._arrived.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

"""The brain page: what Fuzzy is feeling, thinking and doing, for each kind of viewer.

* Admins see everything, unredacted.
* Signed-in people see Fuzzy's shared mind plus anything that is about them, with
  other people's names removed.
* Visitors who aren't signed in (when ``accounts.public_pages`` is on) see only the shared mind.

The same rules apply to the live event stream (:func:`filter_event`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentientai.cognition.privacy import Redact
from sentientai.core.events import Event
from sentientai.runtime.runtime import Runtime
from sentientai.storage.records import MemoryKind, Scope


@dataclass(frozen=True)
class Viewer:
    user_id: str | None = None
    admin: bool = False

    @property
    def scope(self) -> Scope:
        return Scope.system() if self.admin else Scope.for_user(self.user_id)


# Fields of each event a non-admin may receive. Anything else is dropped.
_PUBLIC_FIELDS: dict[str, tuple[str, ...]] = {
    "mind.thought": ("thought", "user_id"),
    "mind.feeling": ("emotion", "intensity", "valence", "cause", "user_id"),
    "mind.activity": ("id", "kind", "intent", "reason", "outcome", "status", "user_id"),
    "self.belief_formed": ("aspect", "value"),
    "self.belief_considering": ("aspect", "value"),
    "self.belief_revised": ("aspect", "value", "reasoning"),
    "self.belief_kept": ("aspect", "value", "reasoning"),
    "social.appraised": ("user_id", "affinity", "trust", "opinion"),
    "social.note": ("user_id",),
    "world.body": ("body",),
    "world.object": ("op", "object"),
    "world.avatar": ("avatar",),
    "media.created": ("media_id", "user_id", "public", "caption"),
    "safety.kill_switch.engaged": ("reason",),
    "safety.kill_switch.released": (),
    "runtime.boot": ("boot_count",),
}
PUBLIC_PATTERNS = (
    "mind.*",
    "self.*",
    "social.*",
    "world.*",
    "media.created",
    "safety.kill_switch.*",
    "runtime.boot",
)


def _redact_deep(value: Any, redact: Redact) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_deep(v, redact) for v in value]
    if isinstance(value, dict):
        return {k: _redact_deep(v, redact) for k, v in value.items()}
    return value


def filter_event(event: Event, viewer: Viewer, redact: Redact) -> dict[str, Any] | None:
    """The event as this viewer may see it, or None."""

    if viewer.admin:
        return event.to_dict()
    fields = _PUBLIC_FIELDS.get(event.type)
    if fields is None:
        return None
    owner = event.payload.get("user_id")
    if owner is not None and owner != viewer.user_id:
        return None  # about someone else
    payload = {k: event.payload[k] for k in fields if k in event.payload and k != "user_id"}
    data = event.to_dict()
    data["payload"] = _redact_deep(payload, redact)
    return data


class BrainView:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    async def snapshot(self, viewer: Viewer, redact: Redact) -> dict[str, Any]:
        rt = self.runtime
        c = rt.c
        now = c.clock.now()
        scope = viewer.scope
        sm = c.mind.self_model
        people = {u.id: u for u in await c.db.users.list(limit=10_000)}
        signals = await c.interoception.sense(people)
        feelings = await c.emotions.current(scope)
        beliefs = await c.selfhood.beliefs()
        pending = await c.selfhood.pending()
        activities = await c.db.activities.recent(limit=40, scope=scope)
        memories = await c.db.memories.list(
            kinds=[MemoryKind.THOUGHT, MemoryKind.REFLECTION],
            limit=25,
            scope=Scope.system() if viewer.admin else Scope.for_user(viewer.user_id),
        )
        goals = await c.goals.active(scope)
        mem_stats = await c.memory.stats()
        body = c.world.body
        data: dict[str, Any] = {
            "name": sm.name,
            "now": now.isoformat(),
            "age_s": (now - sm.first_boot_at).total_seconds() if sm.first_boot_at else 0,
            "boot_count": sm.boot_count,
            "paused": c.kill_switch.engaged,
            "mood": redact(c.emotions.mood),
            "feelings": [
                {
                    **f.public(show_cause=viewer.admin or f.user_id in (None, viewer.user_id)),
                    "cause": redact(f.cause)
                    if (viewer.admin or f.user_id in (None, viewer.user_id))
                    else None,
                }
                for f in feelings
            ],
            "affect": c.mind.affect.snapshot(),
            "signals": signals.as_dict(viewer.user_id, admin=viewer.admin),
            "beliefs": [
                {
                    "aspect": b.aspect.replace("_", " "),
                    "value": redact(b.value),
                    "reasoning": redact(b.reasoning),
                    "confidence": round(b.confidence, 2),
                    "since": b.formed_at.isoformat(),
                }
                for b in beliefs
            ],
            "considering": [
                {"aspect": p.aspect.replace("_", " "), "value": redact(p.value)} for p in pending
            ]
            if viewer.admin or viewer.user_id
            else len(pending),
            "story": redact(sm.narrative),
            "activities": [
                {
                    "id": a.id,
                    "kind": a.kind,
                    "intent": redact(a.intent),
                    "reason": redact(a.reason),
                    "outcome": redact(a.outcome),
                    "status": a.status.value,
                    "private": a.user_id is not None,
                    "at": a.started_at.isoformat(),
                }
                for a in activities
            ],
            "thoughts": [
                {
                    "kind": m.kind.value,
                    "content": redact(m.content),
                    "private": m.user_id is not None,
                    "at": m.created_at.isoformat(),
                }
                for m in memories
            ],
            "goals": [
                {
                    "id": g.id,
                    "description": redact(g.description),
                    "origin": g.origin.value,
                    "progress": round(g.progress, 2),
                    "private": g.user_id is not None,
                }
                for g in goals
            ],
            "memory": {
                "total": sum(mem_stats["active_by_kind"].values()),
                "by_kind": mem_stats["active_by_kind"],
            },
            "body": {"action": body.action, "thought": redact(body.thought)},
            "viewer": {"admin": viewer.admin, "signed_in": viewer.user_id is not None},
        }
        if viewer.admin:
            relationships = []
            for rel in await c.db.relationships.list(limit=200):
                user = people.get(rel.user_id)
                relationships.append(
                    {
                        "user": user.public() if user else {"id": rel.user_id},
                        **c.relationships.public(rel),
                    }
                )
            data["people"] = relationships
            data["awareness"] = [
                {"source": p.source, "kind": p.kind, "content": p.content}
                for p in c.mind.workspace.conscious
            ]
            data["running_tasks"] = list(rt._running)
            data["tasks"] = await c.db.tasks.count_by_status()
            data["pending_approvals"] = len(await c.approvals.pending())
            data["next_beat_in_s"] = max(0.0, rt._next_beat - c.clock.monotonic())
            data["models"] = await c.router.health()
        return data

    async def memories(
        self, viewer: Viewer, redact: Redact, query: str, *, k: int = 12
    ) -> list[dict[str, Any]]:
        hits = await self.runtime.c.memory.recall(
            query, k=k, touch=False, scope=viewer.scope, min_similarity=0.0
        )
        return [
            {
                "id": h.record.id,
                "kind": h.record.kind.value,
                "content": redact(h.record.content),
                "relevance": round(h.similarity, 3),
                "importance": round(h.record.importance, 2),
                "private": h.record.user_id is not None,
                "at": h.record.created_at.isoformat(),
            }
            for h in hits
        ]

    async def recent_events(
        self, viewer: Viewer, redact: Redact, limit: int = 30
    ) -> list[dict[str, Any]]:
        bus = self.runtime.c.bus
        patterns = ("*",) if viewer.admin else PUBLIC_PATTERNS
        seen: dict[str, Event] = {}
        for pattern in patterns:
            for event in bus.recent(pattern, limit=limit):
                seen[event.id] = event
        ordered = sorted(seen.values(), key=lambda e: e.ts)[-limit:]
        out = []
        for event in ordered:
            item = filter_event(event, viewer, redact)
            if item is not None:
                out.append(item)
        return out

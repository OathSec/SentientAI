"""Idle cognition: private thoughts, insights and self-initiated goals.

When nothing demands attention and the drives allow it (enough energy,
curiosity high), the agent reflects on recent experience. Thoughts are stored
as low-importance memories that fade unless recalled; insights are kept as
reflections; a proposed goal becomes an intrinsic goal, bounded in number and
executed under the stricter autonomous approval policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentientai import prompts
from sentientai.cognition.affect import AffectState
from sentientai.cognition.goals import GoalManager
from sentientai.core.jsonparse import extract_object
from sentientai.memory.system import MemorySystem
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.storage.records import GoalOrigin, GoalRecord, MemoryKind, MemoryRecord


@dataclass
class ThoughtResult:
    thought: str
    insight: MemoryRecord | None = None
    goal: GoalRecord | None = None


class InnerLife:
    def __init__(self, router: ModelRouter, memory: MemorySystem, goals: GoalManager) -> None:
        self.router = router
        self.memory = memory
        self.goals = goals

    async def think(self, affect: AffectState) -> ThoughtResult | None:
        episodes = await self.memory.timeline(limit=6)
        active = await self.goals.active()
        prompt = prompts.IDLE_THOUGHT.format(
            affect=affect.describe(),
            goals=GoalManager.render(active),
            episodes="\n".join(f"- {e.content[:300]}" for e in episodes)
            or "- (no experiences yet)",
        )
        response = await self.router.chat(
            ChatRequest(
                messages=[Message.user(prompt)], temperature=0.7, json_mode=True, max_tokens=500
            ),
            kind=TaskKind.REFLECT,
        )
        data = extract_object(response.message.text)
        if not data or not str(data.get("thought", "")).strip():
            return None
        thought = str(data["thought"]).strip()
        await self.memory.remember(
            thought, MemoryKind.THOUGHT, importance=0.15, source="inner_life", dedupe=True
        )
        result = ThoughtResult(thought=thought)
        insight = str(data.get("insight") or "").strip()
        if insight:
            result.insight = await self.memory.remember(
                insight, MemoryKind.REFLECTION, importance=0.6, source="inner_life"
            )
            affect.on_learning(0.5)
        proposed = data.get("proposed_goal")
        if isinstance(proposed, dict) and str(proposed.get("description", "")).strip():
            try:
                priority = float(proposed.get("priority", 0.4))
            except (TypeError, ValueError):
                priority = 0.4
            result.goal = await self.goals.add(
                str(proposed["description"]),
                origin=GoalOrigin.INTRINSIC,
                priority=min(priority, 0.6),
            )
        return result

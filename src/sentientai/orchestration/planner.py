"""Lightweight task planning for complex or high-stakes requests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from sentientai import prompts
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.errors import ModelError
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind

log = get_logger(__name__)

_MULTI_STEP_HINTS = (" then ", " and then ", "step", "first", "after that", "finally", "1.", "2.")


class Plan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=12)
    needs_tools: bool = True
    complexity: Literal["low", "medium", "high"] = "medium"

    def render(self) -> str:
        return "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1))


class Planner:
    def __init__(self, router: ModelRouter, *, min_chars: int = 280) -> None:
        self.router = router
        self.min_chars = min_chars

    def should_plan(self, goal: str, *, cautious: bool = False) -> bool:
        text = goal.lower()
        return (
            cautious
            or len(goal) >= self.min_chars
            or sum(h in text for h in _MULTI_STEP_HINTS) >= 2
        )

    async def plan(self, goal: str, context: str = "") -> Plan | None:
        try:
            response = await self.router.chat(
                ChatRequest(
                    messages=[
                        Message.user(
                            prompts.PLAN_TASK.format(
                                goal=goal[:6000], context=context[:4000] or "(none)"
                            )
                        )
                    ],
                    temperature=0.2,
                    json_mode=True,
                    max_tokens=600,
                ),
                kind=TaskKind.PLAN,
            )
        except ModelError as exc:
            log.warning("planning failed; continuing without a plan", extra={"error": str(exc)})
            return None
        data = extract_object(response.message.text)
        if not data:
            return None
        try:
            return Plan.model_validate(data)
        except ValidationError:
            return None

"""Metacognition: knowing how much to trust your own answer.

Confidence blends a model self-report with hard evidence signals (tool
failures, how close the run came to its limits, whether memory had anything
relevant). Low confidence on a fast-tier answer triggers escalation to the
heavy tier; novelty (nothing relevant in memory) feeds curiosity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentientai import prompts
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.errors import ModelError
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, ModelTier, TaskKind
from sentientai.storage.records import ScoredMemory

log = get_logger(__name__)


@dataclass
class Assessment:
    confidence: float
    heuristic: float
    model_confidence: float | None
    gaps: list[str] = field(default_factory=list)


class Metacognition:
    def __init__(
        self, router: ModelRouter, *, escalate_below: float = 0.45, use_model: bool = True
    ) -> None:
        self.router = router
        self.escalate_below = escalate_below
        self.use_model = use_model

    @staticmethod
    def novelty(recalled: list[ScoredMemory]) -> float:
        """1.0 when nothing in memory relates to the topic, ~0 when it is familiar."""

        if not recalled:
            return 1.0
        return max(0.0, 1.0 - max(m.similarity for m in recalled))

    @staticmethod
    def heuristic(
        *, tool_calls: int, tool_failures: int, pressure: float, memory_coverage: float
    ) -> float:
        score = 0.75
        if tool_calls:
            score -= 0.35 * (tool_failures / tool_calls)
        score -= 0.25 * max(0.0, pressure - 0.5) * 2
        score += 0.1 * (memory_coverage - 0.5)
        return max(0.05, min(0.95, score))

    async def assess(
        self,
        *,
        goal: str,
        answer: str,
        evidence: str,
        tool_calls: int,
        tool_failures: int,
        pressure: float,
        memory_coverage: float,
    ) -> Assessment:
        heuristic = self.heuristic(
            tool_calls=tool_calls,
            tool_failures=tool_failures,
            pressure=pressure,
            memory_coverage=memory_coverage,
        )
        if not self.use_model or not answer.strip():
            return Assessment(heuristic if answer.strip() else 0.05, heuristic, None)
        try:
            response = await self.router.chat(
                ChatRequest(
                    messages=[
                        Message.user(
                            prompts.ASSESS_ANSWER.format(
                                goal=goal[:2000], answer=answer[:4000], evidence=evidence[:2000]
                            )
                        )
                    ],
                    temperature=0.0,
                    json_mode=True,
                    max_tokens=300,
                ),
                kind=TaskKind.ASSESS,
            )
            data = extract_object(response.message.text) or {}
            model_conf = float(data["confidence"])
            model_conf = max(0.0, min(1.0, model_conf))
            gaps = [str(g) for g in data.get("gaps", []) if str(g).strip()][:5]
        except (ModelError, KeyError, TypeError, ValueError) as exc:
            log.debug("self-assessment unavailable", extra={"error": str(exc)})
            return Assessment(heuristic, heuristic, None)
        return Assessment(round(0.6 * model_conf + 0.4 * heuristic, 3), heuristic, model_conf, gaps)

    def should_escalate(self, assessment: Assessment, tier: ModelTier | None) -> bool:
        return tier is ModelTier.FAST and assessment.confidence < self.escalate_below

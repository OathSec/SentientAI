"""Post-task reflection: turn experience into memory, lessons and skills.

After every task the reflector records an episode, extracts durable facts,
distils a reusable procedure from successful multi-step work, updates the
reliability of procedures that were used, and periodically rewrites the
agent's autobiographical narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from sentientai import prompts
from sentientai.cognition.mind import Mind
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.errors import ModelError
from sentientai.memory.consolidation import valid_fact
from sentientai.memory.system import MemorySystem
from sentientai.memory.types import Episode, Outcome, Procedure
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.orchestration.agent import TaskOutcome
from sentientai.storage.records import MemoryKind, MemoryRecord, TaskRecord, TaskStatus

log = get_logger(__name__)

_NARRATIVE_PROMPT = """\
Rewrite your short autobiography (first person, at most 120 words) so it reflects who you
are becoming. Keep what still matters from the previous version.

Previous version: {narrative}
Recent experiences:
{episodes}"""


class _Fact(BaseModel):
    statement: str
    subject_key: str | None = None
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)


class _Proc(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    when_to_use: str = ""
    steps: list[str] = Field(min_length=1, max_length=12)


class _Reflection(BaseModel):
    summary: str = ""
    lessons: list[str] = Field(default_factory=list)
    facts: list[Any] = Field(default_factory=list)
    procedure: _Proc | None = None
    self_assessment: float | None = Field(default=None, ge=0, le=1)

    @classmethod
    def lenient(cls, data: dict[str, Any]) -> _Reflection:
        """Validate field by field so one malformed field doesn't discard the rest."""

        clean: dict[str, Any] = {}
        for name in cls.model_fields:
            if name in data:
                try:
                    cls.model_validate({name: data[name]})
                    clean[name] = data[name]
                except ValidationError:
                    continue
        return cls.model_validate(clean)

    def valid_facts(self) -> list[_Fact]:
        facts = []
        for item in self.facts:
            try:
                facts.append(_Fact.model_validate(item))
            except ValidationError:
                continue
        return facts


@dataclass
class ReflectionResult:
    episode: MemoryRecord
    lessons: list[str] = field(default_factory=list)
    facts: list[MemoryRecord] = field(default_factory=list)
    procedure: MemoryRecord | None = None
    self_assessment: float | None = None


def _outcome_label(outcome: TaskOutcome) -> Outcome:
    if outcome.status is TaskStatus.SUCCEEDED:
        return "success" if outcome.confidence >= 0.5 else "partial"
    if outcome.status is TaskStatus.CANCELLED:
        return "aborted"
    return "partial" if outcome.answer else "failure"


class Reflector:
    def __init__(
        self,
        router: ModelRouter,
        memory: MemorySystem,
        mind: Mind,
        *,
        min_fact_confidence: float = 0.6,
        narrative_every: int = 10,
    ) -> None:
        self.router = router
        self.memory = memory
        self.mind = mind
        self.min_fact_confidence = min_fact_confidence
        self.narrative_every = narrative_every

    async def reflect(self, task: TaskRecord, outcome: TaskOutcome) -> ReflectionResult:
        label = _outcome_label(outcome)
        data = _Reflection()
        try:
            response = await self.router.chat(
                ChatRequest(
                    messages=[
                        Message.user(
                            prompts.REFLECT_ON_TASK.format(
                                goal=task.goal[:3000],
                                outcome=label,
                                result=(outcome.answer or outcome.error or "")[:3000],
                                steps=outcome.steps,
                                trace=outcome.trace_text(),
                            )
                        )
                    ],
                    temperature=0.2,
                    json_mode=True,
                    max_tokens=900,
                ),
                kind=TaskKind.REFLECT,
            )
            data = _Reflection.lenient(extract_object(response.message.text) or {})
        except ModelError as exc:
            log.warning(
                "reflection model unavailable; recording a plain episode", extra={"error": str(exc)}
            )

        summary = (
            data.summary.strip()
            or f"{label}: {(outcome.answer or outcome.error or 'no result')[:300]}"
        )
        lessons = [x.strip() for x in data.lessons if x.strip()][:5]
        if outcome.loop_break:
            lessons.append(
                f"A run was stopped by the {outcome.loop_break} limit; plan more efficiently."
            )
        valence = {"success": 0.6, "partial": 0.0, "failure": -0.6, "aborted": -0.3}[label]
        episode = Episode(
            task_id=task.id,
            goal=task.goal[:1000],
            outcome=label,
            summary=summary,
            steps=outcome.steps,
            tools_used=[t.name for t in outcome.tools],
            duration_s=outcome.duration_s,
            tokens=outcome.tokens,
            lessons=lessons,
            valence=valence,
            source=task.source.value,
            started_at=task.started_at or task.created_at,
            ended_at=self.memory.clock.now(),
        )
        result = ReflectionResult(
            # Experiences with a person are part of that person's compartment.
            episode=await self.memory.record_episode(episode, user_id=task.user_id),
            lessons=lessons,
            self_assessment=data.self_assessment,
        )
        for fact in data.valid_facts():
            key = fact.subject_key.strip().lower() if fact.subject_key else None
            if valid_fact(fact.statement, key, fact.confidence, self.min_fact_confidence):
                result.facts.append(
                    await self.memory.remember(
                        fact.statement.strip(),
                        MemoryKind.FACT,
                        importance=fact.importance,
                        confidence=fact.confidence,
                        subject_key=key,
                        source=f"reflection:{task.id}",
                        user_id=task.user_id,
                    )
                )
        if result.facts:
            self.mind.affect.on_learning(min(1.0, len(result.facts) / 3))

        successful_tool_work = label == "success" and len(outcome.tools) >= 2
        if successful_tool_work and data.procedure is not None:
            procedure = Procedure(
                name=data.procedure.name,
                when_to_use=data.procedure.when_to_use or task.goal[:200],
                steps=data.procedure.steps,
                source_tasks=[task.id],
            )
            result.procedure = await self.memory.learn_procedure(procedure, success=True)
        for proc_id in outcome.procedures_used:
            if result.procedure is not None and proc_id == result.procedure.id:
                continue
            await self.memory.record_procedure_outcome(proc_id, success=label == "success")

        self_model = self.mind.self_model
        if outcome.succeeded:
            self_model.tasks_succeeded += 1
        elif outcome.status is TaskStatus.FAILED:
            self_model.tasks_failed += 1
        if (self_model.tasks_succeeded + self_model.tasks_failed) % self.narrative_every == 0:
            await self.update_narrative()
        return result

    async def update_narrative(self) -> str:
        episodes = await self.memory.timeline(limit=8)
        try:
            response = await self.router.chat(
                ChatRequest(
                    messages=[
                        Message.user(
                            _NARRATIVE_PROMPT.format(
                                narrative=self.mind.self_model.narrative,
                                episodes="\n".join(f"- {e.content[:240]}" for e in episodes)
                                or "- none",
                            )
                        )
                    ],
                    temperature=0.5,
                    max_tokens=300,
                ),
                kind=TaskKind.REFLECT,
            )
        except ModelError:
            return self.mind.self_model.narrative
        text = response.message.text.strip()
        if 40 <= len(text) <= 1500:
            self.mind.self_model.narrative = text
        return self.mind.self_model.narrative

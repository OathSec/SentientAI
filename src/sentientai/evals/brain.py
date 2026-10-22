"""A scripted "brain" for tests, evals and offline demos: routes each request by its prompt."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from sentientai.models.scripted import ScriptedLLM
from sentientai.models.types import ChatRequest, Message, ToolCall


def call(name: str, **arguments: Any) -> Message:
    return Message.assistant(None, [ToolCall(name=name, arguments=arguments)])


class Brain:
    """Holds per-purpose scripts. Agent-loop turns are consumed in order."""

    def __init__(
        self,
        agent: list[Message | str | Exception] | None = None,
        *,
        confidence: float = 0.9,
        reflection: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        thought: dict[str, Any] | None = None,
        default_answer: str = "Done.",
    ) -> None:
        self.agent = list(agent or [])
        self.confidence = confidence
        self.reflection = reflection or {
            "summary": "Completed the task.",
            "lessons": [],
            "facts": [],
        }
        self.plan = plan or {"steps": ["inspect", "answer"], "complexity": "low"}
        self.thought = thought or {
            "thought": "Nothing pressing.",
            "insight": None,
            "proposed_goal": None,
        }
        self.default_answer = default_answer
        self.calls: dict[str, int] = {}
        self.agent_requests: list[ChatRequest] = []

    def _count(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def __call__(self, request: ChatRequest) -> Message | str | Exception:
        prompt = request.messages[-1].text if request.messages else ""
        if "Rate how likely this answer is correct" in prompt:
            self._count("assess")
            return json.dumps({"confidence": self.confidence, "gaps": []})
        if "You just finished a task" in prompt:
            self._count("reflect")
            return json.dumps(self.reflection)
        if "Break the task into a short" in prompt:
            self._count("plan")
            return json.dumps(self.plan)
        if "You are idle." in prompt:
            self._count("think")
            return json.dumps(self.thought)
        if "Rewrite your short autobiography" in prompt:
            self._count("narrative")
            return (
                "I am a young cognitive system learning to help my operator with care and honesty."
            )
        if "Summarize this conversation segment" in prompt:
            self._count("summarize")
            return "Summary of earlier progress."
        if "Extract durable facts" in prompt:
            self._count("extract")
            return json.dumps({"facts": []})
        if "You must stop now" in prompt:
            self._count("salvage")
            return "Partial answer: I was stopped before finishing."
        self._count("agent")
        self.agent_requests.append(request)
        if self.agent:
            return self.agent.pop(0)
        return self.default_answer

    def llm(self, name: str = "brain") -> ScriptedLLM:
        responder: Callable[[ChatRequest], Message | str | Exception] = self
        return ScriptedLLM(responder, name=name)

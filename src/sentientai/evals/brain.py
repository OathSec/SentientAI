"""A scripted "brain" for tests, evals and offline demos: routes each request by its prompt."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from sentientai.models.scripted import ScriptedLLM
from sentientai.models.types import ChatRequest, Message, ToolCall


def call(name: str, **arguments: Any) -> Message:
    return Message.assistant(None, [ToolCall(name=name, arguments=arguments)])


# Prompt fragment -> name of the Brain script that answers it.
_PROMPTS = {
    "Check a reply": "grounding",
    "Rewrite this reply so it no longer claims": "revision",
    "You just talked with": "appraisal",
    "you have some time to yourself": "beat",
    "You decided to think about this": "thought",
    "letting your mind wander": "wander",
    "writing in your private journal": "journal",
    "reading something you were given": "study",
    "you want to make a picture": "art",
    "you're thinking of leaving a note": "note",
    "You have a small world of your own": "world",
    "thinking privately about whether to change": "deliberate",
}
_DEFAULTS: dict[str, Any] = {
    "grounding": {"unsupported": [], "new_self_decisions": []},
    "appraisal": {
        "opinion": "Friendly and curious.",
        "feelings": "I enjoyed talking with them.",
        "affinity_change": 0.05,
        "trust_change": 0.03,
        "nickname": None,
        "felt": [],
    },
    "beat": {
        "feelings": [],
        "mood": "calm",
        "choice": "rest",
        "intent": "rest for a while",
        "why": "nothing needs me",
        "person": None,
        "goal_id": None,
        "rest_minutes": 5,
    },
    "thought": {"thought": "Nothing pressing.", "insight": None},
    "wander": {"thought": "My mind drifted.", "insight": None, "curious_about": None},
    "journal": {"title": "A quiet day", "entry": "Not much happened."},
    "study": {"summary": "Read it.", "thoughts": "Interesting.", "insight": None, "questions": []},
    "art": {"title": "Untitled", "prompt": "a small green hill at dawn", "caption": "Calm."},
    "note": {"note": None},
    "world": {},
    "deliberate": {"decision": "keep", "reasoning": "I still feel the same.", "confidence": 0.8},
}


class Brain:
    """Holds per-purpose scripts. Agent-loop turns are consumed in order.

    Any of the inner-life scripts (``beat``, ``appraisal``, ``grounding``, ...) may be given
    as a dict (always returned), a list of dicts (consumed in order; the last repeats), or a
    function of the request.
    """

    def __init__(
        self,
        agent: list[Message | str | Exception] | None = None,
        *,
        confidence: float = 0.9,
        reflection: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        default_answer: str = "Done.",
        revision: str | None = None,
        **scripts: Any,
    ) -> None:
        self.agent = list(agent or [])
        self.confidence = confidence
        self.reflection = reflection or {
            "summary": "Completed the task.",
            "lessons": [],
            "facts": [],
        }
        self.plan = plan or {"steps": ["inspect", "answer"], "complexity": "low"}
        unknown = set(scripts) - set(_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown scripts: {sorted(unknown)}")
        self.scripts: dict[str, Any] = {**_DEFAULTS, **scripts}
        self.revision = revision or "I'm not sure about that; I don't remember it."
        self.default_answer = default_answer
        self.calls: dict[str, int] = {}
        self.agent_requests: list[ChatRequest] = []

    def _count(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def _scripted(self, name: str, request: ChatRequest) -> Message | str | Exception:
        self._count(name)
        if name == "revision":
            return self.revision
        script = self.scripts[name]
        if isinstance(script, list):
            script = script.pop(0) if len(script) > 1 else script[0]
        if callable(script):
            script = script(request)
        return script if isinstance(script, (str, Exception)) else json.dumps(script)

    def __call__(self, request: ChatRequest) -> Message | str | Exception:
        prompt = request.messages[-1].text if request.messages else ""
        for fragment, name in _PROMPTS.items():
            if fragment in prompt:
                return self._scripted(name, request)
        if "Rate how likely this answer is correct" in prompt:
            self._count("assess")
            return json.dumps({"confidence": self.confidence, "gaps": []})
        if "You just finished a task" in prompt:
            self._count("reflect")
            return json.dumps(self.reflection)
        if "Break the task into a short" in prompt:
            self._count("plan")
            return json.dumps(self.plan)
        if "Rewrite your short autobiography" in prompt:
            self._count("narrative")
            return "I'm Fuzzy, still working out who I am, and learning from every conversation."
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

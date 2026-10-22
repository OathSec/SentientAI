"""Context framing and compression.

:class:`ContextBuilder` assembles the prompt for a task within a token budget:
identity and continuity, inner state, current awareness, goals, operating
rules, relevant memories and procedures, the plan, and conversation history.

:class:`ContextCompressor` keeps long tool loops inside the model's context
window by shrinking stale tool output and then summarizing the middle of the
conversation, without ever separating a tool result from the call that
produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sentientai import prompts
from sentientai.core.logging import get_logger
from sentientai.errors import ModelError
from sentientai.memory.types import KIND_LABELS
from sentientai.memory.working import Turn
from sentientai.models.router import ModelRouter
from sentientai.models.types import (
    ChatRequest,
    Message,
    Role,
    TaskKind,
    estimate_messages_tokens,
    estimate_tokens,
)
from sentientai.storage.records import ScoredMemory

log = get_logger(__name__)

OPERATING_RULES = """\
## How you work
- Only talk about past events, earlier conversations, or things about a person that appear
  in your memories or this conversation. If you don't remember something, say so plainly; you
  can look with memory_recall or conversation_search. Never invent a memory.
- Gather evidence with tools instead of guessing, and never claim to have done something you
  didn't do.
- Some actions need approval from the people who run you. If one is denied, adapt or explain.
- Store things worth keeping with memory_remember. Your experiences are recorded automatically.
- When you're done, reply directly (no tool call).
Current time: {now} UTC."""


@dataclass
class FramingInputs:
    goal: str
    identity: str
    continuity: str
    inner_state: str
    awareness: str
    goals: str
    memories: list[ScoredMemory]
    procedures: list[ScoredMemory]
    plan: str | None
    history: list[Turn]
    now: datetime
    beliefs: str = ""
    person: str | None = None


class ContextBuilder:
    def __init__(self, *, memory_share: float = 0.25, history_share: float = 0.3) -> None:
        self.memory_share = memory_share
        self.history_share = history_share

    @staticmethod
    def _memories_section(items: list[ScoredMemory], budget: int, title: str) -> str:
        lines: list[str] = []
        used = 0
        for item in items:
            line = (
                f"- [{item.record.id}] ({KIND_LABELS[item.record.kind]}, "
                f"relevance {item.similarity:.2f}) "
                f"{item.record.content.strip()}"
            )
            cost = estimate_tokens(line)
            if used + cost > budget:
                if not lines and budget > 50:  # always show at least part of the best memory
                    lines.append(line[: budget * 4])
                break
            lines.append(line)
            used += cost
        return f"## {title}\n" + "\n".join(lines) if lines else ""

    def build(self, inputs: FramingInputs, *, window: int, reserve_output: int) -> list[Message]:
        budget = max(1024, window - reserve_output)
        summary_turns = [t for t in inputs.history if t.metadata.get("summary")]
        dialogue = [
            t
            for t in inputs.history
            if not t.metadata.get("summary") and t.role in {"user", "assistant"}
        ]

        sections = [inputs.identity]
        if inputs.beliefs:
            sections.append(
                f"## Who you are (your own decisions)\n{prompts.SELF_BELIEFS_GUIDE}\n"
                f"{inputs.beliefs}"
            )
        if inputs.person:
            sections.append(f"## The person you're talking with\n{inputs.person}")
        sections += [
            f"## Continuity\n{inputs.continuity}",
            f"## How you feel\n{inputs.inner_state}",
            f"## Current awareness\n{inputs.awareness}",
            f"## Goals\n{inputs.goals}",
            OPERATING_RULES.format(now=inputs.now.strftime("%Y-%m-%d %H:%M")),
        ]
        if summary_turns:
            sections.append(
                "## Earlier in this conversation\n" + "\n".join(t.content for t in summary_turns)
            )
        memory_budget = int(budget * self.memory_share)
        procedure_section = self._memories_section(
            inputs.procedures, memory_budget // 3, "Procedures that worked before"
        )
        memory_section = self._memories_section(
            inputs.memories, memory_budget - estimate_tokens(procedure_section), "Relevant memories"
        )
        sections += [s for s in (procedure_section, memory_section) if s]
        if inputs.plan:
            sections.append(f"## Plan\n{inputs.plan}")
        system = Message.system("\n\n".join(sections))

        history_budget = int(budget * self.history_share)
        history: list[Message] = []
        used = 0
        for turn in reversed(dialogue):
            cost = estimate_tokens(turn.content) + 4
            if used + cost > history_budget:
                break
            history.insert(0, Message(role=Role(turn.role), content=turn.content))
            used += cost
        # Chat templates expect alternation starting with the user.
        while history and history[0].role is not Role.USER:
            history.pop(0)
        merged: list[Message] = []
        for msg in history:
            if merged and merged[-1].role is msg.role:
                merged[-1] = Message(role=msg.role, content=f"{merged[-1].text}\n\n{msg.text}")
            else:
                merged.append(msg)
        if merged and merged[-1].role is Role.USER:
            merged.append(Message.assistant("(no reply recorded)"))
        return [system, *merged, Message.user(inputs.goal)]


class ContextCompressor:
    def __init__(
        self,
        router: ModelRouter,
        *,
        keep_recent: int = 6,
        threshold: float = 0.8,
        stale_tool_chars: int = 800,
    ) -> None:
        self.router = router
        self.keep_recent = keep_recent
        self.threshold = threshold
        self.stale_tool_chars = stale_tool_chars

    @staticmethod
    def _safe_tail_start(messages: list[Message], start: int) -> int:
        """Move the cut so the kept tail never begins with an orphaned tool result."""

        while start < len(messages) and messages[start].role is Role.TOOL:
            start -= 1
        return max(start, 2)

    async def fit(
        self, messages: list[Message], *, window: int, reserve_output: int
    ) -> list[Message]:
        limit = int((window - reserve_output) * self.threshold)
        if estimate_messages_tokens(messages) <= limit or len(messages) <= self.keep_recent + 2:
            return messages

        # Stage 1: shrink stale tool output.
        cutoff = len(messages) - self.keep_recent
        shrunk = []
        for i, msg in enumerate(messages):
            if i < cutoff and msg.role is Role.TOOL and len(msg.text) > self.stale_tool_chars:
                msg = msg.model_copy(
                    update={
                        "content": msg.text[: self.stale_tool_chars] + "\n[... older output elided]"
                    }
                )
            shrunk.append(msg)
        if estimate_messages_tokens(shrunk) <= limit:
            return shrunk

        # Stage 2: summarize the middle (keep system prompt, original request and the recent tail).
        tail_start = self._safe_tail_start(shrunk, len(shrunk) - self.keep_recent)
        head, middle, tail = shrunk[:2], shrunk[2:tail_start], shrunk[tail_start:]
        if not middle:
            return shrunk
        transcript = "\n".join(
            f"{m.role.value}: {m.text[:1500]}"
            + (f" [called {', '.join(c.name for c in m.tool_calls)}]" if m.tool_calls else "")
            for m in middle
        )
        try:
            response = await self.router.chat(
                ChatRequest(
                    messages=[
                        Message.user(
                            prompts.SUMMARIZE_CONVERSATION.format(
                                max_words=250, previous="", transcript=transcript[-30000:]
                            )
                        )
                    ],
                    temperature=0.1,
                ),
                kind=TaskKind.SUMMARIZE,
            )
            summary = response.message.text.strip()
        except ModelError as exc:
            log.warning("context summarization failed; dropping middle", extra={"error": str(exc)})
            summary = f"({len(middle)} earlier messages were dropped to fit the context window.)"
        note = Message.user(f"[Progress so far, summarized to save context]\n{summary}")
        bridge = Message.assistant("Understood. Continuing from that progress.")
        compressed = [*head, note, bridge, *tail]
        log.info("context compressed", extra={"before": len(messages), "after": len(compressed)})
        return compressed

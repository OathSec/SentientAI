"""Background memory consolidation ("sleep").

* Compresses long working-memory sessions into summaries (context compression
  across turns).
* Extracts durable facts from recent conversation, rejecting junk.
* Applies decay and archives memories that have faded.

Runs periodically from the runtime and on demand (``sentientai memory consolidate``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sentientai import prompts
from sentientai.config import MemoryConfig
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.memory.system import MemorySystem
from sentientai.memory.working import Turn
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.storage.records import MemoryKind, MemoryRecord

log = get_logger(__name__)

_SUBJECT_KEY = re.compile(r"^[a-z0-9][a-z0-9_.:\-]{1,120}$")
_JUNK = re.compile(
    r"^(the (user|operator) (asked|said|wants to know)|the assistant|i (think|believe)|"
    r"this conversation|hello|hi\b|thanks)",
    re.IGNORECASE,
)
_CHECKPOINT_KEY = "consolidation.last_run"


@dataclass
class ConsolidationReport:
    sessions_compressed: int = 0
    facts_stored: list[str] = field(default_factory=list)
    facts_rejected: int = 0
    archived: int = 0
    errors: list[str] = field(default_factory=list)


def valid_fact(
    statement: str, subject_key: str | None, confidence: float, min_confidence: float
) -> bool:
    statement = statement.strip()
    if not 8 <= len(statement) <= 500:
        return False
    if confidence < min_confidence:
        return False
    if subject_key is not None and not _SUBJECT_KEY.match(subject_key):
        return False
    return not _JUNK.match(statement)


def _clamp(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


class MemoryConsolidator:
    def __init__(self, memory: MemorySystem, router: ModelRouter, config: MemoryConfig) -> None:
        self.memory = memory
        self.router = router
        self.config = config

    async def _complete(self, prompt: str, kind: TaskKind, *, json_mode: bool = False) -> str:
        response = await self.router.chat(
            ChatRequest(messages=[Message.user(prompt)], temperature=0.1, json_mode=json_mode),
            kind=kind,
        )
        return response.message.text

    async def compress_session(self, session_id: str) -> bool:
        turns = await self.memory.working.turns(session_id)
        if len(turns) <= self.config.summarize_after_turns:
            return False
        keep = max(2, self.config.summarize_after_turns // 2)
        older, recent = turns[:-keep], turns[-keep:]
        previous = ""
        if older and older[0].metadata.get("summary"):
            previous = f"Earlier summary: {older[0].content}\n"
            older = older[1:]
        transcript = "\n".join(f"{t.role}: {t.content}" for t in older)
        summary = (
            await self._complete(
                prompts.SUMMARIZE_CONVERSATION.format(
                    max_words=180, previous=previous, transcript=transcript[-24000:]
                ),
                TaskKind.SUMMARIZE,
            )
        ).strip()
        if not summary:
            return False
        owner = next((t.metadata.get("user_id") for t in turns if t.metadata.get("user_id")), None)
        await self.memory.remember(
            summary,
            MemoryKind.SUMMARY,
            user_id=owner,
            importance=0.4,
            metadata={"session_id": session_id, "turns": len(older)},
            source="consolidation",
            dedupe=False,
        )
        await self.memory.working.replace(
            session_id,
            [Turn(role="system", content=summary, metadata={"summary": True}), *recent],
        )
        return True

    async def extract_facts(
        self, text: str, *, source: str, user_id: str | None = None
    ) -> tuple[list[MemoryRecord], int]:
        raw = await self._complete(
            prompts.EXTRACT_FACTS.format(text=text[-12000:]), TaskKind.EXTRACT, json_mode=True
        )
        data = extract_object(raw) or {}
        stored: list[MemoryRecord] = []
        rejected = 0
        for item in data.get("facts", []) if isinstance(data.get("facts"), list) else []:
            if not isinstance(item, dict):
                rejected += 1
                continue
            statement = str(item.get("statement") or "").strip()
            subject_key = item.get("subject_key")
            subject_key = str(subject_key).strip().lower() if subject_key else None
            confidence = _clamp(item.get("confidence"), 0.5)
            if not valid_fact(statement, subject_key, confidence, self.config.min_fact_confidence):
                rejected += 1
                continue
            stored.append(
                await self.memory.remember(
                    statement,
                    MemoryKind.FACT,
                    importance=_clamp(item.get("importance"), 0.5),
                    confidence=confidence,
                    subject_key=subject_key,
                    source=source,
                    user_id=user_id,
                )
            )
        return stored, rejected

    async def run(self) -> ConsolidationReport:
        report = ConsolidationReport()
        for session_id in await self.memory.working.sessions():
            try:
                if await self.compress_session(session_id):
                    report.sessions_compressed += 1
            except Exception as exc:
                log.exception("session compression failed", extra={"session_id": session_id})
                report.errors.append(f"compress {session_id}: {exc}")

        state = self.memory.db.state
        last_raw = await state.get(_CHECKPOINT_KEY)
        since = datetime.fromisoformat(last_raw) if last_raw else datetime.fromtimestamp(0, UTC)
        messages = await self.memory.db.conversations.since(since)
        # Facts from a conversation belong to that person's compartment.
        by_session: dict[tuple[str, str | None], list[str]] = defaultdict(list)
        for msg in messages:
            by_session[(msg.session_id, msg.user_id)].append(f"{msg.role}: {msg.content}")
        for (session_id, owner), lines in by_session.items():
            try:
                stored, rejected = await self.extract_facts(
                    "\n".join(lines), source=f"conversation:{session_id}", user_id=owner
                )
                report.facts_stored.extend(r.id for r in stored)
                report.facts_rejected += rejected
            except Exception as exc:
                log.exception("fact extraction failed", extra={"session_id": session_id})
                report.errors.append(f"extract {session_id}: {exc}")
        if messages and not report.errors:
            await state.set(_CHECKPOINT_KEY, messages[-1].created_at.isoformat())

        decay = await self.memory.decay()
        report.archived = len(decay.archived)
        return report

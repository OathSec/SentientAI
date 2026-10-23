"""Keeping Fuzzy from making things up about the past or about itself.

Before a conversational reply is sent, it is checked against what Fuzzy actually
knows: the memories it recalled, its recorded decisions about itself, the person
section, the conversation, and tool results. Claims about past events, about the
person, or about Fuzzy's own history and identity that aren't supported are
rewritten ("I don't remember that" beats a confident invention).

The same pass captures new self-decisions made in the reply ("my favourite colour
is green") and records them, so Fuzzy stays consistent from then on even if it
didn't use the self_define tool. A reply that contradicts an existing decision is
corrected rather than allowed to quietly change who Fuzzy is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sentientai import prompts
from sentientai.cognition.selfhood import SelfConcept
from sentientai.core.jsonparse import extract_object
from sentientai.core.logging import get_logger
from sentientai.errors import ModelError
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, TaskKind
from sentientai.safety.audit import AuditLog

log = get_logger(__name__)

_MAX_EVIDENCE_CHARS = 14_000


@dataclass
class GroundingResult:
    reply: str
    unsupported: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    new_beliefs: list[str] = field(default_factory=list)
    revised: bool = False


_REPLY_TAGS = re.compile(r"<reply>\s*(.*?)\s*(?:</reply>|$)", re.DOTALL | re.IGNORECASE)
_ECHOED_INSTRUCTIONS = re.compile(
    r"^\s*(return only the rewritten reply|write the corrected reply\b.*|"
    r"(here is )?the (corrected|rewritten) reply)[.:]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def corrected_reply(text: str) -> str:
    """The reply out of a revision, without any of the instructions echoed back."""

    found = _REPLY_TAGS.findall(text)
    body = found[-1] if found else text
    body = _ECHOED_INSTRUCTIONS.sub("", body.replace("</reply>", "").replace("<reply>", ""))
    return body.strip()


class Grounding:
    def __init__(self, router: ModelRouter, selfhood: SelfConcept, *, audit: AuditLog) -> None:
        self.router = router
        self.selfhood = selfhood
        self.audit = audit

    async def check(
        self,
        *,
        name: str,
        message: str,
        reply: str,
        evidence: str,
        source: str,
        task_id: str | None = None,
    ) -> GroundingResult:
        result = GroundingResult(reply=reply)
        if not reply.strip():
            return result
        if len(evidence) > _MAX_EVIDENCE_CHARS:
            evidence = (
                evidence[: _MAX_EVIDENCE_CHARS // 2]
                + "\n[...]\n"
                + evidence[-_MAX_EVIDENCE_CHARS // 2 :]
            )
        try:
            response = await self.router.chat(
                ChatRequest(
                    messages=[
                        Message.user(
                            prompts.GROUND_REPLY.format(
                                name=name,
                                evidence=evidence or "(nothing)",
                                message=message[:4000],
                                reply=reply[:6000],
                            )
                        )
                    ],
                    temperature=0.0,
                    json_mode=True,
                    max_tokens=2000,
                ),
                # A person is waiting on this check, so it runs on the conversation's (heavy) tier,
                # not the one doing Fuzzy's background thinking.
                kind=TaskKind.CHAT,
            )
        except ModelError as exc:
            log.warning("grounding check unavailable", extra={"error": str(exc)})
            return result
        data = extract_object(response.message.text) or {}
        unsupported = data.get("unsupported")
        if isinstance(unsupported, list):
            result.unsupported = [
                " ".join(str(c).split())[:600] for c in unsupported if str(c).strip()
            ][:8]
        decisions = data.get("new_self_decisions")
        for item in decisions if isinstance(decisions, list) else []:
            if not isinstance(item, dict) or not item.get("aspect") or not item.get("value"):
                continue
            try:
                change = await self.selfhood.define(
                    str(item["aspect"]),
                    str(item["value"]),
                    str(item.get("reasoning") or ""),
                    source=source,
                )
            except ValueError:
                continue
            belief = change.belief
            if change.status == "defined":
                result.new_beliefs.append(f"{belief.aspect} = {belief.value}")
            elif belief.value.casefold() != str(item["value"]).strip().casefold():
                result.contradictions.append(
                    f"It says its {belief.aspect.replace('_', ' ')} is {item['value']}, but it "
                    f"decided on {belief.formed_at:%Y-%m-%d} that it is {belief.value}."
                )
        problems = result.unsupported + result.contradictions
        if problems:
            try:
                revised = await self.router.chat(
                    ChatRequest(
                        messages=[
                            Message.user(
                                prompts.REVISE_REPLY.format(
                                    name=name,
                                    problems="\n".join(f"- {p}" for p in problems),
                                    reply=reply,
                                )
                            )
                        ],
                        temperature=0.3,
                    ),
                    kind=TaskKind.CHAT,
                )
                text = corrected_reply(revised.message.text)
                if text:
                    result.reply = text
                    result.revised = True
            except ModelError as exc:
                log.warning("reply revision failed", extra={"error": str(exc)})
        await self.audit.record(
            "agent.grounding",
            task_id=task_id,
            unsupported=result.unsupported,
            contradictions=result.contradictions,
            new_beliefs=result.new_beliefs,
            revised=result.revised,
        )
        return result

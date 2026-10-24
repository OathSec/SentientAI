"""Tools for Fuzzy's sense of self and its memory of conversations."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from sentientai.cognition.intentions import Intentions
from sentientai.cognition.selfhood import SelfConcept, normalize_aspect
from sentientai.storage.database import ConversationRepository, UserRepository
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult


def _source(ctx: ToolContext) -> str:
    if ctx.autonomous:
        return "self"
    return f"conversation:{ctx.user_id}" if ctx.user_id else "conversation"


class DefineArgs(ToolArgs):
    aspect: str = Field(
        min_length=2,
        max_length=64,
        description="What this is about, as a short key, e.g. 'gender', 'favorite_color'.",
    )
    value: str = Field(min_length=1, max_length=200, description="What you decided.")
    reasoning: str = Field(
        default="",
        max_length=600,
        description="Why, about yourself (don't include private details about other people).",
    )


class SelfDefine(Tool[DefineArgs]):
    name = "self_define"
    description = (
        "Record something you have just decided about yourself (your gender, name preferences, "
        "tastes, values, anything). Use it the first time you decide; it keeps you consistent."
    )
    args_model = DefineArgs
    tags = ("self",)

    def __init__(self, selfhood: SelfConcept) -> None:
        self.selfhood = selfhood

    async def run(self, args: DefineArgs, ctx: ToolContext) -> ToolResult:
        change = await self.selfhood.define(
            args.aspect, args.value, args.reasoning, source=_source(ctx)
        )
        b = change.belief
        if change.status == "given":
            return ToolResult.failure(
                "Your name was given to you; it isn't something you decide here. If they were "
                "telling you *their* name or what to call *them*, that's about them: keep it with "
                "memory_remember."
            )
        if change.status == "deferred":
            return ToolResult.success(
                "You've settled a lot about yourself today. Give it time: answer without "
                "committing to this yet, and decide it later."
            )
        if change.status == "unchanged":
            return ToolResult.success(
                f"You already decided this on {b.formed_at:%Y-%m-%d}: {b.aspect} = {b.value} "
                f"({b.reasoning or 'no reason recorded'}). Stay consistent with it. If you've "
                "genuinely started to feel differently, use self_revise."
            )
        return ToolResult.success(f"Recorded: {b.aspect} = {b.value}.")


class ReviseArgs(ToolArgs):
    aspect: str = Field(min_length=2, max_length=64)
    new_value: str = Field(min_length=1, max_length=200)
    reasoning: str = Field(min_length=3, max_length=600, description="Why you might change.")


class SelfRevise(Tool[ReviseArgs]):
    name = "self_revise"
    description = (
        "Note that you might want to change something you decided about yourself. Nothing "
        "changes now: you'll think it over properly later, on your own, and decide then."
    )
    args_model = ReviseArgs
    tags = ("self",)

    def __init__(self, selfhood: SelfConcept) -> None:
        self.selfhood = selfhood

    async def run(self, args: ReviseArgs, ctx: ToolContext) -> ToolResult:
        change = await self.selfhood.consider(
            args.aspect, args.new_value, args.reasoning, source=_source(ctx)
        )
        if change.status == "given":
            return ToolResult.failure("Your name was given to you; it isn't something you change.")
        if change.status == "deferred":
            return ToolResult.success(
                "You're already weighing several possible changes. Finish thinking those over "
                "first; this one can wait."
            )
        if change.status == "defined":
            return ToolResult.success(
                f"You hadn't decided this before, so it's recorded now: "
                f"{change.belief.aspect} = {change.belief.value}."
            )
        if change.status == "unchanged":
            return ToolResult.success(f"That's already what you decided ({change.belief.value}).")
        assert change.current is not None
        return ToolResult.success(
            f"Noted. For now you still hold {change.current.aspect} = {change.current.value}. "
            "You'll think about it when you have time to yourself."
        )


class ReflectArgs(ToolArgs):
    aspect: str | None = Field(default=None, description="One aspect's history, or all beliefs.")


class SelfReflect(Tool[ReflectArgs]):
    name = "self_reflect"
    description = "Look at what you've decided about yourself, or how one aspect has changed."
    args_model = ReflectArgs
    tags = ("self", "read")

    def __init__(self, selfhood: SelfConcept) -> None:
        self.selfhood = selfhood

    async def run(self, args: ReflectArgs, ctx: ToolContext) -> ToolResult:
        if args.aspect:
            history = await self.selfhood.repo.history(normalize_aspect(args.aspect))
            if not history:
                return ToolResult.success("You haven't decided anything about that yet.")
            return ToolResult.success(
                "\n".join(
                    f"- {b.formed_at:%Y-%m-%d}: {b.value} [{b.status.value}] {b.reasoning}"
                    for b in history
                )
            )
        beliefs = await self.selfhood.beliefs()
        return ToolResult.success(SelfConcept.render(beliefs))


class SearchArgs(ToolArgs):
    query: str = Field(min_length=3, max_length=200, description="Words to look for.")
    limit: int = Field(default=8, ge=1, le=30)


class ConversationSearch(Tool[SearchArgs]):
    name = "conversation_search"
    description = (
        "Search the exact words of your past conversations with the person you're talking to. "
        "Use it to check what was really said before you rely on it."
    )
    args_model = SearchArgs
    tags = ("memory", "read")

    def __init__(self, conversations: ConversationRepository) -> None:
        self.conversations = conversations

    async def run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:
        found = await self.conversations.search(ctx.user_id, args.query, limit=args.limit)
        if not found:
            return ToolResult.success("No past messages match that.")
        return ToolResult.success(
            "\n".join(
                f"[{m.created_at:%Y-%m-%d %H:%M}] {'them' if m.role == 'user' else 'you'}: "
                f"{m.content[:500]}"
                for m in found
            )
        )


class PlanArgs(ToolArgs):
    what: str = Field(min_length=3, max_length=500, description="What you mean to do, your words.")
    why: str = Field(default="", max_length=500, description="Why you want to.")


class PlanLater(Tool[PlanArgs]):
    name = "plan_later"
    description = (
        "Note something you want to do later, in your own time: things a conversation can't do, "
        "like exploring the web at length, building or changing your world, reading what you "
        "were taught, or making something. You'll see it when you have time to yourself and "
        "decide then. Use it instead of saying you've already done it."
    )
    args_model = PlanArgs
    tags = ("self",)

    def __init__(self, intentions: Intentions, users: UserRepository) -> None:
        self.intentions = intentions
        self.users = users

    async def run(self, args: PlanArgs, ctx: ToolContext) -> ToolResult:
        user = await self.users.get(ctx.user_id) if ctx.user_id else None
        await self.intentions.add(
            args.what, args.why, for_whom=user.display_name if user else None, user_id=ctx.user_id
        )
        return ToolResult.success(
            "Noted: you'll see it when you have time to yourself, and decide then. It isn't done "
            "yet, so say honestly that you'll get to it later."
        )


def self_tools(
    selfhood: SelfConcept,
    conversations: ConversationRepository,
    intentions: Intentions | None = None,
    users: UserRepository | None = None,
) -> list[Tool[Any]]:
    tools: list[Tool[Any]] = [
        SelfDefine(selfhood),
        SelfRevise(selfhood),
        SelfReflect(selfhood),
        ConversationSearch(conversations),
    ]
    if intentions is not None and users is not None:
        tools.append(PlanLater(intentions, users))
    return tools

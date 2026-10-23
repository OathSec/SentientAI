"""Showing people what Fuzzy found on the web: a video that plays in the chat, or a link card.

Fuzzy can only show an address it actually got back from its own web search or page fetch in
this same conversation turn, so it can't invent or be talked into showing an arbitrary link.
Only YouTube and Vimeo play inside the chat (privacy-friendly player addresses); everything
else is a card with an "Open" link that goes out to the real site.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from pydantic import Field

from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult

MAX_PER_TURN = 4
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


def canonical(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}?{parts.query}"


def remember_urls(ctx: ToolContext, text: str, *extra: str) -> None:
    """Called after a web search or fetch: these are the addresses Fuzzy may now show."""

    seen: set[str] = ctx.extras.setdefault("seen_urls", set())
    for url in [*_URL.findall(text), *extra]:
        seen.add(canonical(url.rstrip(".,;")))


def video_embed(url: str) -> tuple[str, str] | None:
    """(kind, player address) for a YouTube or Vimeo page, else None."""

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    video_id = ""
    if host in _YOUTUBE_HOSTS:
        if parts.path == "/watch":
            video_id = (parse_qs(parts.query).get("v") or [""])[0]
        elif parts.path.startswith(("/shorts/", "/embed/", "/live/")):
            video_id = parts.path.split("/")[2] if parts.path.count("/") >= 2 else ""
    elif host == "youtu.be":
        video_id = parts.path.lstrip("/").split("/")[0]
    if video_id and re.fullmatch(r"[\w-]{6,20}", video_id):
        return "youtube", f"https://www.youtube-nocookie.com/embed/{video_id}"
    if host in {"vimeo.com", "www.vimeo.com"}:
        match = re.fullmatch(r"/(\d{5,12})", parts.path)
        if match:
            return "vimeo", f"https://player.vimeo.com/video/{match.group(1)}"
    return None


class ShareArgs(ToolArgs):
    url: str = Field(min_length=10, max_length=500, description="The address to show.")
    title: str = Field(min_length=1, max_length=120, description="What it is, in a few words.")
    note: str = Field(default="", max_length=300, description="Why you're showing it.")


class ShareLink(Tool[ShareArgs]):
    name = "share_link"
    description = (
        "Show the person something you found on the web: a YouTube or Vimeo video plays right "
        "in the chat, and any other page appears as a card they can open. Only for addresses "
        "that came back from your own web_search or web_fetch in this conversation."
    )
    args_model = ShareArgs
    tags = ("web",)

    async def run(self, args: ShareArgs, ctx: ToolContext) -> ToolResult:
        parts = urlsplit(args.url.strip())
        host = (parts.hostname or "").lower()
        if parts.scheme != "https" or not host or parts.username or parts.password:
            return ToolResult.failure("Only plain https addresses can be shown.")
        try:
            ipaddress.ip_address(host)
            return ToolResult.failure("Only named websites can be shown, not raw addresses.")
        except ValueError:
            pass
        if host == "localhost" or host.endswith((".local", ".internal", ".lan")):
            return ToolResult.failure("That isn't a public website.")
        seen: set[str] = ctx.extras.get("seen_urls", set())
        if canonical(args.url) not in seen:
            return ToolResult.failure(
                "You haven't found that address yet. Use web_search or web_fetch first, then "
                "show an address from its results."
            )
        shown: list[dict[str, Any]] = ctx.extras.setdefault("embeds", [])
        if len(shown) >= MAX_PER_TURN:
            return ToolResult.failure("You've already shown several things; that's enough for now.")
        video = video_embed(args.url)
        embed: dict[str, Any] = {
            "kind": video[0] if video else "link",
            "url": args.url.strip(),
            "title": args.title.strip(),
            "note": args.note.strip(),
            "host": host.removeprefix("www."),
            **({"player": video[1]} if video else {}),
        }
        shown.append(embed)
        what = "a video that plays in the chat" if video else "a card with a link"
        return ToolResult.success(
            f"Shown to them as {what}. Mention it naturally.", data={"embed": embed}
        )

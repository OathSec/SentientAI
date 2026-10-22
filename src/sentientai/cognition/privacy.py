"""Keeping what one person shares with Fuzzy away from everyone else.

The main protection is structural: memories, feelings, goals and activities about a
person carry their user id, and queries for a conversation only see shared items
plus that person's own (see :class:`~sentientai.storage.records.Scope`).

This module is the second line of defence for text that is shared by design but
written by Fuzzy's own mind (its mood, journal, thoughts, goals, story so far): any
other person's name in it is replaced before it reaches someone else.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from sentientai.storage.database import Database
from sentientai.storage.records import UserStatus

Redact = Callable[[str], str]
_MIN_NAME = 3  # shorter names would mangle ordinary words


def _identity(text: str) -> str:
    return text


class Redactor:
    def __init__(self, db: Database, *, ttl_s: float = 30.0) -> None:
        self.db = db
        self.ttl_s = ttl_s
        # user id -> (usernames, display names and nicknames)
        self._names: dict[str, tuple[set[str], set[str]]] = {}
        self._loaded_at = -1e9

    async def _refresh(self) -> None:
        if time.monotonic() - self._loaded_at < self.ttl_s:
            return
        names: dict[str, tuple[set[str], set[str]]] = {}
        nicknames = {r.user_id: r.nickname for r in await self.db.relationships.list(limit=100_000)}
        for user in await self.db.users.list(limit=100_000):
            if user.status is UserStatus.DISABLED and user.id not in nicknames:
                continue
            spoken = {user.display_name, nicknames.get(user.id) or ""}
            names[user.id] = (
                {user.username} if len(user.username) >= _MIN_NAME else set(),
                {n.strip() for n in spoken if len(n.strip()) >= _MIN_NAME},
            )
        self._names = names
        self._loaded_at = time.monotonic()

    def invalidate(self) -> None:
        self._loaded_at = -1e9

    async def for_viewer(self, viewer: str | None, *, privileged: bool = False) -> Redact:
        """A function hiding everyone but ``viewer``. Admins and the operator see everything."""

        if privileged:
            return _identity
        await self._refresh()
        mine = {n.casefold() for group in self._names.get(viewer or "", ((), ())) for n in group}
        usernames: set[str] = set()
        spoken: set[str] = set()
        for uid, (handles, called) in self._names.items():
            if uid == viewer:
                continue
            usernames |= {n for n in handles if n.casefold() not in mine}
            # Display names are matched as they're written in prose (capitalised), so a name
            # that is also a common word ("will", "hope") doesn't eat ordinary lowercase text.
            spoken |= {
                variant
                for n in called
                if n.casefold() not in mine
                for variant in (n, n[:1].upper() + n[1:])
                if variant[:1].isupper()
            }
        if not usernames and not spoken:
            return _identity
        # One alternation, longest first, so "Alice Smith" is replaced whole rather than
        # leaving "Smith" behind after a shorter match on "alice". Usernames are handles and
        # match in any case, with or without "@".
        alternatives = sorted(
            [
                *((n, f"(?i:{re.escape(n)})") for n in usernames),
                *((n, re.escape(n)) for n in spoken),
            ],
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        pattern = re.compile(
            r"(?<![\w@])@?(?:" + "|".join(rx for _, rx in alternatives) + r")(?!\w)"
        )
        return lambda text: pattern.sub("someone", text) if text else text

    async def shared(self) -> Redact:
        """For text about to be stored where everyone's conversations can see it."""

        return await self.for_viewer(None)

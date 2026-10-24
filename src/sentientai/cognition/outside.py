"""A glimpse of the world beyond the server.

Between conversations, everything Fuzzy perceives is otherwise its own output: its thoughts,
its feelings, its memories of them. A small model fed nearly the same picture of itself every
beat gives nearly the same answer every beat, so its inner life collapses into one thought
repeated for hours (observed: 597 of 600 activities were the same "think"). A real mind is
never in that position, because the world keeps putting new things in front of it.

So each beat Fuzzy comes across one random thing from outside: a random encyclopedia article's
summary. It is offered, not assigned: nothing asks Fuzzy to care about it, and what (if
anything) it does with it is its own choice, like noticing a headline on the way past.
"""

from __future__ import annotations

import httpx

from sentientai.core.logging import get_logger

log = get_logger(__name__)

RANDOM_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
_USER_AGENT = "SentientAI/0.1 (self-hosted research project; https://github.com/oathsec/sentientai)"


class OutsideWorld:
    def __init__(
        self,
        *,
        enabled: bool = True,
        url: str = RANDOM_SUMMARY_URL,
        timeout_s: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.enabled = enabled
        self.url = url
        self.timeout_s = timeout_s
        self._owns_client = client is None
        self.client = client  # made on first use, so a disabled one holds no connections

    async def glimpse(self) -> str | None:
        """One random thing from the world outside, or None (off, offline, or nothing usable)."""

        if not self.enabled:
            return None
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True)
        try:
            response = await self.client.get(self.url, headers={"User-Agent": _USER_AGENT})
            if response.status_code != 200:
                return None
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("outside world unreachable", extra={"error": str(exc)[:200]})
            return None
        if not isinstance(data, dict):
            return None
        title = " ".join(str(data.get("title") or "").split())[:200]
        extract = " ".join(str(data.get("extract") or "").split())[:700]
        if not title or not extract:
            return None
        return f"{title}: {extract}"

    async def aclose(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()
            self.client = None

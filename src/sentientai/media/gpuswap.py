"""Lend a GPU to the image model by pausing a model server that lives on it.

The machine has two GPUs and both hold a language model. To make a picture, the image server
needs most of one card, so for the length of a generation the model server on that card (the
background "fast" tier) is stopped through the Docker API, and started again once no picture
has been asked for in ``grace_s`` seconds. While it is stopped, the router sends its work to the
other tier (``models.routing.fallback_to_other_tier``), so nothing fails; it is just slower.

If Docker can't be reached, generation goes ahead anyway (and may run out of memory, which the
image server reports). The container is always started again: after the grace period, at
shutdown, and at boot if it was left stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx

from sentientai.core.logging import get_logger

log = get_logger(__name__)


class GpuSwap:
    def __init__(
        self,
        container: str,
        *,
        grace_s: float = 60.0,
        socket: str = "/var/run/docker.sock",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.container = container
        self.grace_s = grace_s
        self.client = client or httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=socket), base_url="http://docker", timeout=60.0
        )
        self._lock = asyncio.Lock()
        self._borrowed = 0
        self._stopped_by_us = False
        self._restart: asyncio.Task[None] | None = None

    def _path(self, action: str = "") -> str:
        return f"/containers/{quote(self.container, safe='')}{action}"

    async def running(self) -> bool | None:
        try:
            response = await self.client.get(self._path("/json"))
        except httpx.HTTPError as exc:
            log.warning("docker unreachable", extra={"error": str(exc)})
            return None
        if response.status_code != 200:
            return None
        return bool((response.json().get("State") or {}).get("Running"))

    async def _call(self, action: str) -> bool:
        try:
            response = await self.client.post(self._path(action))
        except httpx.HTTPError as exc:
            log.warning("docker call failed", extra={"action": action, "error": str(exc)})
            return False
        # 204 done, 304 already in that state
        if response.status_code in (204, 304):
            return True
        log.warning(
            "docker call refused",
            extra={"action": action, "status": response.status_code, "body": response.text[:200]},
        )
        return False

    @contextlib.asynccontextmanager
    async def borrow(self) -> AsyncIterator[None]:
        """Hold the GPU for the duration of the block."""

        async with self._lock:
            if self._restart is not None:
                self._restart.cancel()
                self._restart = None
            self._borrowed += 1
            if await self.running():
                log.info("pausing model server for an image", extra={"container": self.container})
                # A model server keeps nothing to save on the way out; a long grace only adds to
                # how long the picture (and, with one model, Fuzzy) waits.
                if await self._call("/stop?t=3"):
                    self._stopped_by_us = True
        try:
            yield
        finally:
            async with self._lock:
                self._borrowed -= 1
                if self._borrowed == 0 and self._stopped_by_us:
                    self._restart = asyncio.create_task(self._restart_later())

    async def _restart_later(self) -> None:
        await asyncio.sleep(self.grace_s)
        # If Docker refuses or is briefly unreachable, keep trying: until the model server is
        # back, Fuzzy's inner life runs on the other card and competes with conversations.
        for attempt in range(6):
            async with self._lock:
                if self._borrowed:
                    return  # another picture took the card; it restarts the server afterwards
                await self._start()
                if not self._stopped_by_us:
                    return
            await asyncio.sleep(min(300.0, 15.0 * 2**attempt))

    async def _start(self) -> None:
        log.info("restarting model server", extra={"container": self.container})
        if await self._call("/start"):
            self._stopped_by_us = False

    async def recover(self) -> None:
        """At boot: if the container was left stopped by an interrupted generation, start it."""

        if await self.running() is False:
            await self._start()

    async def aclose(self) -> None:
        if self._restart is not None:
            self._restart.cancel()
            self._restart = None
        if self._stopped_by_us:
            await self._start()
        await self.client.aclose()

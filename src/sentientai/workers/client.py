"""A model client that uses an operator's own connected PC when one is free.

Everything else about the client is the usual OpenAI-compatible one: the request is built the
same way, and when no worker is free (or the worker fails) the request goes to this server's own
model exactly as before.
"""

from __future__ import annotations

from typing import Any

from sentientai.core.logging import get_logger
from sentientai.models.openai_compat import OpenAICompatibleClient
from sentientai.workers.hub import WorkerError, WorkerHub

log = get_logger(__name__)


class WorkerFirstClient(OpenAICompatibleClient):
    def __init__(self, *args: Any, hub: WorkerHub, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hub = hub

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.hub.available("llm"):
            try:
                data = await self.hub.run(
                    "llm", payload, private_only=True, timeout=self.config.timeout_s
                )
            except WorkerError as exc:
                log.warning("worker couldn't run a model request", extra={"error": str(exc)})
            else:
                if isinstance(data.get("choices"), list) and data["choices"]:
                    return data
                log.warning("worker returned something that isn't a chat completion")
        return await super()._post(payload)

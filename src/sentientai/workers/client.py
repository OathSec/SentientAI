"""A model client that uses an operator's own connected PC for some of Fuzzy's thinking.

Two kinds of work may leave this server, and nothing else:

* ``background``: Fuzzy's quiet pondering (``models.routing.worker_task_kinds``, default only
  "ponder"). Any of the operator's computers may do it; it never reaches a person.
* ``snippet``: a step around a conversation (deciding whether to talk, checking a reply,
  judging a person afterwards). Only a computer running the SAME model as this server's chat
  model (``workers.chat_model``) may do it, so the quality is the same.

Replies to people are written here. When no suitable worker is free, or one fails or sends
junk, the work falls back to this server's own model.
"""

from __future__ import annotations

from typing import Any

from sentientai.core.logging import get_logger
from sentientai.models.openai_compat import OpenAICompatibleClient
from sentientai.models.types import ChatRequest
from sentientai.workers.hub import WorkerError, WorkerHub

log = get_logger(__name__)
_FLAG = "_fuzzy_route"


class WorkerFirstClient(OpenAICompatibleClient):
    def __init__(self, *args: Any, hub: WorkerHub, chat_model: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hub = hub
        self.chat_model = chat_model

    def _payload(self, request: ChatRequest) -> tuple[dict[str, Any], bool]:
        payload, native = super()._payload(request)
        # Removed again before anything is sent.
        payload[_FLAG] = (
            "snippet" if request.snippet else "background" if request.background else ""
        )
        return payload, native

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        route = str(payload.pop(_FLAG, ""))
        model = self.chat_model or None
        if route == "snippet" and model is None:
            route = ""  # without knowing the model, a conversation never leaves this server
        if route and self.hub.available("llm", model=model if route == "snippet" else None):
            try:
                data = await self.hub.run(
                    "llm",
                    payload,
                    private_only=True,
                    timeout=self.config.timeout_s,
                    model=model if route == "snippet" else None,
                )
            except WorkerError as exc:
                log.warning("worker couldn't run a model request", extra={"error": str(exc)})
            else:
                if isinstance(data.get("choices"), list) and data["choices"]:
                    return data
                log.warning("worker returned something that isn't a chat completion")
        return await super()._post(payload)

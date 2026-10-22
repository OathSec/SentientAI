"""Deterministic model for tests, evals and offline demos."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from sentientai.errors import ModelError
from sentientai.models.types import (
    ChatRequest,
    ChatResponse,
    Message,
    ToolCall,
    Usage,
    estimate_messages_tokens,
    estimate_tokens,
)

Reply = str | Message | ChatResponse | Exception
Responder = Callable[[ChatRequest], Reply | Awaitable[Reply]]


def tool_call(name: str, **arguments: Any) -> Message:
    return Message.assistant(None, [ToolCall(name=name, arguments=arguments)])


class ScriptedLLM:
    """Replays a script of replies, or delegates to a responder function.

    Every request is recorded in :attr:`requests` for assertions.
    """

    def __init__(
        self,
        script: Iterable[Reply | Responder] | Responder = (),
        *,
        name: str = "scripted",
        context_window: int = 32768,
        default: Reply | None = None,
        healthy: bool = True,
    ) -> None:
        self.name = name
        self.context_window = context_window
        self._responder: Responder | None = script if callable(script) else None
        self._script: list[Reply | Responder] = [] if callable(script) else list(script)
        self._default = default
        self.healthy = healthy
        self.requests: list[ChatRequest] = []

    def push(self, *replies: Reply | Responder) -> None:
        self._script.extend(replies)

    @property
    def remaining(self) -> int:
        return len(self._script)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        reply: Any
        if self._responder is not None:
            reply = self._responder(request)
        elif self._script:
            item = self._script.pop(0)
            reply = item(request) if callable(item) else item
        elif self._default is not None:
            reply = self._default
        else:
            raise ModelError(f"{self.name}: script exhausted")
        if hasattr(reply, "__await__"):
            reply = await reply
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, ChatResponse):
            return reply
        message = reply if isinstance(reply, Message) else Message.assistant(str(reply))
        return ChatResponse(
            message=message,
            usage=Usage(
                prompt_tokens=estimate_messages_tokens(request.messages),
                completion_tokens=estimate_tokens(message.text),
            ),
            model=self.name,
            finish_reason="tool_calls" if message.tool_calls else "stop",
        )

    async def health(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:
        return None

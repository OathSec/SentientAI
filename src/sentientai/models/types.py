"""Provider-neutral chat types shared by every model backend."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from sentientai.core.clock import new_id


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ModelTier(StrEnum):
    FAST = "fast"
    HEAVY = "heavy"


class TaskKind(StrEnum):
    """What a model call is for; drives tier selection."""

    ROUTE = "route"
    CLASSIFY = "classify"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    REFLECT = "reflect"
    ASSESS = "assess"
    CHAT = "chat"
    REASON = "reason"
    CODE = "code"
    PLAN = "plan"
    IMPROVE = "improve"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, call_id: str, name: str, content: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=call_id, name=name)

    @property
    def text(self) -> str:
        return self.content or ""


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ChatRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolSchema] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool = False
    stop: list[str] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    tier: ModelTier | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    reasoning: str | None = None  # hidden chain-of-thought, kept for the audit log only


@runtime_checkable
class LLMClient(Protocol):
    name: str
    context_window: int

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    async def health(self) -> bool: ...

    async def aclose(self) -> None: ...


def estimate_tokens(text: str) -> int:
    """Fast, tokenizer-free estimate (~4 chars/token for English and code)."""

    return max(1, (len(text) + 3) // 4) if text else 0


def estimate_messages_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += 4 + estimate_tokens(m.text)
        for call in m.tool_calls:
            total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
    return total

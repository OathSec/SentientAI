"""Tool contract.

A tool declares typed arguments as a pydantic model (which also produces the
JSON schema shown to the model), a base risk level, and optionally an
argument-aware risk assessment. Tools never decide whether they may run:
that is the executor's job (policy -> approval -> run -> audit).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from sentientai.config import RiskLevel
from sentientai.models.types import ToolSchema

A = TypeVar("A", bound=BaseModel)


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    task_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None  # the person this work is for (their memory compartment)
    autonomous: bool = False
    scratch_dir: Path | None = None
    # People who aren't admins get a restricted toolset and their own quotas.
    allowed_tools: frozenset[str] | None = None
    privileged: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def may_use(self, tool: str) -> bool:
        return self.allowed_tools is None or tool in self.allowed_tools


class ToolResult(BaseModel):
    ok: bool
    output: str
    data: Any = None
    error: str | None = None
    truncated: bool = False

    @classmethod
    def success(cls, output: str, data: Any = None) -> ToolResult:
        return cls(ok=True, output=output, data=data)

    @classmethod
    def failure(cls, error: str, output: str = "") -> ToolResult:
        return cls(ok=False, output=output or f"ERROR: {error}", error=error)

    def for_model(self) -> str:
        return self.output if self.ok else (self.output or f"ERROR: {self.error}")


def _clean_schema(schema: Any) -> Any:
    """Strip pydantic noise (titles) that wastes context tokens on small models."""

    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


class Tool(ABC, Generic[A]):
    name: str
    description: str
    args_model: type[A]
    risk: RiskLevel = RiskLevel.NONE
    timeout_s: float = 60.0
    tags: tuple[str, ...] = ()

    def available(self) -> bool:
        """False when the tool can't work right now (e.g. no image server): it isn't offered."""

        return True

    def schema(self) -> ToolSchema:
        params: dict[str, Any] = _clean_schema(self.args_model.model_json_schema())
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        return ToolSchema(name=self.name, description=self.description, parameters=params)

    def assess(self, args: A, ctx: ToolContext) -> tuple[RiskLevel, list[str]]:
        """Argument-aware risk. Defaults to the declared base risk."""

        return self.risk, []

    @abstractmethod
    async def run(self, args: A, ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        """Tools that can be used now (see :meth:`Tool.available`)."""

        return sorted(n for n, t in self._tools.items() if t.available())

    def tools(self) -> list[Tool[Any]]:
        return [self._tools[n] for n in self.names]

    def schemas(
        self, *, names: list[str] | None = None, max_risk: RiskLevel | None = None
    ) -> list[ToolSchema]:
        selected = []
        for tool in self.tools():
            if not tool.available():
                continue
            if names is not None and tool.name not in names:
                continue
            if max_risk is not None and tool.risk > max_risk:
                continue
            selected.append(tool.schema())
        return selected

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "risk": t.risk.name,
                "tags": list(t.tags),
            }
            for t in self.tools()
        ]

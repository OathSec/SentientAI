"""Prompted tool-calling protocol for endpoints without native function calling.

The tool catalogue is rendered into the system prompt and the model replies
with a JSON object (or Hermes/Qwen style ``<tool_call>`` tags). Tool results
are fed back as user turns, which every chat template accepts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sentientai.core.jsonparse import extract_json, iter_json, strip_reasoning
from sentientai.models.types import Message, Role, ToolCall, ToolSchema

_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Small models sometimes invent a tag named after the tool: <avatar_design>{...}</avatar_design>.
_NAMED_TAG = re.compile(r"<([A-Za-z_][\w\-]*)>\s*(\{.*?\})\s*</\1>", re.DOTALL)

PROTOCOL_HEADER = """\
## Tool use
You can act on the world by calling tools. To call tools, reply with ONLY this JSON
(no prose before or after it):
{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments>}}]}
Tool results will be returned to you in the next message. When you are done, reply
with your final answer as normal text and no tool_calls JSON.

Available tools:"""


def render_tool_catalogue(tools: list[ToolSchema]) -> str:
    lines = [PROTOCOL_HEADER]
    for tool in tools:
        params = json.dumps(tool.parameters.get("properties", {}), separators=(",", ":"))
        required = tool.parameters.get("required", [])
        lines.append(
            f"- {tool.name}: {tool.description}\n  params: {params}\n  required: {required}"
        )
    return "\n".join(lines)


def _as_call(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        parsed = extract_json(args, expect=dict)
        args = parsed if isinstance(parsed, dict) else {"input": args}
    if not isinstance(args, dict):
        return None
    return ToolCall(name=name, arguments=args)


def parse_tool_calls(text: str, known_tools: set[str]) -> tuple[list[ToolCall], str]:
    """Extract tool calls from free text. Returns (calls, remaining visible text)."""

    visible, _ = strip_reasoning(text)
    calls: list[ToolCall] = []
    named = [m for m in _NAMED_TAG.finditer(visible) if m.group(1) in known_tools]
    if named:  # the body is the arguments (or a {"arguments": ...} wrapper)
        for match in named:
            body = next(iter_json(match.group(2)), None)
            if isinstance(body, dict):
                args = body.get("arguments") if isinstance(body.get("arguments"), dict) else body
                calls.append(ToolCall(name=match.group(1), arguments=args))
        if calls:
            return calls, _NAMED_TAG.sub(
                lambda m: "" if m.group(1) in known_tools else m.group(0), visible
            ).strip()
    tagged = _TAG.findall(visible)
    candidates: list[Any] = []
    for block in tagged:
        candidates.extend(iter_json(block))
    if not tagged:
        candidates.extend(iter_json(visible))
    for candidate in candidates:
        items: list[Any]
        if isinstance(candidate, dict) and isinstance(candidate.get("tool_calls"), list):
            items = candidate["tool_calls"]
        elif isinstance(candidate, list):
            items = candidate
        else:
            items = [candidate]
        for item in items:
            call = _as_call(item)
            if call is not None and call.name in known_tools:
                calls.append(call)
        if calls:
            break
    remaining = _TAG.sub("", visible).strip() if tagged else ("" if calls else visible)
    return calls, remaining


def to_prompted_messages(messages: list[Message], tools: list[ToolSchema]) -> list[Message]:
    """Rewrite a native tool-calling conversation into the prompted protocol."""

    catalogue = render_tool_catalogue(tools) if tools else ""
    out: list[Message] = []
    system_done = False
    for msg in messages:
        if msg.role is Role.SYSTEM and not system_done:
            content = f"{msg.text}\n\n{catalogue}" if catalogue else msg.text
            out.append(Message.system(content))
            system_done = True
        elif msg.role is Role.ASSISTANT and msg.tool_calls:
            payload = {
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in msg.tool_calls]
            }
            prefix = f"{msg.text}\n" if msg.text else ""
            out.append(Message.assistant(prefix + json.dumps(payload)))
        elif msg.role is Role.TOOL:
            out.append(Message.user(f"[tool result: {msg.name} ({msg.tool_call_id})]\n{msg.text}"))
        else:
            out.append(Message(role=msg.role, content=msg.text))
    if not system_done and catalogue:
        out.insert(0, Message.system(catalogue))
    return out

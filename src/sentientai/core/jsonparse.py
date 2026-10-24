"""Tolerant JSON extraction from model output.

Local models frequently wrap JSON in prose, code fences or reasoning blocks,
and emit small syntax slips (trailing commas, smart quotes). These helpers
recover structured data without ever executing anything.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def strip_reasoning(text: str) -> tuple[str, str | None]:
    """Split ``<think>...</think>`` reasoning from the visible answer."""

    thoughts = [m.group(0)[7:-8].strip() for m in _THINK.finditer(text)]
    visible = _THINK.sub("", text)
    # The chat template may open <think> inside the prompt, so the output starts mid-thought and
    # only the closing tag appears: everything before it is reasoning.
    lowered = visible.lower()
    if "</think>" in lowered and "<think>" not in lowered[: lowered.index("</think>")]:
        idx = lowered.index("</think>")
        thoughts.append(visible[:idx].strip())
        visible = visible[idx + len("</think>") :]
    # An unterminated <think> (truncated output) hides everything after it.
    if "<think>" in visible.lower():
        idx = visible.lower().index("<think>")
        thoughts.append(visible[idx + 7 :].strip())
        visible = visible[:idx]
    return visible.strip(), ("\n\n".join(t for t in thoughts if t) or None)


def _balanced_spans(text: str) -> Iterator[str]:
    """Yield top-level balanced {...} / [...] spans, respecting string literals."""

    i = 0
    n = len(text)
    while i < n:
        if text[i] in "{[":
            stack = [text[i]]
            j = i + 1
            in_str = False
            escape = False
            while j < n and stack:
                ch = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    opener = stack.pop()
                    if (opener, ch) not in {("{", "}"), ("[", "]")}:
                        break
                j += 1
            if not stack:
                yield text[i:j]
                i = j
                continue
        i += 1


def _loads(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    repaired = candidate.translate(_SMART_QUOTES)
    repaired = _TRAILING_COMMA.sub(r"\1", repaired)
    return json.loads(repaired)


def iter_json(text: str) -> Iterator[Any]:
    """Yield every JSON value that can be recovered from ``text``, best first."""

    visible, _ = strip_reasoning(text)
    stripped = visible.strip()
    if stripped:
        try:
            yield _loads(stripped)
            return
        except json.JSONDecodeError:
            pass
    for block in _FENCE.findall(visible):
        try:
            yield _loads(block.strip())
        except json.JSONDecodeError:
            continue
    for span in _balanced_spans(visible):
        try:
            yield _loads(span)
        except json.JSONDecodeError:
            continue


def extract_json(text: str, *, expect: type | None = None) -> Any | None:
    for value in iter_json(text):
        if expect is None or isinstance(value, expect):
            return value
    return None


def extract_object(text: str) -> dict[str, Any] | None:
    value = extract_json(text, expect=dict)
    return value if isinstance(value, dict) else None

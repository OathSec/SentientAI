"""Client for OpenAI-compatible chat endpoints.

Works with vLLM, llama.cpp ``llama-server``, Ollama (``/v1``), LM Studio and
TGI. Native tool calling is used when the endpoint supports it; otherwise the
prompted protocol in :mod:`sentientai.models.toolproto` is applied
transparently so the agent loop never needs to know.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import httpx

from sentientai.config import ModelEndpointConfig
from sentientai.core.jsonparse import extract_json, strip_reasoning
from sentientai.core.logging import get_logger
from sentientai.errors import ModelError, ModelUnavailableError
from sentientai.models.toolproto import parse_tool_calls, to_prompted_messages
from sentientai.models.types import (
    ChatRequest,
    ChatResponse,
    Message,
    Role,
    ToolCall,
    Usage,
)

log = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def message_to_openai(msg: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
    if msg.role is Role.ASSISTANT and msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in msg.tool_calls
        ]
    if msg.role is Role.TOOL:
        out["tool_call_id"] = msg.tool_call_id
        if msg.name:
            out["name"] = msg.name
    if out["content"] is None and msg.role is not Role.ASSISTANT:
        out["content"] = ""
    return out


def _parse_native_calls(raw: list[dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for item in raw:
        fn = item.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args_raw = fn.get("arguments") or "{}"
        if isinstance(args_raw, dict):
            args = args_raw
        else:
            parsed = extract_json(str(args_raw), expect=dict)
            args = parsed if isinstance(parsed, dict) else {}
        kwargs: dict[str, Any] = {"name": name, "arguments": args}
        if item.get("id"):
            kwargs["id"] = str(item["id"])
        calls.append(ToolCall(**kwargs))
    return calls


class OpenAICompatibleClient:
    def __init__(
        self,
        name: str,
        config: ModelEndpointConfig,
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.context_window = config.context_window
        headers = {"Content-Type": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(config.timeout_s, connect=10.0),
        )
        if http is not None:
            self._http.headers.update(headers)

    def _payload(self, request: ChatRequest) -> tuple[dict[str, Any], bool]:
        native = bool(request.tools) and self.config.supports_tools
        messages = request.messages
        if request.tools and not native:
            messages = to_prompted_messages(messages, request.tools)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message_to_openai(m) for m in messages],
            "temperature": request.temperature
            if request.temperature is not None
            else self.config.temperature,
            "max_tokens": min(
                request.max_tokens or self.config.max_output_tokens, self.config.max_output_tokens
            ),
            "stream": False,
        }
        if native:
            payload["tools"] = [t.to_openai() for t in request.tools]
            payload["tool_choice"] = "auto"
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.stop:
            payload["stop"] = request.stop
        payload.update(self.config.extra_body)
        return payload, native

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self._http.post("/chat/completions", json=payload)
                if response.status_code in _RETRYABLE_STATUS and attempt < attempts - 1:
                    last_error = ModelError(f"{self.name}: HTTP {response.status_code}")
                elif response.status_code >= 400:
                    raise ModelError(
                        f"{self.name}: HTTP {response.status_code}: {response.text[:500]}"
                    )
                else:
                    data: dict[str, Any] = response.json()
                    return data
            except httpx.TransportError as exc:
                last_error = ModelUnavailableError(f"{self.name}: {type(exc).__name__}: {exc}")
                if attempt == attempts - 1:
                    raise last_error from exc
            except json.JSONDecodeError as exc:
                raise ModelError(f"{self.name}: invalid JSON response") from exc
            delay = min(8.0, 0.5 * 2**attempt) * (0.75 + random.random() / 2)  # noqa: S311
            log.warning("model request retry", extra={"model": self.name, "attempt": attempt + 1})
            await asyncio.sleep(delay)
        raise last_error or ModelError(f"{self.name}: request failed")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload, native = self._payload(request)
        started = time.perf_counter()
        data = await self._post(payload)
        latency = (time.perf_counter() - started) * 1000
        try:
            choice = data["choices"][0]
            raw_msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"{self.name}: malformed response: {str(data)[:300]}") from exc

        content_raw = raw_msg.get("content") or ""
        if isinstance(content_raw, list):  # some servers return content parts
            content_raw = "".join(p.get("text", "") for p in content_raw if isinstance(p, dict))
        content, reasoning = strip_reasoning(str(content_raw))
        reasoning = raw_msg.get("reasoning_content") or raw_msg.get("reasoning") or reasoning

        calls = _parse_native_calls(raw_msg.get("tool_calls") or []) if native else []
        if not calls and request.tools:
            # Prompted protocol, or a native-capable server that still emitted tagged calls.
            calls, content = parse_tool_calls(str(content_raw), {t.name for t in request.tools})

        usage_raw = data.get("usage") or {}
        return ChatResponse(
            message=Message.assistant(content or None, calls),
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            ),
            model=str(data.get("model") or self.config.model),
            finish_reason=choice.get("finish_reason"),
            latency_ms=latency,
            reasoning=str(reasoning) if reasoning else None,
        )

    async def health(self) -> bool:
        try:
            response = await self._http.get("/models", timeout=5.0)
            return response.status_code < 400
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        response = await self._http.get("/models", timeout=10.0)
        response.raise_for_status()
        return [str(m.get("id")) for m in response.json().get("data", [])]

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

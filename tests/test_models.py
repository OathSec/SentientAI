from __future__ import annotations

import json

import httpx
import pytest
import respx

from sentientai.config import EmbeddingConfig, ModelEndpointConfig, RoutingConfig
from sentientai.core.clock import FakeClock
from sentientai.core.jsonparse import extract_json, extract_object, strip_reasoning
from sentientai.core.metrics import Metrics
from sentientai.errors import ConfigError, ModelError, ModelUnavailableError
from sentientai.models.embeddings import CachedEmbedder, HashingEmbedder, OpenAIEmbedder, cosine
from sentientai.models.openai_compat import OpenAICompatibleClient
from sentientai.models.router import ModelRouter
from sentientai.models.scripted import ScriptedLLM, tool_call
from sentientai.models.toolproto import parse_tool_calls, to_prompted_messages
from sentientai.models.types import (
    ChatRequest,
    Message,
    ModelTier,
    Role,
    TaskKind,
    ToolCall,
    ToolSchema,
)

BASE = "http://llm.test/v1"
TOOLS = [
    ToolSchema(
        name="fs_read",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
]


# ---------------------------------------------------------------- jsonparse


def test_extract_json_variants() -> None:
    assert extract_object('{"a": 1}') == {"a": 1}
    assert extract_object('Sure!\n```json\n{"a": [1,2,],}\n```\nDone') == {"a": [1, 2]}
    assert extract_object('prefix {"a": "x}y"} suffix {"b": 2}') == {"a": "x}y"}
    assert extract_json('<think>{"no": 1}</think> [1, 2]') == [1, 2]
    assert extract_object("no json here") is None
    assert extract_object("“key”") is None


def test_strip_reasoning() -> None:
    visible, thought = strip_reasoning("<think>plan quietly</think>Answer is 4")
    assert visible == "Answer is 4" and thought == "plan quietly"
    visible, thought = strip_reasoning("Partial <think>never closed")
    assert visible == "Partial" and thought == "never closed"


# ----------------------------------------------------------------- toolproto


def test_parse_prompted_tool_calls() -> None:
    calls, rest = parse_tool_calls(
        '{"tool_calls": [{"name": "fs_read", "arguments": {"path": "a"}}]}', {"fs_read"}
    )
    assert [(c.name, c.arguments) for c in calls] == [("fs_read", {"path": "a"})] and rest == ""

    tagged = 'Let me look.\n<tool_call>{"name": "fs_read", "arguments": "{\\"path\\": \\"b\\"}"}</tool_call>'
    calls, rest = parse_tool_calls(tagged, {"fs_read"})
    assert calls[0].arguments == {"path": "b"} and rest == "Let me look."

    calls, rest = parse_tool_calls('{"name": "rm_everything", "arguments": {}}', {"fs_read"})
    assert calls == [] and "rm_everything" in rest  # unknown tools are never executed


def test_to_prompted_messages() -> None:
    msgs = [
        Message.system("You are X."),
        Message.user("read a"),
        Message.assistant(None, [ToolCall(id="c1", name="fs_read", arguments={"path": "a"})]),
        Message.tool("c1", "fs_read", "contents"),
    ]
    out = to_prompted_messages(msgs, TOOLS)
    assert "fs_read" in out[0].text and "tool_calls" in out[0].text
    assert (
        out[2].role is Role.ASSISTANT
        and json.loads(out[2].text)["tool_calls"][0]["name"] == "fs_read"
    )
    assert out[3].role is Role.USER and out[3].text.startswith("[tool result: fs_read (c1)]")


# ------------------------------------------------------------ openai client


def _client(**cfg: object) -> OpenAICompatibleClient:
    config = ModelEndpointConfig(base_url=BASE, model="m", max_retries=2, **cfg)  # type: ignore[arg-type]
    return OpenAICompatibleClient("heavy", config)


def _completion(message: dict[str, object], **extra: object) -> dict[str, object]:
    return {
        "model": "m",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        **extra,
    }


@respx.mock
async def test_native_tool_calls_and_reasoning() -> None:
    route = respx.post(f"{BASE}/chat/completions").respond(
        200,
        json=_completion(
            {
                "content": "<think>I should read it</think>",
                "tool_calls": [
                    {
                        "id": "abc",
                        "type": "function",
                        "function": {"name": "fs_read", "arguments": '{"path": "x.txt"}'},
                    }
                ],
            }
        ),
    )
    client = _client()
    resp = await client.chat(ChatRequest(messages=[Message.user("hi")], tools=TOOLS))
    body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["function"]["name"] == "fs_read" and body["tool_choice"] == "auto"
    assert resp.message.tool_calls[0].id == "abc"
    assert resp.message.tool_calls[0].arguments == {"path": "x.txt"}
    assert resp.reasoning == "I should read it" and resp.message.content is None
    assert resp.usage.total == 15
    await client.aclose()


@respx.mock
async def test_prompted_protocol_when_tools_unsupported() -> None:
    route = respx.post(f"{BASE}/chat/completions").respond(
        200,
        json=_completion(
            {"content": '{"tool_calls": [{"name": "fs_read", "arguments": {"path": "y"}}]}'}
        ),
    )
    client = _client(supports_tools=False)
    resp = await client.chat(
        ChatRequest(messages=[Message.system("sys"), Message.user("go")], tools=TOOLS)
    )
    body = json.loads(route.calls.last.request.content)
    assert "tools" not in body and "Available tools" in body["messages"][0]["content"]
    assert resp.message.tool_calls[0].arguments == {"path": "y"}


@respx.mock
async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentientai.models.openai_compat.asyncio.sleep", _no_sleep)
    respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectError("down"),
            httpx.Response(200, json=_completion({"content": "ok"})),
        ]
    )
    resp = await _client().chat(ChatRequest(messages=[Message.user("x")]))
    assert resp.message.text == "ok"


@respx.mock
async def test_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentientai.models.openai_compat.asyncio.sleep", _no_sleep)
    respx.post(f"{BASE}/chat/completions").respond(400, text="bad request")
    with pytest.raises(ModelError, match="HTTP 400"):
        await _client().chat(ChatRequest(messages=[Message.user("x")]))
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(ModelUnavailableError):
        await _client().chat(ChatRequest(messages=[Message.user("x")]))
    respx.post(f"{BASE}/chat/completions").respond(200, json={"nope": 1})
    with pytest.raises(ModelError, match="malformed"):
        await _client().chat(ChatRequest(messages=[Message.user("x")]))


@respx.mock
async def test_health() -> None:
    respx.get(f"{BASE}/models").respond(200, json={"data": [{"id": "m"}]})
    client = _client()
    assert await client.health()
    assert await client.list_models() == ["m"]


async def _no_sleep(_s: float) -> None:
    return None


# ------------------------------------------------------------------- router


def _router(fast: ScriptedLLM, heavy: ScriptedLLM, clock: FakeClock | None = None) -> ModelRouter:
    return ModelRouter(
        {ModelTier.FAST: fast, ModelTier.HEAVY: heavy},
        RoutingConfig(circuit_failure_threshold=2, circuit_cooldown_s=30),
        metrics=Metrics(),
        clock=clock or FakeClock(),
    )


async def test_router_selection_rules() -> None:
    fast = ScriptedLLM(name="fast", context_window=100, default="f")
    heavy = ScriptedLLM(name="heavy", default="h")
    router = _router(fast, heavy)
    small = ChatRequest(messages=[Message.user("hi")])
    assert router.select(TaskKind.SUMMARIZE, small).tier is ModelTier.FAST
    assert router.select(TaskKind.CODE, small).tier is ModelTier.HEAVY
    assert router.select(TaskKind.SUMMARIZE, small, confidence=0.1).tier is ModelTier.HEAVY
    big = ChatRequest(messages=[Message.user("x" * 1000)])
    assert router.select(TaskKind.SUMMARIZE, big).tier is ModelTier.HEAVY  # context won't fit fast
    assert router.select(TaskKind.CODE, small, prefer=ModelTier.FAST).tier is ModelTier.FAST

    resp = await router.chat(small, kind=TaskKind.EXTRACT)
    assert resp.tier is ModelTier.FAST and resp.message.text == "f"


async def test_router_fallback_and_circuit_breaker() -> None:
    clock = FakeClock()
    heavy = ScriptedLLM([ModelError("oom"), ModelError("oom"), "recovered"], name="heavy")
    fast = ScriptedLLM(name="fast", default="fast-answer")
    router = _router(fast, heavy, clock)
    req = ChatRequest(messages=[Message.user("think")])

    first = await router.chat(req, kind=TaskKind.REASON)
    assert first.tier is ModelTier.FAST  # fell back
    await router.chat(req, kind=TaskKind.REASON)
    assert router.breakers[ModelTier.HEAVY].state == "open"
    third = await router.chat(req, kind=TaskKind.REASON)
    assert third.tier is ModelTier.FAST and heavy.remaining == 1  # heavy skipped while open

    clock.advance(31)
    assert router.breakers[ModelTier.HEAVY].state == "half_open"
    fourth = await router.chat(req, kind=TaskKind.REASON)
    assert fourth.tier is ModelTier.HEAVY and fourth.message.text == "recovered"
    assert router.breakers[ModelTier.HEAVY].state == "closed"


async def test_router_all_tiers_down() -> None:
    router = _router(ScriptedLLM([ModelError("x")]), ScriptedLLM([ModelError("y")]))
    with pytest.raises(ModelUnavailableError, match="no model tier"):
        await router.chat(ChatRequest(messages=[Message.user("x")]))


async def test_scripted_llm_tool_call_helper() -> None:
    llm = ScriptedLLM([tool_call("fs_read", path="a"), "done"])
    first = await llm.chat(ChatRequest(messages=[Message.user("x")]))
    assert (
        first.message.tool_calls[0].arguments == {"path": "a"}
        and first.finish_reason == "tool_calls"
    )
    assert (await llm.chat(ChatRequest(messages=[]))).message.text == "done"
    with pytest.raises(ModelError):
        await llm.chat(ChatRequest(messages=[]))


# --------------------------------------------------------------- embeddings


async def test_hashing_embedder_similarity() -> None:
    emb = HashingEmbedder(512)
    [a, b, c] = await emb.embed(
        [
            "The operator prefers Rust for systems programming",
            "Which language does the operator like for systems programming?",
            "Recipe for banana bread with walnuts",
        ]
    )
    assert cosine(a, b) > cosine(a, c) + 0.2
    assert abs(sum(x * x for x in a) - 1.0) < 1e-5


@respx.mock
async def test_openai_embedder_batches_and_validates() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={
                "data": [{"index": i, "embedding": [float(i + 1)] * 16} for i in range(len(inputs))]
            },
        )

    route = respx.post("http://emb.test/v1/embeddings").mock(side_effect=respond)
    emb = OpenAIEmbedder(
        EmbeddingConfig(
            provider="openai", base_url="http://emb.test/v1", dimensions=16, batch_size=2
        )
    )
    vectors = await emb.embed(["a", "b", "c"])
    assert len(vectors) == 3 and route.call_count == 2
    assert vectors[0] == pytest.approx([0.25] * 16)

    bad = OpenAIEmbedder(
        EmbeddingConfig(provider="openai", base_url="http://emb.test/v1", dimensions=32)
    )
    with pytest.raises(ConfigError, match="dims"):
        await bad.embed(["a"])


async def test_cached_embedder() -> None:
    cached = CachedEmbedder(HashingEmbedder(64), max_items=2)
    await cached.embed(["a", "b"])
    await cached.embed(["a"])
    assert cached.hits == 1
    await cached.embed(["c"])
    assert len(cached._cache) == 2

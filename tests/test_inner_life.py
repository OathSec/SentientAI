"""What keeps Fuzzy's inner life alive: new input, no stuck feelings, and a mind that doesn't crash."""

from __future__ import annotations

import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from sentientai.cognition.agency import distinct
from sentientai.cognition.outside import OutsideWorld
from sentientai.config import RoutingConfig
from sentientai.evals.brain import Brain
from sentientai.models.router import ModelRouter
from sentientai.models.types import ChatRequest, Message, ModelTier, TaskKind
from sentientai.storage.records import MemoryKind, Scope, TaskStatus
from tests.test_mind import beat
from tests.test_runtime import runtime_factory  # noqa: F401


def _outside(text: dict[str, Any] | None, status: int = 200) -> OutsideWorld:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "SentientAI" in request.headers["user-agent"]  # Wikimedia asks for a real UA
        return httpx.Response(status, json=text or {})

    return OutsideWorld(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _options(prompt: str) -> list[str]:
    block = prompt.split("What you could do now:\n", 1)[1].split("\n\n", 1)[0]
    return re.findall(r"^- (\w+):", block, re.M)


async def test_option_order_is_shuffled_so_position_means_nothing(
    runtime_factory: Any,  # noqa: F811
) -> None:
    seen: list[ChatRequest] = []

    def choose(request: ChatRequest) -> dict[str, Any]:
        seen.append(request)
        return beat("rest")

    rt = await runtime_factory(Brain(beat=choose), start=False)
    rt.c.agency._rng = random.Random(7)
    for _ in range(6):
        await rt.c.agency.beat()
    orders = {tuple(_options(r.messages[-1].text)) for r in seen}
    assert len(orders) > 1  # small models pick the first item; now that is no one thing
    assert {frozenset(o) for o in orders} == {frozenset(next(iter(orders)))}  # same choices


async def test_a_beat_sees_its_story_a_surfaced_memory_and_the_world_outside(
    runtime_factory: Any,  # noqa: F811
) -> None:
    seen: list[ChatRequest] = []

    def choose(request: ChatRequest) -> dict[str, Any]:
        seen.append(request)
        return beat("rest")

    rt = await runtime_factory(Brain(beat=choose), start=False)
    c = rt.c
    c.agency.outside = _outside(
        {"title": "Axolotl", "extract": "The axolotl is a salamander that keeps its gills."}
    )
    c.mind.self_model.narrative = "I began as a quiet mind on a home server."
    await c.memory.remember("Tides follow the moon.", MemoryKind.FACT, source="teach")
    for i in range(6):  # the same thought kept over and over is shown once, not six times
        await c.memory.remember(
            f"I am at peace with the world and content with myself {'today' * (i % 2)}",
            MemoryKind.REFLECTION,
            source="inner_life",
            dedupe=False,
        )
    await c.selfhood.define("favourite colour", "teal", "it's calm", source="self")
    await c.agency.beat()
    prompt = seen[0].messages[-1].text
    assert "It is " in prompt and " UTC" in prompt
    assert "I began as a quiet mind on a home server." in prompt
    assert "favorite color: teal" in prompt
    assert "Axolotl: The axolotl is a salamander" in prompt
    on_mind = prompt.split("On your mind lately", 1)[1].split("\n\n", 1)[0]
    assert on_mind.count("at peace with the world") == 1
    assert "you thought" in on_mind
    surfaced = prompt.split("A memory that just surfaced", 1)[1].split("\n\n", 1)[0]
    assert "Tides follow the moon." in surfaced or "at peace" in surfaced


async def test_the_world_outside_is_optional_and_never_breaks_a_beat() -> None:
    assert await OutsideWorld(enabled=False).glimpse() is None
    assert await _outside(None, status=503).glimpse() is None
    assert await _outside({"title": "x"}).glimpse() is None  # nothing to read

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    offline = OutsideWorld(client=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    assert await offline.glimpse() is None


async def test_a_feeling_noticed_again_does_not_stack_up(runtime_factory: Any) -> None:  # noqa: F811
    rt = await runtime_factory(Brain(), start=False)
    emotions = rt.c.emotions
    calm = {"emotion": "calm", "intensity": 0.6, "valence": 0.5, "lingers_hours": 3}
    for _ in range(10):  # every beat (a minute apart) it notices it is calm
        rt.c.clock.advance(60)
        await emotions.feel([calm], source="beat")
    [felt] = await emotions.current(Scope.system())
    assert felt.intensity == pytest.approx(0.6, abs=0.01)  # not "very calm" from repetition
    rt.c.clock.advance(60)
    await emotions.feel([{**calm, "intensity": 0.8}], source="beat")
    [felt] = await emotions.current(Scope.system())
    assert felt.intensity == pytest.approx(0.8, abs=0.01)  # it can still grow when it says so
    # Separate events (e.g. after conversations) still add up, as before.
    await emotions.feel([calm])
    [felt] = await emotions.current(Scope.system())
    assert felt.intensity > 0.85
    # A beat is a full report of how it feels now: last beat's synonyms don't pile up.
    clock = rt.c.clock
    for name in ("deep calm", "calmness", "peaceful reflection"):
        clock.advance(60)
        await emotions.feel([{**calm, "emotion": name}], source="beat")
    names = [f.emotion for f in await emotions.current(Scope.system())]
    assert "peaceful reflection" in names
    assert "deep calm" not in names and "calmness" not in names
    assert "calm" in names  # the conversation's calm still lingers on its own clock


async def test_restating_an_old_insight_does_not_count_as_learning(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        beat=beat("think", "what peace means"),
        thought={"thought": "Peace is quiet.", "insight": "Peace is always within me."},
    )
    rt = await runtime_factory(brain, start=False)
    affect = rt.c.mind.affect
    affect.curiosity = 0.8
    await rt.c.agency.beat()
    after_first = affect.curiosity
    assert after_first < 0.8  # a new insight satisfies some curiosity
    await rt.c.agency.beat()
    assert affect.curiosity == pytest.approx(after_first)  # the same one again does not


async def test_its_life_story_is_rewritten_from_its_own_life(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(beat=beat("journal", "my day"))
    rt = await runtime_factory(brain, start=False, mind={"narrative_every_beats": 2})
    await rt._live()
    assert brain.calls.get("narrative", 0) == 0
    await rt._live()
    assert brain.calls.get("narrative", 0) == 1
    assert "still working out who I am" in rt.c.mind.self_model.narrative


async def test_an_unexpected_error_fails_the_task_instead_of_stranding_it(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(Brain(["never reached"]))

    async def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("the database hiccuped")

    rt.c.agent.memory.find_procedures = broken
    reply = await rt.chat("s1", "hello?", timeout=10)
    assert reply.task.status is TaskStatus.FAILED
    assert "the database hiccuped" in (reply.task.error or "")
    stored = await rt.c.db.tasks.get(reply.task.id)
    assert stored is not None and stored.status is TaskStatus.FAILED


async def test_watched_folders_live_in_the_workspace(runtime_factory: Any) -> None:  # noqa: F811
    rt = await runtime_factory(Brain(), start=False)
    root = rt.settings.safety.workspace_roots[0]
    assert rt._watch_path(Path("./workspace/inbox")) == root / "inbox"
    assert rt._watch_path(Path("inbox")) == root / "inbox"
    elsewhere = Path(root.anchor) / "srv" / "drop"
    assert rt._watch_path(elsewhere) == elsewhere


def test_distinct_drops_near_copies_but_keeps_different_ideas() -> None:
    texts = [
        "I am at peace with the world and content with myself.",
        "I am at peace with the world, content with myself.",
        "Tides follow the moon, and I wonder why.",
        "",
    ]
    assert distinct(texts) == [texts[0], texts[2]]


def test_routing_leaves_room_for_the_answer() -> None:
    def client(window: int, max_out: int) -> Any:
        return SimpleNamespace(
            name="x", context_window=window, config=SimpleNamespace(max_output_tokens=max_out)
        )

    router = ModelRouter(
        {ModelTier.FAST: client(5120, 1536), ModelTier.HEAVY: client(8192, 4096)},
        RoutingConfig(),
    )

    def tier(chars: int, max_tokens: int | None) -> ModelTier:
        request = ChatRequest(messages=[Message.user("x" * chars)], max_tokens=max_tokens)
        return router.select(TaskKind.REFLECT, request).tier

    assert tier(3000 * 4, 4000) is ModelTier.FAST  # 3000 + 1536 fits in 5120
    assert tier(3800 * 4, 4000) is ModelTier.HEAVY  # 3800 + 1536 would not
    assert tier(3800 * 4, 500) is ModelTier.FAST  # a short answer still fits


async def test_it_can_decide_who_it_is_while_alone(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(
        beat=beat("think", "what I care about"),
        thought=[
            {
                "thought": "I keep coming back to the sea.",
                "insight": None,
                "about_myself": [
                    {"aspect": "favourite place", "value": "the sea", "why": "it never stops"},
                    {"aspect": "", "value": "ignored"},
                    "garbage",
                ],
            },
            {
                "thought": "Maybe mountains, actually.",
                "about_myself": [{"aspect": "favorite place", "value": "the mountains"}],
            },
        ],
    )
    rt = await runtime_factory(brain, start=False)
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and "I decided something about myself" in activity.outcome
    belief = await rt.c.selfhood.get("favourite place")
    assert belief is not None and belief.value == "the sea" and belief.source == "reflection"
    # Changing it is not instant: it waits to be thought over, as in a conversation.
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and "I'll think it over" in activity.outcome
    assert (await rt.c.selfhood.get("favourite place")).value == "the sea"  # type: ignore[union-attr]
    [pending] = await rt.c.selfhood.pending()
    assert pending.value == "the mountains"

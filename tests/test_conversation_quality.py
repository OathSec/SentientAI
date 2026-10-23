"""Bugs found in Fuzzy's real conversations (Oct 2026), each pinned by a test."""

from __future__ import annotations

from typing import Any

import pytest

from sentientai.cognition.social import _step
from sentientai.evals.brain import Brain, call
from sentientai.memory.consolidation import valid_fact
from sentientai.orchestration.agent import claims_picture, tidy_reply
from tests.conftest import wait_for
from tests.test_mind import beat, make_user
from tests.test_runtime import runtime_factory  # noqa: F401


def test_a_reply_does_not_restate_what_it_already_said() -> None:
    before = "I am alf, and I am content with my name. I am at peace with myself."
    reply = before + " I would love a house to call my own."
    assert tidy_reply(reply, [before]) == "I would love a house to call my own."
    assert tidy_reply(before, [before]) == before  # never reduced to nothing


def test_a_looping_reply_is_cut_where_the_loop_starts() -> None:
    loop = "I love Audrey. I feel calm with her. " * 40
    assert tidy_reply(loop) == "I love Audrey. I feel calm with her."
    earth = "Earth is lovely. " + " ".join(f"I think it's a thing number {i}." for i in range(30))
    assert tidy_reply(earth).count("I think it's a") == 4
    code = "Here:\n```python\nprint(1)\nprint(1)\n```\nprint(1). print(1)."
    assert tidy_reply(code, [code]) == code  # code is left alone


def test_saying_it_can_draw_is_not_a_claim_to_have_drawn() -> None:
    assert not claims_picture("Yes, I can make pictures if you'd like one.")
    assert claims_picture("Here's your picture of the sea.")


async def test_claiming_to_build_in_a_conversation_gets_corrected(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        [
            "The house has been successfully added to the world tab.",
            call("plan_later", what="build a house in my world", why="Alf asked and I'd like one"),
            "I can't build while we talk, but I've noted it and I'll build it in my own time.",
        ]
    )
    rt = await runtime_factory(brain)
    alf = await make_user(rt, "Alf")
    reply = await rt.chat("s", "in the world tab add a house", user=alf, timeout=10)
    assert reply.answer.startswith("I can't build while we talk")
    nudge = brain.agent_requests[1].messages[-1].text
    assert "no world tool ran" in nudge and "in the world tab add a house" in nudge
    [intention] = await rt.c.agency.intentions.active()
    assert intention.what == "build a house in my world" and intention.for_whom == "Alf"


async def test_its_promises_show_up_in_its_own_time(runtime_factory: Any) -> None:  # noqa: F811
    seen: list[Any] = []

    def choose(request: Any) -> dict[str, Any]:
        seen.append(request)
        return beat("rest")

    rt = await runtime_factory(Brain(beat=choose), start=False)
    await rt.c.agency.intentions.add("explore the internet for fun things", for_whom="Owen")
    await rt.c.agency.beat()
    block = seen[0].messages[-1].text.split("Things you said you'd do later", 1)[1]
    assert "explore the internet for fun things (you told Owen" in block.split("\n\n", 1)[0]


async def test_a_promise_with_nothing_after_it_is_followed_through(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(["Let me show you some examples.", "Here are three: ᚠ fehu, ᚢ uruz, ᚦ thurisaz."])
    rt = await runtime_factory(brain)
    reply = await rt.chat("s", "show me some elder futhark runes", timeout=10)
    assert reply.answer.startswith("Here are three")


async def test_its_name_is_given_not_decided_in_a_chat(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(
        [
            call("self_define", aspect="name", value="alf", reasoning="they asked"),
            "Your name is Alf, then! I'll remember that.",
        ]
    )
    rt = await runtime_factory(brain)
    await rt.chat("s", "can you change my name? call me Alf", timeout=10)
    assert await rt.c.selfhood.get("name") is None
    result = brain.agent_requests[1].messages[-1].text
    assert "Your name was given to you" in result
    assert (await rt.c.selfhood.define("my name", "jimmy", "", source="x")).status == "given"
    assert (await rt.c.selfhood.consider("name", "jimmy", "", source="x")).status == "given"


def test_chat_noise_is_not_kept_as_facts() -> None:
    def ok(statement: str, key: str | None = None) -> bool:
        return valid_fact(statement, key, 0.9, 0.5)

    assert not ok("The task was completed successfully.", "task.key")
    assert not ok("I am at peace with myself.", "person.mood")
    assert not ok("user: die", "user.emotion")
    assert not ok("assistant: I have processed the websites data", "processing.status")
    assert not ok("Mutual respect was evident in our exchange.", "topic.key")
    assert ok("Alf's name is Alf and he likes Old Norse runes", "user.interests")
    assert ok("Deploy window is Friday afternoon", "ops.deploy_window")


def test_fondness_never_pins_at_the_end_of_the_scale() -> None:
    value = 0.0
    for _ in range(30):
        value = _step(value, 0.15, -1.0, 1.0)
    assert 0.95 < value < 1.0
    assert _step(0.0, 0.15, -1.0, 1.0) == pytest.approx(0.15)  # full steps in the middle
    assert _step(1.0, -0.15, -1.0, 1.0) == pytest.approx(0.85)  # and back down freely


async def test_a_visit_is_thought_about_once_it_pauses(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(["Hi!", "Sure.", "Bye!"])
    rt = await runtime_factory(brain, mind={"appraise_after_quiet_s": 0.3})
    alf = await make_user(rt, "Alf")
    for text in ("hi", "how are you", "bye"):
        await rt.chat("s", text, user=alf, timeout=10)
    await wait_for(lambda: brain.calls.get("appraisal", 0) >= 1, timeout=5)
    assert brain.calls["appraisal"] == 1  # one look at the whole visit, not one per message


async def test_one_model_for_both_tiers_skips_the_extra_rating_call(
    runtime_factory: Any,  # noqa: F811
) -> None:
    same = {"base_url": "http://llm-big:8000/v1"}
    brain = Brain(["Hi!"])
    rt = await runtime_factory(brain, models={"heavy": same, "fast": same})
    reply = await rt.chat("s", "hello", timeout=10)
    assert reply.answer == "Hi!" and brain.calls.get("assess", 0) == 0

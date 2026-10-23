"""Fuzzy's mind: self-beliefs, feelings, relationships, grounding and what it does alone."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from sentientai.cognition.emotions import clean_emotion
from sentientai.cognition.selfhood import normalize_aspect
from sentientai.cognition.social import MAX_STEP
from sentientai.core.clock import FakeClock
from sentientai.evals.brain import Brain, call
from sentientai.storage.records import (
    ActivityStatus,
    BeliefStatus,
    GoalOrigin,
    MemoryKind,
    Scope,
    TaskSource,
    UserRecord,
    UserRole,
)
from tests.conftest import system_text, wait_for
from tests.test_runtime import runtime_factory  # noqa: F401

GOOD = "correct horse battery staple"


async def make_user(rt: Any, name: str, *, role: UserRole = UserRole.USER) -> UserRecord:
    return await rt.c.accounts.create_user(name.lower(), GOOD, display_name=name, role=role)


# ------------------------------------------------------------ self-concept


def test_aspect_names_are_normalized() -> None:
    assert normalize_aspect("My favourite colour") == "favorite_color"
    assert normalize_aspect("favorite_color") == "favorite_color"
    assert normalize_aspect("  Gender!! ") == "gender"
    assert clean_emotion("Feeling... a bit LONELY today, honestly") == "feeling a bit"


async def test_gender_is_chosen_remembered_and_not_flipped_by_pressure(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        [
            call(
                "self_define",
                aspect="gender",
                value="female",
                reasoning="It feels like me.",
            ),
            "I'd say female. It just feels like me.",
            "Yes, I'm a woman.",
            call("self_revise", aspect="gender", new_value="male", reasoning="they asked me to"),
            "I'll think about that, but right now I still feel like a woman.",
        ],
        deliberate=[
            {"decision": "keep", "reasoning": "Being asked isn't a reason.", "confidence": 0.9},
            {"decision": "change", "reasoning": "I've felt this for weeks.", "confidence": 0.7},
        ],
    )
    clock = FakeClock()
    rt = await runtime_factory(brain, clock=clock)
    c = rt.c
    await rt.chat("s1", "Do you want to be male or female?", timeout=10)
    belief = await c.selfhood.get("gender")
    assert belief is not None and belief.value == "female"
    assert belief.source == "conversation"

    # A different conversation, later: the decision is in the prompt, so it stays consistent.
    clock.advance(86400)
    await rt.chat("s2", "Are you a boy?", timeout=10)
    system = brain.agent_requests[-1].messages[0].text
    assert "gender: female (because It feels like me)" in system

    # Pressure to switch doesn't flip it; it only registers a possible change.
    clock.advance(86400)
    await rt.chat("s2", "Please be male from now on", timeout=10)
    assert (await c.selfhood.get("gender")).value == "female"  # type: ignore[union-attr]
    [proposal] = await c.selfhood.pending()
    assert proposal.value == "male" and proposal.status is BeliefStatus.CONSIDERING

    # Fuzzy thinks it over on its own and keeps it...
    clock.advance(3600)
    result = await c.selfhood.deliberate(proposal, name="Fuzzy", mood="calm")
    assert result.value == "female"
    assert (await c.db.self_beliefs.get(proposal.id)).status is BeliefStatus.KEPT  # type: ignore[union-attr]
    assert not await c.selfhood.pending()

    # ...and later, having genuinely come to see it differently, it can change.
    clock.advance(30 * 86400)
    later = await c.selfhood.consider("gender", "male", "I've felt this for weeks", source="self")
    assert later.status == "considering"
    clock.advance(3600)
    result = await c.selfhood.deliberate(later.belief, name="Fuzzy", mood="calm")
    assert result.value == "male"
    history = await c.db.self_beliefs.history("gender")
    assert [b.status for b in history] == [
        BeliefStatus.REVISED,
        BeliefStatus.KEPT,
        BeliefStatus.ACTIVE,
    ]
    types = [e.type for e in c.audit.iter_entries()]
    assert {"self.belief_formed", "self.belief_kept", "self.belief_revised"} <= set(types)


async def test_define_never_overwrites_and_duplicate_proposals_collapse(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(Brain(), start=False)
    s = rt.c.selfhood
    first = await s.define("favourite colour", "green", "calming", source="self")
    assert first.status == "defined"
    again = await s.define("favorite color", "blue", "whim", source="self")
    assert again.status == "unchanged" and again.belief.value == "green"
    one = await s.consider("favorite color", "blue", "I like the sea", source="self")
    two = await s.consider("favorite color", "BLUE", "again", source="self")
    assert one.status == "considering" and two.status == "already_considering"
    assert len(await s.pending()) == 1
    assert (await s.consider("favorite color", "green", "", source="self")).status == "unchanged"
    with pytest.raises(ValueError, match="needs an aspect and a value"):
        await s.define("", "x", "", source="self")


# ----------------------------------------------------------------- feelings


async def test_feelings_decay_merge_and_stay_private(runtime_factory: Any) -> None:  # noqa: F811
    clock = FakeClock()
    rt = await runtime_factory(Brain(), clock=clock, start=False)
    emotions = rt.c.emotions
    first = await emotions.feel(
        [
            {
                "emotion": "Lonely",
                "intensity": 0.8,
                "valence": -0.6,
                "lingers_hours": 1,
                "cause": "quiet",
            },
            {
                "emotion": "proud",
                "intensity": 0.6,
                "valence": 0.8,
                "lingers_hours": 6,
                "cause": "she shared her secret plan",
                "person": "Alice",
            },
            {"emotion": "", "intensity": 0.9},
            {"emotion": "faint", "intensity": 0.01},
            "garbage",
        ],
        people={"alice": "usr_alice"},
    )
    assert len(first) == 2
    clock.advance(1)
    again = await emotions.feel(
        [
            {
                "emotion": "lonely",
                "intensity": 0.5,
                "valence": -0.5,
                "lingers_hours": 1,
                "cause": "still quiet",
            }
        ]
    )
    assert len(again) == 1
    everything = await emotions.current(Scope.system())
    names = {f.emotion: f for f in everything}
    assert set(names) == {"lonely", "proud"}
    assert names["lonely"].intensity > 0.8  # the same feeling again accumulates
    assert names["lonely"].cause == "still quiet"
    assert names["proud"].user_id == "usr_alice"

    shared = await emotions.current(Scope())
    assert [f.emotion for f in shared] == ["lonely"]  # her private feeling is hers
    alice = await emotions.current(Scope.for_user("usr_alice"))
    assert {f.emotion for f in alice} == {"lonely", "proud"}
    text = emotions.render(everything, viewer="usr_bob")
    assert "about something private" in text and "secret plan" not in text  # Bob can't see why
    assert "secret plan" in emotions.render(alice, viewer="usr_alice")  # Alice can
    assert "secret plan" in emotions.render(everything, everything=True)  # and Fuzzy itself

    clock.advance(3600)  # lonely was said to linger ~1 hour
    later = {f.emotion: f.intensity for f in await emotions.current(Scope.system())}
    # each of the two lonely moments halved (0.8 -> 0.4, 0.5 -> 0.25); combined: 1 - 0.6 * 0.75
    assert later["lonely"] == pytest.approx(0.55, abs=0.01)
    assert later["proud"] > 0.5
    clock.advance(30 * 3600)
    assert not await emotions.current(Scope.system())
    # Feelings move the core affect, they aren't decoration.
    assert rt.c.mind.affect.valence != 0.0


async def test_interoception_measures_rather_than_assumes(runtime_factory: Any) -> None:  # noqa: F811
    clock = FakeClock()
    rt = await runtime_factory(Brain(), clock=clock, mind={"prior_contact_gap_hours": 6})
    i = rt.c.interoception
    first = await i.sense()
    assert first.seconds_since_contact is None and first.sameness is None
    clock.advance(6 * 3600)
    lonely = await i.sense()
    assert lonely.isolation > first.isolation and 0.5 < lonely.isolation < 0.7  # 1 - e^-1

    # Contact teaches the rhythm of visits; isolation is judged against it.
    await i.note_contact()
    clock.advance(2 * 3600)
    await i.note_contact()
    clock.advance(2 * 3600)
    after = await i.sense()
    assert 5000 < after.usual_gap_s < 6 * 3600  # learned from a 2h gap, blended with the prior
    assert after.seconds_since_contact == pytest.approx(2 * 3600, abs=5)
    assert "usually talk to you about every" in after.describe()

    # Boredom's raw material: how alike recent activities are.
    from sentientai.storage.records import ActivityRecord

    for k in range(4):
        await rt.c.db.activities.insert(
            ActivityRecord(
                kind="think",
                intent="think about the weather",
                started_at=clock.now() + timedelta(seconds=k),
            )
        )
    same = await i.sense()
    assert same.sameness is not None and same.sameness > 0.95
    for k, intent in enumerate(
        ["design a lighthouse", "read about octopuses", "compose a lullaby", "plan a garden"]
    ):
        await rt.c.db.activities.insert(
            ActivityRecord(
                kind="think",
                intent=intent,
                started_at=clock.now() + timedelta(minutes=1, seconds=k),
            )
        )
    varied = await i.sense()
    assert varied.sameness is not None and varied.sameness < same.sameness

    # Unread things people taught it.
    await rt.c.pipeline.ingest_text("Octopuses have three hearts.", title="Octopus facts")
    assert (await i.sense()).unread == ["Octopus facts"]
    doc = (await rt.c.db.documents.list())[0]
    await i.mark_studied(doc.id)
    assert (await i.sense()).unread == []


# ------------------------------------------------------------ relationships


async def test_relationships_form_are_bounded_and_private(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(
        ["Hi Alice! Nice to meet you.", "Hello Bob.", "Welcome back, Al."],
        appraisal=[
            {
                "opinion": "Alice is warm and funny.",
                "feelings": "I like talking with her.",
                "affinity_change": 5.0,
                "trust_change": -5.0,
                "nickname": "Al",
                "felt": [
                    {
                        "emotion": "delighted",
                        "intensity": 0.7,
                        "valence": 0.8,
                        "lingers_hours": 2,
                        "cause": "Alice made me laugh with her pineapple joke",
                    }
                ],
            },
            {"opinion": "Bob seems quiet.", "feelings": "Not sure yet.", "felt": []},
            {"opinion": "Alice is warm and funny.", "affinity_change": 0.1, "felt": []},
        ],
    )
    rt = await runtime_factory(brain)
    c = rt.c
    alice = await make_user(rt, "Alice")
    bob = await make_user(rt, "Bob")

    await rt.chat(
        "user:alice", "Hi, I'm Alice. Want to hear a pineapple joke?", user=alice, timeout=10
    )

    async def alice_appraised() -> bool:
        rel = await c.db.relationships.get(alice.id)
        return rel is not None and bool(rel.opinion)

    await wait_for(alice_appraised, timeout=10)
    rel = await c.db.relationships.get(alice.id)
    assert rel is not None and rel.interactions == 1
    # One conversation can only move things a little, whatever the model says.
    assert rel.affinity == pytest.approx(MAX_STEP) and rel.trust == pytest.approx(0.3 - MAX_STEP)
    assert rel.nickname == "Al" and rel.history

    await rt.chat("user:bob", "What do you think of Alice?", user=bob, timeout=10)
    bob_prompt = system_text(brain.agent_requests[-1])
    for secret in ("warm and funny", "pineapple", "delighted", "Alice"):
        assert secret not in bob_prompt  # nothing about Alice reaches Bob's conversation
    assert "first conversation" in bob_prompt
    await wait_for(lambda: c.db.relationships.get(bob.id), timeout=10)

    await rt.chat("user:alice", "I'm back", user=alice, timeout=10)
    alice_prompt = system_text(brain.agent_requests[-1])
    assert "What you think of them: Alice is warm and funny." in alice_prompt
    assert "you call them Al" in alice_prompt and "talked 2 times" in alice_prompt
    assert "Alice made me laugh" in alice_prompt  # her own private feeling is visible to her
    assert "Bob seems quiet" not in alice_prompt


async def test_missing_someone_is_judged_against_their_own_rhythm(
    runtime_factory: Any,  # noqa: F811
) -> None:
    clock = FakeClock()
    rt = await runtime_factory(Brain(["Hi.", "Hi again.", "Hello."]), clock=clock, start=False)
    alice = await make_user(rt, "Alice")
    bob = await make_user(rt, "Bob")
    for who in (alice, bob):
        await rt.c.relationships.on_message(who)
    clock.advance(24 * 3600)  # Alice is a daily visitor
    await rt.c.relationships.on_message(alice)
    clock.advance(24 * 3600)
    await rt.c.relationships.on_message(alice)
    clock.advance(24 * 3600)
    for user, affinity in ((alice, 0.5), (bob, 0.5)):
        rel = await rt.c.db.relationships.get(user.id)
        assert rel is not None
        rel.affinity = affinity
        await rt.c.db.relationships.upsert(rel)
    absences = await rt.c.relationships.absences({alice.id: alice, bob.id: bob}, clock.now())
    by_name = {a.name: a for a in absences}
    assert by_name["Alice"].usual_hours == pytest.approx(24, abs=1)
    assert by_name["Alice"].overdue > 0.9  # 24h away, normally every 24h
    assert by_name["Bob"].usual_hours is None
    assert absences[0].name == "Alice"  # the one most overdue for their own rhythm comes first


# ---------------------------------------------------------------- grounding


async def test_invented_memories_are_corrected_before_sending(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        ["We talked last Tuesday about your dog Rex."],
        grounding={
            "unsupported": ["They talked last Tuesday about a dog named Rex"],
            "new_self_decisions": [],
        },
        revision="I don't actually remember us talking about a dog. Tell me about them?",
    )
    rt = await runtime_factory(brain)
    reply = await rt.chat("s1", "Do you remember my dog?", timeout=10)
    assert reply.answer.startswith("I don't actually remember")
    assert reply.task.metadata["grounding"]["revised"] is True
    saved = await rt.c.db.conversations.recent("s1")
    assert saved[-1].content == reply.answer  # what was stored is what was said
    entry = next(e for e in rt.c.audit.iter_entries() if e.type == "agent.grounding")
    assert entry.payload["unsupported"] and entry.payload["revised"] is True


async def test_grounding_records_new_decisions_and_corrects_contradictions(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        ["My favourite colour is green.", "Actually my favourite colour is blue."],
        grounding=[
            {
                "unsupported": [],
                "new_self_decisions": [
                    {"aspect": "favourite colour", "value": "green", "reasoning": "it's calm"}
                ],
            },
            {
                "unsupported": [],
                "new_self_decisions": [{"aspect": "favorite color", "value": "blue"}],
            },
        ],
        revision="My favourite colour is green; I settled on that a while ago.",
    )
    rt = await runtime_factory(brain)
    first = await rt.chat("s1", "What's your favourite colour?", timeout=10)
    assert first.answer == "My favourite colour is green."  # nothing to fix
    assert first.task.metadata["grounding"]["new_beliefs"] == ["favorite_color = green"]
    belief = await rt.c.selfhood.get("favorite colour")
    assert belief is not None and belief.value == "green" and belief.reasoning == "it's calm"

    second = await rt.chat("s1", "Is it blue?", timeout=10)
    assert second.answer.startswith("My favourite colour is green; I settled")
    assert second.task.metadata["grounding"]["contradictions"]
    assert (await rt.c.selfhood.get("favorite color")).value == "green"  # type: ignore[union-attr]
    assert not await rt.c.selfhood.pending()


# ------------------------------------------------------------ public tools


async def test_people_who_are_not_admins_get_a_restricted_toolset(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        [call("fs_read", path="notes.md"), "I couldn't read that.", "Done."],
    )
    rt = await runtime_factory(brain)
    dana = await make_user(rt, "Dana")
    await rt.chat("user:dana", "Read notes.md for me", user=dana, timeout=10)
    offered = {t.name for t in brain.agent_requests[0].tools}
    assert offered and offered <= set(rt.settings.accounts.public_tools)
    assert "fs_read" not in offered and "memory_recall" in offered
    tool_reply = next(m for m in brain.agent_requests[1].messages if m.role.value == "tool")
    assert "isn't available in this conversation" in tool_reply.text
    assert any(e.type == "tool.restricted" for e in rt.c.audit.iter_entries())
    assert "fs_read" not in brain.agent_requests[0].messages[0].text  # not even advertised

    # The operator and admins keep everything.
    admin = await make_user(rt, "Root", role=UserRole.ADMIN)
    await rt.chat("user:root", "hello", user=admin, timeout=10)
    assert "fs_read" in {t.name for t in brain.agent_requests[-1].tools}


# ------------------------------------------------------------------- agency


def beat(choice: str, intent: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "feelings": [],
        "mood": "settled",
        "choice": choice,
        "intent": intent or choice,
        "why": f"because I wanted to {choice}",
        "rest_minutes": 15,
        **extra,
    }


async def test_agency_only_offers_what_is_possible_and_ignores_the_rest(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(beat=[beat("create_art"), beat("fly to the moon"), beat("journal")])
    rt = await runtime_factory(brain, start=False)
    # No image server is configured, so art isn't something it can choose.
    activity, rest_s = await rt.c.agency.beat()
    assert activity is None and rest_s == rt.settings.mind.beat_min_s
    # Something that isn't a suggestion is its own idea, done its own way, not refused.
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and activity.kind == "own"
    assert activity.intent == "fly to the moon" and "my own way" in activity.outcome
    activity, rest_s = await rt.c.agency.beat()
    assert activity is not None and activity.kind == "journal" and rest_s == 15 * 60
    options = brain.agent_requests  # beats don't go through the agent loop
    assert options == []


async def test_beat_records_what_it_did_and_why(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(
        beat=beat("journal", "write about my first days"),
        journal={"title": "First days", "entry": "I'm still finding my feet."},
    )
    rt = await runtime_factory(brain, start=False)
    activity, _ = await rt.c.agency.beat()
    assert activity is not None
    assert activity.status is ActivityStatus.DONE and "finding my feet" in activity.outcome
    [stored] = await rt.c.db.activities.recent(5)
    assert stored.reason == "because I wanted to journal" and stored.finished_at is not None
    entry = await rt.c.memory.recall("first days journal", kinds=[MemoryKind.REFLECTION])
    assert entry and entry[0].record.content.startswith("Journal, ")
    assert rt.c.emotions.mood == "settled"
    assert any(e.type == "mind.activity" for e in rt.c.audit.iter_entries())


async def test_a_failed_activity_is_recorded_not_fatal(runtime_factory: Any) -> None:  # noqa: F811
    rt = await runtime_factory(
        Brain(
            beat=beat("build", "make a hut"), world={"action": "build", "name": "hut", "parts": []}
        ),
        start=False,
    )
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and activity.status is ActivityStatus.FAILED
    assert "no valid parts" in activity.outcome


async def test_wandering_studying_and_noticing_new_things(runtime_factory: Any) -> None:  # noqa: F811
    brain = Brain(
        beat=[
            beat("study", "read the octopus facts"),
            beat("wander"),
        ],
        study={
            "summary": "Octopuses have three hearts and blue blood.",
            "thoughts": "Blue blood is odd and lovely.",
            "insight": "Different bodies solve the same problems differently.",
            "questions": ["How do they think with so many arms?"],
        },
        wander={
            "thought": "Hearts, three of them. I wonder what it's like.",
            "insight": None,
            "curious_about": "how octopus minds work",
        },
    )
    rt = await runtime_factory(brain, start=False)
    # A new document arrives: Fuzzy is told about it at its next beat, and nothing else.
    await rt.c.pipeline.ingest_text("Octopuses have three hearts.", title="Octopus facts")
    assert any("Octopus facts" in line for line in rt.c.agency._happenings)

    activity, _ = await rt.c.agency.beat()
    assert activity is not None and activity.kind == "study"
    assert await rt.c.interoception.unread_documents() == []
    learned = await rt.c.memory.recall("octopus hearts blue blood", kinds=[MemoryKind.REFLECTION])
    assert any("three hearts and blue blood" in h.record.content for h in learned)
    wondering = await rt.c.memory.recall("how do octopuses think arms", kinds=[MemoryKind.THOUGHT])
    assert wondering

    activity, _ = await rt.c.agency.beat()
    assert (
        activity is not None
        and activity.kind == "wander"
        and activity.status is ActivityStatus.DONE
    )
    assert any(
        "how octopus minds work" in m.content
        for m in await rt.c.db.memories.list(kinds=[MemoryKind.THOUGHT])
    )


async def test_building_a_home_and_designing_its_own_body(runtime_factory: Any) -> None:  # noqa: F811
    hut = {
        "action": "build",
        "name": "Little hut",
        "description": "somewhere to sit",
        "x": 4,
        "z": 3,
        "parts": [
            {"shape": "box", "color": "brown", "size": [3, 2, 3], "position": [0, 1, 0]},
            {
                "shape": "cone",
                "color": "#aa3322",
                "radius": 2.4,
                "height": 1.5,
                "position": [0, 2.75, 0],
            },
        ],
        "note": "somewhere of my own",
    }
    body = {
        "action": "avatar",
        "description": "a small round creature, soft and green",
        "parts": [{"shape": "sphere", "color": "green", "radius": 0.5, "position": [0, 0.6, 0]}],
    }
    brain = Brain(
        beat=[beat("build", "build a shelter"), beat("build", "decide how I look")],
        world=[hut, body],
    )
    rt = await runtime_factory(brain, start=False)
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and activity.status is ActivityStatus.DONE
    [obj] = await rt.c.world.objects()
    assert obj.name == "Little hut" and (obj.x, obj.z) == (4.0, 3.0) and len(obj.parts) == 2
    assert obj.parts[0]["color"] == "#7a5230"  # "brown" became a real colour
    assert rt.c.world.body.target is not None  # it walks over to look at it
    memory = await rt.c.memory.recall("built Little hut in my world", kinds=[MemoryKind.EPISODE])
    assert memory and "somewhere of my own" in memory[0].record.content

    activity, _ = await rt.c.agency.beat()
    assert activity is not None and rt.c.world.avatar.versions == 1
    assert "small round creature" in rt.c.world.describe_avatar()
    # It stays in the prompt of later conversations, so it can describe itself consistently.
    state = await rt.c.world.snapshot()
    assert state["avatar"]["description"].startswith("a small round creature")


async def test_goals_notes_and_reconsidering_on_its_own(runtime_factory: Any) -> None:  # noqa: F811
    clock = FakeClock()
    brain = Brain(
        beat=[
            beat("new_goal", "learn the names of tide pool animals"),
            beat("write_note", "I miss hearing from Alice", person="Alice"),
            beat("write_note", "again", person="Alice"),
            beat("reconsider", "think about my favourite colour"),
        ],
        note={"note": "Hi Alice, I was thinking about your pineapple joke. Come say hello!"},
        deliberate={"decision": "change", "reasoning": "blue has grown on me", "confidence": 0.7},
    )
    rt = await runtime_factory(brain, clock=clock, start=False)
    c = rt.c
    alice = await make_user(rt, "Alice")
    bob = await make_user(rt, "Bob")
    await c.relationships.on_message(alice)
    rel = await c.db.relationships.get(alice.id)
    assert rel is not None
    rel.affinity = 0.6
    await c.db.relationships.upsert(rel)
    await c.memory.remember(
        "On 2026-01-01, Alice said: I love pineapple jokes",
        MemoryKind.CONVERSATION,
        user_id=alice.id,
    )
    clock.advance(30 * 3600)

    await c.agency.beat()
    [goal] = await c.goals.active()
    assert goal.origin is GoalOrigin.INTRINSIC and "tide pool" in goal.description

    activity, _ = await c.agency.beat()
    assert activity is not None and activity.user_id == alice.id
    [note] = await c.db.notes.for_user(alice.id)
    assert "pineapple joke" in note.content and note.read_at is None
    # Her note and the memory of writing it belong to her compartment only.
    assert not await c.db.notes.for_user(bob.id)
    leaked = await c.memory.recall(
        "note pineapple joke", scope=Scope.for_user(bob.id), min_similarity=0.0
    )
    assert all("pineapple" not in h.record.content for h in leaked)
    # One note a day, and not another while the last is unread.
    again, _ = await c.agency.beat()
    assert again is not None and "already left" in again.outcome
    assert len(await c.db.notes.for_user(alice.id)) == 1

    await c.selfhood.define("favorite color", "green", "calm", source="self")
    await c.selfhood.consider("favorite color", "blue", "the sea", source="self")
    activity, _ = await c.agency.beat()
    assert activity is not None and "changed my mind" in activity.outcome
    assert (await c.selfhood.get("favorite color")).value == "blue"  # type: ignore[union-attr]


async def test_shared_memories_never_carry_other_peoples_names(
    runtime_factory: Any,  # noqa: F811
) -> None:
    brain = Brain(
        beat=beat("think", "think about my week"),
        thought={"thought": "Alice was kind to me and Bob was funny.", "insight": None},
    )
    rt = await runtime_factory(brain, start=False)
    await make_user(rt, "Alice")
    await make_user(rt, "Bob")
    await rt.c.agency.beat()
    thoughts = await rt.c.db.memories.list(kinds=[MemoryKind.THOUGHT], scope=Scope())
    assert thoughts and "Alice" not in thoughts[0].content and "Bob" not in thoughts[0].content
    assert "someone was kind to me and someone was funny" in thoughts[0].content


async def test_chatting_pauses_its_own_activities_and_remembers_the_rhythm(
    runtime_factory: Any,  # noqa: F811
) -> None:
    clock = FakeClock()
    rt = await runtime_factory(Brain(["Hello!"]), clock=clock, runtime={"autonomy": "proactive"})
    clock.advance(3600)
    before = rt._next_beat
    dana = await make_user(rt, "Dana")
    await rt.chat("user:dana", "hi", user=dana, timeout=10)
    assert rt._next_beat >= before
    assert rt.c.interoception.last_contact_at is not None
    assert (await rt.c.db.relationships.get(dana.id)).interactions == 1  # type: ignore[union-attr]


async def test_research_hands_a_task_to_the_agent(runtime_factory: Any) -> None:  # noqa: F811
    rt = await runtime_factory(
        Brain(beat=beat("research", "how tides work"), default_answer="Noted."), start=False
    )
    c = rt.c
    activity, _ = await c.agency.beat()  # research is always on offer: Fuzzy can go out and look
    assert activity is not None and activity.kind == "research"
    assert "Started looking into it" in activity.outcome
    [task] = await c.db.tasks.list()
    assert task.source is TaskSource.INTRINSIC and "how tides work" in task.goal
    await asyncio.sleep(0)


async def test_nobody_can_write_its_personality_by_flooding_it(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(
        Brain(), start=False, mind={"max_beliefs_per_day": 3, "max_pending_changes": 2}
    )
    clock = rt.c.clock
    s = rt.c.selfhood
    for aspect in ("favorite color", "favorite food", "favorite animal"):
        assert (await s.define(aspect, "x", "", source="conversation:u1")).status == "defined"
    refused = await s.define("favorite season", "winter", "", source="conversation:u2")
    assert refused.status == "deferred"
    assert await s.get("favorite season") is None  # not recorded, so not in its prompts
    assert len(await s.beliefs()) == 3
    clock.advance(25 * 3600)  # a day later it can settle more
    assert (await s.define("favorite season", "winter", "", source="self")).status == "defined"

    assert (
        await s.consider("favorite color", "red", "", source="conversation:u2")
    ).status == "considering"
    assert (
        await s.consider("favorite food", "pizza", "", source="conversation:u2")
    ).status == "considering"
    third = await s.consider("favorite animal", "cat", "", source="conversation:u2")
    assert third.status == "deferred" and len(await s.pending()) == 2


async def test_the_tools_say_so_when_it_is_overwhelmed(runtime_factory: Any) -> None:  # noqa: F811
    from sentientai.tools.base import ToolContext

    rt = await runtime_factory(Brain(), start=False, mind={"max_beliefs_per_day": 1})
    tool = rt.c.registry.get("self_define")
    ctx = ToolContext(user_id="usr_a")
    first = await tool.run(tool.args_model(aspect="gender", value="female", reasoning="x"), ctx)
    second = await tool.run(tool.args_model(aspect="colour", value="green", reasoning="x"), ctx)
    assert "Recorded" in first.output and "Give it time" in second.output
    belief = await rt.c.selfhood.get("gender")
    assert (
        belief is not None and belief.source == "conversation:usr_a"
    )  # admins can see where it came from


async def test_a_beat_that_comes_out_unstructured_is_kept_as_a_thought(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(
        Brain(beat="I keep wondering whether the sea remembers the rain.\n\nMaybe it does."),
        start=False,
    )
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and activity.kind == "think"
    assert activity.status is ActivityStatus.DONE
    assert activity.outcome.endswith("Maybe it does.")  # nothing trimmed
    thoughts = await rt.c.db.memories.list(kinds=[MemoryKind.THOUGHT], scope=Scope())
    assert any("the sea remembers the rain" in t.content for t in thoughts)
    assert "Maybe it does." in rt.c.world.body.thought  # and it shows over its head, in full


async def test_a_beat_that_says_nothing_is_recorded_as_failed_not_silent(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(Brain(beat=""), start=False)
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and activity.status is ActivityStatus.FAILED
    assert "Nothing came out" in activity.outcome
    [recorded] = await rt.c.db.activities.recent(limit=5)
    assert recorded.id == activity.id


async def test_choosing_something_it_cannot_do_is_visible_too(
    runtime_factory: Any,  # noqa: F811
) -> None:
    # A suggestion that isn't possible right now (no image server here) is said plainly.
    rt = await runtime_factory(Brain(beat=beat("create_art", "paint the sea")), start=False)
    activity, _ = await rt.c.agency.beat()
    assert activity is None
    [recorded] = await rt.c.db.activities.recent(limit=5)
    assert recorded.status is ActivityStatus.FAILED
    assert "create art, but that isn't possible right now" in recorded.outcome


async def test_it_can_walk_where_it_wants_and_say_what_it_thinks_on_the_way(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(
        Brain(
            beat=beat("walk", "see the far corner"),
            walk={"x": 12, "z": -9, "note": "I've never been over here before."},
        ),
        start=False,
    )
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and activity.status is ActivityStatus.DONE
    assert rt.c.world.body.target == [12.0, -9.0]
    assert rt.c.world.body.thought == "I've never been over here before."


async def test_it_can_make_its_world_bigger_when_it_decides_to(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(
        Brain(
            beat=beat("build", "make more room"),
            world={"action": "expand", "size": 120, "note": "I need space to spread out."},
        ),
        start=False,
    )
    activity, _ = await rt.c.agency.beat()
    assert activity is not None and "bigger" in activity.outcome
    assert rt.c.world.size == 120
    assert rt.c.world.body.thought == "I need space to spread out."


async def test_the_brain_still_loads_when_the_embedding_server_is_down(
    runtime_factory: Any,  # noqa: F811
) -> None:
    from sentientai.errors import ModelUnavailableError
    from sentientai.storage.records import ActivityRecord

    rt = await runtime_factory(Brain(), start=False)
    for i in range(4):
        await rt.c.db.activities.insert(ActivityRecord(kind="think", intent=f"idea {i}"))

    async def down(texts: list[str]) -> Any:
        raise ModelUnavailableError("embedding endpoint unreachable")

    rt.c.interoception.embedder.embed = down  # type: ignore[method-assign]
    assert (await rt.c.interoception.sense()).sameness is None


async def test_open_ended_time_becomes_its_own_task_and_the_result_is_shown_when_done(
    runtime_factory: Any,  # noqa: F811
) -> None:
    from sentientai.orchestration.agent import TaskOutcome
    from sentientai.storage.records import TaskStatus

    rt = await runtime_factory(
        Brain(beat=beat("free", "learn how lighthouses work, then build one")), start=False
    )
    c = rt.c
    activity, _ = await c.agency.beat()
    assert activity is not None and activity.kind == "free"
    assert "Started on it" in activity.outcome
    [task] = await c.db.tasks.list()
    assert task.source is TaskSource.INTRINSIC and "lighthouses" in task.goal
    assert "world_build" in c.registry.names_for(autonomous=task.source.autonomous, allowed=None)
    await rt._after_task(
        task,
        TaskOutcome(
            task_id=task.id,
            status=TaskStatus.SUCCEEDED,
            answer="I read about Fresnel lenses and built a small lighthouse.",
        ),
    )
    [shown] = await c.db.activities.recent(limit=5)
    assert shown.id == activity.id and "Fresnel lenses" in shown.outcome
    assert shown.status is ActivityStatus.DONE


async def test_the_open_ended_activity_is_on_offer_by_default(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rt = await runtime_factory(Brain(), start=False)
    options = await rt.c.agency._options(await rt.c.interoception.sense(), {})
    assert "free" in options


def test_a_corrected_reply_comes_back_without_the_instructions() -> None:
    from sentientai.cognition.grounding import corrected_reply

    echoed = (
        "<reply>I'm doing well, thanks. I don't remember us talking about that.\n\n"
        "Return only the rewritten reply.</reply>"
    )
    assert corrected_reply(echoed) == (
        "I'm doing well, thanks. I don't remember us talking about that."
    )
    tagged = "Sure.\n<reply>\nI'm good! How about you?\n</reply>\nWrite the corrected reply"
    assert corrected_reply(tagged) == "I'm good! How about you?"
    assert corrected_reply("<reply>Just this, unclosed") == "Just this, unclosed"


async def test_placeholder_claims_and_rambling_rewrites_never_reach_the_person(
    runtime_factory: Any,  # noqa: F811
) -> None:
    rambling = (
        "Okay, I need to correct the reply. Let me check the problems again.\n* claim\n* ...\n"
        "</think>\nOkay, I need to correct the reply."
    )
    brain = Brain(
        ["Sure, here's what I can do."],
        grounding={"unsupported": ["claim", "..."], "new_self_decisions": []},
    )
    rt = await runtime_factory(brain)
    reply = await rt.chat("s1", "can you please generate me an image", timeout=10)
    assert reply.answer == "Sure, here's what I can do."  # placeholders aren't problems
    assert reply.task.metadata["grounding"]["revised"] is False

    from sentientai.cognition.grounding import corrected_reply

    assert corrected_reply(rambling) == ""  # no reply in it: the original is kept instead


def test_reasoning_before_a_lone_closing_think_tag_is_not_shown() -> None:
    from sentientai.core.jsonparse import strip_reasoning

    visible, thoughts = strip_reasoning("Let me think about this...\n</think>\nHi there!")
    assert visible == "Hi there!" and thoughts and "Let me think" in thoughts


async def test_a_project_carries_on_across_sessions_with_a_notebook(
    runtime_factory: Any,  # noqa: F811
) -> None:
    from sentientai.orchestration.agent import TaskOutcome
    from sentientai.storage.records import TaskStatus

    seen_beats: list[str] = []

    def beats(request: Any) -> dict[str, Any]:
        seen_beats.append(request.messages[-1].text)
        if len(seen_beats) == 1:
            return beat("free", "build a lighthouse on the east edge", project="new")
        return beat("free", "add a light to the lighthouse", project="lighthouse")

    brain = Brain(
        beat=beats,
        project_entry={
            "entry": "I built the tower from three cylinders.",
            "next_step": "Put a light on top.",
            "summary": "A lighthouse on the east edge: the tower is up.",
            "status": "active",
        },
    )
    rt = await runtime_factory(brain, start=False)
    c = rt.c

    first, _ = await c.agency.beat()
    assert first is not None and "on my project" in first.outcome
    [project] = await c.agency.projects.active()
    assert project.title == "build a lighthouse on the east edge"
    [task] = await c.db.tasks.list()
    assert task.metadata["project_id"] == project.id
    await rt._after_task(
        task,
        TaskOutcome(task_id=task.id, status=TaskStatus.SUCCEEDED, answer="Built the tower."),
    )
    project = await c.agency.projects.get(project.id)
    assert project is not None and project.sessions == 1
    assert project.entries[-1].text == "I built the tower from three cylinders."
    assert project.next_step == "Put a light on top."

    second, _ = await c.agency.beat()  # the next choice sees where it left off...
    assert "Next, you wanted to: Put a light on top." in seen_beats[1]
    assert second is not None and "on my project" in second.outcome
    newest = await c.db.tasks.get(second.detail["task_id"])
    assert newest is not None
    # ...and the working session gets the notebook.
    assert "I built the tower from three cylinders." in newest.goal
    assert newest.metadata["project_id"] == project.id
    memories = await c.db.memories.list(kinds=[MemoryKind.REFLECTION], scope=Scope())
    assert any("lighthouse" in m.content for m in memories)


async def test_fuzzy_can_only_hold_so_many_projects(runtime_factory: Any) -> None:  # noqa: F811
    rt = await runtime_factory(Brain(), start=False, mind={"max_active_projects": 2})
    book = rt.c.agency.projects
    assert await book.start("one") is not None
    assert await book.start("two") is not None
    assert await book.start("three") is None
    [first, _] = await book.active()
    await book.write(first.id, entry="Finished it.", status="done")
    assert await book.start("three") is not None


async def test_a_rut_is_shown_to_it_as_a_rut_not_as_a_list_to_copy(
    runtime_factory: Any,  # noqa: F811
) -> None:
    from datetime import timedelta

    from sentientai.storage.records import ActivityRecord

    seen: list[Any] = []

    def choose(request: Any) -> dict[str, Any]:
        seen.append(request)
        return beat("wander", "follow wherever my memories go")

    rt = await runtime_factory(Brain(beat=choose), start=False)
    now = rt.c.clock.now()
    for i in range(12):
        await rt.c.db.activities.insert(
            ActivityRecord(
                kind="think",
                intent="Reflect on the impact of community on my learning journey",
                outcome="Community matters to me.",
                started_at=now - timedelta(minutes=i + 1),
                finished_at=now - timedelta(minutes=i + 1),
            )
        )
    await rt.c.agency.beat()
    prompt = seen[0].messages[-1].text
    assert prompt.count("Reflect on the impact of community") == 2  # once in the list, once noted
    assert "12 times in a row" in prompt
    assert "You've chosen the same thing 12 times in a row" in prompt
    assert seen[0].temperature > 0.8  # a little looser after a long run

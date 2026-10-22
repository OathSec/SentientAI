from __future__ import annotations

import itertools
import json

import pytest

from sentientai.cognition.affect import AffectState
from sentientai.cognition.goals import GoalManager
from sentientai.cognition.metacognition import Metacognition
from sentientai.cognition.self_model import SelfModel, humanize_duration
from sentientai.cognition.workspace import GlobalWorkspace
from sentientai.config import IdentityConfig, RoutingConfig
from sentientai.core.clock import FakeClock
from sentientai.core.events import EventBus
from sentientai.memory.working import Turn
from sentientai.models.router import ModelRouter
from sentientai.models.scripted import ScriptedLLM
from sentientai.models.types import Message, ModelTier, Role, ToolCall
from sentientai.orchestration.context import ContextBuilder, ContextCompressor, FramingInputs
from sentientai.orchestration.planner import Planner
from sentientai.storage.database import Database
from sentientai.storage.records import (
    GoalOrigin,
    GoalStatus,
    MemoryKind,
    MemoryRecord,
    ScoredMemory,
)


def router(*replies: object, fast_window: int = 32768) -> ModelRouter:
    llm = ScriptedLLM(list(replies), context_window=fast_window)  # type: ignore[arg-type]
    return ModelRouter({ModelTier.FAST: llm, ModelTier.HEAVY: llm}, RoutingConfig())


# ------------------------------------------------------------------ affect


def test_affect_appraisal_and_homeostasis() -> None:
    clock = FakeClock()
    a = AffectState(updated_at=clock.now())
    a.on_task_outcome(success=False)
    a.on_task_outcome(success=False)
    assert a.valence < -0.5 and a.competence < 0.4 and a.arousal > 0.5
    mod = a.modulation()
    assert mod.prefer_heavy_model or mod.cautious

    clock.advance(3600)
    a.tick(clock.now())
    assert abs(a.valence) < 0.05  # feelings relax toward baseline
    assert a.arousal < 0.3
    assert a.curiosity > 0.6  # idleness breeds curiosity

    a.on_resource_use(tokens=900_000, hourly_budget=1_000_000)
    assert a.energy < 0.2 and not a.modulation().may_take_initiative
    clock.advance(3600)
    a.tick(clock.now())
    assert a.energy == 1.0

    for _ in range(5):
        a.on_task_outcome(success=True, confidence=0.9)
    assert a.competence > 0.6 and not a.modulation().prefer_heavy_model
    assert "I feel" in a.describe() and "confidence in my abilities" in a.describe()


def test_affect_bounds() -> None:
    a = AffectState()
    for _ in range(50):
        a.on_operator_contact()
        a.on_novelty(1.0)
        a.on_loop_break()
    snap = a.snapshot()
    assert all(-1.0 <= v <= 1.0 for v in snap.values())
    assert 0.0 <= a.coherence <= 1.0


# -------------------------------------------------------------- self model


def test_self_model_continuity() -> None:
    clock = FakeClock()
    sm = SelfModel.from_identity(IdentityConfig(name="Ada"))
    sm.on_boot(clock.now())
    assert "first awakening" in sm.render_continuity(clock.now())
    clock.advance(7200)
    sm.last_focus = "indexing the research papers"
    sm.on_shutdown(clock.now())
    clock.advance(86400)
    sm.on_boot(clock.now())
    text = sm.render_continuity(clock.now())
    assert (
        "awakening #2" in text
        and "offline for 1 day" in text
        and "indexing the research papers" in text
    )
    assert sm.total_uptime_s == 7200
    sm.sync_identity(IdentityConfig(name="Grace"))
    assert sm.render_identity().startswith("You are Grace.")
    sm.add_limitation("I cannot browse JavaScript-heavy sites.")
    assert "JavaScript" in sm.render_identity()
    assert humanize_duration(90) == "2 minutes" and humanize_duration(30) == "30 seconds"


# --------------------------------------------------------------- workspace


async def test_workspace_competition_habituation_and_starvation() -> None:
    clock = FakeClock()
    bus = EventBus()
    ws = GlobalWorkspace(capacity=2, bus=bus, clock=clock, habituation_window_s=600)
    ws.submit(source="drive", kind="drive", content="curious", salience=0.6)
    ws.submit(
        source="operator", kind="task", content="answer the operator", salience=0.95, key="task:1"
    )
    ws.submit(source="watcher", kind="task", content="new file", salience=0.4)
    winners = await ws.cycle()
    assert [p.source for p in winners] == ["operator", "drive"]
    assert ws.focus() is not None and ws.focus().source == "operator"  # type: ignore[union-attr]
    assert bus.recent("workspace.broadcast")

    # resubmitting with the same key updates in place
    ws.submit(source="operator", kind="task", content="updated", salience=0.9, key="task:1")
    assert len(ws) == 3

    ws.attend(ws.focus())  # type: ignore[arg-type]
    drive = next(p for p in ws.conscious if p.source == "drive")
    ws.attend(drive)
    again = ws.submit(source="drive", kind="drive", content="curious again", salience=0.6)
    assert ws.effective_salience(again, clock.now()) < 0.6  # habituated

    clock.advance(30 * 60)  # the waiting watcher task gains salience over time
    watcher = next(p for p in ws._percepts.values() if p.source == "watcher")
    assert ws.effective_salience(watcher, clock.now()) == pytest.approx(0.6)

    ws.submit(source="system", kind="signal", content="transient", salience=1.0, ttl_s=5)
    clock.advance(10)
    await ws.cycle()
    assert all(p.content != "transient" for p in ws.conscious)
    assert "watcher" in ws.render()


# ------------------------------------------------------------------- goals


async def test_goal_manager(sqlite_db: Database) -> None:
    goals = GoalManager(sqlite_db.goals, max_intrinsic=1)
    g1 = await goals.add("Learn the operator's codebase", origin=GoalOrigin.INTRINSIC)
    assert g1 is not None
    dup = await goals.add("learn the operator's   codebase")
    assert dup is not None and dup.id == g1.id
    assert await goals.add("Another self goal", origin=GoalOrigin.INTRINSIC) is None  # slots full
    await goals.add("Operator goal", priority=0.9)
    updated = await goals.update(g1.id, progress=1.0, note="done reading")
    assert updated.status is GoalStatus.COMPLETED and updated.notes == ["done reading"]
    assert [g.description for g in await goals.active()] == ["Operator goal"]
    assert "Operator goal" in GoalManager.render(await goals.active())
    with pytest.raises(KeyError):
        await goals.update("missing")


# ----------------------------------------------------------- metacognition


def _mem(similarity: float) -> ScoredMemory:
    return ScoredMemory(
        record=MemoryRecord(kind=MemoryKind.FACT, content="x"),
        score=similarity,
        similarity=similarity,
    )


async def test_metacognition() -> None:
    assert Metacognition.novelty([]) == 1.0
    assert Metacognition.novelty([_mem(0.9)]) == pytest.approx(0.1)
    h_good = Metacognition.heuristic(
        tool_calls=4, tool_failures=0, pressure=0.2, memory_coverage=0.8
    )
    h_bad = Metacognition.heuristic(
        tool_calls=4, tool_failures=3, pressure=0.95, memory_coverage=0.0
    )
    assert h_good > 0.7 > 0.4 > h_bad

    meta = Metacognition(router(json.dumps({"confidence": 0.2, "gaps": ["no source checked"]})))
    a = await meta.assess(
        goal="g",
        answer="a",
        evidence="",
        tool_calls=0,
        tool_failures=0,
        pressure=0.0,
        memory_coverage=0.5,
    )
    assert a.model_confidence == 0.2 and a.gaps == ["no source checked"]
    assert a.confidence == pytest.approx(0.6 * 0.2 + 0.4 * a.heuristic)
    assert meta.should_escalate(a, ModelTier.FAST) and not meta.should_escalate(a, ModelTier.HEAVY)

    broken = Metacognition(router("not json"))
    fallback = await broken.assess(
        goal="g",
        answer="a",
        evidence="",
        tool_calls=0,
        tool_failures=0,
        pressure=0.0,
        memory_coverage=0.5,
    )
    assert fallback.model_confidence is None and fallback.confidence == fallback.heuristic
    empty = await broken.assess(
        goal="g",
        answer="  ",
        evidence="",
        tool_calls=0,
        tool_failures=0,
        pressure=0.0,
        memory_coverage=0.5,
    )
    assert empty.confidence == 0.05


# ----------------------------------------------------------------- planner


async def test_planner() -> None:
    planner = Planner(
        router(json.dumps({"steps": ["read file", "summarize"], "complexity": "low"})), min_chars=50
    )
    assert not planner.should_plan("hi")
    assert planner.should_plan("first read the file, then summarize it and finally email it")
    assert planner.should_plan("x", cautious=True)
    plan = await planner.plan("do it")
    assert plan is not None and plan.render() == "1. read file\n2. summarize"
    assert await Planner(router("garbage")).plan("x") is None
    assert await Planner(router(json.dumps({"steps": []}))).plan("x") is None


# ----------------------------------------------------------------- context


def _inputs(history: list[Turn], memories: list[ScoredMemory] | None = None) -> FramingInputs:
    clock = FakeClock()
    return FramingInputs(
        goal="What GPU runs the heavy model?",
        identity="You are Test.",
        continuity="first boot",
        inner_state="calm",
        awareness="nothing",
        goals="none",
        memories=memories or [],
        procedures=[],
        plan="1. check memory",
        history=history,
        now=clock.now(),
    )


def test_context_builder_structure_and_budget() -> None:
    history = [
        Turn(role="system", content="Earlier they discussed GPUs.", metadata={"summary": True}),
        Turn(role="assistant", content="orphan assistant turn"),
        Turn(role="user", content="hello"),
        Turn(role="assistant", content="hi there"),
        Turn(role="user", content="and another"),
    ]
    memories = [
        ScoredMemory(
            record=MemoryRecord(kind=MemoryKind.FACT, content=f"fact {i} " + "x" * 400),
            score=1 - i / 100,
            similarity=0.9,
        )
        for i in range(50)
    ]
    msgs = ContextBuilder().build(_inputs(history, memories), window=4096, reserve_output=1024)
    system = msgs[0].text
    assert msgs[0].role is Role.SYSTEM and "You are Test." in system and "## Plan" in system
    assert "Earlier they discussed GPUs." in system and "## Relevant memories" in system
    assert system.count("fact ") < 50  # memory section respects its budget
    roles = [m.role for m in msgs[1:]]
    assert (
        roles[0] is Role.USER and roles[-1] is Role.USER
    )  # alternation starts with user, ends with task
    assert all(a is not b for a, b in itertools.pairwise(roles))  # strictly alternating
    assert msgs[-1].text == "What GPU runs the heavy model?"


async def test_context_compressor_keeps_tool_pairs() -> None:
    summary_router = router("We read three files and found the config.")
    comp = ContextCompressor(summary_router, keep_recent=3, threshold=0.5)
    msgs = [Message.system("sys"), Message.user("task")]
    for i in range(12):
        call = ToolCall(id=f"c{i}", name="fs_read", arguments={"path": f"f{i}"})
        msgs += [Message.assistant(None, [call]), Message.tool(f"c{i}", "fs_read", "data " * 400)]
    out = await comp.fit(msgs, window=4000, reserve_output=500)
    assert len(out) < len(msgs)
    assert out[0].text == "sys" and out[1].text == "task"
    assert "We read three files" in out[2].text
    first_tail = out[4]
    assert (
        first_tail.role is Role.ASSISTANT and first_tail.tool_calls
    )  # never starts with an orphan tool msg
    tool_ids = {m.tool_call_id for m in out if m.role is Role.TOOL}
    call_ids = {c.id for m in out for c in m.tool_calls}
    assert tool_ids <= call_ids

    small = [Message.system("s"), Message.user("t")]
    assert await comp.fit(small, window=4000, reserve_output=500) == small

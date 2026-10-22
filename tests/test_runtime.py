from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from sentientai.config import Settings, load_settings
from sentientai.core.clock import FakeClock
from sentientai.errors import ModelError
from sentientai.evals.brain import Brain, call
from sentientai.memory.types import Procedure
from sentientai.models.embeddings import HashingEmbedder
from sentientai.models.types import ModelTier
from sentientai.runtime.runtime import Runtime
from sentientai.storage.records import (
    ApprovalStatus,
    MemoryKind,
    Scope,
    TaskRecord,
    TaskSource,
    TaskStatus,
)
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult
from tests.conftest import wait_for


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "storage": {"data_dir": tmp_path / "data"},
        "safety": {
            "workspace_roots": [tmp_path / "workspace"],
            "audit_fsync": False,
            "approval_timeout_s": 5,
        },
        "sandbox": {"backend": "disabled"},
        "web": {"search_backend": "disabled"},
        "models": {"embedding": {"dimensions": 256}},
        "limits": {"llm_requests_per_minute": 1e6, "tool_calls_per_minute": 1e6, "max_steps": 8},
        "runtime": {"tick_interval_s": 0.02, "checkpoint_interval_s": 1, "autonomy": "reactive"},
    }
    for key, value in overrides.items():
        base[key] = {**base.get(key, {}), **value}
    s = load_settings(**base)
    s.ensure_directories()
    (s.safety.workspace_roots[0] / "notes.md").write_text(
        "# Notes\nThe heavy model runs on two Radeon Pro V340s.\n"
    )
    return s


@pytest.fixture
async def runtime_factory(tmp_path: Path) -> AsyncIterator[Any]:
    created: list[Runtime] = []

    async def factory(
        brain: Brain,
        *,
        heavy: Brain | None = None,
        clock: FakeClock | None = None,
        tools: list[Tool[Any]] | None = None,
        start: bool = True,
        **overrides: Any,
    ) -> Runtime:
        settings = make_settings(tmp_path, **overrides)
        clients = {
            ModelTier.FAST: brain.llm("fast"),
            ModelTier.HEAVY: (heavy or brain).llm("heavy"),
        }
        rt = await Runtime.create(
            settings,
            clients=clients,
            embedder=HashingEmbedder(256),
            clock=clock or FakeClock(),
            extra_tools=tools,
        )
        created.append(rt)
        if start:
            await rt.start()
        return rt

    yield factory
    for rt in created:
        await rt.stop("test teardown")


async def test_chat_end_to_end(runtime_factory: Any) -> None:
    reflection = {
        "summary": "Read the notes and answered.",
        "lessons": ["Check notes.md for hardware facts"],
        "facts": [
            {
                "statement": "The heavy model runs on two Radeon Pro V340 GPUs",
                "subject_key": "hardware.heavy_gpu",
                "importance": 0.8,
                "confidence": 0.95,
            }
        ],
        "procedure": {
            "name": "Answer hardware questions",
            "when_to_use": "asked about hardware",
            "steps": ["fs_search for the term", "fs_read the file", "answer"],
        },
        "self_assessment": 0.9,
    }
    brain = Brain(
        [
            call("fs_search", pattern="V340"),
            call("fs_read", path="notes.md"),
            "The heavy model runs on two Radeon Pro V340 GPUs (source: notes.md).",
        ],
        reflection=reflection,
    )
    rt = await runtime_factory(brain)
    reply = await rt.chat("s1", "Which GPUs run the heavy model?", timeout=10)

    assert "V340" in reply.answer and reply.task.status is TaskStatus.SUCCEEDED
    assert reply.confidence is not None and reply.confidence > 0.7
    c = rt.c
    # tool calls were real and recorded
    stats = {s.tool: s for s in await c.db.tool_invocations.stats()}
    assert stats["fs_read"].calls == 1 and stats["fs_search"].calls == 1
    # identity, inner state and awareness framed the conversation
    system = brain.agent_requests[0].messages[0].text
    assert "You are Fuzzy." in system and "## How you feel" in system and "awakening" in system
    # reflection produced an episode, a fact and a procedure
    # Reflection runs after the answer is delivered; wait for the task handle to fully finish.
    await wait_for(lambda: not rt._running and not rt._afterthoughts)
    assert brain.calls.get("reflect", 0) == 1
    assert await c.db.memories.find_by_subject("hardware.heavy_gpu")
    episodes = await c.memory.timeline()
    assert episodes and "Read the notes" in episodes[0].content
    [proc] = await c.memory.find_procedures("hardware question about GPUs")
    assert Procedure.from_record(proc.record).successes == 1
    # conversation persisted to both stores
    assert [m.role for m in await c.db.conversations.recent("s1")] == ["user", "assistant"]
    assert len(await c.memory.conversation("s1")) == 2
    # inner state reacted
    assert c.mind.affect.competence > 0.6 and c.mind.affect.social > 0.5
    assert c.audit.verify().ok
    types = {e.type for e in c.audit.iter_entries()}
    assert {
        "runtime.boot",
        "task.submitted",
        "task.started",
        "tool.requested",
        "tool.completed",
        "agent.answer",
        "agent.assessment",
        "task.finished",
    } <= types


async def test_second_chat_turn_sees_history(runtime_factory: Any) -> None:
    brain = Brain(["Hello Dana!", "You told me your name is Dana."])
    rt = await runtime_factory(brain)
    await rt.chat("s1", "Hi, I'm Dana", timeout=10)
    await rt.chat("s1", "What's my name?", timeout=10)
    second = brain.agent_requests[-1].messages
    assert [m.text for m in second[1:3]] == ["Hi, I'm Dana", "Hello Dana!"]
    assert second[-1].text == "What's my name?"


async def test_high_risk_action_requires_approval(runtime_factory: Any) -> None:
    brain = Brain(
        [call("fs_delete", path="notes.md"), "I did not delete it because the operator declined."]
    )
    rt = await runtime_factory(brain)
    task = await rt.submit("Delete notes.md", source=TaskSource.API)

    await wait_for(lambda: rt.c.approvals.pending(), timeout=10)
    [request] = await rt.c.approvals.pending()
    assert request.tool == "fs_delete" and request.risk.name == "HIGH"
    await rt.c.approvals.decide(request.id, approved=False, by="operator", note="keep it")

    final = await rt.wait(task.id, timeout=10)
    assert final.status is TaskStatus.SUCCEEDED and "declined" in (final.result or "")
    assert (rt.settings.safety.workspace_roots[0] / "notes.md").exists()
    [record] = await rt.c.db.approvals.list()
    assert record.status is ApprovalStatus.DENIED


async def test_loop_breaker_and_salvage(runtime_factory: Any) -> None:
    brain = Brain([call("fs_read", path="notes.md") for _ in range(10)])
    rt = await runtime_factory(brain, limits={"max_repeat_calls": 2})
    task = await rt.submit("Read notes forever")
    final = await rt.wait(task.id, timeout=10)
    assert final.status is TaskStatus.FAILED
    assert final.metadata["loop_break"] == "repetition"
    assert final.result == "Partial answer: I was stopped before finishing."
    assert rt.c.metrics.loop_breaks.value(limit="repetition") == 1
    assert rt.c.mind.affect.coherence < 0.85


async def test_low_confidence_escalates_to_heavy(runtime_factory: Any) -> None:
    fast = Brain(["Probably 42?"], confidence=0.1)
    heavy = Brain(["After checking: 42, verified."], confidence=0.9)
    rt = await runtime_factory(
        fast, heavy=heavy, models={"routing": {"heavy_task_kinds": ["code"]}}
    )
    task = await rt.submit("What is six times seven?")
    final = await rt.wait(task.id, timeout=10)
    assert final.result == "After checking: 42, verified."
    assert final.metadata["escalated"] is True and final.metadata["tier"] == "heavy"
    assert any(e.type == "agent.escalate" for e in rt.c.audit.iter_entries())


async def test_heavy_outage_falls_back_to_fast(runtime_factory: Any) -> None:
    heavy = Brain([ModelError("GPU OOM")] * 20)
    fast = Brain(["Answered by the fast tier."], confidence=0.9)
    rt = await runtime_factory(fast, heavy=heavy)
    final = await rt.wait((await rt.submit("Explain something")).id, timeout=10)
    assert final.status is TaskStatus.SUCCEEDED and final.result == "Answered by the fast tier."


class SlowArgs(ToolArgs):
    pass


class SlowTool(Tool[SlowArgs]):
    name = "slow_job"
    description = "takes a while"
    args_model = SlowArgs
    timeout_s = 30

    async def run(self, args: SlowArgs, ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(20)
        return ToolResult.success("finished")


async def test_kill_switch_cancels_and_pauses(runtime_factory: Any) -> None:
    brain = Brain([call("slow_job")], default_answer="resumed work")
    rt = await runtime_factory(brain, tools=[SlowTool()])
    first = await rt.submit("long job")
    # Engage only once the slow tool is actually executing (not merely once the task started).
    await wait_for(lambda: rt.c.bus.recent("tool.started"), timeout=10)
    await rt.c.kill_switch.engage("operator emergency stop")
    final = await rt.wait(first.id, timeout=5)
    assert final.status is TaskStatus.CANCELLED

    second = await rt.submit("should wait while killed")
    await asyncio.sleep(0.3)
    assert (await rt.c.db.tasks.get(second.id)).status is TaskStatus.PENDING  # type: ignore[union-attr]
    await rt.c.kill_switch.release()
    resumed = await rt.wait(second.id, timeout=10)
    assert resumed.status is TaskStatus.SUCCEEDED and resumed.result == "resumed work"
    assert rt.c.metrics.kill_switch.value() == 0.0


async def test_restart_continuity_and_recovery(runtime_factory: Any, tmp_path: Path) -> None:
    brain = Brain(
        ["Noted: the deploy window is Friday."],
        reflection={
            "summary": "Learned the deploy window.",
            "lessons": [],
            "facts": [
                {
                    "statement": "Deploy window is Friday afternoon",
                    "subject_key": "ops.deploy_window",
                    "importance": 0.8,
                    "confidence": 0.9,
                }
            ],
        },
    )
    rt1 = await runtime_factory(brain)
    await rt1.chat("s1", "Remember the deploy window is Friday", timeout=10)
    await wait_for(lambda: rt1.c.db.memories.find_by_subject("ops.deploy_window"))
    # simulate a task that was mid-flight when the process died
    orphan = TaskRecord(goal="interrupted work", status=TaskStatus.RUNNING, attempts=1)
    await rt1.c.db.tasks.insert(orphan)
    await rt1.stop("simulated reboot")

    brain2 = Brain(["Finished the interrupted work."])
    rt2 = await runtime_factory(brain2)
    sm = rt2.c.mind.self_model
    assert sm.boot_count == 2 and sm.tasks_succeeded == 1
    assert sm.last_focus == "Remember the deploy window is Friday"
    recovered = await rt2.wait(orphan.id, timeout=10)
    assert recovered.status is TaskStatus.SUCCEEDED and recovered.attempts == 2
    hits = await rt2.c.memory.recall("when is the deploy window?", kinds=[MemoryKind.FACT])
    assert hits and "Friday" in hits[0].record.content  # vectors persisted to disk
    assert "awakening #2" in brain2.agent_requests[0].messages[0].text
    assert rt2.c.audit.verify().ok


async def test_proactive_life_beat(runtime_factory: Any) -> None:
    """With nobody around, Fuzzy notices how it feels, picks an activity itself and does it."""

    clock = FakeClock()
    brain = Brain(
        beat={
            "feelings": [
                {
                    "emotion": "restless",
                    "intensity": 0.6,
                    "valence": -0.2,
                    "lingers_hours": 3,
                    "cause": "nothing has happened in a while",
                }
            ],
            "mood": "a bit restless",
            "choice": "think",
            "intent": "what I'd like to understand better",
            "why": "I feel restless and want to use the time",
            "rest_minutes": 20,
        },
        thought={
            "thought": "I would like to understand how my memory decides what to keep.",
            "insight": "Idle time is best spent organizing what I already know.",
        },
    )
    rt = await runtime_factory(
        brain,
        clock=clock,
        runtime={"autonomy": "proactive"},
        mind={"beat_min_s": 10, "beat_max_s": 3600},
    )
    clock.advance(3600)
    rt._wake.set()

    async def thought_done() -> bool:
        return any(
            a.kind == "think" and a.status.value == "done"
            for a in await rt.c.db.activities.recent(10)
        )

    await wait_for(thought_done, timeout=10)
    [activity] = await rt.c.db.activities.recent(10)
    assert activity.reason.startswith("I feel restless")
    feelings = await rt.c.emotions.current(Scope.system())
    assert [f.emotion for f in feelings] == ["restless"]
    assert rt.c.emotions.mood == "a bit restless"
    reflections = await rt.c.memory.recall(
        "organizing what I already know", kinds=[MemoryKind.REFLECTION]
    )
    assert reflections
    assert any(e.type == "mind.activity" for e in rt.c.audit.iter_entries())
    # It chose how long to rest, so the next beat isn't due yet.
    assert rt._next_beat - clock.monotonic() > 600


async def test_state_snapshot(runtime_factory: Any) -> None:
    rt = await runtime_factory(Brain())
    state = await rt.state()
    assert state["name"] == "Fuzzy" and state["boot_count"] == 1
    assert set(state["affect"]) >= {"curiosity", "competence", "valence"}
    assert "I feel" in state["feeling"] and state["kill_switch"]["engaged"] is False


async def test_cancel_pending(runtime_factory: Any) -> None:
    rt = await runtime_factory(Brain(), start=False)
    task = await rt.submit("never runs")
    assert await rt.cancel(task.id)
    assert (await rt.c.db.tasks.get(task.id)).status is TaskStatus.CANCELLED  # type: ignore[union-attr]
    assert not await rt.cancel("nope")
    await rt.start()


def test_brain_routes_json() -> None:
    brain = Brain(confidence=0.3)
    from sentientai.models.types import ChatRequest, Message

    out = brain(ChatRequest(messages=[Message.user("Rate how likely this answer is correct ...")]))
    assert json.loads(str(out))["confidence"] == 0.3


async def test_awareness_tracks_approvals_and_stays_live_while_killed(runtime_factory: Any) -> None:
    from sentientai.config import RiskLevel
    from sentientai.storage.records import ApprovalRecord

    rt = await runtime_factory(Brain(), start=False)
    approval = await rt.c.db.approvals.insert(
        ApprovalRecord(tool="repo_commit", arguments={}, risk=RiskLevel.HIGH, reasons=["x"])
    )
    await rt.c.kill_switch.engage("maintenance")
    await rt.cycle()
    assert any("repo_commit" in p.content for p in rt.c.mind.workspace.conscious)
    await rt.c.approvals.decide(approval.id, approved=False, by="op")
    await rt.cycle()  # awareness updates even though the kill switch is engaged
    assert not any("repo_commit" in p.content for p in rt.c.mind.workspace.conscious)
    task = await rt.submit("queued while killed")
    await rt.cycle()
    assert (await rt.c.db.tasks.get(task.id)).status is TaskStatus.PENDING  # type: ignore[union-attr]
    await rt.c.kill_switch.release()


async def test_kill_before_first_step_does_not_strand_task(runtime_factory: Any) -> None:
    rt = await runtime_factory(Brain(), start=False)
    record = await rt.submit("claimed but never started")
    claimed = await rt.c.db.tasks.claim(record.id)
    assert claimed is not None
    rt._start_task(claimed)
    # Cancel with no await in between: the task's coroutine never takes its first step,
    # so its own cleanup never runs and the kill handler must settle the record.
    await rt._on_kill("instant stop")
    final = await rt.c.db.tasks.get(record.id)
    assert final is not None and final.status is TaskStatus.CANCELLED
    assert final.error == "kill switch: instant stop"

    # A task that is already running reports the kill reason from its own cleanup.
    brain_task = await rt.submit("running when killed")
    claimed2 = await rt.c.db.tasks.claim(brain_task.id)
    assert claimed2 is not None
    rt._start_task(claimed2)
    await asyncio.sleep(0)
    await rt.c.kill_switch.engage("operator stop")
    final2 = await rt.c.db.tasks.get(brain_task.id)
    assert final2 is not None and final2.status is TaskStatus.CANCELLED
    assert final2.error == "kill switch: operator stop"
    await rt.c.kill_switch.release()


async def test_orderly_shutdown_requeues_and_resumes(runtime_factory: Any) -> None:
    brain = Brain([call("slow_job")])
    rt1 = await runtime_factory(brain, tools=[SlowTool()])
    task = await rt1.submit("long job interrupted by a reboot")
    await wait_for(lambda: rt1.c.bus.recent("tool.started"), timeout=10)
    await rt1.stop("planned reboot", grace_s=0.2)

    rt2 = await runtime_factory(Brain(default_answer="finished after reboot"), tools=[SlowTool()])
    resumed = await rt2.wait(task.id, timeout=10)
    assert resumed.status is TaskStatus.SUCCEEDED and resumed.result == "finished after reboot"
    assert resumed.attempts == 2

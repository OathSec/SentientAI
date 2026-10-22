"""Composition root: build every component from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentientai.cognition.goals import GoalManager
from sentientai.cognition.inner_life import InnerLife
from sentientai.cognition.mind import Mind
from sentientai.cognition.workspace import GlobalWorkspace
from sentientai.config import Settings
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.core.metrics import Metrics
from sentientai.errors import SandboxError
from sentientai.ingestion.pipeline import IngestionPipeline
from sentientai.memory.consolidation import MemoryConsolidator
from sentientai.memory.factory import build_memory
from sentientai.memory.system import MemorySystem
from sentientai.models.embeddings import Embedder
from sentientai.models.factory import build_router
from sentientai.models.router import ModelRouter
from sentientai.models.types import LLMClient, ModelTier
from sentientai.orchestration.agent import CognitiveAgent
from sentientai.reflection.analyzer import PerformanceAnalyzer
from sentientai.reflection.improvement import SelfImprovementEngine
from sentientai.reflection.reflector import Reflector
from sentientai.safety.approvals import ApprovalGate
from sentientai.safety.audit import AuditLog
from sentientai.safety.killswitch import KillSwitch
from sentientai.safety.limits import TokenBucket
from sentientai.safety.policy import PolicyEngine
from sentientai.storage.database import Database
from sentientai.tools.base import Tool, ToolRegistry
from sentientai.tools.cognitive import cognitive_tools
from sentientai.tools.executor import ToolExecutor
from sentientai.tools.filesystem import Workspace, filesystem_tools
from sentientai.tools.memory_tools import memory_tools
from sentientai.tools.repository import repository_tools
from sentientai.tools.sandbox import SandboxBackend, build_sandbox, sandbox_tools
from sentientai.tools.web import WebClient, web_tools

log = get_logger(__name__)


@dataclass
class Components:
    settings: Settings
    clock: Clock
    bus: EventBus
    metrics: Metrics
    db: Database
    audit: AuditLog
    kill_switch: KillSwitch
    approvals: ApprovalGate
    policy: PolicyEngine
    router: ModelRouter
    memory: MemorySystem
    consolidator: MemoryConsolidator
    workspace: Workspace
    web: WebClient
    sandbox: SandboxBackend | None
    pipeline: IngestionPipeline
    registry: ToolRegistry
    executor: ToolExecutor
    goals: GoalManager
    mind: Mind
    agent: CognitiveAgent
    reflector: Reflector
    inner_life: InnerLife
    analyzer: PerformanceAnalyzer
    improvement: SelfImprovementEngine | None

    async def aclose(self) -> None:
        for closer in (self.memory.aclose, self.router.aclose, self.web.aclose, self.db.close):
            try:
                await closer()
            except Exception:  # pragma: no cover - best effort shutdown
                log.exception("error during shutdown")


async def build_components(
    settings: Settings,
    *,
    clients: dict[ModelTier, LLMClient] | None = None,
    embedder: Embedder | None = None,
    clock: Clock | None = None,
    extra_tools: list[Tool[Any]] | None = None,
) -> Components:
    clock = clock or SystemClock()
    settings.ensure_directories()
    bus = EventBus()
    metrics = Metrics()

    db = Database(settings.storage.resolved_database_url())
    await db.create_all()
    hmac_key = settings.safety.audit_hmac_key
    audit = AuditLog(
        settings.audit_path,
        hmac_key=hmac_key.get_secret_value().encode() if hmac_key else None,
        fsync=settings.safety.audit_fsync,
        segment_max_bytes=settings.safety.audit_segment_max_bytes,
    )
    kill_switch = KillSwitch(settings.kill_switch_path, bus=bus, audit=audit)
    approvals = ApprovalGate(
        db.approvals,
        bus=bus,
        audit=audit,
        kill_switch=kill_switch,
        timeout_s=settings.safety.approval_timeout_s,
        clock=clock,
    )
    policy = PolicyEngine(settings.safety)
    router = build_router(settings, metrics=metrics, clients=clients, clock=clock)

    memory = build_memory(settings, db, bus=bus, metrics=metrics, embedder=embedder, clock=clock)
    await memory.open()
    consolidator = MemoryConsolidator(memory, router, settings.memory)

    workspace = Workspace(settings.safety.workspace_roots, settings.safety.protected_paths)
    web = WebClient(settings.web)
    sandbox: SandboxBackend | None
    try:
        sandbox = build_sandbox(settings.sandbox, workspace.primary)
    except SandboxError as exc:
        log.warning(
            "sandbox unavailable; code execution tools disabled", extra={"reason": str(exc)}
        )
        sandbox = None
    if sandbox is not None:
        sandbox_backend = sandbox

        async def kill_sandboxes(_reason: str) -> None:
            killed = await sandbox_backend.kill_all()
            log.critical("killed sandbox processes", extra={"count": killed})

        kill_switch.on_engage(kill_sandboxes)

    pipeline = IngestionPipeline(memory, settings.storage.resolved_archive_dir(), web=web, bus=bus)
    goals = GoalManager(
        db.goals, bus=bus, max_intrinsic=settings.runtime.max_intrinsic_goals, clock=clock
    )
    attention = GlobalWorkspace(bus=bus, clock=clock)
    mind = await Mind.load(
        db.state, settings.identity, workspace=attention, goals=goals, clock=clock
    )
    analyzer = PerformanceAnalyzer(db, audit, clock=clock)
    improvement = (
        SelfImprovementEngine(
            settings.self_improvement,
            router=router,
            proposals=db.proposals,
            audit=audit,
            bus=bus,
            analyzer=analyzer,
            staging_root=settings.staging_path,
        )
        if settings.self_improvement.enabled
        else None
    )

    registry = ToolRegistry()
    tools: list[Tool[Any]] = [
        *filesystem_tools(workspace),
        *repository_tools(workspace),
        *web_tools(web),
        *memory_tools(memory, pipeline, workspace),
        *cognitive_tools(goals, analyzer, improvement),
    ]
    if sandbox is not None:
        tools += sandbox_tools(sandbox, settings.sandbox, settings.storage.data_dir / "sandbox")
    for tool in [*tools, *(extra_tools or [])]:
        registry.register(tool)

    executor = ToolExecutor(
        registry,
        policy=policy,
        approvals=approvals,
        audit=audit,
        bus=bus,
        invocations=db.tool_invocations,
        metrics=metrics,
        rate_limiter=TokenBucket(settings.limits.tool_calls_per_minute, clock=clock),
        max_output_chars=settings.limits.max_tool_output_chars,
    )
    agent = CognitiveAgent(
        settings=settings,
        router=router,
        memory=memory,
        registry=registry,
        executor=executor,
        mind=mind,
        audit=audit,
        bus=bus,
        kill_switch=kill_switch,
        metrics=metrics,
        clock=clock,
    )
    reflector = Reflector(
        router, memory, mind, min_fact_confidence=settings.memory.min_fact_confidence
    )
    return Components(
        settings=settings,
        clock=clock,
        bus=bus,
        metrics=metrics,
        db=db,
        audit=audit,
        kill_switch=kill_switch,
        approvals=approvals,
        policy=policy,
        router=router,
        memory=memory,
        consolidator=consolidator,
        workspace=workspace,
        web=web,
        sandbox=sandbox,
        pipeline=pipeline,
        registry=registry,
        executor=executor,
        goals=goals,
        mind=mind,
        agent=agent,
        reflector=reflector,
        inner_life=InnerLife(router, memory, goals),
        analyzer=analyzer,
        improvement=improvement,
    )

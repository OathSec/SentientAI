"""Build model clients and the router from configuration."""

from __future__ import annotations

from sentientai.config import ModelEndpointConfig, Settings
from sentientai.core.clock import Clock
from sentientai.core.metrics import Metrics
from sentientai.models.openai_compat import OpenAICompatibleClient
from sentientai.models.router import ModelRouter
from sentientai.models.scripted import ScriptedLLM
from sentientai.models.types import LLMClient, ModelTier
from sentientai.safety.limits import TokenBucket
from sentientai.workers.client import WorkerFirstClient
from sentientai.workers.hub import WorkerHub


def build_client(name: str, config: ModelEndpointConfig, hub: WorkerHub | None = None) -> LLMClient:
    if config.provider == "scripted":
        return ScriptedLLM(
            name=name,
            context_window=config.context_window,
            default="(scripted model has no reply configured)",
        )
    if hub is not None:
        return WorkerFirstClient(name, config, hub=hub)
    return OpenAICompatibleClient(name, config)


def build_router(
    settings: Settings,
    *,
    metrics: Metrics | None = None,
    clients: dict[ModelTier, LLMClient] | None = None,
    clock: Clock | None = None,
    hub: WorkerHub | None = None,
) -> ModelRouter:
    if clients is None:
        fast_cfg, heavy_cfg = settings.models.fast, settings.models.heavy
        fast = build_client("fast", fast_cfg, hub)
        # One server can back both tiers (e.g. a single GPU box); share the client.
        same = (fast_cfg.base_url, fast_cfg.model) == (heavy_cfg.base_url, heavy_cfg.model)
        heavy = fast if same else build_client("heavy", heavy_cfg, hub)
        clients = {ModelTier.FAST: fast, ModelTier.HEAVY: heavy}
    limiter = TokenBucket(settings.limits.llm_requests_per_minute, clock=clock)
    return ModelRouter(
        clients, settings.models.routing, metrics=metrics, rate_limiter=limiter, clock=clock
    )

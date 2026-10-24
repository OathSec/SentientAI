"""Tiered model routing.

Two tiers map onto the hardware plan:

* ``fast``  - small quantized model on the secondary GPU: routing,
  classification, summarization, extraction, reflection, self-assessment.
* ``heavy`` - large model across the primary GPUs: reasoning, planning,
  code generation, open conversation.

Selection considers the task kind, whether the prompt fits the tier's context
window, prompt complexity, and metacognitive confidence (low confidence
escalates to heavy). Each tier sits behind a circuit breaker; when a tier is
down the router falls back to the other one.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass

from sentientai.config import RoutingConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.logging import get_logger
from sentientai.core.metrics import Metrics
from sentientai.errors import ModelError, ModelUnavailableError
from sentientai.models.types import (
    ChatRequest,
    ChatResponse,
    LLMClient,
    ModelTier,
    TaskKind,
    estimate_messages_tokens,
)
from sentientai.safety.limits import TokenBucket

log = get_logger(__name__)

# Set while a person is waiting on the answer (see ModelRouter.foreground).
_FOREGROUND: ContextVar[bool] = ContextVar("foreground", default=False)


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float, clock: Clock) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock.monotonic() - self.opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock.monotonic()


@dataclass(frozen=True)
class RouteDecision:
    tier: ModelTier
    reason: str


class ModelRouter:
    def __init__(
        self,
        clients: dict[ModelTier, LLMClient],
        config: RoutingConfig,
        *,
        metrics: Metrics | None = None,
        rate_limiter: TokenBucket | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not clients:
            raise ValueError("ModelRouter needs at least one client")
        self.clients = clients
        self.config = config
        self.metrics = metrics
        self.rate_limiter = rate_limiter
        self.clock = clock or SystemClock()
        self.breakers = {
            tier: CircuitBreaker(
                config.circuit_failure_threshold, config.circuit_cooldown_s, self.clock
            )
            for tier in clients
        }
        # Conversations come first: while one is being answered, Fuzzy's own work (its beats,
        # projects, reflections) holds its next model request, so the person gets the whole
        # model; with nobody talking, its own thinking has all of it. (Measured on the 14B: a
        # reply written at ~3.6 tok/s while a project ran beside it, ~30 tok/s alone.)
        self._foreground = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @contextlib.asynccontextmanager
    async def foreground(self) -> AsyncIterator[None]:
        """Everything inside (and every task it starts) is a person waiting: it goes first."""

        token = _FOREGROUND.set(True)
        self._foreground += 1
        self._idle.clear()
        try:
            yield
        finally:
            _FOREGROUND.reset(token)
            self._foreground -= 1
            if self._foreground == 0:
                self._idle.set()

    async def _yield_to_conversations(self) -> None:
        if _FOREGROUND.get() or self._foreground == 0:
            return
        with contextlib.suppress(TimeoutError):  # never starve its own life completely
            await asyncio.wait_for(self._idle.wait(), timeout=self.config.background_yield_s)

    def _available(self, tier: ModelTier) -> ModelTier:
        if tier in self.clients:
            return tier
        return next(iter(self.clients))

    def select(
        self,
        kind: TaskKind,
        request: ChatRequest,
        *,
        prefer: ModelTier | None = None,
        confidence: float | None = None,
    ) -> RouteDecision:
        if prefer is not None:
            return RouteDecision(self._available(prefer), "explicit preference")
        tokens = estimate_messages_tokens(request.messages)
        fast = self.clients.get(ModelTier.FAST)
        if confidence is not None and confidence < self.config.escalate_below_confidence:
            return RouteDecision(
                self._available(ModelTier.HEAVY), f"low confidence {confidence:.2f}"
            )
        if fast is not None and self._overflows(fast, tokens, request):
            return RouteDecision(
                self._available(ModelTier.HEAVY), "prompt exceeds fast context window"
            )
        if kind.value in self.config.heavy_task_kinds:
            return RouteDecision(
                self._available(ModelTier.HEAVY), f"{kind.value} is a heavy task kind"
            )
        if kind.value in self.config.fast_task_kinds:
            return RouteDecision(
                self._available(ModelTier.FAST), f"{kind.value} is a fast task kind"
            )
        if tokens > self.config.complexity_token_threshold:
            return RouteDecision(self._available(ModelTier.HEAVY), "complex prompt")
        return RouteDecision(self._available(ModelTier.FAST), "default")

    @staticmethod
    def _overflows(client: LLMClient, tokens: int, request: ChatRequest) -> bool:
        """The prompt plus the room asked for the answer won't fit the client's context.

        (A server whose context fills up mid-answer stops there, and a half-written JSON answer
        reads as nonsense.)"""

        if tokens > int(client.context_window * 0.85):
            return True
        cap = getattr(getattr(client, "config", None), "max_output_tokens", None)
        wanted = request.max_tokens or cap
        if not isinstance(wanted, int):
            return False
        if isinstance(cap, int):
            wanted = min(wanted, cap)
        return tokens + wanted > client.context_window

    async def chat(
        self,
        request: ChatRequest,
        *,
        kind: TaskKind = TaskKind.CHAT,
        prefer: ModelTier | None = None,
        confidence: float | None = None,
    ) -> ChatResponse:
        await self._yield_to_conversations()
        if kind.value in self.config.worker_task_kinds:
            request = request.model_copy(update={"background": True})
        elif kind is TaskKind.CHAT and self.config.replies_on_matching_workers:
            request = request.model_copy(update={"snippet": True})  # same-model computers only
        decision = self.select(kind, request, prefer=prefer, confidence=confidence)
        order = [decision.tier]
        if self.config.fallback_to_other_tier:
            order += [t for t in self.clients if t not in order]
        # Every server can be briefly unreachable at once: a restart, or (with one model on both
        # cards) while it is paused for a picture. Then wait for one to come back rather than
        # fail a person's reply; real errors (a bad request, a crash mid-answer) still fail fast.
        deadline = time.monotonic() + self.config.wait_for_model_s
        waited = False
        while True:
            errors: list[str] = []
            unreachable = True
            for tier in order:
                breaker = self.breakers[tier]
                if not breaker.allow():
                    errors.append(f"{tier.value}: circuit open")
                    continue
                client = self.clients[tier]
                if self.rate_limiter is not None:
                    await self.rate_limiter.acquire(timeout=120)
                try:
                    response = await client.chat(request)
                except ModelError as exc:
                    breaker.failure()
                    errors.append(f"{tier.value}: {exc}")
                    unreachable = unreachable and isinstance(exc, ModelUnavailableError)
                    self._observe(tier, "error", None)
                    log.warning(
                        "model tier failed", extra={"tier": tier.value, "error": str(exc)[:200]}
                    )
                    continue
                breaker.success()
                response.tier = tier
                self._observe(tier, "ok" if tier == decision.tier else "fallback", response)
                return response
            remaining = deadline - time.monotonic()
            if not unreachable or remaining <= 0:
                raise ModelUnavailableError("no model tier available: " + "; ".join(errors))
            if not waited:
                log.warning("no model server reachable; waiting for one to come back")
                waited = True
            await asyncio.sleep(min(5.0, remaining))

    def _observe(self, tier: ModelTier, outcome: str, response: ChatResponse | None) -> None:
        if self.metrics is None:
            return
        self.metrics.llm_requests.inc(tier=tier.value, outcome=outcome)
        if response is not None:
            self.metrics.llm_latency.observe(response.latency_ms / 1000, tier=tier.value)
            self.metrics.llm_tokens.inc(
                response.usage.prompt_tokens, tier=tier.value, kind="prompt"
            )
            self.metrics.llm_tokens.inc(
                response.usage.completion_tokens, tier=tier.value, kind="completion"
            )

    def context_window(self, tier: ModelTier) -> int:
        return self.clients[self._available(tier)].context_window

    async def health(self) -> dict[str, bool]:
        return {tier.value: await client.health() for tier, client in self.clients.items()}

    async def aclose(self) -> None:
        seen: set[int] = set()
        for client in self.clients.values():
            if id(client) not in seen:
                seen.add(id(client))
                await client.aclose()

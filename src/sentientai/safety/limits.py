"""Deterministic loop breakers and rate limiting.

:class:`LoopGuard` bounds every task run by hard limits that do not depend on
model judgement: steps, tool calls, wall time, token spend, identical repeated
calls, consecutive failures and stalled (repeated) model output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from sentientai.config import LimitsConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.errors import LoopBreakError
from sentientai.safety.killswitch import KillSwitch


def call_digest(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"n": name, "a": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class TokenBucket:
    """Async token bucket: ``rate_per_minute`` sustained, ``burst`` capacity."""

    def __init__(
        self, rate_per_minute: float, burst: float | None = None, clock: Clock | None = None
    ) -> None:
        self.rate = rate_per_minute / 60.0
        self.capacity = burst if burst is not None else max(1.0, rate_per_minute / 6.0)
        self.clock = clock or SystemClock()
        self._tokens = self.capacity
        self._updated = self.clock.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self.clock.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        async with self._lock:
            while not self.try_acquire(tokens):
                wait = (tokens - self._tokens) / self.rate if self.rate > 0 else 1.0
                if deadline is not None and loop.time() + wait > deadline:
                    raise TimeoutError("rate limit wait exceeds timeout")
                await asyncio.sleep(min(wait, 1.0))


@dataclass
class GuardSnapshot:
    steps: int
    tool_calls: int
    tokens: int
    elapsed_s: float
    consecutive_failures: int
    pressure: float  # 0..1, how close the run is to its tightest limit


class LoopGuard:
    def __init__(
        self,
        limits: LimitsConfig,
        *,
        kill_switch: KillSwitch | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.limits = limits
        self.kill_switch = kill_switch
        self.clock = clock or SystemClock()
        self.started = self.clock.monotonic()
        self.steps = 0
        self.tool_calls = 0
        self.tokens = 0
        self.consecutive_failures = 0
        self._call_counts: Counter[str] = Counter()
        self._recent_outputs: deque[str] = deque(maxlen=3)

    @property
    def elapsed(self) -> float:
        return self.clock.monotonic() - self.started

    def check(self) -> None:
        if self.kill_switch is not None:
            self.kill_switch.check()
        if self.elapsed > self.limits.max_wall_time_s:
            raise LoopBreakError(
                f"wall time exceeded {self.limits.max_wall_time_s:.0f}s", limit="wall_time"
            )
        if self.tokens > self.limits.max_tokens_per_task:
            raise LoopBreakError(f"token budget exceeded ({self.tokens})", limit="tokens")

    def begin_step(self) -> None:
        self.check()
        if self.steps >= self.limits.max_steps:
            raise LoopBreakError(f"step limit {self.limits.max_steps} reached", limit="steps")
        self.steps += 1

    def record_tokens(self, count: int) -> None:
        self.tokens += max(0, count)

    def before_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        self.check()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise LoopBreakError(
                f"tool call limit {self.limits.max_tool_calls} reached", limit="tool_calls"
            )
        digest = call_digest(name, arguments)
        self._call_counts[digest] += 1
        if self._call_counts[digest] > self.limits.max_repeat_calls:
            raise LoopBreakError(
                f"identical call to {name} repeated {self._call_counts[digest]} times",
                limit="repetition",
            )
        self.tool_calls += 1

    def after_tool_call(self, ok: bool) -> None:
        self.consecutive_failures = 0 if ok else self.consecutive_failures + 1
        if self.consecutive_failures >= self.limits.max_consecutive_failures:
            raise LoopBreakError(
                f"{self.consecutive_failures} consecutive tool failures",
                limit="consecutive_failures",
            )

    def record_output(self, text: str) -> None:
        """Detect a stalled model emitting the same output over and over."""

        normalized = " ".join(text.split())[:2000]
        if not normalized:
            return
        self._recent_outputs.append(normalized)
        if (
            len(self._recent_outputs) == self._recent_outputs.maxlen
            and len(set(self._recent_outputs)) == 1
        ):
            raise LoopBreakError("model output stalled (identical responses)", limit="stall")

    def snapshot(self) -> GuardSnapshot:
        ratios = [
            self.steps / self.limits.max_steps,
            self.elapsed / self.limits.max_wall_time_s,
            self.tokens / self.limits.max_tokens_per_task,
        ]
        if self.limits.max_tool_calls:
            ratios.append(self.tool_calls / self.limits.max_tool_calls)
        return GuardSnapshot(
            steps=self.steps,
            tool_calls=self.tool_calls,
            tokens=self.tokens,
            elapsed_s=round(self.elapsed, 3),
            consecutive_failures=self.consecutive_failures,
            pressure=min(1.0, max(ratios)),
        )

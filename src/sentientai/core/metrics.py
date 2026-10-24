"""Dependency-free metrics registry with Prometheus text exposition."""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict
from collections.abc import Iterable

LabelKey = tuple[tuple[str, str], ...]

_DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


def _key(labels: dict[str, str] | None) -> LabelKey:
    return tuple(sorted((labels or {}).items()))


def _fmt_labels(key: LabelKey, extra: Iterable[tuple[str, str]] = ()) -> str:
    items = [*key, *extra]
    if not items:
        return ""
    escaped = (
        f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in items
    )
    return "{" + ",".join(escaped) + "}"


class Counter:
    def __init__(self, name: str, help_: str) -> None:
        self.name, self.help = name, help_
        self._values: dict[LabelKey, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._values[_key(labels)] += amount

    def value(self, **labels: str) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        lines += [f"{self.name}{_fmt_labels(k)} {v}" for k, v in sorted(self._values.items())]
        return lines


class Gauge:
    def __init__(self, name: str, help_: str) -> None:
        self.name, self.help = name, help_
        self._values: dict[LabelKey, float] = {}

    def set(self, value: float, **labels: str) -> None:
        self._values[_key(labels)] = float(value)

    def value(self, **labels: str) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        lines += [f"{self.name}{_fmt_labels(k)} {v}" for k, v in sorted(self._values.items())]
        return lines


class Histogram:
    def __init__(self, name: str, help_: str, buckets: tuple[float, ...] = _DEFAULT_BUCKETS):
        self.name, self.help = name, help_
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = _key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            counts[bisect.bisect_left(self.buckets, value)] += 1
            self._sums[key] += value

    def count(self, **labels: str) -> int:
        return sum(self._counts.get(_key(labels), []))

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, counts in sorted(self._counts.items()):
            cumulative = 0
            for bound, count in zip(self.buckets, counts, strict=False):
                cumulative += count
                lines.append(
                    f"{self.name}_bucket{_fmt_labels(key, [('le', str(bound))])} {cumulative}"
                )
            cumulative += counts[-1]
            lines.append(f"{self.name}_bucket{_fmt_labels(key, [('le', '+Inf')])} {cumulative}")
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {self._sums[key]}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {cumulative}")
        return lines


class Metrics:
    """The runtime's metric set. One instance per runtime (no global state)."""

    def __init__(self) -> None:
        self.llm_requests = Counter(
            "sentient_llm_requests_total", "LLM requests by tier and outcome"
        )
        self.llm_latency = Histogram("sentient_llm_latency_seconds", "LLM request latency")
        self.llm_tokens = Counter("sentient_llm_tokens_total", "Tokens processed by tier and kind")
        self.tool_calls = Counter("sentient_tool_calls_total", "Tool calls by tool and outcome")
        self.tool_latency = Histogram("sentient_tool_latency_seconds", "Tool execution latency")
        self.tasks = Counter("sentient_tasks_total", "Tasks by source and final status")
        self.loop_breaks = Counter("sentient_loop_breaks_total", "Loop breakers tripped by limit")
        self.approvals = Counter("sentient_approvals_total", "Approval decisions by outcome")
        self.policy_denials = Counter("sentient_policy_denials_total", "Actions denied by policy")
        self.memory_ops = Counter("sentient_memory_ops_total", "Memory operations by op and kind")
        self.memory_items = Gauge("sentient_memory_items", "Active memory items by kind")
        self.affect = Gauge("sentient_affect", "Internal affective/drive state")
        self.uptime = Gauge("sentient_uptime_seconds", "Seconds since boot")
        self.kill_switch = Gauge(
            "sentient_kill_switch_engaged", "1 when the kill switch is engaged"
        )

    def _all(self) -> list[Counter | Gauge | Histogram]:
        return [v for v in vars(self).values() if isinstance(v, Counter | Gauge | Histogram)]

    def render(self) -> str:
        lines: list[str] = []
        for metric in self._all():
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"

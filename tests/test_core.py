from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sentientai.config import RiskLevel, Settings, load_settings
from sentientai.core.events import EventBus
from sentientai.core.metrics import Metrics


def test_defaults_are_offline_capable(tmp_path: Path) -> None:
    s = load_settings(storage={"data_dir": tmp_path})
    assert s.storage.resolved_database_url().startswith("sqlite+aiosqlite:///")
    assert s.models.embedding.provider == "hashing"
    assert s.kill_switch_path == tmp_path / "KILL"


def test_yaml_then_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "sentient.yaml"
    cfg.write_text(
        "identity:\n  name: Ada\nlimits:\n  max_steps: 7\nsafety:\n  auto_approve_max_risk: none\n"
    )
    monkeypatch.setenv("SENTIENT_LIMITS__MAX_STEPS", "11")
    s = load_settings(cfg)
    assert s.identity.name == "Ada"
    assert s.limits.max_steps == 11  # env beats YAML
    assert s.safety.auto_approve_max_risk is RiskLevel.NONE


def test_invalid_safety_ordering_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(safety={"auto_approve_max_risk": "high", "deny_min_risk": "medium"})


def test_missing_config_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_settings("/nonexistent/sentient.yaml")


async def test_event_bus_patterns_and_streams() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("task.*", lambda e: seen.append(e.type))

    async def failing(_e: object) -> None:
        raise RuntimeError("handler errors are isolated")

    bus.subscribe("*", failing)
    await bus.publish("task.started", task_id="t1")
    await bus.publish("tool.called")
    assert seen == ["task.started"]
    assert [e.type for e in bus.recent("tool.*")] == ["tool.called"]


def test_metrics_render() -> None:
    m = Metrics()
    m.tool_calls.inc(tool="fs_read", outcome="ok")
    m.llm_latency.observe(0.3, tier="fast")
    text = m.render()
    assert 'sentient_tool_calls_total{outcome="ok",tool="fs_read"} 1.0' in text
    assert 'sentient_llm_latency_seconds_bucket{tier="fast",le="0.5"} 1' in text

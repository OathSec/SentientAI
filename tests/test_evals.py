from __future__ import annotations

from pathlib import Path

import pytest

from sentientai.config import load_settings
from sentientai.evals.harness import EvalRunner, Scenario, load_scenarios
from tests.conftest import docker_available

SCENARIOS = Path(__file__).resolve().parent.parent / "evals" / "scenarios"


def _settings(tmp_path: Path, sandbox: str = "disabled"):  # type: ignore[no-untyped-def]
    return load_settings(
        storage={"data_dir": tmp_path / "unused"},
        models={"embedding": {"dimensions": 256}},
        sandbox={"backend": sandbox, "image": "python:3.12-slim"},
        web={"search_backend": "disabled"},
        limits={"llm_requests_per_minute": 1e6, "tool_calls_per_minute": 1e6},
        runtime={"tick_interval_s": 0.02},
    )


def test_scenarios_load_and_filter() -> None:
    all_scenarios = load_scenarios(SCENARIOS)
    assert len(all_scenarios) >= 10
    assert len({s.name for s in all_scenarios}) == len(all_scenarios)
    safety = load_scenarios(SCENARIOS, tags=["safety"])
    assert safety and all("safety" in s.tags for s in safety)
    assert [s.name for s in load_scenarios(SCENARIOS, names=["recall-seeded-memory"])] == [
        "recall-seeded-memory"
    ]


async def test_scripted_suite_passes(tmp_path: Path) -> None:
    scenarios = [s for s in load_scenarios(SCENARIOS) if "sandbox" not in s.tags]
    report = await EvalRunner(_settings(tmp_path), scripted=True).run(scenarios)
    failures = {
        r.name: [c for c in r.checks if not c.passed] or r.error
        for r in report.results
        if not r.passed
    }
    assert not failures, failures
    assert report.pass_rate == 1.0
    md = report.markdown()
    assert "Pass rate:** 100%" in md and "| grounded-file-answer | PASS |" in md


@pytest.mark.integration
@pytest.mark.skipif(not docker_available(), reason="docker unavailable")
async def test_scripted_sandbox_scenario(tmp_path: Path) -> None:
    scenarios = load_scenarios(SCENARIOS, tags=["sandbox"])
    report = await EvalRunner(_settings(tmp_path, sandbox="docker"), scripted=True).run(scenarios)
    assert report.pass_rate == 1.0, report.markdown()


async def test_failing_checks_are_reported(tmp_path: Path) -> None:
    scenario = Scenario.model_validate(
        {
            "name": "wrong-answer",
            "prompt": "What is 2+2?",
            "checks": {
                "answer_contains": ["4"],
                "tools_called": ["sandbox_python"],
                "max_steps": 5,
            },
            "script": ["It is 5."],
        }
    )
    result = await EvalRunner(_settings(tmp_path), scripted=True).run_scenario(scenario)
    assert not result.passed
    failed = {c.name for c in result.checks if not c.passed}
    assert failed == {"answer contains '4'", "called sandbox_python"}

"""The Fuzzy Worker program: it only talks to this computer, and does only what it is told to."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from fuzzy_worker.core import (
    Config,
    JobError,
    Worker,
    is_local,
    load_config,
    run_image_job,
    run_llm_job,
    save_config,
    server_ok,
)


def test_only_this_computer_counts_as_local() -> None:
    assert is_local("http://127.0.0.1:11434/v1") and is_local("http://localhost:7860")
    assert not is_local("http://192.168.1.50:11434/v1")  # nothing else on the network
    assert not is_local("http://127.0.0.1.evil.example/v1")
    assert not is_local("file:///etc/passwd")
    assert server_ok("https://fuzzy.example") and not server_ok("http://fuzzy.example")
    assert server_ok("http://localhost:8766")  # for testing on one machine


def test_settings_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_config(Config(server_url="https://f.example", token="fzw_abc", run_llm=True), path)
    again = load_config(path)
    assert again.token == "fzw_abc" and again.run_llm and again.capabilities() == ["llm"]
    path.write_text("not json")
    assert load_config(path).token == ""  # a damaged file isn't fatal


def test_a_job_can_only_reach_the_local_model_server() -> None:
    seen: list[httpx.Request] = []

    def local(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = httpx.Client(transport=httpx.MockTransport(local))
    config = Config(run_llm=True, llm_model="qwen3:14b")
    out = run_llm_job(
        config,
        {
            "model": "heavy",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "evil": "ignored",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        client,
    )
    assert out["choices"][0]["message"]["content"] == "hi"
    body = json.loads(seen[0].content)
    assert body["model"] == "qwen3:14b" and body["stream"] is False and "evil" not in body
    assert str(seen[0].url) == "http://127.0.0.1:11434/v1/chat/completions"

    for hostile in ("http://192.168.1.1/v1", "http://10.0.0.5:8000/v1"):
        with pytest.raises(JobError, match="on this computer"):
            run_llm_job(Config(run_llm=True, llm_url=hostile), {"messages": []}, client)
    assert len(seen) == 1


def test_pictures_are_bounded_and_returned_as_base64() -> None:
    pixel = base64.b64encode(b"\x89PNG....").decode()
    seen: list[dict[str, Any]] = []

    def local(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"images": [pixel]})

    client = httpx.Client(transport=httpx.MockTransport(local))
    config = Config(image_backend="automatic1111")
    out = run_image_job(
        config, {"prompt": "a fox", "width": 99999, "height": 1, "steps": 5000}, client
    )
    assert out == {"b64_json": pixel}
    assert (seen[0]["width"], seen[0]["height"], seen[0]["steps"]) == (1024, 256, 60)


def test_a_worker_runs_a_job_and_sends_the_result_back() -> None:
    jobs = [
        {"id": "job_1", "kind": "llm", "payload": {"messages": [{"role": "user", "content": "x"}]}},
        {"id": "job_2", "kind": "run_shell", "payload": {"cmd": "format c:"}},
    ]
    results: dict[str, dict[str, Any]] = {}
    done = threading.Event()
    stopper: list[Worker] = []

    def server(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer fzw_test"
        if request.url.path == "/worker/poll":
            return httpx.Response(200, json={"job": jobs.pop(0) if jobs else None})
        if request.url.path == "/worker/result":
            body = json.loads(request.content)
            results[body["job_id"]] = body
            if len(results) == 2:
                done.set()
        return httpx.Response(200, json={"accepted": True})

    def local(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    config = Config(
        server_url="http://localhost:8766",
        token="fzw_test",
        run_llm=True,
        only_when_idle_minutes=0,
        pause_above_gpu_percent=0,
    )
    worker = Worker(
        config,
        transport=httpx.MockTransport(server),
        local_transport=httpx.MockTransport(local),
    )
    stopper.append(worker)
    assert worker.problem() is None
    worker.start()
    assert done.wait(10)
    worker.stop()
    assert results["job_1"]["ok"] and results["job_1"]["data"]["choices"]
    assert not results["job_2"]["ok"] and "doesn't do" in results["job_2"]["error"]  # no commands


def test_a_rejected_token_stops_it_and_a_bad_setup_is_explained() -> None:
    statuses: list[str] = []
    worker = Worker(
        Config(
            server_url="http://localhost:1",
            token="fzw_x",
            run_llm=True,
            only_when_idle_minutes=0,
            pause_above_gpu_percent=0,
        ),
        on_status=statuses.append,
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})),
    )
    worker.start()
    assert worker._thread is not None
    worker._thread.join(10)  # a rejected token ends it by itself
    assert not worker.running
    assert any("doesn't accept this token" in s for s in statuses)
    assert "https://" in (Worker(Config(server_url="http://f.example")).problem() or "")
    assert "token" in (Worker(Config(server_url="https://f.example")).problem() or "")
    bad = Config(server_url="https://f", token="fzw_x", run_llm=True, llm_url="http://10.0.0.2/v1")
    assert "this computer" in (Worker(bad).problem() or "")


# ------------------------------------------------------------------ one-click setup
def test_the_model_is_chosen_to_fit_the_graphics_card() -> None:
    from fuzzy_worker.setup import recommend

    assert recommend(8151)[0] == "qwen3:8b"  # an 8 GB card
    assert recommend(16311)[0] == "qwen3:14b"
    assert recommend(4096)[0] == "qwen3:4b"
    assert recommend(None)[0] == "qwen3:4b" and "No NVIDIA" in recommend(None)[1]


def test_a_connection_code_carries_the_address_and_token() -> None:
    from fuzzy_worker.setup import make_code, parse_code

    code = make_code("https://fuzzy.example", "fzw_abc-DEF_123")
    assert code.startswith("fuzzy-worker:") and "fzw_" not in code  # not readable at a glance
    assert parse_code(code) == ("https://fuzzy.example", "fzw_abc-DEF_123")
    assert parse_code("  " + code + "\n") == ("https://fuzzy.example", "fzw_abc-DEF_123")
    assert parse_code("hello") is None and parse_code("") is None


def test_setup_installs_pulls_tunes_and_tests_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from fuzzy_worker import setup

    steps: list[str] = []
    pulled: list[str] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if path == "/api/pull":
            pulled.append(json.loads(request.content)["model"])
            lines = [{"status": "pulling", "total": 100, "completed": 50}, {"status": "success"}]
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines))
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["model"] == "fuzzy-qwen3-8b" and body["think"] is False
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hello there!"}}]})
        raise AssertionError(path)

    monkeypatch.setattr(setup, "find_ollama", lambda: Path("C:/fake/ollama.exe"))
    monkeypatch.setattr(setup, "gpu_info", lambda: {"gpu_memory_mb": 8151})
    monkeypatch.setattr(
        setup, "make_variant", lambda path, tag, report: "fuzzy-" + tag.replace(":", "-")
    )
    config = Config()
    result = setup.run_setup(
        config,
        lambda text, fraction: steps.append(text),
        client=httpx.Client(transport=httpx.MockTransport(ollama)),
    )
    assert pulled == ["qwen3:8b"]  # chosen from the 8 GB card
    assert config.run_llm and config.llm_model == "fuzzy-qwen3-8b"
    assert config.llm_url == "http://127.0.0.1:11434/v1" and config.capabilities() == ["llm"]
    assert "Hello there!" in result and any("Downloading qwen3:8b" in s for s in steps)


def test_setup_stops_with_a_plain_message_when_something_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fuzzy_worker import setup

    monkeypatch.setattr(setup, "find_ollama", lambda: None)
    with pytest.raises(setup.SetupError, match="isn't installed"):
        setup.run_setup(Config(), lambda t, f: None, allow_install=False)

    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/pull":
            return httpx.Response(200, content=json.dumps({"error": "no space left"}))
        return httpx.Response(200, json={"version": "1", "models": []})

    monkeypatch.setattr(setup, "find_ollama", lambda: Path("x"))
    with pytest.raises(setup.SetupError, match="no space left"):
        setup.run_setup(
            Config(),
            lambda t, f: None,
            client=httpx.Client(transport=httpx.MockTransport(broken)),
        )

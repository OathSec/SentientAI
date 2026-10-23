"""The part of the Fuzzy worker that does the work (no window in here, so it can be tested).

What it does, and all it does: it connects OUT to a Fuzzy server with a token, asks "any work?",
and runs one of two fixed kinds of job against a model server on THIS computer:

* ``llm``:   one chat completion, sent to your local OpenAI-style server (Ollama, LM Studio, ...)
* ``image``: one picture, from your local image server (Stable Diffusion WebUI, ...)

It never runs code or commands sent by the server, never reads your files, and only talks to
addresses on this computer (``127.0.0.1`` / ``localhost``) besides the Fuzzy server itself.
"""

from __future__ import annotations

import base64
import contextlib
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

VERSION = "1.0.0"
PROTOCOL = 1
MAX_JOB_BYTES = 4_000_000
POLL_SECONDS = 8

IMAGE_BACKENDS = ("none", "automatic1111", "openai")


@dataclass
class Config:
    server_url: str = ""
    token: str = ""
    run_llm: bool = False
    llm_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = ""
    image_backend: str = "none"
    image_url: str = "http://127.0.0.1:7860"
    # Limits, so it stays out of your way. 0 turns a limit off.
    only_when_idle_minutes: int = 10  # only work after this long with no keyboard or mouse
    pause_above_gpu_percent: int = 40  # stop taking work if the GPU is already this busy
    connect_on_start: bool = False
    start_with_windows: bool = False
    # Lend the processor too: Fuzzy sends batches of text to turn into search vectors (reading
    # documents people taught it). Only ever for the operator's own computers.
    use_cpu: bool = False
    cpu_percent: int = 50  # how much of the processor Fuzzy may use for it
    embed_model: str = ""
    llm_base: str = ""  # the model it runs, e.g. qwen3:14b: the server compares it with its own
    linked_name: str = ""  # what the server called this computer when it was linked
    linked_trust: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def capabilities(self) -> list[str]:
        caps = []
        if self.run_llm:
            caps.append("llm")
        if self.image_backend != "none":
            caps.append("image")
        if self.use_cpu and self.embed_model:
            caps.append("embed")
        return caps


def config_path() -> Path:
    if sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:  # Windows: %APPDATA%; Linux: $XDG_CONFIG_HOME or ~/.config
        base = (
            os.environ.get("APPDATA")
            or os.environ.get("XDG_CONFIG_HOME")
            or str(Path.home() / ".config")
        )
    return Path(base) / "FuzzyWorker" / "config.json"


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    config = Config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return config
    known = {f.name for f in fields(Config)}
    for key, value in data.items() if isinstance(data, dict) else []:
        if key in known and key != "extra":
            setattr(config, key, value)
    return config


def save_config(config: Config, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def is_local(url: str) -> bool:
    """Only this computer: the server can't aim the worker at your network."""

    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and parts.hostname in ("127.0.0.1", "localhost", "::1")


def server_ok(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" or (parts.scheme == "http" and is_local(url))


# ------------------------------------------------------------------ this computer
def idle_seconds() -> float:
    """Seconds since the last keyboard or mouse input. If it can't be measured (a Linux session
    without xprintidle, Wayland), the computer is treated as idle rather than never working."""

    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["ioreg", "-c", "IOHIDSystem"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
            line = next(x for x in out.splitlines() if "HIDIdleTime" in x)
            return int(line.split("=")[-1].strip()) / 1e9
        except (OSError, StopIteration, ValueError, subprocess.SubprocessError):
            return 1e9
    if sys.platform != "win32":
        try:
            out = subprocess.run(
                ["xprintidle"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
            return int(out.strip()) / 1000.0
        except (OSError, ValueError, subprocess.SubprocessError):
            return 1e9

    class LastInput(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LastInput()
    info.cbSize = ctypes.sizeof(LastInput)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
        return 0.0
    return float((ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0  # type: ignore[attr-defined]


def gpu_info() -> dict[str, Any]:
    """GPU name, memory and load from nvidia-smi, if there is one. On an Apple Silicon Mac the
    graphics share the computer's memory; about two thirds of it is usable for a model."""

    if sys.platform == "darwin":
        try:
            import platform

            if platform.machine() == "arm64":
                total = int(
                    subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    ).stdout.strip()
                )
                return {
                    "gpu": "Apple Silicon",
                    "gpu_memory_mb": int(total / 1_048_576 * 0.66),
                    "gpu_percent": 0,
                }
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=0x08000000 if sys.platform == "win32" else 0,  # no console window
        ).stdout.strip()
        name, memory, util = (p.strip() for p in out.splitlines()[0].split(","))
        return {"gpu": name, "gpu_memory_mb": int(memory), "gpu_percent": int(util)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def check_link(
    server_url: str, token: str, *, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Ask the server whether this address and token are real. Raises JobError with a plain
    reason if not; otherwise the computer's name and kind as the server knows them."""

    if not server_ok(server_url):
        raise JobError("The address in the code must start with https://")
    try:
        with httpx.Client(
            base_url=server_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}", "User-Agent": f"FuzzyWorker/{VERSION}"},
            timeout=20.0,
            transport=transport,
        ) as client:
            response = client.post("/worker/hello")
    except httpx.HTTPError as exc:
        raise JobError(
            f"Couldn't reach Fuzzy ({type(exc).__name__}). Check your internet."
        ) from exc
    if response.status_code == 401:
        raise JobError(
            "Fuzzy didn't recognise this code. It may have been unlinked; make a new one."
        )
    if response.status_code != 200:
        raise JobError(f"Fuzzy answered with an error ({response.status_code}). Try again soon.")
    data = response.json()
    if not isinstance(data, dict) or not data.get("linked"):
        raise JobError("That didn't look like a Fuzzy server.")
    return data


# ------------------------------------------------------------------------- jobs
class JobError(Exception):
    pass


def run_llm_job(config: Config, payload: dict[str, Any], client: httpx.Client) -> dict[str, Any]:
    if not config.run_llm:
        raise JobError("this computer isn't set up to run chat models")
    if not is_local(config.llm_url):
        raise JobError("the model server must be on this computer")
    body = {k: v for k, v in payload.items() if k in _LLM_FIELDS}
    body["stream"] = False
    # Fuzzy asks its models not to "think out loud" (it wastes time); Ollama spells that
    # differently from llama.cpp, and ignores what it doesn't know.
    body["think"] = False
    body["reasoning_effort"] = "none"
    if config.llm_model:
        body["model"] = config.llm_model
    try:
        response = client.post(config.llm_url.rstrip("/") + "/chat/completions", json=body)
    except httpx.HTTPError as exc:
        raise JobError(f"your local model server didn't answer ({type(exc).__name__})") from exc
    if response.status_code >= 400:
        raise JobError(f"your local model server said {response.status_code}")
    data = response.json()
    if not isinstance(data, dict):
        raise JobError("your local model server answered with something unexpected")
    return data


# The only request fields passed on to the local model server.
_LLM_FIELDS = {
    "model",
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "stop",
    "response_format",
    "chat_template_kwargs",
}


def run_embed_job(config: Config, payload: dict[str, Any], client: httpx.Client) -> dict[str, Any]:
    """Turn a batch of texts into vectors with the local embedding model."""

    if not (config.use_cpu and config.embed_model):
        raise JobError("this computer isn't set up to share its processor")
    if not is_local(config.llm_url):
        raise JobError("the model server must be on this computer")
    texts = payload.get("input")
    if not isinstance(texts, list) or not texts or len(texts) > 64:
        raise JobError("that isn't a batch of texts I can do")
    texts = [str(t)[:8000] or " " for t in texts]
    try:
        response = client.post(
            config.llm_url.rstrip("/") + "/embeddings",
            json={"model": config.embed_model, "input": texts},
        )
    except httpx.HTTPError as exc:
        raise JobError(f"your local model server didn't answer ({type(exc).__name__})") from exc
    if response.status_code >= 400:
        raise JobError(f"your local model server said {response.status_code}")
    try:
        items = sorted(response.json()["data"], key=lambda d: d["index"])
        vectors = [[float(x) for x in item["embedding"]] for item in items]
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError("your local model server returned no vectors") from exc
    if len(vectors) != len(texts):
        raise JobError("your local model server returned the wrong number of vectors")
    return {"embeddings": vectors}


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def run_image_job(config: Config, payload: dict[str, Any], client: httpx.Client) -> dict[str, Any]:
    if config.image_backend == "none":
        raise JobError("this computer isn't set up to draw pictures")
    if not is_local(config.image_url):
        raise JobError("the image server must be on this computer")
    prompt = str(payload.get("prompt") or "")[:2000]
    if len(prompt) < 3:
        raise JobError("no picture was described")
    width = _clamp(payload.get("width"), 256, 1024, 512)
    height = _clamp(payload.get("height"), 256, 1024, 512)
    steps = _clamp(payload.get("steps"), 1, 60, 25)
    negative = str(payload.get("negative_prompt") or "")[:500]
    base = config.image_url.rstrip("/")
    try:
        if config.image_backend == "automatic1111":
            response = client.post(
                base + "/sdapi/v1/txt2img",
                json={
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "batch_size": 1,
                    "n_iter": 1,
                },
            )
        else:
            response = client.post(
                base + "/v1/images/generations",
                json={
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "size": f"{width}x{height}",
                    "steps": steps,
                    "n": 1,
                    "response_format": "b64_json",
                },
            )
    except httpx.HTTPError as exc:
        raise JobError(f"your local image server didn't answer ({type(exc).__name__})") from exc
    if response.status_code >= 400:
        raise JobError(f"your local image server said {response.status_code}")
    data = response.json()
    try:
        b64 = (
            data["images"][0]
            if config.image_backend == "automatic1111"
            else data["data"][0]["b64_json"]
        )
        base64.b64decode(b64, validate=False)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise JobError("your local image server returned no picture") from exc
    return {"b64_json": b64}


# ----------------------------------------------------------------------- worker
class Worker:
    """Runs in a background thread: connect, ask for work, do it, repeat."""

    def __init__(
        self,
        config: Config,
        *,
        on_status: Callable[[str], None] = lambda s: None,
        on_log: Callable[[str], None] = lambda s: None,
        transport: httpx.BaseTransport | None = None,
        local_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self._local_transport = local_transport
        self.on_status = on_status
        self.on_log = on_log
        self._transport = transport
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.jobs_done = 0

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fuzzy-worker", daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        """Ask it to stop. The window passes ``wait=False``: it never freezes, and polls
        ``running`` to see when the worker has really finished (after its current job)."""

        self._stop.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=60)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def problem(self) -> str | None:
        """Why it can't start, in plain words, or None."""

        c = self.config
        if not server_ok(c.server_url):
            return "The server address must start with https://"
        if not c.token.startswith("fzw_"):
            return "Paste the token you were given (it starts with fzw_)."
        if c.use_cpu and not c.embed_model:
            return "Processor sharing needs setting up first: press Set everything up."
        if not c.capabilities():
            return "Turn on at least one thing to share (chat model or pictures)."
        for what, on, url in (
            ("model", c.run_llm, c.llm_url),
            ("image", c.image_backend != "none", c.image_url),
        ):
            if on and not is_local(url):
                return f"The {what} server address must be on this computer (127.0.0.1)."
        return None

    # --------------------------------------------------------------------- loop
    def _allowed_now(self) -> str | None:
        """None if it may work now, else why it is waiting."""

        c = self.config
        if c.only_when_idle_minutes and idle_seconds() < c.only_when_idle_minutes * 60:
            return (
                f"waiting: you're using this computer (needs {c.only_when_idle_minutes} min idle)"
            )
        if c.pause_above_gpu_percent:
            busy = gpu_info().get("gpu_percent", 0)
            if busy >= c.pause_above_gpu_percent:
                return f"waiting: your GPU is busy ({busy}%)"
        return None

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"version": VERSION, "os": sys.platform, **gpu_info()}
        if self.config.run_llm and self.config.llm_base:
            info["base_model"] = self.config.llm_base
        return info

    def _run(self) -> None:
        c = self.config
        headers = {"Authorization": f"Bearer {c.token}", "User-Agent": f"FuzzyWorker/{VERSION}"}
        server = httpx.Client(
            base_url=c.server_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(60.0, read=POLL_SECONDS + 20.0),
            transport=self._transport,
        )
        local = httpx.Client(
            timeout=httpx.Timeout(900.0, connect=5.0), transport=self._local_transport
        )
        delay = 2.0
        connected = False
        self.on_status("starting")
        try:
            while not self._stop.is_set():
                reason = self._allowed_now()
                if reason is not None:
                    if connected:
                        self._say_bye(server)
                        connected = False
                    self.on_status(reason)
                    self._stop.wait(10)
                    continue
                try:
                    response = server.post(
                        "/worker/poll",
                        json={
                            "capabilities": c.capabilities(),
                            "info": self._info(),
                            "protocol": PROTOCOL,
                            "wait": POLL_SECONDS,  # short, so Stop takes effect within seconds
                        },
                    )
                except httpx.HTTPError as exc:
                    connected = False
                    self.on_status(f"can't reach the server ({type(exc).__name__}); retrying")
                    self._stop.wait(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                if response.status_code == 401:
                    self.on_status("stopped: the server doesn't accept this token")
                    self.on_log("The token was rejected (wrong, or revoked).")
                    return
                if response.status_code == 426:
                    self.on_status("stopped: this program is out of date; get the new one")
                    return
                if response.status_code != 200:
                    self.on_status(f"the server said {response.status_code}; retrying")
                    self._stop.wait(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                delay = 2.0
                if not connected:
                    connected = True
                    self.on_log("Connected to the server.")
                job = response.json().get("job")
                if not job:
                    self.on_status("connected; waiting for work")
                    continue
                self._do(job, server, local)
        finally:
            if connected:
                self._say_bye(server)
            server.close()
            local.close()
            self.on_status("stopped")

    def _say_bye(self, server: httpx.Client) -> None:
        with contextlib.suppress(httpx.HTTPError):
            server.post("/worker/bye", timeout=5.0)

    def _do(self, job: dict[str, Any], server: httpx.Client, local: httpx.Client) -> None:
        kind = str(job.get("kind"))
        self.on_status(f"working: {kind}")
        started = time.monotonic()
        ok, data, error = False, None, ""
        try:
            payload = job.get("payload")
            if not isinstance(payload, dict) or len(json.dumps(payload)) > MAX_JOB_BYTES:
                raise JobError("the job was malformed")
            if kind == "llm":
                data = run_llm_job(self.config, payload, local)
            elif kind == "image":
                data = run_image_job(self.config, payload, local)
            elif kind == "embed":
                data = run_embed_job(self.config, payload, local)
            else:
                raise JobError(f"this program doesn't do {kind!r} jobs")
            ok = True
        except JobError as exc:
            error = str(exc)
        except Exception as exc:  # never let one job stop the worker
            error = f"unexpected error ({type(exc).__name__})"
        took = time.monotonic() - started
        self.on_log(f"{kind} job {'done' if ok else 'failed: ' + error} in {took:.0f}s")
        try:
            server.post(
                "/worker/result",
                json={"job_id": job.get("id"), "ok": ok, "data": data, "error": error[:400]},
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            self.on_log(f"couldn't send the result ({type(exc).__name__})")
        if ok:
            self.jobs_done += 1

"""A small OpenAI-compatible image server for Fuzzy (``POST /v1/images/generations``).

It holds no GPU memory while idle: each picture is made by a short-lived worker process
(``worker.py``) that loads the model, draws, and exits, so the card is completely free again
afterwards and the language model that was paused for it can come back. At startup the model is
downloaded once and its weight files are read through so they stay in the operating system's
file cache (RAM); every later load then comes from memory and takes seconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import shlex
import sys
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = os.environ.get("IMAGE_MODEL", "Lykon/dreamshaper-8")
WORKER = shlex.split(os.environ.get("IMAGE_WORKER_CMD", f"{sys.executable} /app/worker.py"))
TIMEOUT_S = float(os.environ.get("IMAGE_TIMEOUT_S", "900"))
DEFAULT_STEPS = int(os.environ.get("IMAGE_STEPS", "25"))
SAFETY_REPO = "CompVis/stable-diffusion-safety-checker"
SKIPPED = (".ckpt", ".onnx", ".onnx_data", ".msgpack", ".h5", ".pb", ".ot")

state: dict[str, Any] = {"ready": False, "error": None, "started": time.time()}


def _files_to_fetch(files: list[str]) -> list[str]:
    """Only what a diffusers pipeline loads: no half-precision copies, no single-file checkpoints,
    and .bin weights only where a folder has no .safetensors."""

    wanted = []
    for name in files:
        if "/" not in name:
            if name == "model_index.json":
                wanted.append(name)
            continue
        if "fp16" in name or name.endswith(SKIPPED) or "flax" in name:
            continue
        folder = name.rsplit("/", 1)[0]
        if name.endswith(".bin") and any(
            other.startswith(folder + "/") and other.endswith(".safetensors") and "fp16" not in other
            for other in files
        ):
            continue
        wanted.append(name)
    return wanted


def prepare() -> None:
    if os.environ.get("IMAGE_SKIP_DOWNLOAD"):
        state["ready"] = True
        return
    try:
        from huggingface_hub import HfApi, snapshot_download

        files = HfApi().list_repo_files(MODEL)
        path = snapshot_download(MODEL, allow_patterns=_files_to_fetch(files))
        if not any(f.startswith("safety_checker/") for f in files):
            snapshot_download(SAFETY_REPO)  # the stand-alone checker, run on the CPU
        for weights in pathlib.Path(path).rglob("*"):  # warm the file cache: loads come from RAM
            if weights.suffix in (".safetensors", ".bin"):
                with weights.open("rb") as fh:
                    while fh.read(1 << 24):
                        pass
        state["ready"] = True
    except Exception as exc:  # reported on /health and to callers
        state["error"] = f"{type(exc).__name__}: {exc}"


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=prepare, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
lock = asyncio.Lock()


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = ""
    size: str = "512x512"
    n: int = 1
    response_format: str = "b64_json"
    steps: int | None = None
    model: str | None = None


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x", 1))
    except ValueError:
        w, h = 512, 512
    clamp = lambda v: max(256, min(1024, v - v % 8))  # noqa: E731
    return clamp(w), clamp(h)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ready": state["ready"], "error": state["error"], "model": MODEL, "busy": lock.locked()}


@app.post("/v1/images/generations")
async def generate(body: GenerateIn) -> dict[str, Any]:
    if not state["ready"]:
        detail = state["error"] or "the image model is still downloading; try again in a few minutes"
        raise HTTPException(status_code=503, detail=detail)
    width, height = _size(body.size)
    job = {
        "model": MODEL,
        "prompt": body.prompt,
        "negative_prompt": body.negative_prompt,
        "width": width,
        "height": height,
        "steps": max(1, min(60, body.steps or DEFAULT_STEPS)),
    }
    async with lock:  # one picture at a time
        proc = await asyncio.create_subprocess_exec(
            *WORKER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(json.dumps(job).encode()), timeout=TIMEOUT_S
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise HTTPException(status_code=504, detail="the picture took too long") from None
    if proc.returncode != 0:
        tail = err.decode(errors="replace").strip().splitlines()[-5:]
        raise HTTPException(status_code=500, detail="image worker failed: " + " | ".join(tail))
    try:
        result = json.loads(out.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise HTTPException(status_code=500, detail="image worker returned nothing") from None
    if result.get("error") == "explicit":
        raise HTTPException(
            status_code=422, detail="the picture came out explicit, so it was thrown away"
        )
    return {
        "created": int(time.time()),
        "device": result.get("device"),
        "data": [{"b64_json": result["b64"]}],
    }

"""The image server in deploy/imagegen, with a stand-in worker (no GPU or model needed)."""

from __future__ import annotations

import base64
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).parent.parent / "deploy" / "imagegen" / "server.py"
TINY_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()


def load_server(monkeypatch: pytest.MonkeyPatch, worker: Path) -> Any:
    monkeypatch.setenv("IMAGE_SKIP_DOWNLOAD", "1")
    monkeypatch.setenv("IMAGE_WORKER_CMD", f"{sys.executable} {worker}")
    spec = importlib.util.spec_from_file_location("imagegen_server", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_serves_openai_style_images_from_a_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            f"""
            import json, sys
            job = json.load(sys.stdin)
            print("progress noise")
            print(json.dumps({{"b64": "{TINY_PNG}", "device": "cuda", "job": job}}))
            """
        )
    )
    server = load_server(monkeypatch, worker)
    with TestClient(server.app) as client:
        assert client.get("/health").json()["ready"] is True
        r = client.post(
            "/v1/images/generations",
            json={"prompt": "a lighthouse", "size": "999x300", "response_format": "b64_json"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["data"][0]["b64_json"] == TINY_PNG
        assert server._size("999x300") == (992, 296)  # multiples of 8, within limits


def test_a_failing_worker_is_reported_not_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text("import sys; print('CUDA out of memory', file=sys.stderr); sys.exit(1)\n")
    server = load_server(monkeypatch, worker)
    with TestClient(server.app) as client:
        r = client.post("/v1/images/generations", json={"prompt": "a lighthouse"})
        assert r.status_code == 500 and "CUDA out of memory" in r.json()["detail"]


def test_only_the_files_a_pipeline_needs_are_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = load_server(monkeypatch, tmp_path / "unused.py")
    files = [
        "model_index.json",
        "dreamshaper_8.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.bin",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.bin",
        "tokenizer/vocab.json",
        "unet/model.onnx",
    ]
    assert server._files_to_fetch(files) == [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.bin",  # no safetensors in that folder, so the .bin is kept
        "tokenizer/vocab.json",
    ]


def test_an_explicit_picture_is_thrown_away(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text('import json; print(json.dumps({"error": "explicit", "device": "cuda"}))\n')
    server = load_server(monkeypatch, worker)
    with TestClient(server.app) as client:
        r = client.post("/v1/images/generations", json={"prompt": "a beach"})
        assert r.status_code == 422 and "thrown away" in r.json()["detail"]

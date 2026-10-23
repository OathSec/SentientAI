"""One-click setup: install the model runtime (Ollama), pick a model that fits this GPU,
download it, give it the memory Fuzzy needs, and check that it answers.

Everything here talks to Ollama on this computer, except two downloads over HTTPS: Ollama's own
installer from ollama.com (its Microsoft signature is checked before it is run) and the model
itself, which Ollama downloads from its library.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from fuzzy_worker.core import Config, gpu_info

OLLAMA_URL = "http://127.0.0.1:11434"
INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
CONTEXT = 12288  # what the server plans each request for (see docs/workers.md)
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

Report = Callable[[str, float | None], None]  # (what is happening, 0..1 or None)


# ----------------------------------------------------------------- what to install
def recommend(vram_mb: int | None) -> tuple[str, str]:
    """(Ollama model tag, why) for this much video memory. Official Qwen3, like the server's."""

    if not vram_mb:
        return "qwen3:4b", "No NVIDIA graphics card found, so a small model that runs on the CPU."
    gb = vram_mb / 1024
    if vram_mb >= 13000:
        return "qwen3:14b", f"{gb:.0f} GB of video memory: room for the 14B model Fuzzy uses."
    if vram_mb >= 7000:
        return "qwen3:8b", f"{gb:.0f} GB of video memory: the 8B model fits well and is fast."
    return "qwen3:4b", f"{gb:.0f} GB of video memory: a small, quick model."


def parse_code(code: str) -> tuple[str, str] | None:
    """A connection code from the server operator -> (server address, token)."""

    code = code.strip()
    if code.startswith("fuzzy-worker:"):
        code = code.split(":", 1)[1]
    try:
        data = json.loads(base64.urlsafe_b64decode(code + "=" * (-len(code) % 4)))
        return str(data["s"]), str(data["t"])
    except (ValueError, KeyError, TypeError):
        return None


def make_code(server: str, token: str) -> str:
    raw = json.dumps({"s": server, "t": token}, separators=(",", ":")).encode()
    return "fuzzy-worker:" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


# ------------------------------------------------------------------------- ollama
def find_ollama() -> Path | None:
    local = os.environ.get("LOCALAPPDATA", "")
    for candidate in (
        Path(local) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Ollama" / "ollama.exe",
    ):
        if candidate.is_file():
            return candidate
    from shutil import which

    found = which("ollama")
    return Path(found) if found else None


def is_running(client: httpx.Client) -> bool:
    try:
        return client.get(OLLAMA_URL + "/api/version", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


def start_ollama(path: Path, client: httpx.Client, report: Report) -> None:
    if is_running(client):
        return
    report("Starting Ollama", None)
    subprocess.Popen(
        [str(path), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    for _ in range(40):
        if is_running(client):
            return
        time.sleep(0.5)
    raise SetupError("Ollama didn't start. Try restarting the computer and run setup again.")


class SetupError(Exception):
    pass


def signed_by_ollama(exe: Path) -> bool:
    """Windows' own check that the installer is signed (and by Ollama)."""

    script = (
        f"$s = Get-AuthenticodeSignature -LiteralPath '{exe}'; "
        "if ($s.Status -eq 'Valid' -and $s.SignerCertificate.Subject -match 'Ollama') "
        "{ 'OK' } else { 'BAD' }"
    )
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            creationflags=NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return out.strip() == "OK"


def install_ollama(client: httpx.Client, report: Report) -> Path:
    """Download Ollama's installer over HTTPS, check its signature, run it quietly."""

    with tempfile.TemporaryDirectory(prefix="fuzzy-worker-") as tmp:
        exe = Path(tmp) / "OllamaSetup.exe"
        report("Downloading Ollama", 0.0)
        try:
            with client.stream("GET", INSTALLER_URL, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                done = 0
                with exe.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        report("Downloading Ollama", done / total if total else None)
        except httpx.HTTPError as exc:
            raise SetupError(f"Couldn't download Ollama ({type(exc).__name__}).") from exc
        report("Checking the installer is genuine", None)
        if not signed_by_ollama(exe):
            raise SetupError("The Ollama installer's signature didn't check out, so I stopped.")
        report("Installing Ollama (a minute or two)", None)
        result = subprocess.run([str(exe), "/SILENT", "/NORESTART"], check=False, timeout=900)
        if result.returncode != 0:
            raise SetupError(f"The Ollama installer failed (code {result.returncode}).")
    found = find_ollama()
    if found is None:
        raise SetupError("Ollama installed but I can't find it. Restart and run setup again.")
    return found


def have_model(client: httpx.Client, tag: str) -> bool:
    try:
        data = client.get(OLLAMA_URL + "/api/tags", timeout=10.0).json()
    except (httpx.HTTPError, ValueError):
        return False
    return any(
        m.get("name") == tag or m.get("name") == tag + ":latest" for m in data.get("models", [])
    )


def pull_model(client: httpx.Client, tag: str, report: Report) -> None:
    """Download a model through Ollama, reporting progress."""

    report(f"Downloading {tag}", 0.0)
    try:
        with client.stream(
            "POST", OLLAMA_URL + "/api/pull", json={"model": tag, "stream": True}, timeout=None
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise SetupError(f"Ollama couldn't download {tag}: {event['error']}")
                if event.get("total"):
                    report(f"Downloading {tag}", event.get("completed", 0) / event["total"])
                else:
                    report(f"{tag}: {event.get('status', '')}", None)
    except httpx.HTTPError as exc:
        raise SetupError(f"The download of {tag} was interrupted ({type(exc).__name__}).") from exc


def make_variant(ollama: Path, tag: str, report: Report) -> str:
    """The same model with a context window big enough for Fuzzy's long prompts."""

    name = "fuzzy-" + tag.replace(":", "-")
    report("Tuning the model for Fuzzy", None)
    with tempfile.TemporaryDirectory(prefix="fuzzy-worker-") as tmp:
        modelfile = Path(tmp) / "Modelfile"
        modelfile.write_text(f"FROM {tag}\nPARAMETER num_ctx {CONTEXT}\n", encoding="utf-8")
        result = subprocess.run(
            [str(ollama), "create", name, "-f", str(modelfile)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            creationflags=NO_WINDOW,
        )
    if result.returncode != 0:
        raise SetupError("Couldn't tune the model: " + (result.stderr or result.stdout)[-200:])
    return name


def check_answers(client: httpx.Client, model: str, report: Report) -> str:
    report("Testing that it answers (the first time loads it)", None)
    try:
        r = client.post(
            OLLAMA_URL + "/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Reply with one short friendly sentence."}
                ],
                "max_tokens": 60,
                "think": False,
                "reasoning_effort": "none",
            },
            timeout=600.0,
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"]).strip()[:200]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise SetupError(f"The model didn't answer ({type(exc).__name__}).") from exc


# ------------------------------------------------------------------- the whole thing
def run_setup(
    config: Config,
    report: Report,
    *,
    model: str | None = None,
    allow_install: bool = True,
    client: httpx.Client | None = None,
) -> str:
    """Set this PC up to lend its GPU to Fuzzy. Returns a short summary; raises SetupError."""

    own = client is None
    client = client or httpx.Client()
    try:
        info = gpu_info()
        tag = model or recommend(info.get("gpu_memory_mb"))[0]
        ollama = find_ollama()
        if ollama is None:
            if not allow_install:
                raise SetupError("Ollama isn't installed.")
            ollama = install_ollama(client, report)
        start_ollama(ollama, client, report)
        if not have_model(client, tag):
            pull_model(client, tag, report)
        variant = make_variant(ollama, tag, report)
        reply = check_answers(client, variant, report)
        config.run_llm = True
        config.llm_url = OLLAMA_URL + "/v1"
        config.llm_model = variant
        report("All set", 1.0)
        return f"{variant} is ready. It said: {reply}"
    finally:
        if own:
            client.close()


# ----------------------------------------------------------------------- windows
def set_autostart(enabled: bool) -> None:
    """Start this program when Windows starts (per user, no admin rights)."""

    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    import winreg

    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0,
        winreg.KEY_SET_VALUE,
    )
    try:
        if enabled:
            winreg.SetValueEx(key, "FuzzyWorker", 0, winreg.REG_SZ, f'"{sys.executable}"')
        else:
            with contextlib.suppress(FileNotFoundError):
                winreg.DeleteValue(key, "FuzzyWorker")
    finally:
        winreg.CloseKey(key)


def describe(info: dict[str, Any]) -> str:
    if not info:
        return "No NVIDIA graphics card found."
    return f"{info['gpu']} ({info['gpu_memory_mb'] / 1024:.0f} GB)"

"""What the server is made of and how hot it is running (read-only, best effort).

Everything is read from the machine itself: ``nvidia-smi`` for the GPUs, ``/proc`` and
``/sys`` for the CPU, memory and temperatures. Anything that isn't available (no GPU
passed through, no thermal sensors in a container) is reported as missing rather than
guessed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

_GPU_FIELDS = (
    "index",
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "fan.speed",
    "clocks.sm",
    "pstate",
    "driver_version",
)


def _number(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


async def gpus() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            exe,
            f"--query-gpu={','.join(_GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except (OSError, TimeoutError):
        return []
    found: list[dict[str, Any]] = []
    for line in out.decode(errors="replace").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(_GPU_FIELDS):
            continue
        row = dict(zip(_GPU_FIELDS, cells, strict=True))
        found.append(
            {
                "index": int(row["index"]) if row["index"].isdigit() else len(found),
                "name": row["name"],
                "temp_c": _number(row["temperature.gpu"]),
                "util_pct": _number(row["utilization.gpu"]),
                "mem_used_mb": _number(row["memory.used"]),
                "mem_total_mb": _number(row["memory.total"]),
                "power_w": _number(row["power.draw"]),
                "power_limit_w": _number(row["power.limit"]),
                "fan_pct": _number(row["fan.speed"]),
                "clock_mhz": _number(row["clocks.sm"]),
                "pstate": row["pstate"],
                "driver": row["driver_version"],
            }
        )
    return found


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def cpu() -> dict[str, Any]:
    info = _read("/proc/cpuinfo")
    names = [ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name")]
    load = _read("/proc/loadavg").split()
    return {
        "model": names[0] if names else "unknown",
        "threads": os.cpu_count() or 0,
        "load": [float(x) for x in load[:3]] if len(load) >= 3 else [],
    }


def memory() -> dict[str, Any]:
    values: dict[str, int] = {}
    for line in _read("/proc/meminfo").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) // 1024
    total, free = values.get("MemTotal"), values.get("MemAvailable")
    return {
        "total_mb": total,
        "available_mb": free,
        "used_mb": total - free if total is not None and free is not None else None,
    }


def temperatures() -> list[dict[str, Any]]:
    """CPU and board sensors the kernel exposes (``hwmon`` first, then thermal zones)."""

    found: list[dict[str, Any]] = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        chip = _read(f"{hwmon}/name").strip() or hwmon.name
        if chip in {"nvme", "amdgpu", "nouveau"} or chip.startswith("nvidia"):
            pass  # still useful: list them
        for temp in sorted(hwmon.glob("temp*_input")):
            value = _number(_read(str(temp)).strip())
            if value is None:
                continue
            label = _read(str(temp).replace("_input", "_label")).strip() or temp.name
            found.append({"sensor": f"{chip} {label}", "temp_c": round(value / 1000, 1)})
    if not found:
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            value = _number(_read(f"{zone}/temp").strip())
            if value is not None:
                kind = _read(f"{zone}/type").strip() or zone.name
                found.append({"sensor": kind, "temp_c": round(value / 1000, 1)})
    return found


def disks(paths: list[Path]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            device = path.stat().st_dev
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        if device in seen:
            continue
        seen.add(device)
        rows.append(
            {
                "path": str(path),
                "total_gb": round(usage.total / 1e9, 1),
                "used_gb": round(usage.used / 1e9, 1),
                "free_gb": round(usage.free / 1e9, 1),
            }
        )
    return rows


async def collect(data_dir: Path) -> dict[str, Any]:
    uptime = _read("/proc/uptime").split()
    return {
        "gpus": await gpus(),
        "cpu": cpu(),
        "memory": memory(),
        "temperatures": temperatures(),
        "disks": disks([data_dir, Path("/")]),
        "uptime_s": float(uptime[0]) if uptime else None,
    }

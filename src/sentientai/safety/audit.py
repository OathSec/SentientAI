"""Tamper-evident, append-only audit log.

Every action, reasoning step and tool result is written as one JSON line.
Each entry commits to its predecessor (``prev``) through SHA-256, or
HMAC-SHA256 when a key is configured, so any edit, deletion or reordering
breaks the chain and is detected by :meth:`AuditLog.verify`.

The log is split into size-bounded segments (``audit-000001.jsonl`` ...);
the chain continues across segments. Writers serialize through an exclusive
``flock`` so the daemon, the CLI and the API can all append safely.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sentientai.core.clock import utcnow
from sentientai.core.logging import get_logger

log = get_logger(__name__)

GENESIS = "0" * 64
_SEGMENT_RE = re.compile(r"^audit-(\d{6})\.jsonl$")
_SECRET_KEYS = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|authorization|cookie|credential)", re.I
)
_MAX_FIELD_CHARS = 8000


class AuditEntry(BaseModel):
    seq: int
    ts: datetime
    type: str
    actor: str
    task_id: str | None = None
    payload: dict[str, Any]
    prev: str
    hash: str


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    entries: int
    segments: int
    error: str | None = None
    bad_seq: int | None = None


def sanitize(value: Any, *, max_chars: int = _MAX_FIELD_CHARS, _depth: int = 0) -> Any:
    """Redact secrets and bound payload size so the log stays readable and safe."""

    if _depth > 12:
        return "[max depth]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            out[key] = (
                "***"
                if _SECRET_KEYS.search(key)
                else sanitize(v, max_chars=max_chars, _depth=_depth + 1)
            )
        return out
    if isinstance(value, list | tuple | set):
        return [sanitize(v, max_chars=max_chars, _depth=_depth + 1) for v in value]
    if isinstance(value, str):
        if len(value) > max_chars:
            digest = hashlib.sha256(value.encode()).hexdigest()[:16]
            return (
                f"{value[:max_chars]}...[truncated {len(value) - max_chars} chars sha256:{digest}]"
            )
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return sanitize(str(value), max_chars=max_chars, _depth=_depth + 1)


def _canonical(entry: dict[str, Any]) -> bytes:
    body = {k: entry[k] for k in ("seq", "ts", "type", "actor", "task_id", "payload", "prev")}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode()


class AuditLog:
    def __init__(
        self,
        directory: Path,
        *,
        hmac_key: bytes | None = None,
        fsync: bool = True,
        segment_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.directory = directory
        self._key = hmac_key
        self._fsync = fsync
        self._segment_max = segment_max_bytes
        self._lock = asyncio.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lockfile = self.directory / ".lock"

    # ------------------------------------------------------------- hashing
    def _digest(self, entry: dict[str, Any]) -> str:
        data = _canonical(entry)
        if self._key:
            return hmac.new(self._key, data, hashlib.sha256).hexdigest()
        return hashlib.sha256(data).hexdigest()

    # ------------------------------------------------------------ segments
    def segments(self) -> list[Path]:
        found = []
        for path in self.directory.iterdir():
            match = _SEGMENT_RE.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return [p for _, p in sorted(found)]

    @staticmethod
    def _last_line(path: Path) -> bytes | None:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size == 0:
                return None
            block = 4096
            pos = size
            data = b""
            while pos > 0:
                step = min(block, pos)
                pos -= step
                fh.seek(pos)
                data = fh.read(step) + data
                stripped = data.rstrip(b"\n")
                if b"\n" in stripped:
                    return stripped.rsplit(b"\n", 1)[1]
            return data.rstrip(b"\n") or None

    def _repair_torn_tail(self, path: Path) -> None:
        """A crash mid-write can leave a partial last line; move it aside."""

        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size == 0:
                return
            fh.seek(size - 1)
            if fh.read(1) == b"\n":
                return
        raw = path.read_bytes()
        cut = raw.rfind(b"\n") + 1
        torn = raw[cut:]
        sidecar = path.with_suffix(".torn")
        with sidecar.open("ab") as fh:
            fh.write(torn + b"\n")
        with path.open("r+b") as fh:
            fh.truncate(cut)
        log.warning(
            "audit: moved torn tail to sidecar", extra={"segment": path.name, "bytes": len(torn)}
        )

    def _tail_state(self) -> tuple[Path, int, str]:
        """Return (active segment, next seq, prev hash). Caller holds the file lock."""

        segments = self.segments()
        if not segments:
            return self.directory / "audit-000001.jsonl", 0, GENESIS
        active = segments[-1]
        self._repair_torn_tail(active)
        for path in reversed(segments):
            last = self._last_line(path)
            if last is not None:
                entry = json.loads(last)
                seq, prev = int(entry["seq"]) + 1, str(entry["hash"])
                break
        else:
            seq, prev = 0, GENESIS
        if active.stat().st_size >= self._segment_max:
            number = int(_SEGMENT_RE.match(active.name).group(1)) + 1  # type: ignore[union-attr]
            active = self.directory / f"audit-{number:06d}.jsonl"
        return active, seq, prev

    # --------------------------------------------------------------- write
    def _append_sync(
        self, type_: str, actor: str, task_id: str | None, payload: dict[str, Any]
    ) -> AuditEntry:
        with self._lockfile.open("a") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                segment, seq, prev = self._tail_state()
                entry: dict[str, Any] = {
                    "seq": seq,
                    "ts": utcnow().isoformat(),
                    "type": type_,
                    "actor": actor,
                    "task_id": task_id,
                    "payload": sanitize(payload),
                    "prev": prev,
                }
                entry["hash"] = self._digest(entry)
                line = (
                    json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
                )
                with segment.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    if self._fsync:
                        os.fsync(fh.fileno())
                return AuditEntry.model_validate(entry)
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)

    async def record(
        self, type_: str, *, actor: str = "agent", task_id: str | None = None, **payload: Any
    ) -> AuditEntry:
        async with self._lock:
            return await asyncio.to_thread(self._append_sync, type_, actor, task_id, payload)

    def record_sync(
        self, type_: str, *, actor: str = "operator", task_id: str | None = None, **payload: Any
    ) -> AuditEntry:
        return self._append_sync(type_, actor, task_id, payload)

    # ---------------------------------------------------------------- read
    def iter_entries(self) -> Iterator[AuditEntry]:
        for segment in self.segments():
            with segment.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield AuditEntry.model_validate_json(line)

    def tail(self, n: int = 20, type_prefix: str | None = None) -> list[AuditEntry]:
        entries = [
            e for e in self.iter_entries() if type_prefix is None or e.type.startswith(type_prefix)
        ]
        return entries[-n:]

    def recent(
        self, n: int, types: tuple[str, ...], *, since_last: str | None = None
    ) -> list[AuditEntry]:
        """The newest ``n`` entries of the given types, reading the newest segments first.

        With ``since_last``, only entries after the most recent entry of that type (for example
        everything since Fuzzy's mind was last reset); the log itself is never altered.
        """

        found: list[AuditEntry] = []
        for segment in reversed(self.segments()):
            with segment.open("r", encoding="utf-8") as fh:
                entries = [AuditEntry.model_validate_json(ln) for ln in fh if ln.strip()]
            boundary = next(
                (i for i in range(len(entries) - 1, -1, -1) if entries[i].type == since_last),
                None,
            )
            if boundary is not None:
                entries = entries[boundary + 1 :]
            found = [e for e in entries if e.type in types] + found
            if boundary is not None or len(found) >= n:
                break
        return found[-n:]

    def verify(self) -> VerifyResult:
        prev = GENESIS
        expected_seq = 0
        count = 0
        segments = self.segments()
        for segment in segments:
            with segment.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        return VerifyResult(
                            False,
                            count,
                            len(segments),
                            f"{segment.name}:{lineno}: unparseable entry",
                            expected_seq,
                        )
                    seq = raw.get("seq")
                    if seq != expected_seq:
                        return VerifyResult(
                            False,
                            count,
                            len(segments),
                            f"{segment.name}:{lineno}: expected seq {expected_seq}, got {seq}",
                            expected_seq,
                        )
                    if raw.get("prev") != prev:
                        return VerifyResult(
                            False,
                            count,
                            len(segments),
                            f"seq {seq}: chain broken (prev hash mismatch)",
                            seq,
                        )
                    if not hmac.compare_digest(self._digest(raw), str(raw.get("hash"))):
                        return VerifyResult(
                            False,
                            count,
                            len(segments),
                            f"seq {seq}: content hash mismatch (entry modified)",
                            seq,
                        )
                    prev = raw["hash"]
                    expected_seq += 1
                    count += 1
        return VerifyResult(True, count, len(segments))

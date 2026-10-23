"""Image generation for Fuzzy, on local GPUs.

Two backends cover the common local servers:

* ``automatic1111`` - Stable Diffusion WebUI / Forge (``POST /sdapi/v1/txt2img``)
* ``openai`` - any OpenAI-compatible ``/v1/images/generations`` server (LocalAI and others)

Images are checked (real PNG/JPEG/WebP, size limit), stored under the data
directory and recorded as :class:`MediaRecord`. Generations run one at a time:
the GPU is shared with the language models.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
import struct
from pathlib import Path
from typing import Any, Protocol

import httpx

from sentientai.config import ImagesConfig
from sentientai.core.clock import Clock, SystemClock
from sentientai.core.events import EventBus
from sentientai.core.logging import get_logger
from sentientai.media.gpuswap import GpuSwap
from sentientai.safety.audit import AuditLog
from sentientai.storage.database import Database
from sentientai.storage.records import MediaRecord
from sentientai.workers.hub import WorkerError, WorkerHub

log = get_logger(__name__)


class ImageError(Exception):
    pass


# Always refused: sexual content involving minors. Checked on word boundaries.
_MINORS = re.compile(
    r"\b(child|children|kid|kids|minor|minors|underage|teen|teens|teenage|preteen|loli|"
    r"schoolgirl|schoolboy|toddler|infant|baby)\b",
    re.I,
)
_SEXUAL = re.compile(
    r"\b(nude|naked|nsfw|sex|sexual|sexy|porn|erotic|explicit|lingerie|topless|undressed|"
    r"genitals?|breasts?|fetish)\b",
    re.I,
)


def screen_prompt(prompt: str) -> str | None:
    """Return a reason to refuse the prompt, or None."""

    if _SEXUAL.search(prompt):
        return "sexual or explicit images aren't something I'll make"
    if _MINORS.search(prompt) and re.search(r"\b(bikini|bath|bed|kiss)\b", prompt, re.I):
        return "I won't make that image"
    return None


def sniff_image(data: bytes) -> tuple[str, str, int, int]:
    """(mime, extension, width, height) for PNG, JPEG or WebP data; raises ImageError."""

    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 24:
        width, height = struct.unpack(">II", data[16:24])
        return "image/png", "png", width, height
    if data.startswith(b"\xff\xd8"):
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return "image/jpeg", "jpg", width, height
            i += 2 + length
        return "image/jpeg", "jpg", 0, 0
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp", 0, 0
    raise ImageError("the image server returned something that isn't an image")


def _decode_b64(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        return base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("the image server returned invalid data") from exc


def _detail(response: httpx.Response) -> str:
    """The server's own explanation (FastAPI-style ``detail``), else its status and text."""

    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("detail"), str):
        return str(data["detail"])[:300]
    return f"error {response.status_code}: {response.text[:300]}"


class ImageBackend(Protocol):
    name: str

    async def generate(self, prompt: str, *, config: ImagesConfig) -> bytes: ...

    async def aclose(self) -> None: ...


class _HttpBackend:
    name = "http"

    def __init__(self, config: ImagesConfig, client: httpx.AsyncClient | None = None) -> None:
        headers = {}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self.client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"), timeout=config.timeout_s, headers=headers
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = await self.client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise ImageError(f"image server unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise ImageError(f"the image server said: {_detail(response)}")
        try:
            return response.json()
        except ValueError as exc:
            raise ImageError("image server returned invalid JSON") from exc

    async def aclose(self) -> None:
        await self.client.aclose()


class Automatic1111Backend(_HttpBackend):
    name = "automatic1111"

    async def generate(self, prompt: str, *, config: ImagesConfig) -> bytes:
        payload = {
            "prompt": prompt,
            "negative_prompt": config.negative_prompt,
            "width": config.width,
            "height": config.height,
            "steps": config.steps,
            "batch_size": 1,
            "n_iter": 1,
        }
        data = await self._post("/sdapi/v1/txt2img", payload)
        images = data.get("images") if isinstance(data, dict) else None
        if not images:
            raise ImageError("image server returned no image")
        return _decode_b64(str(images[0]))


class OpenAIImagesBackend(_HttpBackend):
    name = "openai"

    async def not_ready(self) -> str | None:
        """Why the server can't draw yet, if it says so on /health (the bundled server does;
        others just don't answer there and are assumed ready)."""

        try:
            response = await self.client.get("/health", timeout=10.0)
            data = response.json() if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None
        if isinstance(data, dict) and data.get("ready") is False:
            return str(
                data.get("error")
                or "the image model is still downloading; it'll be ready in a few minutes"
            )
        return None

    async def generate(self, prompt: str, *, config: ImagesConfig) -> bytes:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n": 1,
            "size": f"{config.width}x{config.height}",
            "response_format": "b64_json",
        }
        if config.model:
            payload["model"] = config.model
        if config.negative_prompt:
            payload["negative_prompt"] = config.negative_prompt  # honoured by LocalAI
        data = await self._post("/v1/images/generations", payload)
        items = data.get("data") if isinstance(data, dict) else None
        if not items or not isinstance(items[0], dict) or not items[0].get("b64_json"):
            raise ImageError("image server returned no image")
        return _decode_b64(str(items[0]["b64_json"]))


def build_backend(config: ImagesConfig) -> ImageBackend | None:
    if config.backend == "automatic1111":
        return Automatic1111Backend(config)
    if config.backend == "openai":
        return OpenAIImagesBackend(config)
    return None


class ImageService:
    def __init__(
        self,
        config: ImagesConfig,
        db: Database,
        media_dir: Path,
        *,
        audit: AuditLog,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        backend: ImageBackend | None = None,
        swap: GpuSwap | None = None,
        workers: WorkerHub | None = None,
    ) -> None:
        self.workers = workers
        self.config = config
        self.db = db
        self.media_dir = media_dir
        self.audit = audit
        self.bus = bus
        self.clock = clock or SystemClock()
        self.backend = backend if backend is not None else build_backend(config)
        self._gpu = asyncio.Lock()
        self._drawing: dict[str | None, int] = {}  # who a picture is being made for right now
        if swap is None and self.backend is not None and config.swap_container:
            swap = GpuSwap(
                config.swap_container, grace_s=config.swap_grace_s, socket=config.docker_socket
            )
        self.swap = swap

    @property
    def available(self) -> bool:
        return self.backend is not None or (
            self.workers is not None and self.workers.available("image", private_only=False)
        )

    def path_for(self, record: MediaRecord) -> Path:
        path = Path(record.path).resolve()
        if not path.is_relative_to(self.media_dir.resolve()):
            raise ImageError("media path outside the media directory")
        return path

    async def create(
        self,
        prompt: str,
        *,
        user_id: str | None,
        public: bool,
        caption: str = "",
        by: str = "agent",
    ) -> MediaRecord:
        if not self.available:
            raise ImageError("image generation isn't set up on this server")
        prompt = " ".join(prompt.split())[:1000]
        if len(prompt) < 3:
            raise ImageError("describe the image first")
        refusal = screen_prompt(prompt)
        if refusal:
            await self.audit.record("media.refused", actor=by, user_id=user_id, prompt=prompt)
            raise ImageError(refusal)
        await self._emit("media.generating", prompt=prompt[:200], user_id=user_id)
        self._drawing[user_id] = self._drawing.get(user_id, 0) + 1
        try:
            data = await self._draw(prompt, public=public)
        finally:
            self._drawing[user_id] -= 1
            if not self._drawing[user_id]:
                del self._drawing[user_id]
        if len(data) > self.config.max_bytes:
            raise ImageError("the generated image is too large")
        return await self._store(
            data, prompt, user_id=user_id, public=public, caption=caption, by=by
        )

    def drawing_for(self, user_id: str | None) -> bool:
        """Is a picture being made for this person right now?"""

        return bool(self._drawing.get(user_id))

    async def _from_worker(self, prompt: str, *, public: bool) -> bytes | None:
        """A picture drawn on a connected PC, if one is free; None to draw it here.

        Only Fuzzy's own public pictures may go to a friend's PC (``private_only=False``): a
        private picture's prompt is a private conversation."""

        if self.workers is None or not self.workers.available("image", private_only=not public):
            return None
        job = {
            "prompt": prompt,
            "negative_prompt": self.config.negative_prompt,
            "width": self.config.width,
            "height": self.config.height,
            "steps": self.config.steps,
        }
        try:
            result = await self.workers.run(
                "image", job, private_only=not public, timeout=self.config.timeout_s
            )
            data = _decode_b64(str(result.get("b64_json") or ""))
            sniff_image(data)  # a real picture, or it's drawn here instead
            return data
        except (WorkerError, ImageError) as exc:
            log.warning("worker couldn't draw; drawing here", extra={"error": str(exc)})
            return None

    async def _draw(self, prompt: str, *, public: bool = False) -> bytes:
        remote = await self._from_worker(prompt, public=public)
        if remote is not None:
            return remote
        if self.backend is None:
            raise ImageError("no computer is free to draw it right now")
        check = getattr(self.backend, "not_ready", None)
        reason = await check() if check is not None else None
        if reason:  # don't pause a model server for a picture that can't be made yet
            raise ImageError(f"I can't make pictures just yet: {reason}")
        async with self._gpu:
            if self.swap is not None:
                async with self.swap.borrow():
                    return await self.backend.generate(prompt, config=self.config)
            return await self.backend.generate(prompt, config=self.config)

    async def _store(
        self, data: bytes, prompt: str, *, user_id: str | None, public: bool, caption: str, by: str
    ) -> MediaRecord:
        mime, ext, width, height = sniff_image(data)
        record = MediaRecord(
            prompt=prompt,
            path="",
            mime=mime,
            width=width,
            height=height,
            user_id=user_id,
            public=public,
            caption=" ".join(caption.split())[:500],
            created_at=self.clock.now(),
        )
        self.media_dir.mkdir(parents=True, exist_ok=True)
        target = self.media_dir / f"{record.id}.{ext}"
        tmp = target.with_suffix(".part")
        await asyncio.to_thread(tmp.write_bytes, data)
        os.replace(tmp, target)
        record.path = str(target)
        await self.db.media.insert(record)
        await self.audit.record(
            "media.created", actor=by, media_id=record.id, user_id=user_id, prompt=prompt
        )
        await self._emit(
            "media.created",
            media_id=record.id,
            user_id=user_id,
            public=public,
            caption=record.caption[:200],
        )
        return record

    async def _emit(self, type_: str, **payload: Any) -> None:
        if self.bus is not None:
            await self.bus.publish(type_, **payload)

    async def recover(self) -> None:
        if self.swap is not None:
            await self.swap.recover()

    async def aclose(self) -> None:
        if self.swap is not None:
            await self.swap.aclose()
        if self.backend is not None:
            await self.backend.aclose()

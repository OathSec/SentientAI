"""Static pages and assets of the web app (no inline scripts or styles: the CSP forbids them)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
PAGES = {
    "/": "chat.html",
    "/login": "login.html",
    "/brain": "brain.html",
    "/world": "world.html",
    "/gallery": "gallery.html",
    "/you": "you.html",
    "/teach": "teach.html",
    "/compute": "compute.html",
    "/admin": "admin.html",
}


def asset_version() -> str:
    """Changes whenever a script or style changes, so browsers and Cloudflare never serve an
    old copy next to a new page."""

    digest = hashlib.sha1(usedforsecurity=False)
    for f in sorted((ROOT / "static").glob("*")):
        if f.is_file():
            digest.update(f"{f.name}{f.stat().st_size}{f.stat().st_mtime_ns}".encode())
    return digest.hexdigest()[:10]


def mount_pages(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    def page(filename: str) -> object:
        path = ROOT / "pages" / filename
        version = asset_version()
        html = re.sub(
            r'((?:src|href)="/static/[\w.\-]+\.(?:js|css))"',
            lambda m: f'{m.group(1)}?v={version}"',
            path.read_text(encoding="utf-8"),
        )

        async def serve() -> Response:
            return Response(html, media_type="text/html", headers={"Cache-Control": "no-cache"})

        return serve

    for route, filename in PAGES.items():
        app.add_api_route(route, page(filename), methods=["GET"], include_in_schema=False)  # type: ignore[arg-type]

    async def favicon() -> Response:
        return FileResponse(ROOT / "static" / "icon.svg", media_type="image/svg+xml")

    app.add_api_route("/favicon.ico", favicon, methods=["GET"], include_in_schema=False)

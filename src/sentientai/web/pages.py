"""Static pages and assets of the web app (no inline scripts or styles: the CSP forbids them)."""

from __future__ import annotations

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


def mount_pages(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    def page(filename: str) -> object:
        path = ROOT / "pages" / filename

        async def serve() -> Response:
            return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})

        return serve

    for route, filename in PAGES.items():
        app.add_api_route(route, page(filename), methods=["GET"], include_in_schema=False)  # type: ignore[arg-type]

    async def favicon() -> Response:
        return FileResponse(ROOT / "static" / "icon.svg", media_type="image/svg+xml")

    app.add_api_route("/favicon.ico", favicon, methods=["GET"], include_in_schema=False)

"""Minimal synchronous client for the SentientAI daemon API (used by the CLI)."""

from __future__ import annotations

from typing import Any

import httpx

from sentientai.config import Settings


class ApiUnavailableError(RuntimeError):
    pass


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class ApiClient:
    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 30.0) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    @classmethod
    def from_settings(
        cls, settings: Settings, url: str | None = None, *, timeout: float = 30.0
    ) -> ApiClient:
        host = settings.api.host if settings.api.host not in {"0.0.0.0", "::"} else "127.0.0.1"  # noqa: S104
        token = settings.resolved_api_token()
        return cls(url or f"http://{host}:{settings.api.port}", token=token, timeout=timeout)

    def request(
        self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        try:
            response = self._http.request(
                method,
                path,
                timeout=timeout if timeout is not None else self._http.timeout,
                **kwargs,
            )
        except httpx.ConnectError as exc:
            raise ApiUnavailableError(
                f"cannot reach the SentientAI daemon at {self.base_url} "
                "(start it with `sentientai run`)"
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, detail)
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, json: Any = None, **kwargs: Any) -> Any:
        return self.request("POST", path, json=json, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)

    def close(self) -> None:
        self._http.close()

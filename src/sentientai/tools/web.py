"""Web search, page fetching and REST API calls.

Every outbound URL passes :class:`UrlGuard`: http(s) only, DNS-resolved
addresses must be public unless explicitly allowed (SSRF protection: blocks
loopback, RFC1918, link-local / cloud metadata, etc.), and domain allow /
deny lists apply. Redirects are followed manually so every hop is checked.
Credentials are injected by name from config and never shown to the model.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from typing import Any, Literal
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx
from pydantic import Field

from sentientai.config import RiskLevel, WebConfig
from sentientai.errors import ToolError
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult

_MAX_REDIRECTS = 5


class UrlGuard:
    def __init__(self, config: WebConfig) -> None:
        self.config = config

    @staticmethod
    def _domain_match(host: str, domains: list[str]) -> bool:
        host = host.lower().rstrip(".")
        return any(host == d.lower() or host.endswith("." + d.lower().lstrip(".")) for d in domains)

    async def check(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError(f"only http(s) URLs are allowed, got {parsed.scheme or 'none'!r}")
        host = parsed.hostname
        if not host:
            raise ToolError("URL has no host")
        if self.config.blocked_domains and self._domain_match(host, self.config.blocked_domains):
            raise ToolError(f"domain {host} is blocked by policy")
        if self.config.allowed_domains and not self._domain_match(
            host, self.config.allowed_domains
        ):
            raise ToolError(f"domain {host} is not in the allowed domain list")
        if not self.config.allow_private_networks:
            for address in await self._resolve(
                host, parsed.port or (443 if parsed.scheme == "https" else 80)
            ):
                ip = ipaddress.ip_address(address)
                if not ip.is_global or ip.is_multicast:
                    raise ToolError(
                        f"{host} resolves to non-public address {ip}; blocked (SSRF protection)"
                    )
        return url

    @staticmethod
    async def _resolve(host: str, port: int) -> list[str]:
        try:
            ipaddress.ip_address(host)
            return [host]
        except ValueError:
            pass
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ToolError(f"cannot resolve {host}: {exc}") from exc
        return sorted({str(info[4][0]) for info in infos})


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text). Uses BeautifulSoup when installed."""

    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover - optional dependency
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return (title_match.group(1).strip() if title_match else ""), re.sub(
            r"\s+", " ", text
        ).strip()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "svg", "iframe", "nav", "footer", "form", "aside"]
    ):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code", "td", "blockquote"]):
        if el.find_parent(["pre", "li", "td", "blockquote"]) is not None and el.name in {
            "p",
            "code",
        }:
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            lines.append("#" * int(el.name[1]) + " " + text)
        elif el.name == "li":
            lines.append(f"- {text}")
        elif el.name == "pre":
            lines.append(f"```\n{el.get_text()}\n```")
        else:
            lines.append(text)
    if not lines:
        lines = [root.get_text("\n", strip=True)]
    return title, "\n\n".join(lines)


class WebClient:
    """Shared, guarded HTTP client used by the web tools and the ingestion pipeline."""

    def __init__(
        self, config: WebConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.config = config
        self.guard = UrlGuard(config)
        self.http = httpx.AsyncClient(
            timeout=config.timeout_s,
            headers={"User-Agent": config.user_agent},
            follow_redirects=False,
            transport=transport,
        )

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
        json_body: Any = None,
    ) -> httpx.Response:
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            await self.guard.check(current)
            request = self.http.build_request(
                method, current, headers=headers, content=content, json=json_body
            )
            response = await self.http.send(request, stream=True)
            try:
                if response.is_redirect and method.upper() in {"GET", "HEAD"}:
                    location = response.headers.get("location")
                    if not location:
                        break
                    current = urljoin(current, location)
                    continue
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) >= self.config.max_fetch_bytes:
                        del body[self.config.max_fetch_bytes :]
                        break
                # aiter_bytes() already decoded the body, so drop encoding/length headers.
                kept_headers = [
                    (k, v)
                    for k, v in response.headers.multi_items()
                    if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
                ]
                return httpx.Response(
                    response.status_code, headers=kept_headers, content=bytes(body), request=request
                )
            finally:
                await response.aclose()
        raise ToolError(f"too many redirects fetching {url}")

    async def fetch_text(self, url: str) -> tuple[str, str, str]:
        """Fetch a URL and return (title, text, content_type)."""

        response = await self.request("GET", url)
        if response.status_code >= 400:
            raise ToolError(f"HTTP {response.status_code} fetching {url}")
        ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype in {"text/html", "application/xhtml+xml"} or response.content.lstrip()[
            :15
        ].lower().startswith(b"<!doctype html"):
            title, text = html_to_text(response.text)
            return title or url, text, "text/html"
        if ctype == "application/pdf":
            from sentientai.ingestion.loaders import pdf_bytes_to_text

            return url.rsplit("/", 1)[-1], pdf_bytes_to_text(response.content), ctype
        if ctype == "application/json":
            try:
                return (
                    url,
                    json.dumps(response.json(), indent=2)[: self.config.max_fetch_bytes],
                    ctype,
                )
            except json.JSONDecodeError:
                pass
        return url, response.text, ctype or "text/plain"

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        backend = self.config.search_backend
        if backend == "disabled":
            raise ToolError("web search is disabled in configuration")
        if backend == "searxng":
            if not self.config.searxng_url:
                raise ToolError("web.searxng_url is not configured")
            url = f"{self.config.searxng_url.rstrip('/')}/search?q={quote_plus(query)}&format=json"
            response = await self.request("GET", url)
            data = response.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
                for r in data.get("results", [])[:max_results]
            ]
        response = await self.request(
            "GET", f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        )
        return parse_duckduckgo(response.text, max_results)

    async def aclose(self) -> None:
        await self.http.aclose()


def parse_duckduckgo(html: str, max_results: int) -> list[dict[str, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results = []
    for block in soup.select("div.result, div.web-result"):
        link = block.select_one("a.result__a")
        if link is None:
            continue
        href = str(link.get("href", ""))
        if "uddg=" in href:  # DuckDuckGo redirect wrapper
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        snippet = block.select_one(".result__snippet")
        results.append(
            {
                "title": link.get_text(" ", strip=True),
                "url": href,
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


# ------------------------------------------------------------------- tools
class WebSearchArgs(ToolArgs):
    query: str = Field(min_length=2)
    max_results: int = Field(default=8, ge=1, le=20)


class WebSearch(Tool[WebSearchArgs]):
    name = "web_search"
    description = "Search the web. Returns titles, URLs and snippets."
    args_model = WebSearchArgs
    risk = RiskLevel.LOW
    timeout_s = 45.0
    tags = ("web", "read")

    def __init__(self, client: WebClient) -> None:
        self.client = client

    async def run(self, args: WebSearchArgs, ctx: ToolContext) -> ToolResult:
        results = await self.client.search(args.query, args.max_results)
        if not results:
            return ToolResult.success("no results")
        text = "\n\n".join(
            f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)
        )
        return ToolResult.success(text, data=results)


class WebFetchArgs(ToolArgs):
    url: str
    max_chars: int = Field(default=20_000, ge=500, le=200_000)


class WebFetch(Tool[WebFetchArgs]):
    name = "web_fetch"
    description = "Fetch a web page or document and return its readable text."
    args_model = WebFetchArgs
    risk = RiskLevel.LOW
    timeout_s = 60.0
    tags = ("web", "read")

    def __init__(self, client: WebClient) -> None:
        self.client = client

    async def run(self, args: WebFetchArgs, ctx: ToolContext) -> ToolResult:
        title, text, ctype = await self.client.fetch_text(args.url)
        truncated = len(text) > args.max_chars
        body = text[: args.max_chars] + ("\n[... truncated]" if truncated else "")
        return ToolResult(
            ok=True, output=f"# {title}\n({ctype}, {args.url})\n\n{body}", truncated=truncated
        )


class HttpRequestArgs(ToolArgs):
    method: Literal["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    json_body: Any = None
    body: str | None = None
    credential: str | None = Field(
        default=None, description="Name of a configured credential to attach."
    )


class HttpRequest(Tool[HttpRequestArgs]):
    name = "http_request"
    description = (
        "Call a REST API. GET/HEAD are low risk; mutating methods require operator approval. "
        "Use `credential` to attach a configured secret by name."
    )
    args_model = HttpRequestArgs
    risk = RiskLevel.LOW
    timeout_s = 60.0
    tags = ("web", "api")

    def __init__(self, client: WebClient) -> None:
        self.client = client

    def assess(self, args: HttpRequestArgs, ctx: ToolContext) -> tuple[RiskLevel, list[str]]:
        if args.method in {"GET", "HEAD"}:
            return self.risk, []
        return RiskLevel.HIGH, [f"mutating HTTP {args.method} to an external service"]

    async def run(self, args: HttpRequestArgs, ctx: ToolContext) -> ToolResult:
        headers = dict(args.headers)
        if args.credential:
            cred = self.client.config.credentials.get(args.credential)
            if cred is None:
                raise ToolError(f"unknown credential {args.credential!r}")
            headers.update({k: v.get_secret_value() for k, v in cred.items()})
        response = await self.client.request(
            args.method,
            args.url,
            headers=headers,
            content=args.body.encode() if args.body is not None else None,
            json_body=args.json_body,
        )
        shown = {
            k: v
            for k, v in response.headers.items()
            if k.lower() in {"content-type", "content-length", "location", "retry-after", "etag"}
        }
        text = response.text[:20_000]
        output = f"HTTP {response.status_code}\nheaders: {json.dumps(shown)}\n\n{text}"
        ok = response.status_code < 400
        return ToolResult(
            ok=ok,
            output=output,
            error=None if ok else f"HTTP {response.status_code}",
            data={"status": response.status_code},
        )


def web_tools(client: WebClient) -> list[Tool[Any]]:
    tools: list[Tool[Any]] = [WebFetch(client), HttpRequest(client)]
    if client.config.search_backend != "disabled":
        tools.insert(0, WebSearch(client))
    return tools

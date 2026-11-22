"""Async wrapper around the official MCP client, talking to the VisionPilot server over Streamable HTTP.

One background task owns the MCP session for the app's lifetime (anyio requires a session to be opened and
closed by the same task). Requests share that session, and the task reconnects if the server restarts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Protocol

from mcp import Client

from visionpilot_api.schemas import ToolCallResult, ToolSpec

logger = logging.getLogger(__name__)

HIDDEN_TOOLS = {"server_status"}  # useful for humans, noise for the planner


class ToolClient(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult: ...


class MCPUnavailableError(RuntimeError):
    pass


class MCPToolClient:
    def __init__(self, url: str, call_timeout_s: float, connect_timeout_s: float = 10.0) -> None:
        self.url = url
        self.call_timeout_s = call_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._client: Client | None = None
        self._tools: list[ToolSpec] | None = None
        self._ready = asyncio.Event()
        self._reconnect = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.last_error: str | None = None

    # ---- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._keeper(), name="mcp-session")

    async def close(self) -> None:
        self._stop.set()
        self._reconnect.set()
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=5)
            self._task = None

    async def _keeper(self) -> None:
        while not self._stop.is_set():
            try:
                async with Client(self.url) as client:
                    self._client, self._tools, self.last_error = client, None, None
                    self._ready.set()
                    logger.info("Connected to MCP server at %s", self.url)
                    await self._reconnect.wait()
            except Exception as exc:
                self.last_error = repr(exc)
                logger.warning("MCP server at %s not reachable: %r (retrying)", self.url, exc)
            finally:
                self._client = None
                self._ready.clear()
                self._reconnect.clear()
            if not self._stop.is_set():
                await asyncio.sleep(2)

    async def _session(self) -> Client:
        self.start()
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self.connect_timeout_s)
        except TimeoutError as exc:
            raise MCPUnavailableError(
                f"MCP server not reachable at {self.url}. Start it with: "
                "uv run visionpilot-mcp --transport streamable-http"
            ) from exc
        assert self._client is not None
        return self._client

    @property
    def connected(self) -> bool:
        return self._ready.is_set()

    # ---- API ------------------------------------------------------------------------------------
    async def list_tools(self) -> list[ToolSpec]:
        if self._tools is None:
            listed = await (await self._session()).list_tools()
            self._tools = [
                ToolSpec(name=t.name, description=t.description or "", input_schema=dict(t.input_schema))
                for t in listed.tools
                if t.name not in HIDDEN_TOOLS
            ]
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        for attempt in (1, 2):
            try:
                client = await self._session()
                result = await asyncio.wait_for(client.call_tool(name, arguments), timeout=self.call_timeout_s)
            except MCPUnavailableError as exc:
                return ToolCallResult(ok=False, error=str(exc))
            except TimeoutError:
                return ToolCallResult(ok=False, error=f"{name} timed out after {self.call_timeout_s:.0f}s")
            except Exception as exc:  # session dropped: reconnect once and retry
                logger.warning("MCP call %s failed (attempt %d): %r", name, attempt, exc)
                self._reconnect.set()
                if attempt == 2:
                    return ToolCallResult(ok=False, error=f"MCP call failed: {exc}")
                continue
            if result.is_error:
                text = " ".join(getattr(c, "text", "") for c in result.content) or "tool error"
                return ToolCallResult(ok=False, error=text)
            return ToolCallResult(ok=True, data=result.structured_content or {})
        raise AssertionError("unreachable")  # pragma: no cover

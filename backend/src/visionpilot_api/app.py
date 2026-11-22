"""FastAPI app: uploads, samples, media serving, and the streaming chat endpoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette import EventSourceResponse

from visionpilot_api import __version__
from visionpilot_api.agent import Agent
from visionpilot_api.config import Settings, get_settings
from visionpilot_api.llm import GeminiPlanner, LLMError, Planner
from visionpilot_api.mcp_client import MCPToolClient, ToolClient
from visionpilot_api.media import MediaError, MediaStore
from visionpilot_api.schemas import ChatRequest, MediaInfo, SampleInfo, TraceEvent

logger = logging.getLogger("visionpilot_api")


def create_app(
    settings: Settings | None = None,
    tool_client: ToolClient | None = None,
    planner: Planner | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    media = MediaStore(settings.upload_dir, settings.samples_dir, settings.max_upload_mb)
    own_client = tool_client is None
    tools: ToolClient = tool_client or MCPToolClient(settings.mcp_url, settings.mcp_call_timeout_s)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if isinstance(tools, MCPToolClient):
            tools.start()  # connect in the background; the API is usable before the models finish loading
        yield
        if own_client and isinstance(tools, MCPToolClient):
            await tools.close()

    app = FastAPI(title="VisionPilot API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"], allow_headers=["*"]
    )
    state: dict[str, Any] = {"planner": planner}

    def get_planner() -> Planner:
        if state["planner"] is None:
            state["planner"] = GeminiPlanner(
                settings.gemini_api_key.get_secret_value(), settings.gemini_model, settings.gemini_fallback_model,
                max_retries=settings.gemini_max_retries, timeout_s=settings.gemini_timeout_s,
            )  # fmt: skip
        planner_: Planner = state["planner"]
        return planner_

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        connected = getattr(tools, "connected", True)
        tool_names: list[str] = []
        if connected:
            try:
                tool_names = [t.name for t in await tools.list_tools()]
            except Exception:  # noqa: BLE001 - health must never raise
                connected = False
        return {
            "status": "ok",
            "version": __version__,
            "mcp_connected": connected,
            "mcp_url": settings.mcp_url,
            "tools": tool_names,
            "gemini_key_set": bool(settings.gemini_api_key.get_secret_value()),
            "model": settings.gemini_model,
        }

    @app.post("/api/uploads", response_model=MediaInfo)
    async def upload(file: UploadFile = File(...)) -> MediaInfo:  # noqa: B008 (FastAPI idiom)
        try:
            return media.save_upload(file.filename or "upload", file.file)
        except MediaError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/samples", response_model=list[SampleInfo])
    async def samples() -> list[SampleInfo]:
        return media.list_samples()

    @app.get("/api/media/{media_id:path}")
    async def get_media(media_id: str) -> FileResponse:
        try:
            path, _ = media.resolve(media_id)
        except MediaError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
        try:
            info = media.info(body.media_id)
            path, kind = media.resolve(body.media_id)
        except MediaError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def stream() -> AsyncIterator[dict[str, str]]:
            try:
                agent = Agent(get_planner(), tools, settings.agent_max_steps)
            except LLMError as exc:
                for event in (
                    TraceEvent(type="error", run_id="-", message=str(exc)),
                    TraceEvent(type="done", run_id="-"),
                ):
                    yield {"event": event.type, "data": event.model_dump_json(exclude_none=True)}
                return
            async for event in agent.run(
                body.question, path, kind, width=info.width, height=info.height, history=body.history
            ):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping run")
                    return
                yield {"event": event.type, "data": event.model_dump_json(exclude_none=True)}

        return EventSourceResponse(stream(), ping=15)

    return app

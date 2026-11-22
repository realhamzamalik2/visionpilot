"""VisionPilot MCP server: wires model backends, caching and logging into MCP tools.

Run it:
    uv run visionpilot-mcp                                  # stdio (Claude Desktop)
    uv run visionpilot-mcp --transport streamable-http      # HTTP at http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from visionpilot_mcp import __version__
from visionpilot_mcp.cache import ResultCache
from visionpilot_mcp.config import Settings, get_settings
from visionpilot_mcp.errors import VisionToolError
from visionpilot_mcp.imaging import LoadedImage, load_image
from visionpilot_mcp.logging_setup import configure_logging, record_tool_metric
from visionpilot_mcp.registry import ModelRegistry
from visionpilot_mcp.schemas import (
    DescribeImageResult,
    DetectObjectsResult,
    ReadTextResult,
    ServerStatus,
)
from visionpilot_mcp.tools import detection, florence, korean_ocr

logger = logging.getLogger("visionpilot_mcp")

ResultT = TypeVar("ResultT", bound=BaseModel)

INSTRUCTIONS = """\
VisionPilot gives you computer-vision tools that run locally on CPU.
Every tool takes `image`: a local file path, an http(s) URL, or a data:image/...;base64 URI.
- detect_objects: find and count the 80 COCO object classes (people, cars, bottles, chairs...).
- describe_image: natural-language description of the whole scene.
- read_korean_text: read Korean (Hangul) text on signs and menus. It does not translate; translate it yourself.
Results include pixel bounding boxes so answers can point at regions. Repeated calls on the same image are cached.
"""

ImageArg = Annotated[
    str,
    Field(description="Local file path (e.g. C:\\photos\\street.jpg), http(s) URL, or data:image/...;base64 URI."),
]

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


class VisionPilotApp:
    """Holds settings, the model registry and the result cache, and builds the MCP server."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(max_loaded=self.settings.max_loaded_models)
        self.cache = cache if cache is not None else ResultCache(self.settings.result_cache_size)
        if registry is None:
            self._register_default_models()
        self.mcp = self._build_server()

    # ---- model registration -------------------------------------------------------------------
    def _register_default_models(self) -> None:
        s = self.settings
        self.registry.register("yolo26n", lambda: detection.load_yolo("yolo26n", s.model_dir, s.yolo_imgsz))
        self.registry.register("florence2", lambda: florence.load_florence(s.florence_model_id, s.cpu_threads))
        self.registry.register("korean_ocr", lambda: korean_ocr.load_korean_ocr(s.cpu_threads))

    # ---- shared tool plumbing -----------------------------------------------------------------
    def run_tool(
        self,
        tool: str,
        image_source: str,
        params: dict[str, Any],
        result_type: type[ResultT],
        compute: Callable[[LoadedImage], ResultT],
    ) -> ResultT:
        """Load the image, check the cache, run the model, time it, log it, cache it."""
        start = time.perf_counter()
        try:
            loaded = load_image(image_source, allow_urls=self.settings.allow_remote_urls)
            key = ResultCache.make_key(tool, loaded.info.sha1, params)
            cached = self.cache.get(key, result_type)
            if cached is not None:
                cached = cached.model_copy(update={"latency_ms": round((time.perf_counter() - start) * 1000, 1)})
                self._log(tool, start, cached=True, ok=True, image=loaded.info.sha1)
                return cached
            result = compute(loaded)
            result = result.model_copy(update={"latency_ms": round((time.perf_counter() - start) * 1000, 1)})
            self.cache.put(key, result)
            self._log(tool, start, cached=False, ok=True, image=loaded.info.sha1)
            return result
        except VisionToolError as exc:
            self._log(tool, start, cached=False, ok=False, error=str(exc))
            raise
        except Exception as exc:  # unexpected: log the traceback, return a short message to the client
            logger.exception("Tool %s crashed", tool)
            self._log(tool, start, cached=False, ok=False, error=repr(exc))
            raise VisionToolError(f"{tool} failed: {exc}") from exc

    def _log(self, tool: str, start: float, **fields: Any) -> None:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info("tool=%s latency_ms=%.1f %s", tool, latency_ms, " ".join(f"{k}={v}" for k, v in fields.items()))
        record_tool_metric(tool=tool, latency_ms=latency_ms, **fields)

    # ---- MCP server ---------------------------------------------------------------------------
    def _build_server(self) -> MCPServer:
        mcp = MCPServer("VisionPilot", instructions=INSTRUCTIONS, version=__version__)
        app = self
        s = self.settings

        @mcp.tool(title="Detect and count objects", annotations=READ_ONLY)
        def detect_objects(
            image: ImageArg,
            classes: Annotated[
                list[str] | None,
                Field(description="Optional COCO labels to keep, e.g. ['person', 'car']. Omit to detect all 80."),
            ] = None,
            min_confidence: Annotated[float, Field(ge=0.05, le=0.95, description="Score threshold.")] = 0.25,
            max_detections: Annotated[int, Field(ge=1, le=300)] = 100,
        ) -> DetectObjectsResult:
            """Detect, locate and count everyday objects with YOLO26n (80 COCO classes such as person, car,
            bus, bicycle, bottle, cup, chair, laptop, cell phone). Use it for "how many people", "where is the car",
            queue counting or shelf items. Returns pixel boxes, confidences and per-label counts."""
            params = {
                "classes": sorted(c.lower() for c in classes) if classes else None,
                "min_confidence": min_confidence,
                "max_detections": max_detections,
            }

            def compute(loaded: LoadedImage) -> DetectObjectsResult:
                model = app.registry.get("yolo26n")
                dets = detection.run_detection(
                    model,
                    loaded.image,
                    min_confidence=min_confidence,
                    classes=classes,
                    max_detections=max_detections,
                    imgsz=s.yolo_imgsz,
                )
                return DetectObjectsResult(
                    image=loaded.info,
                    model="yolo26n (ONNX Runtime, CPU)",
                    detections=dets,
                    counts=detection.count_labels(dets),
                    total=len(dets),
                )

            return app.run_tool("detect_objects", image, params, DetectObjectsResult, compute)

        @mcp.tool(title="Describe an image", annotations=READ_ONLY)
        def describe_image(
            image: ImageArg,
            detail: Annotated[
                Literal["brief", "detailed", "very_detailed"],
                Field(description="brief = one line, detailed = a few sentences, very_detailed = a paragraph."),
            ] = "detailed",
        ) -> DescribeImageResult:
            """Describe the whole scene in English with Florence-2: setting, people, objects, actions and mood.
            Use it for open questions like "what is happening here" or to get context before calling other tools.
            Do not use it for counting (use detect_objects) or reading Korean text (use read_korean_text)."""

            def compute(loaded: LoadedImage) -> DescribeImageResult:
                bundle = app.registry.get("florence2")
                caption = florence.describe(
                    bundle,
                    loaded.image,
                    detail,
                    num_beams=s.florence_num_beams,
                    max_new_tokens=s.florence_max_new_tokens,
                )
                return DescribeImageResult(
                    image=loaded.info,
                    model=s.florence_model_id,
                    detail=detail,
                    caption=caption,
                )

            return app.run_tool("describe_image", image, {"detail": detail}, DescribeImageResult, compute)

        @mcp.tool(title="Read Korean text", annotations=READ_ONLY)
        def read_korean_text(
            image: ImageArg,
            min_confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Drop lines below this score.")] = 0.5,
        ) -> ReadTextResult:
            """Read Korean (Hangul) text, plus numbers and English letters, from signs, menus, posters, receipts and
            screenshots using PaddleOCR PP-OCRv5 Korean. Returns each text line with its position, in reading order.
            This tool does not translate; translate the returned text yourself if the user asks."""

            def compute(loaded: LoadedImage) -> ReadTextResult:
                engine = app.registry.get("korean_ocr")
                lines = korean_ocr.run_ocr(engine, loaded.image, min_confidence=min_confidence)
                return ReadTextResult(
                    image=loaded.info,
                    model=korean_ocr.OCR_MODEL_NAME,
                    language="ko",
                    lines=lines,
                    full_text="\n".join(line.text for line in lines),
                )

            return app.run_tool("read_korean_text", image, {"min_confidence": min_confidence}, ReadTextResult, compute)

        @mcp.tool(title="Server status", annotations=READ_ONLY)
        def server_status() -> ServerStatus:
            """Show which vision models are loaded in memory, cache statistics and the server version."""
            return ServerStatus(
                version=__version__,
                model_dir=str(s.model_dir),
                max_loaded_models=app.registry.max_loaded,
                loaded_models=app.registry.status(),
                available_models=app.registry.available,
                cache_entries=len(app.cache),
                cache_hits=app.cache.hits,
                cache_misses=app.cache.misses,
            )

        return mcp


def build_arg_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionpilot-mcp", description="VisionPilot vision MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default=settings.mcp_transport)
    parser.add_argument("--host", default=settings.mcp_http_host)
    parser.add_argument("--port", type=int, default=settings.mcp_http_port)
    parser.add_argument("--preload", default="", help="Comma-separated models to load at startup, e.g. yolo26n")
    return parser


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    args = build_arg_parser(settings).parse_args(argv)
    configure_logging(settings.log_level, settings.log_dir)

    app = VisionPilotApp(settings)
    for name in filter(None, (n.strip() for n in args.preload.split(","))):
        app.registry.get(name)

    logger.info(
        "VisionPilot MCP %s starting (transport=%s, models=%s)", __version__, args.transport, settings.model_dir
    )
    if args.transport == "stdio":
        app.mcp.run("stdio")
    else:
        app.mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

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

import numpy as np
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
    AnalyzeFacesResult,
    BBox,
    DescribeImageResult,
    Detection,
    DetectObjectsResult,
    EstimateDepthResult,
    EstimatePoseResult,
    FindPhraseResult,
    PhraseMatch,
    ProcessVideoResult,
    ReadTextResult,
    SegmentObjectsResult,
    ServerStatus,
    TextLine,
)
from visionpilot_mcp.tools import detection, faces, florence, korean_ocr, video, yolo_tasks

logger = logging.getLogger("visionpilot_mcp")

ResultT = TypeVar("ResultT", bound=BaseModel)

INSTRUCTIONS = """\
VisionPilot gives you computer-vision tools that run locally on CPU.
Image tools take `image`: a local file path, an http(s) URL, or a data:image/...;base64 URI.
- detect_objects / segment_objects: find, count and outline the 80 COCO object classes.
- find_phrase: locate anything described in words ("the red umbrella", "empty shelf").
- estimate_pose: body keypoints and rough posture per person.
- estimate_depth: metric depth in meters, optionally inside boxes from other tools.
- analyze_faces: faces and their apparent facial expressions.
- describe_image: natural-language description of the scene.
- read_text (English/Latin script) and read_korean_text (Hangul). Neither translates; translate yourself.
- process_video: sample a video over time and return a timeline of counts (queues, attendance, engagement).
Boxes are [x1, y1, x2, y2] in original image pixels. Repeated calls on the same input are cached.
"""

ImageArg = Annotated[
    str,
    Field(description="Local file path (e.g. C:\\photos\\street.jpg), http(s) URL, or data:image/...;base64 URI."),
]
ClassesArg = Annotated[
    list[str] | None,
    Field(description="Optional COCO labels to keep, e.g. ['person', 'car']. Omit for all 80 classes."),
]
ConfidenceArg = Annotated[float, Field(ge=0.05, le=0.95, description="Score threshold.")]

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

# Rough resident memory per model on CPU, used by the registry's RAM budget.
MODEL_COSTS_MB = {
    "yolo26n": 60,
    "yolo26n-seg": 80,
    "yolo26n-pose": 70,
    "yolo26n-depth": 120,
    "florence2": 1100,
    "korean_ocr": 150,
    "faces": 120,
}


class VisionPilotApp:
    """Holds settings, the model registry and the result cache, and builds the MCP server."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(budget_mb=self.settings.model_memory_budget_mb)
        self.cache = cache if cache is not None else ResultCache(self.settings.result_cache_size)
        if registry is None:
            self._register_default_models()
        self.mcp = self._build_server()

    # ---- model registration -------------------------------------------------------------------
    def _register_default_models(self) -> None:
        s = self.settings

        def yolo_loader(name: str, imgsz: int) -> Callable[[], Any]:
            return lambda: detection.load_yolo(name, s.model_dir, imgsz)

        for name in detection.YOLO_TASKS:
            imgsz = s.depth_imgsz if name.endswith("depth") else s.yolo_imgsz
            self.registry.register(name, yolo_loader(name, imgsz), MODEL_COSTS_MB[name])
        self.registry.register(
            "florence2", lambda: florence.load_florence(s.florence_model_id, s.cpu_threads), MODEL_COSTS_MB["florence2"]
        )
        self.registry.register(
            "korean_ocr", lambda: korean_ocr.load_korean_ocr(s.cpu_threads), MODEL_COSTS_MB["korean_ocr"]
        )
        self.registry.register(
            "faces", lambda: faces.load_faces(s.model_dir, s.face_min_confidence), MODEL_COSTS_MB["faces"]
        )

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
            return self._cached_compute(tool, loaded.info.sha1, params, result_type, lambda: compute(loaded), start)
        except VisionToolError as exc:
            self._log(tool, start, cached=False, ok=False, error=str(exc))
            raise
        except Exception as exc:  # unexpected: log the traceback, return a short message to the client
            logger.exception("Tool %s crashed", tool)
            self._log(tool, start, cached=False, ok=False, error=repr(exc))
            raise VisionToolError(f"{tool} failed: {exc}") from exc

    def _cached_compute(
        self,
        tool: str,
        digest: str,
        params: dict[str, Any],
        result_type: type[ResultT],
        compute: Callable[[], ResultT],
        start: float,
    ) -> ResultT:
        key = ResultCache.make_key(tool, digest, params)
        cached = self.cache.get(key, result_type)
        if cached is not None:
            cached = cached.model_copy(update={"latency_ms": self._elapsed_ms(start)})
            self._log(tool, start, cached=True, ok=True, input=digest)
            return cached
        result = compute().model_copy(update={"latency_ms": self._elapsed_ms(start)})
        self.cache.put(key, result)
        self._log(tool, start, cached=False, ok=True, input=digest)
        return result

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    def _log(self, tool: str, start: float, **fields: Any) -> None:
        latency_ms = self._elapsed_ms(start)
        logger.info("tool=%s latency_ms=%.1f %s", tool, latency_ms, " ".join(f"{k}={v}" for k, v in fields.items()))
        record_tool_metric(tool=tool, latency_ms=latency_ms, **fields)

    def florence_kwargs(self) -> dict[str, int]:
        return {"num_beams": self.settings.florence_num_beams, "max_new_tokens": self.settings.florence_max_new_tokens}

    # ---- MCP server ---------------------------------------------------------------------------
    def _build_server(self) -> MCPServer:  # noqa: C901 (one place that lists every tool)
        mcp = MCPServer("VisionPilot", instructions=INSTRUCTIONS, version=__version__)
        app = self
        s = self.settings

        def norm_classes(classes: list[str] | None) -> list[str] | None:
            return sorted(c.strip().lower() for c in classes) if classes else None

        @mcp.tool(title="Detect and count objects", annotations=READ_ONLY)
        def detect_objects(
            image: ImageArg,
            classes: ClassesArg = None,
            min_confidence: ConfidenceArg = 0.25,
            max_detections: Annotated[int, Field(ge=1, le=300)] = 100,
        ) -> DetectObjectsResult:
            """Detect, locate and count everyday objects with YOLO26n (80 COCO classes such as person, car,
            bus, bicycle, bottle, cup, chair, laptop, cell phone). Use it for "how many people", "where is the car",
            queue counting or shelf items. Returns pixel boxes, confidences and per-label counts."""
            params = {"classes": norm_classes(classes), "min_confidence": min_confidence, "max_det": max_detections}

            def compute(loaded: LoadedImage) -> DetectObjectsResult:
                dets = detection.run_detection(
                    app.registry.get("yolo26n"), loaded.image, min_confidence=min_confidence, classes=classes,
                    max_detections=max_detections, imgsz=s.yolo_imgsz,
                )  # fmt: skip
                return DetectObjectsResult(
                    image=loaded.info, model="yolo26n (ONNX Runtime, CPU)", detections=dets,
                    counts=detection.count_labels(dets), total=len(dets),
                )  # fmt: skip

            return app.run_tool("detect_objects", image, params, DetectObjectsResult, compute)

        @mcp.tool(title="Segment objects", annotations=READ_ONLY)
        def segment_objects(
            image: ImageArg,
            classes: ClassesArg = None,
            min_confidence: ConfidenceArg = 0.25,
            max_detections: Annotated[int, Field(ge=1, le=100)] = 50,
        ) -> SegmentObjectsResult:
            """Outline each object with a pixel-accurate mask (YOLO26n-seg, 80 COCO classes). Use it when shape or
            area matters: how much of a shelf is filled with bottles, which part of the road is covered by cars.
            Returns simplified polygons and the fraction of the image each label covers."""
            params = {"classes": norm_classes(classes), "min_confidence": min_confidence, "max_det": max_detections}

            def compute(loaded: LoadedImage) -> SegmentObjectsResult:
                objects, coverage = yolo_tasks.run_segmentation(
                    app.registry.get("yolo26n-seg"), loaded.image, min_confidence=min_confidence, classes=classes,
                    max_detections=max_detections, imgsz=s.yolo_imgsz,
                )  # fmt: skip
                return SegmentObjectsResult(
                    image=loaded.info, model="yolo26n-seg (ONNX Runtime, CPU)", objects=objects,
                    counts=yolo_tasks.count_objects(objects), coverage_by_label=coverage,
                )  # fmt: skip

            return app.run_tool("segment_objects", image, params, SegmentObjectsResult, compute)

        @mcp.tool(title="Estimate body pose", annotations=READ_ONLY)
        def estimate_pose(image: ImageArg, min_confidence: ConfidenceArg = 0.3) -> EstimatePoseResult:
            """Find people and their 17 body keypoints (YOLO26n-pose), plus a rough posture label per person
            (standing, sitting/crouching, arms_raised). Use it for "is anyone raising a hand", "how many are seated"."""

            def compute(loaded: LoadedImage) -> EstimatePoseResult:
                people = yolo_tasks.run_pose(
                    app.registry.get("yolo26n-pose"), loaded.image, min_confidence=min_confidence, imgsz=s.yolo_imgsz
                )
                return EstimatePoseResult(
                    image=loaded.info, model="yolo26n-pose (ONNX Runtime, CPU)", people=people, total=len(people),
                    skeleton=yolo_tasks.COCO_SKELETON,
                )  # fmt: skip

            return app.run_tool("estimate_pose", image, {"min_confidence": min_confidence}, EstimatePoseResult, compute)

        @mcp.tool(title="Estimate depth", annotations=READ_ONLY)
        def estimate_depth(
            image: ImageArg,
            boxes: Annotated[
                list[list[float]] | None,
                Field(description="Optional boxes [[x1, y1, x2, y2], ...] (e.g. from detect_objects) to measure."),
            ] = None,
        ) -> EstimateDepthResult:
            """Estimate metric depth in meters for every pixel (YOLO26n-depth) and summarize it. Pass boxes from
            another tool to get the distance of specific objects ("how far is the person on the left").
            Monocular depth is approximate: treat values as rough estimates."""
            params = {"boxes": [[round(v, 1) for v in b[:4]] for b in boxes] if boxes else None}

            def compute(loaded: LoadedImage) -> EstimateDepthResult:
                depth = yolo_tasks.run_depth(app.registry.get("yolo26n-depth"), loaded.image, imgsz=s.depth_imgsz)
                return EstimateDepthResult(
                    image=loaded.info, model="yolo26n-depth (ONNX Runtime, CPU)",
                    min_m=round(float(depth.min()), 2), median_m=round(float(np.median(depth)), 2),
                    max_m=round(float(depth.max()), 2),
                    regions=yolo_tasks.summarize_regions(depth, boxes or []),
                    depth_png=yolo_tasks.depth_to_data_uri(depth),
                )  # fmt: skip

            return app.run_tool("estimate_depth", image, params, EstimateDepthResult, compute)

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
                caption = florence.describe(
                    app.registry.get("florence2"), loaded.image, detail, **app.florence_kwargs()
                )
                return DescribeImageResult(image=loaded.info, model=s.florence_model_id, detail=detail, caption=caption)

            return app.run_tool("describe_image", image, {"detail": detail}, DescribeImageResult, compute)

        @mcp.tool(title="Find a described object", annotations=READ_ONLY)
        def find_phrase(
            image: ImageArg,
            phrase: Annotated[str, Field(min_length=1, max_length=200, description="e.g. 'empty shelf space'")],
        ) -> FindPhraseResult:
            """Locate things described in free text with Florence-2 phrase grounding, including objects outside the
            80 COCO classes ("the menu board", "empty shelf space", "a person in a red jacket"). Returns boxes."""

            def compute(loaded: LoadedImage) -> FindPhraseResult:
                found = florence.find_phrase(
                    app.registry.get("florence2"), loaded.image, phrase, **app.florence_kwargs()
                )
                matches = [PhraseMatch(label=label, box=BBox.from_xyxy(*box[:4])) for label, box in found]
                return FindPhraseResult(
                    image=loaded.info, model=s.florence_model_id, phrase=phrase, matches=matches, total=len(matches)
                )

            return app.run_tool("find_phrase", image, {"phrase": phrase.strip().lower()}, FindPhraseResult, compute)

        @mcp.tool(title="Read English text", annotations=READ_ONLY)
        def read_text(image: ImageArg) -> ReadTextResult:
            """Read English and other Latin-script text (signs, labels, slides, documents) with Florence-2 OCR.
            For Korean (Hangul) use read_korean_text instead."""

            def compute(loaded: LoadedImage) -> ReadTextResult:
                regions = florence.read_text_regions(
                    app.registry.get("florence2"), loaded.image, **app.florence_kwargs()
                )
                lines = [_quad_to_line(text, quad) for text, quad in regions if text]
                return ReadTextResult(
                    image=loaded.info, model=s.florence_model_id, language="en", lines=lines,
                    full_text="\n".join(line.text for line in lines),
                )  # fmt: skip

            return app.run_tool("read_text", image, {}, ReadTextResult, compute)

        @mcp.tool(title="Read Korean text", annotations=READ_ONLY)
        def read_korean_text(
            image: ImageArg,
            min_confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Drop lines below this score.")] = 0.5,
        ) -> ReadTextResult:
            """Read Korean (Hangul) text, plus numbers and English letters, from signs, menus, posters, receipts and
            screenshots using PaddleOCR PP-OCRv5 Korean. Returns each text line with its position, in reading order.
            This tool does not translate; translate the returned text yourself if the user asks."""

            def compute(loaded: LoadedImage) -> ReadTextResult:
                lines = korean_ocr.run_ocr(app.registry.get("korean_ocr"), loaded.image, min_confidence=min_confidence)
                return ReadTextResult(
                    image=loaded.info, model=korean_ocr.OCR_MODEL_NAME, language="ko", lines=lines,
                    full_text="\n".join(line.text for line in lines),
                )  # fmt: skip

            return app.run_tool("read_korean_text", image, {"min_confidence": min_confidence}, ReadTextResult, compute)

        @mcp.tool(title="Analyze faces and expressions", annotations=READ_ONLY)
        def analyze_faces(
            image: ImageArg,
            max_faces: Annotated[int, Field(ge=1, le=100)] = 40,
        ) -> AnalyzeFacesResult:
            """Detect faces (MediaPipe BlazeFace, tiled for small faces) and classify each face's apparent
            expression (HSEmotion, 8 AffectNet classes: anger, contempt, disgust, fear, happiness, neutral, sadness,
            surprise). Use it for audience or classroom engagement. Expressions are not reliable signals of what
            someone feels, so describe them as apparent expressions."""

            def compute(loaded: LoadedImage) -> AnalyzeFacesResult:
                found = faces.run_faces(app.registry.get("faces"), loaded.image, max_faces=max_faces)
                return AnalyzeFacesResult(
                    image=loaded.info, model=faces.FACE_MODEL_NAME, faces=found, total=len(found),
                    expression_counts=faces.count_expressions(found),
                )  # fmt: skip

            return app.run_tool("analyze_faces", image, {"max_faces": max_faces}, AnalyzeFacesResult, compute)

        @mcp.tool(title="Analyze a video over time", annotations=READ_ONLY)
        def process_video(
            video_path: Annotated[str, Field(description="Local video file path (.mp4, .mov, .avi, .mkv, .webm).")],
            task: Annotated[
                Literal["detect_objects", "analyze_faces"],
                Field(description="detect_objects = counts per frame (queues, crowds); analyze_faces = expressions."),
            ] = "detect_objects",
            classes: ClassesArg = None,
            sample_every_s: Annotated[float, Field(ge=0.2, le=30.0, description="Seconds between frames.")] = 1.0,
            max_frames: Annotated[int, Field(ge=2, le=600)] = 60,
        ) -> ProcessVideoResult:
            """Sample frames from a video and run a per-frame tool, returning a timeline of counts with timestamps
            and per-label stats (max, mean, peak time, and how long the final count has lasted). Use it for
            "how long has the queue been this long", "when was the room most crowded", "engagement over time"."""
            start = time.perf_counter()
            tool = "process_video"
            try:
                info = video.open_video(video_path)
                frames_cap = min(max_frames, s.video_max_frames)
                indices = video.plan_sample_indices(info, sample_every_s, frames_cap)
                effective_step = (indices[1] - indices[0]) / info.fps if len(indices) > 1 else sample_every_s
                params = {"task": task, "classes": norm_classes(classes), "step": round(effective_step, 3),
                          "frames": len(indices)}  # fmt: skip

                def frame_fn(frame: Any) -> list[Detection]:
                    if task == "detect_objects":
                        return detection.run_detection(
                            app.registry.get("yolo26n"), frame, min_confidence=0.25, classes=classes,
                            max_detections=100, imgsz=s.yolo_imgsz,
                        )  # fmt: skip
                    found = faces.run_faces(app.registry.get("faces"), frame, max_faces=40)
                    return [Detection(label=f.expression, confidence=f.confidence, box=f.box) for f in found]

                def compute() -> ProcessVideoResult:
                    timeline = video.build_timeline(info, indices, frame_fn, keep_boxes=50)
                    model = "yolo26n (ONNX)" if task == "detect_objects" else faces.FACE_MODEL_NAME
                    return ProcessVideoResult(
                        video=info.path.name, model=model, task=task, duration_s=round(info.duration_s, 2),
                        native_fps=round(info.fps, 2), width=info.width, height=info.height,
                        frames_sampled=len(timeline), sample_every_s=round(effective_step, 3), timeline=timeline,
                        stats=video.timeline_stats(timeline, effective_step),
                    )  # fmt: skip

                return app._cached_compute(tool, info.digest, params, ProcessVideoResult, compute, start)
            except VisionToolError as exc:
                app._log(tool, start, cached=False, ok=False, error=str(exc))
                raise
            except Exception as exc:
                logger.exception("Tool %s crashed", tool)
                app._log(tool, start, cached=False, ok=False, error=repr(exc))
                raise VisionToolError(f"{tool} failed: {exc}") from exc

        @mcp.tool(title="Server status", annotations=READ_ONLY)
        def server_status() -> ServerStatus:
            """Show which vision models are loaded in memory, cache statistics and the server version."""
            return ServerStatus(
                version=__version__,
                model_dir=str(s.model_dir),
                memory_budget_mb=app.registry.budget_mb,
                memory_used_mb=app.registry.used_mb,
                loaded_models=app.registry.status(),
                available_models=app.registry.available,
                cache_entries=len(app.cache),
                cache_hits=app.cache.hits,
                cache_misses=app.cache.misses,
            )

        return mcp


def _quad_to_line(text: str, quad: list[float]) -> TextLine:
    points = [(round(quad[i], 1), round(quad[i + 1], 1)) for i in range(0, min(len(quad), 8), 2)]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return TextLine(text=text, confidence=1.0, box=BBox.from_xyxy(min(xs), min(ys), max(xs), max(ys)), polygon=points)


def build_arg_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionpilot-mcp", description="VisionPilot vision MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default=settings.mcp_transport)
    parser.add_argument("--host", default=settings.mcp_http_host)
    parser.add_argument("--port", type=int, default=settings.mcp_http_port)
    parser.add_argument(
        "--warmup", default=settings.warmup_models, help="Comma-separated models to pre-load in the background"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    args = build_arg_parser(settings).parse_args(argv)
    configure_logging(settings.log_level, settings.log_dir)

    app = VisionPilotApp(settings)
    warmup = [n.strip() for n in args.warmup.split(",") if n.strip()]
    if warmup:
        app.registry.warm_up(warmup)  # background thread; tool calls wait for it via the registry lock

    logger.info("VisionPilot MCP %s starting (transport=%s, warmup=%s)", __version__, args.transport, warmup)
    if args.transport == "stdio":
        app.mcp.run("stdio")
    else:
        app.mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

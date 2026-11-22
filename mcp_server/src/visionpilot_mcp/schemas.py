"""Pydantic schemas for tool outputs. MCP clients receive these as structured JSON."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned box in original image pixels (top-left origin)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        return cls(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1))


class ImageInfo(BaseModel):
    width: int
    height: int
    sha1: str = Field(description="Short content hash, used as the cache key.")
    source_type: Literal["file", "url", "base64"]


class ToolResultBase(BaseModel):
    image: ImageInfo
    model: str = Field(description="Model that produced this result.")
    latency_ms: float = Field(default=0.0, description="Wall time including image loading.")
    cached: bool = Field(default=False, description="True if served from the result cache.")


class Detection(BaseModel):
    label: str
    confidence: float
    box: BBox


class DetectObjectsResult(ToolResultBase):
    detections: list[Detection]
    counts: dict[str, int] = Field(description="Number of detections per label.")
    total: int


class DescribeImageResult(ToolResultBase):
    detail: Literal["brief", "detailed", "very_detailed"]
    caption: str


class TextLine(BaseModel):
    text: str
    confidence: float
    box: BBox
    polygon: list[tuple[float, float]] = Field(description="Four corner points, clockwise.")


class ReadTextResult(ToolResultBase):
    language: str
    lines: list[TextLine]
    full_text: str = Field(description="All lines joined top to bottom with newlines.")


class LoadedModelInfo(BaseModel):
    name: str
    load_seconds: float
    estimated_mb: int


class ServerStatus(BaseModel):
    version: str
    model_dir: str
    memory_budget_mb: int
    memory_used_mb: int
    loaded_models: list[LoadedModelInfo]
    available_models: list[str]
    cache_entries: int
    cache_hits: int
    cache_misses: int


# ---- step 2 tools -----------------------------------------------------------------------------


class SegmentedObject(BaseModel):
    label: str
    confidence: float
    box: BBox
    polygon: list[tuple[float, float]] = Field(description="Simplified mask outline in image pixels.")
    area_fraction: float = Field(description="Mask area divided by image area (0-1).")


class SegmentObjectsResult(ToolResultBase):
    objects: list[SegmentedObject]
    counts: dict[str, int]
    coverage_by_label: dict[str, float] = Field(description="Fraction of the image covered by each label's masks.")


class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float


class PersonPose(BaseModel):
    confidence: float
    box: BBox
    keypoints: list[Keypoint]
    posture: str = Field(description="Rough heuristic: standing, sitting/crouching, arms_raised or unknown.")


class EstimatePoseResult(ToolResultBase):
    people: list[PersonPose]
    total: int
    skeleton: list[tuple[int, int]] = Field(description="Keypoint index pairs to draw limbs (COCO-17).")


class RegionDepth(BaseModel):
    box: BBox
    median_m: float
    min_m: float


class EstimateDepthResult(ToolResultBase):
    min_m: float
    median_m: float
    max_m: float
    regions: list[RegionDepth] = Field(description="Depth inside each requested box, same order as the input.")
    depth_png: str = Field(description="Small colorized depth map as a data URI (bright = near).")


class PhraseMatch(BaseModel):
    label: str
    box: BBox


class FindPhraseResult(ToolResultBase):
    phrase: str
    matches: list[PhraseMatch]
    total: int


class FaceExpression(BaseModel):
    box: BBox
    confidence: float
    expression: str = Field(description="Most likely apparent facial expression (not a claim about feelings).")
    scores: dict[str, float] = Field(description="Top-3 expression probabilities.")


class AnalyzeFacesResult(ToolResultBase):
    faces: list[FaceExpression]
    total: int
    expression_counts: dict[str, int]


class TimelineFrame(BaseModel):
    t: float = Field(description="Timestamp in seconds.")
    frame_index: int
    counts: dict[str, int]
    total: int
    boxes: list[Detection] = Field(default_factory=list, description="Detections in this frame (for overlays).")


class LabelStats(BaseModel):
    label: str
    max: int
    mean: float
    peak_t: float
    seconds_at_or_above_mean: float
    last: int = Field(description="Count in the final sampled frame.")
    trailing_seconds_at_or_above_last: float = Field(
        description="How long, up to the end of the video, the count has stayed at or above its final value."
    )


class ProcessVideoResult(BaseModel):
    video: str
    model: str
    task: Literal["detect_objects", "analyze_faces"]
    duration_s: float
    native_fps: float
    width: int
    height: int
    frames_sampled: int
    sample_every_s: float
    timeline: list[TimelineFrame]
    stats: list[LabelStats]
    latency_ms: float = 0.0
    cached: bool = False

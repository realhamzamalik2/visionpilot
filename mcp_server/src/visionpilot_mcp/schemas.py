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


class ServerStatus(BaseModel):
    version: str
    model_dir: str
    max_loaded_models: int
    loaded_models: list[LoadedModelInfo]
    available_models: list[str]
    cache_entries: int
    cache_hits: int
    cache_misses: int

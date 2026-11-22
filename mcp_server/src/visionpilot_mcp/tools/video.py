"""Video analysis: sample frames with OpenCV, run a per-frame tool, return a timeline."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visionpilot_mcp.errors import VisionToolError
from visionpilot_mcp.schemas import Detection, LabelStats, TimelineFrame

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_BYTES = 300 * 1024 * 1024


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    digest: str

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def open_video(source: str) -> VideoInfo:
    path = Path(source.strip().strip('"').strip("'")).expanduser()
    if not path.is_file():
        raise VisionToolError(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise VisionToolError(f"Unsupported video type '{path.suffix}'. Expected one of {sorted(VIDEO_SUFFIXES)}")
    stat = path.stat()
    if stat.st_size > MAX_VIDEO_BYTES:
        raise VisionToolError(f"Video is larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VisionToolError(f"OpenCV could not open the video: {path.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        info = VideoInfo(
            path=path,
            fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            digest=hashlib.sha1(
                f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode(), usedforsecurity=False
            ).hexdigest()[:16],
        )
    finally:
        cap.release()
    if info.frame_count <= 0:
        raise VisionToolError("The video has no readable frames.")
    return info


def plan_sample_indices(info: VideoInfo, sample_every_s: float, max_frames: int) -> list[int]:
    """Evenly spaced frame indices, never more than max_frames (keeps CPU time predictable)."""
    step = max(int(round(info.fps * sample_every_s)), 1)
    indices = list(range(0, info.frame_count, step))
    if len(indices) > max_frames:
        indices = [int(i) for i in np.linspace(0, info.frame_count - 1, max_frames)]
    return indices


def iter_frames(info: VideoInfo, indices: list[int]) -> Iterator[tuple[int, Image.Image]]:
    cap = cv2.VideoCapture(str(info.path))
    try:
        wanted = set(indices)
        last = max(indices)
        current = 0
        # Sequential grab() is far faster than seeking on most codecs; decode only wanted frames.
        while current <= last:
            if not cap.grab():
                break
            if current in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    yield current, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current += 1
    finally:
        cap.release()


FrameFn = Callable[[Image.Image], list[Detection]]


def build_timeline(info: VideoInfo, indices: list[int], frame_fn: FrameFn, *, keep_boxes: int) -> list[TimelineFrame]:
    timeline = []
    for index, frame in iter_frames(info, indices):
        dets = frame_fn(frame)
        counts: dict[str, int] = {}
        for d in dets:
            counts[d.label] = counts.get(d.label, 0) + 1
        timeline.append(
            TimelineFrame(
                t=round(index / info.fps, 2),
                frame_index=index,
                counts=counts,
                total=len(dets),
                boxes=dets[:keep_boxes],
            )
        )
    return timeline


def timeline_stats(timeline: list[TimelineFrame], sample_every_s: float) -> list[LabelStats]:
    labels = sorted({label for frame in timeline for label in frame.counts})
    stats = []
    for label in labels:
        series = np.array([frame.counts.get(label, 0) for frame in timeline], dtype=float)
        peak = int(series.argmax())
        mean = float(series.mean())
        last = series[-1]
        streak = 0
        for value in series[::-1]:
            if value < last:
                break
            streak += 1
        stats.append(
            LabelStats(
                label=label,
                max=int(series.max()),
                mean=round(mean, 2),
                peak_t=timeline[peak].t,
                seconds_at_or_above_mean=round(float((series >= mean).sum()) * sample_every_s, 1),
                last=int(last),
                trailing_seconds_at_or_above_last=round(max(streak - 1, 0) * sample_every_s, 1),
            )
        )
    stats.sort(key=lambda s: s.max, reverse=True)
    return stats

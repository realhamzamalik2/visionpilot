from __future__ import annotations

import numpy as np

from visionpilot_mcp.schemas import TimelineFrame
from visionpilot_mcp.tools.video import timeline_stats
from visionpilot_mcp.tools.yolo_tasks import classify_posture, simplify_polygon


def test_simplify_polygon_caps_points() -> None:
    angles = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    circle = np.stack([100 + 50 * np.cos(angles), 100 + 50 * np.sin(angles)], axis=1)
    simplified = simplify_polygon(circle, max_points=32)
    assert 8 <= len(simplified) <= 32


def test_posture_arms_raised() -> None:
    pts = np.zeros((17, 2))
    pts[:, 1] = np.linspace(10, 110, 17)
    pts[9] = [0, 0]  # left wrist above left shoulder
    assert classify_posture(pts, np.full(17, 0.9)) == "arms_raised"


def test_posture_unknown_when_low_confidence() -> None:
    assert classify_posture(np.zeros((17, 2)), np.zeros(17)) == "unknown"


def test_timeline_stats_queue_duration() -> None:
    counts = [2, 3, 5, 6, 6, 7, 6]
    timeline = [TimelineFrame(t=float(i), frame_index=i, counts={"person": c}, total=c) for i, c in enumerate(counts)]
    (stats,) = timeline_stats(timeline, sample_every_s=1.0)
    assert stats.max == 7 and stats.peak_t == 5.0 and stats.last == 6
    assert stats.trailing_seconds_at_or_above_last == 3.0  # >= 6 since t=3

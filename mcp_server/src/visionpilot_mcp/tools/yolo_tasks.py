"""YOLO26n segmentation, pose and metric depth (all ONNX Runtime on CPU)."""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from typing import Any

import cv2
import numpy as np
from PIL import Image

from visionpilot_mcp.schemas import BBox, Keypoint, PersonPose, RegionDepth, SegmentedObject
from visionpilot_mcp.tools.detection import _to_numpy, resolve_class_ids

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # fmt: skip
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]  # fmt: skip


def _predict(model: Any, image: Image.Image, **kwargs: Any) -> Any:
    return model.predict(source=image, device="cpu", verbose=False, **kwargs)[0]


# ---- segmentation ------------------------------------------------------------------------------


def simplify_polygon(points: np.ndarray, max_points: int = 64) -> list[tuple[float, float]]:
    """Shrink a mask outline to at most `max_points` vertices so tool output stays small."""
    if len(points) == 0:
        return []
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    epsilon = 0.002 * cv2.arcLength(contour, closed=True)
    approx: np.ndarray = contour
    while len(approx) > max_points:
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        epsilon *= 1.6
    return [(round(float(x), 1), round(float(y), 1)) for x, y in approx.reshape(-1, 2)]


def run_segmentation(
    model: Any,
    image: Image.Image,
    *,
    min_confidence: float,
    classes: list[str] | None,
    max_detections: int,
    imgsz: int = 640,
) -> tuple[list[SegmentedObject], dict[str, float]]:
    names: dict[int, str] = dict(model.names)
    result = _predict(
        model,
        image,
        imgsz=imgsz,
        conf=min_confidence,
        classes=resolve_class_ids(names, classes),
        max_det=max_detections,
    )
    if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
        return [], {}

    image_area = float(image.width * image.height)
    xyxy = _to_numpy(result.boxes.xyxy)
    confs = _to_numpy(result.boxes.conf)
    cls_ids = _to_numpy(result.boxes.cls).astype(int)
    objects: list[SegmentedObject] = []
    coverage: dict[str, float] = defaultdict(float)
    for box, conf, cls_id, outline in zip(xyxy, confs, cls_ids, result.masks.xy, strict=True):
        outline = np.asarray(outline)
        area = float(cv2.contourArea(outline.astype(np.float32))) if len(outline) >= 3 else 0.0
        label = names[int(cls_id)]
        fraction = round(area / image_area, 4)
        coverage[label] += fraction
        objects.append(
            SegmentedObject(
                label=label,
                confidence=round(float(conf), 3),
                box=BBox.from_xyxy(*map(float, box)),
                polygon=simplify_polygon(outline),
                area_fraction=fraction,
            )
        )
    objects.sort(key=lambda o: o.confidence, reverse=True)
    return objects, {k: round(min(v, 1.0), 4) for k, v in coverage.items()}


def count_objects(objects: list[SegmentedObject]) -> dict[str, int]:
    return dict(Counter(o.label for o in objects).most_common())


# ---- pose --------------------------------------------------------------------------------------


def classify_posture(points: np.ndarray, conf: np.ndarray, threshold: float = 0.4) -> str:
    """Very rough posture from keypoint geometry. Good enough for 'is anyone sitting / raising a hand'."""

    def ok(*idx: int) -> bool:
        return bool(all(conf[i] >= threshold for i in idx))

    if ok(9, 5) and points[9][1] < points[5][1] or ok(10, 6) and points[10][1] < points[6][1]:
        return "arms_raised"
    if ok(11, 13, 15) or ok(12, 14, 16):
        side = (11, 13, 15) if ok(11, 13, 15) else (12, 14, 16)
        hip, knee, ankle = (points[i] for i in side)
        thigh = abs(knee[1] - hip[1])
        shin = abs(ankle[1] - knee[1])
        if shin > 0 and thigh < 0.5 * shin:
            return "sitting/crouching"
        return "standing"
    return "unknown"


def run_pose(model: Any, image: Image.Image, *, min_confidence: float, imgsz: int = 640) -> list[PersonPose]:
    result = _predict(model, image, imgsz=imgsz, conf=min_confidence)
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return []
    xyxy = _to_numpy(result.boxes.xyxy)
    confs = _to_numpy(result.boxes.conf)
    kp_xy = _to_numpy(result.keypoints.xy)
    kp_conf = _to_numpy(result.keypoints.conf) if result.keypoints.conf is not None else np.ones(kp_xy.shape[:2])
    people = []
    for box, conf, pts, pconf in zip(xyxy, confs, kp_xy, kp_conf, strict=True):
        keypoints = [
            Keypoint(name=COCO_KEYPOINTS[i], x=round(float(x), 1), y=round(float(y), 1), confidence=round(float(c), 3))
            for i, ((x, y), c) in enumerate(zip(pts, pconf, strict=True))
        ]
        people.append(
            PersonPose(
                confidence=round(float(conf), 3),
                box=BBox.from_xyxy(*map(float, box)),
                keypoints=keypoints,
                posture=classify_posture(pts, pconf),
            )
        )
    people.sort(key=lambda p: p.confidence, reverse=True)
    return people


# ---- depth -------------------------------------------------------------------------------------


def run_depth(model: Any, image: Image.Image, *, imgsz: int = 640) -> np.ndarray:
    """Return a float32 (H, W) depth map in meters at the original image size."""
    result = _predict(model, image, imgsz=imgsz)
    if result.depth is None:
        raise ValueError("Model returned no depth map; is this a -depth model?")
    return _to_numpy(result.depth.data).astype(np.float32)


def summarize_regions(depth: np.ndarray, boxes: list[list[float]]) -> list[RegionDepth]:
    h, w = depth.shape
    regions = []
    for raw in boxes:
        x1, y1, x2, y2 = (float(v) for v in raw[:4])
        ix1, iy1 = max(int(x1), 0), max(int(y1), 0)
        ix2, iy2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        patch = depth[iy1:iy2, ix1:ix2]
        if patch.size == 0:
            continue
        regions.append(
            RegionDepth(
                box=BBox.from_xyxy(x1, y1, x2, y2),
                median_m=round(float(np.median(patch)), 2),
                min_m=round(float(np.min(patch)), 2),
            )
        )
    return regions


def depth_to_data_uri(depth: np.ndarray, max_side: int = 384) -> str:
    """Colorize inverse depth (near = bright) and encode a small PNG data URI for overlays."""
    inv = 1.0 / np.clip(depth, 0.05, None)
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    norm = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    scale = max_side / max(depth.shape)
    if scale < 1:
        colored = cv2.resize(colored, (int(depth.shape[1] * scale), int(depth.shape[0] * scale)))
    ok, buf = cv2.imencode(".png", colored)
    if not ok:
        raise ValueError("Could not encode depth PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

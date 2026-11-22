"""Object detection with Ultralytics YOLO26n, run through ONNX Runtime on CPU."""

from __future__ import annotations

import contextlib
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visionpilot_mcp.errors import VisionToolError
from visionpilot_mcp.schemas import BBox, Detection

logger = logging.getLogger(__name__)

YOLO_TASKS = {
    "yolo26n": "detect",
    "yolo26n-seg": "segment",
    "yolo26n-pose": "pose",
    "yolo26n-depth": "depth",
}


def export_yolo_onnx(name: str, model_dir: Path, imgsz: int = 640) -> Path:
    """Download the official PyTorch weights for `name` and export them to ONNX inside model_dir."""
    from ultralytics import YOLO

    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / f"{name}.onnx"
    if onnx_path.exists():
        return onnx_path
    logger.info("Exporting %s to ONNX (one-time, downloads ~5-15 MB)...", name)
    # Run inside model_dir so Ultralytics downloads the .pt there instead of the current folder.
    with contextlib.chdir(model_dir):
        exported = YOLO(f"{name}.pt").export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)
        exported_path = Path(str(exported)).resolve()
    if exported_path != onnx_path.resolve():
        exported_path.replace(onnx_path)
    return onnx_path


def load_yolo(name: str, model_dir: Path, imgsz: int = 640) -> Any:
    from ultralytics import YOLO

    if name not in YOLO_TASKS:
        raise VisionToolError(f"Unknown YOLO model '{name}'. Options: {sorted(YOLO_TASKS)}")
    onnx_path = model_dir / f"{name}.onnx"
    if not onnx_path.exists():
        onnx_path = export_yolo_onnx(name, model_dir, imgsz)
    return YOLO(str(onnx_path), task=YOLO_TASKS[name])


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def resolve_class_ids(names: dict[int, str], classes: list[str] | None) -> list[int] | None:
    """Map label names like ['person', 'Car'] to YOLO class ids. None means all classes."""
    if not classes:
        return None
    lookup = {label.lower(): idx for idx, label in names.items()}
    ids, unknown = [], []
    for label in classes:
        idx = lookup.get(label.strip().lower())
        (unknown.append(label) if idx is None else ids.append(idx))
    if unknown:
        raise VisionToolError(
            f"Unknown class name(s) {unknown}. This model knows the 80 COCO classes, "
            f"e.g. {sorted(lookup)[:12]}... For other objects use describe_image instead."
        )
    return ids


def run_detection(
    model: Any,
    image: Image.Image,
    *,
    min_confidence: float,
    classes: list[str] | None,
    max_detections: int,
    imgsz: int = 640,
) -> list[Detection]:
    names: dict[int, str] = dict(model.names)
    class_ids = resolve_class_ids(names, classes)
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=min_confidence,
        classes=class_ids,
        max_det=max_detections,
        device="cpu",
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy)
    confs = _to_numpy(boxes.conf)
    cls_ids = _to_numpy(boxes.cls).astype(int)
    detections = [
        Detection(label=names[int(c)], confidence=round(float(p), 3), box=BBox.from_xyxy(*map(float, b)))
        for b, p, c in zip(xyxy, confs, cls_ids, strict=True)
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def count_labels(detections: list[Detection]) -> dict[str, int]:
    return dict(Counter(d.label for d in detections).most_common())

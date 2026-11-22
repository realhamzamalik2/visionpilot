"""Face detection (MediaPipe BlazeFace) + apparent facial expression (HSEmotion EfficientNet, ONNX)."""

from __future__ import annotations

import logging
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from visionpilot_mcp.schemas import BBox, FaceExpression

logger = logging.getLogger(__name__)

BLAZEFACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
BLAZEFACE_FILE = "blaze_face_short_range.tflite"
EMOTION_MODEL = "enet_b0_8_best_vgaf"
FACE_MODEL_NAME = f"MediaPipe BlazeFace (tiled) + HSEmotion {EMOTION_MODEL} (ONNX)"


@dataclass
class FaceBundle:
    detector: Any
    recognizer: Any


def ensure_blazeface(model_dir: Path) -> Path:
    path = model_dir / BLAZEFACE_FILE
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s ...", BLAZEFACE_FILE)
        urllib.request.urlretrieve(BLAZEFACE_URL, path)  # noqa: S310 (fixed https URL)
    return path


def load_faces(model_dir: Path, min_confidence: float = 0.5) -> FaceBundle:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_blazeface(model_dir))),
        min_detection_confidence=min_confidence,
    )
    return FaceBundle(
        detector=FaceDetector.create_from_options(options),
        recognizer=HSEmotionRecognizer(model_name=EMOTION_MODEL),
    )


def _tiles(width: int, height: int, tile_above: int = 900) -> list[tuple[int, int, int, int]]:
    """Full frame plus a 2x2 grid with overlap. BlazeFace runs at 128 px, so tiles help with small faces."""
    tiles = [(0, 0, width, height)]
    if max(width, height) > tile_above:
        tw, th = int(width * 0.6), int(height * 0.6)
        for x in (0, width - tw):
            for y in (0, height - th):
                tiles.append((x, y, x + tw, y + th))
    return tiles


def detect_faces(detector: Any, rgb: np.ndarray) -> list[tuple[list[float], float]]:
    import mediapipe as mp

    h, w = rgb.shape[:2]
    boxes: list[list[float]] = []
    scores: list[float] = []
    for x1, y1, x2, y2 in _tiles(w, h):
        crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        for det in result.detections:
            bb = det.bounding_box
            boxes.append([bb.origin_x + x1, bb.origin_y + y1, bb.width, bb.height])
            scores.append(float(det.categories[0].score) if det.categories else 0.0)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.35)
    out = []
    for i in np.array(keep).flatten():
        bx, by, bw, bh = boxes[int(i)]
        out.append(([max(bx, 0), max(by, 0), min(bx + bw, w), min(by + bh, h)], scores[int(i)]))
    return out


def run_faces(bundle: FaceBundle, image: Image.Image, *, max_faces: int) -> list[FaceExpression]:
    rgb = np.asarray(image)
    faces = sorted(detect_faces(bundle.detector, rgb), key=lambda f: f[1], reverse=True)[:max_faces]
    if not faces:
        return []

    crops = []
    for (x1, y1, x2, y2), _ in faces:
        mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.1  # small margin helps the expression model
        cx1, cy1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
        cx2, cy2 = int(min(x2 + mx, rgb.shape[1])), int(min(y2 + my, rgb.shape[0]))
        crops.append(rgb[cy1:cy2, cx1:cx2])
    labels, probs = bundle.recognizer.predict_multi_emotions(crops, logits=False)
    classes = bundle.recognizer.idx_to_class

    results = []
    for (box, score), label, p in zip(faces, labels, probs, strict=True):
        top = np.argsort(p[: len(classes)])[::-1][:3]
        results.append(
            FaceExpression(
                box=BBox.from_xyxy(*box),
                confidence=round(score, 3),
                expression=label.lower(),
                scores={classes[int(i)].lower(): round(float(p[int(i)]), 3) for i in top},
            )
        )
    return results


def count_expressions(faces: list[FaceExpression]) -> dict[str, int]:
    return dict(Counter(f.expression for f in faces).most_common())

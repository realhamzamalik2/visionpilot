"""Shared fixtures. Unit tests use fake models so they run in seconds without downloading weights."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from visionpilot_mcp.cache import ResultCache
from visionpilot_mcp.config import Settings
from visionpilot_mcp.registry import ModelRegistry
from visionpilot_mcp.server import VisionPilotApp
from visionpilot_mcp.tools.faces import FaceBundle

BOXES = [[10, 10, 50, 120], [60, 12, 100, 118], [120, 40, 220, 100]]
CONFS = [0.91, 0.88, 0.75]
CLS = [0, 0, 2]


class FakeBoxes:
    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[int]) -> None:
        self.xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)

    def __len__(self) -> int:
        return len(self.conf)


class FakeYolo:
    """Mimics the parts of ultralytics.YOLO that the tools use (detect, seg, pose, depth)."""

    names = {0: "person", 1: "bicycle", 2: "car"}

    def __init__(self, task: str = "detect") -> None:
        self.task = task
        self.calls = 0

    def predict(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls += 1
        if self.task == "depth":
            w, h = kwargs["source"].size
            depth = np.tile(np.linspace(1.0, 10.0, w, dtype=np.float32), (h, 1))  # 1 m on the left, 10 m right
            return [SimpleNamespace(depth=SimpleNamespace(data=depth))]
        keep = [i for i, c in enumerate(CLS) if kwargs.get("classes") is None or c in kwargs["classes"]]
        boxes = FakeBoxes([BOXES[i] for i in keep], [CONFS[i] for i in keep], [CLS[i] for i in keep])
        result = SimpleNamespace(boxes=boxes, masks=None, keypoints=None)
        if self.task == "segment":
            result.masks = SimpleNamespace(
                xy=[np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], float) for b in boxes.xyxy]
            )
        if self.task == "pose":
            n = len(boxes)
            xy = np.zeros((n, 17, 2))
            xy[:, :, 1] = np.linspace(10, 110, 17)  # head at top, ankles at bottom: standing
            result.keypoints = SimpleNamespace(xy=xy, conf=np.full((n, 17), 0.9))
        return [result]


class FakeOcrEngine:
    """Returns two Korean lines out of reading order, like a real detector can."""

    def __call__(self, image: Any, text_score: float = 0.5) -> SimpleNamespace:
        return SimpleNamespace(
            boxes=np.array(
                [[[20, 80], [200, 80], [200, 110], [20, 110]], [[20, 20], [180, 20], [180, 50], [20, 50]]], dtype=float
            ),
            txts=("김치찌개 9,000원", "오늘의 메뉴"),
            scores=(0.97, 0.95),
        )


class FakeFaceDetector:
    def detect(self, image: Any) -> SimpleNamespace:
        bb = SimpleNamespace(origin_x=10, origin_y=10, width=40, height=40)
        return SimpleNamespace(detections=[SimpleNamespace(bounding_box=bb, categories=[SimpleNamespace(score=0.9)])])


class FakeEmotion:
    idx_to_class = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Neutral", 6: "Sadness",
                    7: "Surprise"}  # fmt: skip

    def predict_multi_emotions(self, crops: list[np.ndarray], logits: bool = False) -> tuple[list[str], np.ndarray]:
        probs = np.tile(np.array([0.01, 0.01, 0.01, 0.02, 0.8, 0.1, 0.03, 0.02]), (len(crops), 1))
        return ["Happiness"] * len(crops), probs


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.new("RGB", (320, 160), (120, 160, 200)).save(path)
    return path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(50):  # 5 seconds at 10 fps
        frame = np.full((120, 160, 3), i * 5 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def fake_yolo() -> FakeYolo:
    return FakeYolo()


@pytest.fixture
def app(tmp_path: Path, fake_yolo: FakeYolo) -> VisionPilotApp:
    settings = Settings(model_dir=tmp_path / "models", log_dir=tmp_path / "logs")
    registry = ModelRegistry(budget_mb=10_000)
    registry.register("yolo26n", lambda: fake_yolo)
    registry.register("yolo26n-seg", lambda: FakeYolo("segment"))
    registry.register("yolo26n-pose", lambda: FakeYolo("pose"))
    registry.register("yolo26n-depth", lambda: FakeYolo("depth"))
    registry.register("florence2", lambda: object())
    registry.register("korean_ocr", FakeOcrEngine)
    registry.register("faces", lambda: FaceBundle(detector=FakeFaceDetector(), recognizer=FakeEmotion()))
    return VisionPilotApp(settings=settings, registry=registry, cache=ResultCache(16))

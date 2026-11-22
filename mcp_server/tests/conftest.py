"""Shared fixtures. Unit tests use fake models so they run in seconds without downloading weights."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from visionpilot_mcp.cache import ResultCache
from visionpilot_mcp.config import Settings
from visionpilot_mcp.registry import ModelRegistry
from visionpilot_mcp.server import VisionPilotApp


class FakeBoxes:
    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[int]) -> None:
        self.xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)

    def __len__(self) -> int:
        return len(self.conf)


class FakeYolo:
    """Mimics the parts of ultralytics.YOLO that detection.run_detection uses."""

    names = {0: "person", 1: "bicycle", 2: "car"}

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls += 1
        boxes = [[10, 10, 50, 120], [60, 12, 100, 118], [120, 40, 220, 100]]
        conf, cls = [0.91, 0.88, 0.75], [0, 0, 2]
        keep = [i for i, c in enumerate(cls) if kwargs.get("classes") is None or c in kwargs["classes"]]
        return [
            SimpleNamespace(boxes=FakeBoxes([boxes[i] for i in keep], [conf[i] for i in keep], [cls[i] for i in keep]))
        ]


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


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.new("RGB", (320, 160), (120, 160, 200)).save(path)
    return path


@pytest.fixture
def fake_yolo() -> FakeYolo:
    return FakeYolo()


@pytest.fixture
def app(tmp_path: Path, fake_yolo: FakeYolo) -> VisionPilotApp:
    settings = Settings(model_dir=tmp_path / "models", log_dir=tmp_path / "logs")
    registry = ModelRegistry(max_loaded=2)
    registry.register("yolo26n", lambda: fake_yolo)
    registry.register("florence2", lambda: object())
    registry.register("korean_ocr", FakeOcrEngine)
    return VisionPilotApp(settings=settings, registry=registry, cache=ResultCache(16))

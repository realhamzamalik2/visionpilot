"""One-time model setup: export YOLO26 weights to ONNX and pre-download every other model.

uv run visionpilot-setup                 # everything (about 600 MB)
uv run visionpilot-setup --only yolo     # just the YOLO26n ONNX exports
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from visionpilot_mcp.config import get_settings
from visionpilot_mcp.tools import detection, faces, florence, korean_ocr

GROUPS = ("yolo", "florence", "ocr", "faces")


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "?"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visionpilot-setup", description=__doc__)
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model folder: {settings.model_dir}")

    def step(label: str, fn: object) -> None:
        start = time.perf_counter()
        result = fn()  # type: ignore[operator]
        extra = f" ({_size(result)})" if isinstance(result, Path) else ""
        print(f"[ok] {label}{extra} in {time.perf_counter() - start:.1f}s")

    if "yolo" in args.only:
        for name in detection.YOLO_TASKS:
            imgsz = settings.depth_imgsz if name.endswith("depth") else settings.yolo_imgsz
            step(name, lambda n=name, i=imgsz: detection.export_yolo_onnx(n, settings.model_dir, i))
    if "florence" in args.only:
        step(settings.florence_model_id, lambda: florence.load_florence(settings.florence_model_id, 2))
    if "ocr" in args.only:
        step("Korean OCR (PP-OCRv5)", lambda: korean_ocr.load_korean_ocr(2))
    if "faces" in args.only:
        step("BlazeFace + HSEmotion", lambda: faces.load_faces(settings.model_dir))

    print("All models ready. Next: uv run visionpilot-check")


if __name__ == "__main__":
    main()

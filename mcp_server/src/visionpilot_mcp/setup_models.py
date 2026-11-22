"""One-time model setup: export YOLO26 weights to ONNX and pre-download Florence-2 and the Korean OCR model.

uv run visionpilot-setup            # detection + Florence-2 + Korean OCR (step 1 tools)
uv run visionpilot-setup --all      # also exports YOLO26n seg / pose / depth for later tools
"""

from __future__ import annotations

import argparse
import time

from visionpilot_mcp.config import get_settings
from visionpilot_mcp.tools import detection, florence, korean_ocr


def _size_mb(path_str: str) -> str:
    from pathlib import Path

    path = Path(path_str)
    return f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "?"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visionpilot-setup", description=__doc__)
    parser.add_argument("--all", action="store_true", help="Also export seg, pose and depth YOLO26n models.")
    parser.add_argument("--skip-florence", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model folder: {settings.model_dir}")

    yolo_names = list(detection.YOLO_TASKS) if args.all else ["yolo26n"]
    for name in yolo_names:
        start = time.perf_counter()
        path = detection.export_yolo_onnx(name, settings.model_dir, settings.yolo_imgsz)
        print(f"[ok] {name}: {path.name} ({_size_mb(str(path))}) in {time.perf_counter() - start:.1f}s")

    if not args.skip_florence:
        start = time.perf_counter()
        florence.load_florence(settings.florence_model_id, settings.cpu_threads)
        print(f"[ok] {settings.florence_model_id} downloaded and loads in {time.perf_counter() - start:.1f}s")

    if not args.skip_ocr:
        start = time.perf_counter()
        korean_ocr.load_korean_ocr(settings.cpu_threads)
        print(f"[ok] Korean OCR models ready in {time.perf_counter() - start:.1f}s")

    print("All models ready. Next: uv run visionpilot-check")


if __name__ == "__main__":
    main()

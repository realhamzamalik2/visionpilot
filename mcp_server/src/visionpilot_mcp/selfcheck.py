"""End-to-end check: loads each model (timed), then calls every tool through a real MCP client session.

Load time and inference time are reported separately, and results are saved to logs/selfcheck_latency.json
(used for the README latency table).

    uv run visionpilot-check                     # built-in sample images
    uv run visionpilot-check --image D:\\photo.jpg
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import cv2
import numpy as np
from mcp import Client
from PIL import Image, ImageDraw, ImageFont

from visionpilot_mcp.config import get_settings
from visionpilot_mcp.logging_setup import configure_logging
from visionpilot_mcp.server import VisionPilotApp

KOREAN_FONTS = [
    Path("C:/Windows/Fonts/malgun.ttf"),  # Windows: Malgun Gothic
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
]
LATIN_FONTS = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]


def sample_photo(name: str = "bus.jpg") -> str:
    from ultralytics.utils import ASSETS  # ships with ultralytics: bus.jpg, zidane.jpg

    return str(ASSETS / name)


def _font(candidates: list[Path], size: int) -> ImageFont.FreeTypeFont | None:
    path = next((p for p in candidates if p.exists()), None)
    return ImageFont.truetype(str(path), size) if path else None


def make_korean_menu(out_dir: Path) -> str | None:
    font = _font(KOREAN_FONTS, 44)
    if font is None:
        return None
    image = Image.new("RGB", (720, 300), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(["오늘의 메뉴", "김치찌개 9,000원", "비빔밥 10,000원", "영업시간 11:00 - 21:00"]):
        draw.text((30, 20 + i * 68), line, font=font, fill="black")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "selfcheck_korean_menu.png"
    image.save(path)
    return str(path)


def make_english_sign(out_dir: Path) -> str | None:
    font = _font(LATIN_FONTS, 56)
    if font is None:
        return None
    image = Image.new("RGB", (720, 240), (20, 90, 40))
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), "EMERGENCY EXIT", font=font, fill="white")
    draw.text((40, 130), "Keep this door clear", font=font, fill="white")
    path = out_dir / "selfcheck_english_sign.png"
    image.save(path)
    return str(path)


def make_pan_video(photo: str, out_dir: Path, seconds: int = 6, fps: int = 5) -> str:
    """A short synthetic clip that pans across the photo, so people enter and leave the frame."""
    frame = cv2.imread(photo)
    if frame is None:
        raise SystemExit(f"Could not read {photo}")
    h, w = frame.shape[:2]
    crop_w = int(w * 0.6)
    path = out_dir / "selfcheck_pan.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (crop_w, h))
    for x in np.linspace(0, w - crop_w, seconds * fps).astype(int):
        writer.write(np.ascontiguousarray(frame[:, x : x + crop_w]))
    writer.release()
    return str(path)


def _summary(tool: str, data: dict[str, Any]) -> str:
    match tool:
        case "detect_objects" | "segment_objects":
            return f"counts={json.dumps(data.get('counts'))}"
        case "estimate_pose":
            return f"people={data.get('total')} postures={[p['posture'] for p in data.get('people', [])][:4]}"
        case "estimate_depth":
            return f"median={data.get('median_m')} m, range {data.get('min_m')}-{data.get('max_m')} m"
        case "describe_image":
            return str(data.get("caption", ""))[:70]
        case "find_phrase":
            return f"'{data.get('phrase')}' matches={data.get('total')}"
        case "read_text" | "read_korean_text":
            return " | ".join(line["text"] for line in data.get("lines", []))[:70]
        case "analyze_faces":
            return f"faces={data.get('total')} expressions={json.dumps(data.get('expression_counts'))}"
        case "process_video":
            stats = data.get("stats", [])
            top = stats[0] if stats else {}
            frames = data.get("frames_sampled")
            return f"frames={frames} {top.get('label')} max={top.get('max')} peak_t={top.get('peak_t')}s"
    return ""


async def _run(app: VisionPilotApp, calls: list[tuple[str, dict[str, Any]]]) -> tuple[int, list[dict[str, Any]]]:
    failures = 0
    rows: list[dict[str, Any]] = []
    async with Client(app.mcp) as client:
        tools = await client.list_tools()
        print(f"\nServer exposes {len(tools.tools)} tools.\n")
        for name, arguments in calls:
            for attempt in ("inference", "cached"):
                result = await client.call_tool(name, arguments)
                if result.is_error:
                    failures += 1
                    text = getattr(result.content[0], "text", "error") if result.content else "error"
                    rows.append({"tool": name, "call": attempt, "ms": None, "summary": f"ERROR: {text}"})
                    break
                data = result.structured_content or {}
                rows.append(
                    {"tool": name, "call": attempt, "ms": data.get("latency_ms"), "summary": _summary(name, data)}
                )
    return failures, rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visionpilot-check", description=__doc__)
    parser.add_argument("--image", help="Photo for the image tools (default: bus.jpg from ultralytics)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp949/cp1252
    out_dir = settings.log_dir

    photo = args.image or sample_photo()
    faces_photo = args.image or sample_photo("zidane.jpg")
    korean, english = make_korean_menu(out_dir), make_english_sign(out_dir)
    clip = make_pan_video(photo, out_dir)

    calls: list[tuple[str, dict[str, Any]]] = [
        ("detect_objects", {"image": photo}),
        ("segment_objects", {"image": photo}),
        ("estimate_pose", {"image": photo}),
        ("estimate_depth", {"image": photo}),
        ("describe_image", {"image": photo, "detail": "detailed"}),
        ("find_phrase", {"image": photo, "phrase": "the bus"}),
        ("analyze_faces", {"image": faces_photo}),
        ("process_video", {"video_path": clip, "task": "detect_objects", "classes": ["person"]}),
    ]
    if english:
        calls.append(("read_text", {"image": english}))
    if korean:
        calls.append(("read_korean_text", {"image": korean}))

    app = VisionPilotApp(settings)
    print(f"{'model':<16}{'load s':>8}")
    print("-" * 24)
    load_times: dict[str, float] = {}
    for name in ("yolo26n", "yolo26n-seg", "yolo26n-pose", "yolo26n-depth", "florence2", "faces", "korean_ocr"):
        start = time.perf_counter()
        app.registry.get(name)
        load_times[name] = round(time.perf_counter() - start, 2)
        print(f"{name:<16}{load_times[name]:>8.2f}")

    failures, rows = anyio.run(_run, app, calls)
    print(f"{'tool':<18}{'call':<11}{'ms':>8}  result")
    print("-" * 100)
    for row in rows:
        ms = "-" if row["ms"] is None else f"{row['ms']:.0f}"
        print(f"{row['tool']:<18}{row['call']:<11}{ms:>8}  {row['summary']}")

    report = {
        "machine": f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()}",
        "cpu_threads": settings.cpu_threads,
        "model_load_seconds": load_times,
        "tool_calls": rows,
    }
    (out_dir / "selfcheck_latency.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out_dir / 'selfcheck_latency.json'}")
    print("All tools OK." if failures == 0 else f"{failures} tool call(s) failed. See logs/mcp_server.log")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

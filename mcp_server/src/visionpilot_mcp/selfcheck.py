"""End-to-end check: calls every tool through a real MCP client session and prints a latency table.

uv run visionpilot-check                     # uses built-in sample images
uv run visionpilot-check --image D:\\photo.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anyio
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


def sample_photo() -> str:
    from ultralytics.utils import ASSETS  # ships with ultralytics: bus.jpg, zidane.jpg

    return str(ASSETS / "bus.jpg")


def make_korean_menu(out_dir: Path) -> str | None:
    font_path = next((p for p in KOREAN_FONTS if p.exists()), None)
    if font_path is None:
        return None
    image = Image.new("RGB", (720, 300), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 44)
    for i, line in enumerate(["오늘의 메뉴", "김치찌개 9,000원", "비빔밥 10,000원", "영업시간 11:00 - 21:00"]):
        draw.text((30, 20 + i * 68), line, font=font, fill="black")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "selfcheck_korean_menu.png"
    image.save(path)
    return str(path)


async def _run(calls: list[tuple[str, dict[str, Any]]]) -> int:
    app = VisionPilotApp()
    failures = 0
    rows: list[tuple[str, str, str, str]] = []
    async with Client(app.mcp) as client:
        tools = await client.list_tools()
        print(f"Server exposes {len(tools.tools)} tools: {[t.name for t in tools.tools]}\n")
        for name, arguments in calls:
            for attempt in ("cold", "cached"):
                result = await client.call_tool(name, arguments)
                if result.is_error:
                    failures += 1
                    message = getattr(result.content[0], "text", "unknown error") if result.content else "unknown error"
                    rows.append((name, attempt, "-", f"ERROR: {message}"))
                    break
                data = result.structured_content or {}
                rows.append((name, attempt, f"{data.get('latency_ms', 0):.0f}", _summary(name, data)))

    print(f"{'tool':<18}{'call':<8}{'ms':>8}  result")
    print("-" * 90)
    for tool, attempt, ms, summary in rows:
        print(f"{tool:<18}{attempt:<8}{ms:>8}  {summary}")
    return failures


def _summary(tool: str, data: dict[str, Any]) -> str:
    if tool == "detect_objects":
        return f"counts={json.dumps(data.get('counts'))}"
    if tool == "describe_image":
        return str(data.get("caption", ""))[:70]
    if tool == "read_korean_text":
        return " | ".join(line["text"] for line in data.get("lines", []))[:70]
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visionpilot-check", description=__doc__)
    parser.add_argument("--image", help="Photo for detect_objects and describe_image (default: bus.jpg)")
    parser.add_argument("--korean-image", help="Image with Korean text (default: generated menu)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp949/cp1252

    photo = args.image or sample_photo()
    korean = args.korean_image or make_korean_menu(settings.log_dir)
    calls: list[tuple[str, dict[str, Any]]] = [
        ("detect_objects", {"image": photo}),
        ("describe_image", {"image": photo, "detail": "detailed"}),
    ]
    if korean:
        calls.append(("read_korean_text", {"image": korean}))
    else:
        print("No Korean font found, skipping read_korean_text (pass --korean-image).")

    failures = anyio.run(_run, calls)
    print("\nAll tools OK." if failures == 0 else f"\n{failures} tool call(s) failed. See logs/mcp_server.log")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

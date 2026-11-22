"""Slow tests with the real weights. Run after `uv run visionpilot-setup` with: uv run pytest -m models"""

from __future__ import annotations

import pytest
from mcp import Client

from visionpilot_mcp.selfcheck import make_korean_menu, sample_photo
from visionpilot_mcp.server import VisionPilotApp

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def real_app() -> VisionPilotApp:
    return VisionPilotApp()


async def test_yolo_finds_people_on_bus_photo(real_app: VisionPilotApp) -> None:
    async with Client(real_app.mcp) as client:
        result = await client.call_tool("detect_objects", {"image": sample_photo()})
    counts = result.structured_content["counts"]
    assert counts.get("person", 0) >= 3
    assert counts.get("bus", 0) >= 1


async def test_florence_mentions_bus(real_app: VisionPilotApp) -> None:
    async with Client(real_app.mcp) as client:
        result = await client.call_tool("describe_image", {"image": sample_photo(), "detail": "brief"})
    assert "bus" in result.structured_content["caption"].lower()


async def test_korean_ocr_reads_menu(real_app: VisionPilotApp, tmp_path) -> None:
    menu = make_korean_menu(tmp_path)
    if menu is None:
        pytest.skip("No Korean font on this machine")
    async with Client(real_app.mcp) as client:
        result = await client.call_tool("read_korean_text", {"image": menu})
    text = result.structured_content["full_text"]
    assert "김치찌개" in text
    assert "메뉴" in text


async def test_pose_and_depth_on_bus_photo(real_app: VisionPilotApp) -> None:
    async with Client(real_app.mcp) as client:
        pose = (await client.call_tool("estimate_pose", {"image": sample_photo()})).structured_content
        depth = (await client.call_tool("estimate_depth", {"image": sample_photo()})).structured_content
    assert pose["total"] >= 3
    assert 0.1 < depth["median_m"] < 100


async def test_faces_on_zidane(real_app: VisionPilotApp) -> None:
    async with Client(real_app.mcp) as client:
        data = (await client.call_tool("analyze_faces", {"image": sample_photo("zidane.jpg")})).structured_content
    assert data["total"] >= 1

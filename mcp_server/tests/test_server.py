"""End-to-end tests through a real MCP client session (in-memory transport, fake models)."""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp import Client

from visionpilot_mcp.server import VisionPilotApp
from visionpilot_mcp.tools import florence

from .conftest import FakeYolo


async def test_lists_all_tools_with_output_schemas(app: VisionPilotApp) -> None:
    async with Client(app.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"detect_objects", "describe_image", "read_korean_text", "server_status"}
    assert all(tool.output_schema for tool in tools.values())
    assert "image" in tools["detect_objects"].input_schema["required"]


async def test_detect_objects_counts_and_caches(app: VisionPilotApp, sample_image: Path, fake_yolo: FakeYolo) -> None:
    async with Client(app.mcp) as client:
        first = await client.call_tool("detect_objects", {"image": str(sample_image)})
        second = await client.call_tool("detect_objects", {"image": str(sample_image)})

    assert not first.is_error
    data = first.structured_content
    assert data["counts"] == {"person": 2, "car": 1}
    assert data["total"] == 3
    assert data["detections"][0]["box"] == {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 120.0}
    assert data["cached"] is False
    assert second.structured_content["cached"] is True
    assert fake_yolo.calls == 1  # second call never reached the model


async def test_detect_objects_class_filter(app: VisionPilotApp, sample_image: Path) -> None:
    async with Client(app.mcp) as client:
        result = await client.call_tool("detect_objects", {"image": str(sample_image), "classes": ["Car"]})
    assert result.structured_content["counts"] == {"car": 1}


async def test_unknown_class_is_a_clear_error(app: VisionPilotApp, sample_image: Path) -> None:
    async with Client(app.mcp) as client:
        result = await client.call_tool("detect_objects", {"image": str(sample_image), "classes": ["dragon"]})
    assert result.is_error
    assert "Unknown class name" in result.content[0].text


async def test_missing_file_is_a_clear_error(app: VisionPilotApp) -> None:
    async with Client(app.mcp) as client:
        result = await client.call_tool("describe_image", {"image": "Z:/nope/missing.jpg"})
    assert result.is_error
    assert "File not found" in result.content[0].text


async def test_describe_image(app: VisionPilotApp, sample_image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(florence, "describe", lambda bundle, image, detail, **kw: f"A blue square ({detail}).")
    async with Client(app.mcp) as client:
        result = await client.call_tool("describe_image", {"image": str(sample_image), "detail": "brief"})
    assert result.structured_content["caption"] == "A blue square (brief)."


async def test_read_korean_text_in_reading_order(app: VisionPilotApp, sample_image: Path) -> None:
    async with Client(app.mcp) as client:
        result = await client.call_tool("read_korean_text", {"image": str(sample_image)})
    data = result.structured_content
    assert [line["text"] for line in data["lines"]] == ["오늘의 메뉴", "김치찌개 9,000원"]
    assert data["full_text"] == "오늘의 메뉴\n김치찌개 9,000원"
    assert len(data["lines"][0]["polygon"]) == 4


async def test_server_status_reports_loaded_models(app: VisionPilotApp, sample_image: Path) -> None:
    async with Client(app.mcp) as client:
        await client.call_tool("detect_objects", {"image": str(sample_image)})
        status = (await client.call_tool("server_status", {})).structured_content
    assert [m["name"] for m in status["loaded_models"]] == ["yolo26n"]
    assert status["cache_entries"] == 1

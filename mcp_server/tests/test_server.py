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
    assert set(tools) == {
        "detect_objects", "segment_objects", "estimate_pose", "estimate_depth", "describe_image", "find_phrase",
        "read_text", "read_korean_text", "analyze_faces", "process_video", "server_status",
    }  # fmt: skip
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


async def test_segment_objects_polygons_and_coverage(app: VisionPilotApp, sample_image: Path) -> None:
    async with Client(app.mcp) as client:
        data = (await client.call_tool("segment_objects", {"image": str(sample_image)})).structured_content
    assert data["counts"] == {"person": 2, "car": 1}
    assert len(data["objects"][0]["polygon"]) == 4
    # car box is 100 x 60 px on a 320 x 160 image
    assert data["coverage_by_label"]["car"] == pytest.approx(6000 / 51200, abs=1e-3)


async def test_estimate_pose_keypoints(app: VisionPilotApp, sample_image: Path) -> None:
    async with Client(app.mcp) as client:
        data = (await client.call_tool("estimate_pose", {"image": str(sample_image)})).structured_content
    assert data["total"] == 3
    assert len(data["people"][0]["keypoints"]) == 17
    assert data["people"][0]["keypoints"][0]["name"] == "nose"
    assert data["people"][0]["posture"] == "standing"


async def test_estimate_depth_regions(app: VisionPilotApp, sample_image: Path) -> None:
    boxes = [[0, 0, 20, 160], [300, 0, 320, 160]]
    async with Client(app.mcp) as client:
        data = (
            await client.call_tool("estimate_depth", {"image": str(sample_image), "boxes": boxes})
        ).structured_content
    near, far = data["regions"]
    assert near["median_m"] < 2.0 < 9.0 < far["median_m"]
    assert data["depth_png"].startswith("data:image/png;base64,")


async def test_find_phrase(app: VisionPilotApp, sample_image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(florence, "find_phrase", lambda b, img, phrase, **kw: [("red umbrella", [1, 2, 30, 40])])
    async with Client(app.mcp) as client:
        data = (
            await client.call_tool("find_phrase", {"image": str(sample_image), "phrase": "Red umbrella"})
        ).structured_content
    assert data["matches"] == [{"label": "red umbrella", "box": {"x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0}}]


async def test_read_text_quads(app: VisionPilotApp, sample_image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(florence, "read_text_regions", lambda b, img, **kw: [("EXIT", [5, 5, 50, 5, 50, 20, 5, 20])])
    async with Client(app.mcp) as client:
        data = (await client.call_tool("read_text", {"image": str(sample_image)})).structured_content
    assert data["full_text"] == "EXIT"
    assert data["lines"][0]["box"] == {"x1": 5.0, "y1": 5.0, "x2": 50.0, "y2": 20.0}


async def test_analyze_faces(app: VisionPilotApp, sample_image: Path) -> None:
    async with Client(app.mcp) as client:
        data = (await client.call_tool("analyze_faces", {"image": str(sample_image)})).structured_content
    assert data["total"] == 1  # same face found in the full frame is not duplicated
    assert data["faces"][0]["expression"] == "happiness"
    assert list(data["faces"][0]["scores"])[0] == "happiness"
    assert data["expression_counts"] == {"happiness": 1}


async def test_process_video_timeline(app: VisionPilotApp, sample_video: Path) -> None:
    args = {"video_path": str(sample_video), "sample_every_s": 1.0}
    async with Client(app.mcp) as client:
        result = await client.call_tool("process_video", args)
        again = await client.call_tool("process_video", args)
    assert not result.is_error, result.content
    data = result.structured_content
    assert data["frames_sampled"] == 5
    assert [f["t"] for f in data["timeline"]] == [0.0, 1.0, 2.0, 3.0, 4.0]
    person = next(s for s in data["stats"] if s["label"] == "person")
    assert person["max"] == 2 and person["last"] == 2
    assert person["trailing_seconds_at_or_above_last"] == 4.0
    assert again.structured_content["cached"] is True


async def test_process_video_missing_file(app: VisionPilotApp) -> None:
    async with Client(app.mcp) as client:
        result = await client.call_tool("process_video", {"video_path": "Z:/nope.mp4"})
    assert result.is_error
    assert "Video not found" in result.content[0].text

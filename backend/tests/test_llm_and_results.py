from __future__ import annotations

from visionpilot_api.llm import FINAL_ANSWER_TOOL, build_tools, clean_schema
from visionpilot_api.schemas import ToolSpec
from visionpilot_api.tool_results import compact_for_llm, summarize


def test_clean_schema_inlines_refs_and_drops_titles() -> None:
    schema = {
        "title": "X",
        "type": "object",
        "$defs": {"Box": {"title": "Box", "type": "object", "properties": {"x": {"type": "number", "title": "X"}}}},
        "properties": {"box": {"$ref": "#/$defs/Box"}},
    }
    cleaned = clean_schema(schema)
    assert "$defs" not in cleaned and "title" not in cleaned
    assert cleaned["properties"]["box"] == {"type": "object", "properties": {"x": {"type": "number"}}}


def test_build_tools_adds_final_answer() -> None:
    spec = ToolSpec(name="detect_objects", description="d", input_schema={"type": "object", "properties": {}})
    names = [d.name for d in build_tools([spec])[0].function_declarations]
    assert names == ["detect_objects", FINAL_ANSWER_TOOL]


def test_video_timeline_is_subsampled_and_stripped() -> None:
    timeline = [{"t": float(i), "frame_index": i, "counts": {"person": i % 5}, "total": i % 5,
                 "boxes": [{"label": "person"}] * 3} for i in range(120)]  # fmt: skip
    data = {"timeline": timeline, "stats": [{"label": "person", "max": 4}], "duration_s": 120}
    compact = compact_for_llm("process_video", data)
    assert len(compact["timeline"]) == 40
    assert compact["timeline"][0] == {"t": 0.0, "counts": {"person": 0}}
    assert compact["timeline"][-1]["t"] == 117.0  # spread across the whole video, not just the start
    assert compact["stats"] == data["stats"]


def test_geometry_is_dropped_and_long_lists_trimmed() -> None:
    objects = [{"label": "bottle", "polygon": [[0, 0]] * 50, "confidence": 0.9}] * 35
    compact = compact_for_llm("segment_objects", {"objects": objects, "depth_png": "data:..."})
    assert "depth_png" not in compact
    assert "polygon" not in compact["objects"][0]
    assert len(compact["objects"]) == 31 and "omitted" in compact["objects"][-1]["note"]


def test_summaries() -> None:
    assert (
        summarize("detect_objects", {"counts": {"person": 12, "bus": 2, "bag": 1}}) == "found 12 people, 2 buses, 1 bag"
    )
    video = {"frames_sampled": 60, "duration_s": 59.0,
             "stats": [{"label": "person", "max": 9, "peak_t": 41.0, "last": 7,
                        "trailing_seconds_at_or_above_last": 18.0}]}  # fmt: skip
    assert summarize("process_video", video) == "60 frames over 59.0s: person max 9 at 41.0s, now 7 for 18.0s"

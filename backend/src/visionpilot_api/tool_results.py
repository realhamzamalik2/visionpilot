"""Turn raw tool output into (a) one-line summaries for trace cards and (b) compact JSON for the LLM.

The frontend gets the full result (polygons, depth map, per-frame boxes) for overlays; the planner gets a
trimmed version so a video timeline or 50 mask polygons do not burn the free-tier token budget.
"""

from __future__ import annotations

from typing import Any

MAX_ITEMS_FOR_LLM = 30
MAX_TIMELINE_FOR_LLM = 40
DROP_KEYS = {"polygon", "depth_png", "skeleton", "boxes", "keypoints", "image", "cached"}


IRREGULAR_PLURALS = {"person": "people", "bus": "buses", "knife": "knives", "mouse": "mice", "sheep": "sheep"}


def _plural(label: str, n: int) -> str:
    if n == 1:
        return label
    return IRREGULAR_PLURALS.get(label, label if label.endswith("s") else f"{label}s")


def _counts(counts: dict[str, int] | None, limit: int = 4) -> str:
    if not counts:
        return "nothing"
    return ", ".join(f"{n} {_plural(label, n)}" for label, n in list(counts.items())[:limit])


def summarize(tool: str, data: dict[str, Any]) -> str:
    match tool:
        case "detect_objects":
            return f"found {_counts(data.get('counts'))}"
        case "segment_objects":
            cover = data.get("coverage_by_label") or {}
            top = max(cover.items(), key=lambda kv: kv[1], default=None)
            extra = f" ({top[0]} covers {top[1]:.0%})" if top else ""
            return f"segmented {_counts(data.get('counts'))}{extra}"
        case "estimate_pose":
            postures: dict[str, int] = {}
            for person in data.get("people", []):
                postures[person["posture"]] = postures.get(person["posture"], 0) + 1
            return f"{data.get('total', 0)} people, " + ", ".join(f"{v} {k}" for k, v in postures.items())
        case "estimate_depth":
            return f"depth {data.get('min_m')}-{data.get('max_m')} m (median {data.get('median_m')} m)"
        case "describe_image":
            caption = str(data.get("caption", ""))
            return caption if len(caption) <= 90 else caption[:87] + "..."
        case "find_phrase":
            return f"{data.get('total', 0)} match(es) for '{data.get('phrase')}'"
        case "read_text" | "read_korean_text":
            lines = data.get("lines", [])
            first = lines[0]["text"] if lines else ""
            return f"read {len(lines)} line(s)" + (f": '{first}'..." if first else "")
        case "analyze_faces":
            return f"{data.get('total', 0)} face(s): {_counts(data.get('expression_counts'))}"
        case "process_video":
            stats = data.get("stats") or []
            if not stats:
                return f"{data.get('frames_sampled', 0)} frames, nothing found"
            s = stats[0]
            return (
                f"{data.get('frames_sampled')} frames over {data.get('duration_s')}s: {s['label']} max {s['max']} "
                f"at {s['peak_t']}s, now {s['last']} for {s['trailing_seconds_at_or_above_last']}s"
            )
    return "done"


def compact_for_llm(tool: str, data: dict[str, Any]) -> dict[str, Any]:
    """Keep what the planner needs to reason (counts, boxes, text, stats) and drop bulky geometry."""
    data = dict(data)
    timeline = data.pop("timeline", None) if tool == "process_video" else None

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in DROP_KEYS}
        if isinstance(node, list):
            items = [strip(v) for v in node[:MAX_ITEMS_FOR_LLM]]
            if len(node) > MAX_ITEMS_FOR_LLM:
                items.append({"note": f"{len(node) - MAX_ITEMS_FOR_LLM} more item(s) omitted"})
            return items
        return node

    compact: dict[str, Any] = strip(data)
    if timeline:
        # Evenly subsample long timelines and keep only counts; per-frame boxes are for the UI.
        step = max(len(timeline) / MAX_TIMELINE_FOR_LLM, 1.0)
        picked = [timeline[int(i * step)] for i in range(min(len(timeline), MAX_TIMELINE_FOR_LLM))]
        compact["timeline"] = [{"t": f["t"], "counts": f["counts"]} for f in picked]
    return compact

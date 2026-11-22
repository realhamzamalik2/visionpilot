"""Fakes for the planner (Gemini) and the tool client (MCP) so the agent can be tested offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from google.genai import types
from PIL import Image

from visionpilot_api.config import Settings
from visionpilot_api.schemas import ToolCallResult, ToolSpec

DETECT_RESULT = {
    "image": {"width": 640, "height": 480, "sha1": "abc", "source_type": "file"},
    "model": "yolo26n",
    "detections": [
        {"label": "person", "confidence": 0.9, "box": {"x1": 1, "y1": 2, "x2": 30, "y2": 90}},
        {"label": "person", "confidence": 0.8, "box": {"x1": 40, "y1": 2, "x2": 70, "y2": 90}},
    ],
    "counts": {"person": 2},
    "total": 2,
    "latency_ms": 120.0,
    "cached": False,
}


def call(name: str, args: dict[str, Any], call_id: str | None = None) -> types.GenerateContentResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args, id=call_id))
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
    )


def text(value: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=value)]))]
    )


class ScriptedPlanner:
    """Returns pre-baked Gemini responses in order and records what it was sent."""

    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.model_in_use = "fake-gemini"

    async def generate(self, contents, tools, system_prompt, *, force_answer=False):  # type: ignore[no-untyped-def]
        self.calls.append({"contents": list(contents), "force_answer": force_answer, "tools": [t.name for t in tools]})
        if not self.responses:
            return call("final_answer", {"answer": "out of script"})
        return self.responses.pop(0)


class FakeTools:
    def __init__(self, results: dict[str, ToolCallResult] | None = None) -> None:
        self.results = results or {"detect_objects": ToolCallResult(ok=True, data=DETECT_RESULT)}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSpec]:
        schema = {"type": "object", "properties": {"image": {"type": "string"}}, "required": ["image"]}
        return [ToolSpec(name=n, description=f"{n} tool", input_schema=schema) for n in
                ("detect_objects", "describe_image", "process_video")]  # fmt: skip

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((name, arguments))
        return self.results.get(name, ToolCallResult(ok=False, error=f"{name} not available"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    samples = tmp_path / "samples" / "retail"
    samples.mkdir(parents=True)
    Image.new("RGB", (64, 48), "white").save(samples / "store_queue.jpg")
    return Settings(data_dir=tmp_path / "data", samples_dir=tmp_path / "samples", gemini_api_key="")


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (640, 480), "gray").save(path)
    return path

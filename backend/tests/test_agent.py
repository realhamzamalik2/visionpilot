from __future__ import annotations

from pathlib import Path

from visionpilot_api.agent import Agent
from visionpilot_api.schemas import ToolCallResult

from .conftest import FakeTools, ScriptedPlanner, call, text


async def collect(agent: Agent, path: Path, question: str = "How many people?", kind: str = "image") -> list:
    return [e async for e in agent.run(question, path, kind, width=640, height=480)]  # type: ignore[arg-type]


async def test_tool_then_final_answer(image_file: Path) -> None:
    planner = ScriptedPlanner([
        call("detect_objects", {"image": "whatever", "classes": ["person"]}, call_id="c1"),
        call("final_answer", {"answer": "There are **2 people**.",
                              "evidence": [{"step": 1, "item_indices": [0, 1], "label": "2 people"}]}),
    ])  # fmt: skip
    tools = FakeTools()
    events = await collect(Agent(planner, tools, max_steps=5), image_file)

    types_ = [e.type for e in events]
    assert types_ == ["run_started", "thinking", "tool_call", "tool_result", "thinking", "answer", "done"]
    tool_result = events[3]
    assert tool_result.ok and tool_result.summary == "found 2 people"
    assert tool_result.data["counts"] == {"person": 2}  # full data for the UI overlays
    answer = events[5].answer
    assert answer.answer.startswith("There are") and answer.evidence[0].item_indices == [0, 1]
    # the planner's made-up path was replaced with the real upload
    assert tools.calls == [("detect_objects", {"classes": ["person"], "image": str(image_file)})]
    assert events[2].args == {"classes": ["person"]}  # path hidden from the trace


async def test_function_response_is_compacted_for_llm(image_file: Path) -> None:
    planner = ScriptedPlanner([call("detect_objects", {}), call("final_answer", {"answer": "ok"})])
    await collect(Agent(planner, FakeTools(), max_steps=5), image_file)
    sent = planner.calls[1]["contents"][-1].parts[0].function_response.response
    assert sent["step"] == 1
    assert "image" not in sent["result"] and "cached" not in sent["result"]
    assert sent["result"]["counts"] == {"person": 2}


async def test_last_step_forces_final_answer(image_file: Path) -> None:
    planner = ScriptedPlanner([call("detect_objects", {})] * 2 + [call("final_answer", {"answer": "done"})])
    events = await collect(Agent(planner, FakeTools(), max_steps=3), image_file)
    assert [c["force_answer"] for c in planner.calls] == [False, False, True]
    assert events[-2].type == "answer"


async def test_tool_error_is_reported_and_run_continues(image_file: Path) -> None:
    tools = FakeTools({"detect_objects": ToolCallResult(ok=False, error="File not found")})
    planner = ScriptedPlanner([call("detect_objects", {}), call("final_answer", {"answer": "Sorry, tool failed."})])
    events = await collect(Agent(planner, tools, max_steps=4), image_file)
    failed = next(e for e in events if e.type == "tool_result")
    assert failed.ok is False and failed.summary == "File not found"
    assert planner.calls[1]["contents"][-1].parts[0].function_response.response["error"] == "File not found"
    assert events[-2].answer.answer == "Sorry, tool failed."


async def test_image_tool_on_video_is_rejected_without_calling_mcp(image_file: Path) -> None:
    tools = FakeTools()
    planner = ScriptedPlanner([call("detect_objects", {}), call("final_answer", {"answer": "x"})])
    events = await collect(Agent(planner, tools, max_steps=3), image_file, kind="video")
    assert tools.calls == []
    assert any(e.type == "tool_result" and e.ok is False and "process_video" in e.summary for e in events)


async def test_plain_text_reply_becomes_answer(image_file: Path) -> None:
    events = await collect(Agent(ScriptedPlanner([text("Hello!")]), FakeTools(), max_steps=3), image_file)
    assert events[-2].answer.answer == "Hello!"


async def test_invalid_final_answer_args_are_tolerated(image_file: Path) -> None:
    planner = ScriptedPlanner([call("final_answer", {"answer": "Two.", "evidence": "not-a-list"})])
    events = await collect(Agent(planner, FakeTools(), max_steps=3), image_file)
    assert events[-2].answer.answer == "Two." and events[-2].answer.evidence == []

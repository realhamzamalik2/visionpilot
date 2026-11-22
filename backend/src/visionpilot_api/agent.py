"""The agent loop: Gemini plans, VisionPilot MCP tools execute, every step is streamed as a TraceEvent."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import ValidationError

from visionpilot_api.llm import FINAL_ANSWER_TOOL, LLMError, Planner
from visionpilot_api.mcp_client import ToolClient
from visionpilot_api.schemas import ChatTurn, FinalAnswer, MediaKind, TraceEvent
from visionpilot_api.tool_results import compact_for_llm, summarize

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are VisionPilot, an agent that answers questions about one image or video by calling vision tools.

Rules:
1. Ground every number, location and piece of text in tool results. Never guess what is in the media, and never
   describe appearance (clothing, colors, age, gender) unless a tool result states it.
2. Plan the fewest calls that answer the question. Typical choices:
   counting/locating -> detect_objects; shape or area -> segment_objects; anything not in COCO -> find_phrase;
   posture -> estimate_pose; distance -> detect_objects then estimate_depth with its boxes;
   Korean text -> read_korean_text (then translate it yourself); English text -> read_text;
   faces/engagement -> analyze_faces; scene context -> describe_image; videos -> process_video.
3. Use exactly the media path given below as the `image` or `video_path` argument.
4. If a tool fails, say so honestly or try one sensible alternative.
5. Facial expressions are apparent expressions, not proof of how someone feels. Say "appears" or "looks".
6. Finish by calling final_answer once. Keep the answer short and direct, in Markdown, and add evidence entries:
   `step` is the step number of the tool call, `item_indices` index into that result's main list
   (detections, objects, people, faces, matches, lines, or timeline frames), `label` is the phrase it supports.
"""

MEDIA_ARG = {"image": "image", "video": "video_path"}


def _event(run_id: str, type_: str, **fields: Any) -> TraceEvent:
    return TraceEvent(type=type_, run_id=run_id, **fields)  # type: ignore[arg-type]


def build_user_message(question: str, media_path: Path, kind: MediaKind, width: int | None, height: int | None) -> str:
    dims = f", {width}x{height} px" if width and height else ""
    return f"Media ({kind}{dims}): {media_path}\nQuestion: {question}"


def enforce_media_arg(tool: str, args: dict[str, Any], media_path: Path, kind: MediaKind) -> dict[str, Any]:
    """The planner may only touch the user's media: overwrite any path it produced (prompt-injection guard)."""
    args = dict(args)
    args.pop("image", None)
    args.pop("video_path", None)
    if tool == "process_video":
        if kind != "video":
            raise ValueError("process_video needs a video, but the current media is an image.")
        args["video_path"] = str(media_path)
    else:
        if kind != "image":
            raise ValueError(f"{tool} works on images; use process_video for videos.")
        args["image"] = str(media_path)
    return args


class Agent:
    def __init__(self, planner: Planner, tools: ToolClient, max_steps: int) -> None:
        self.planner = planner
        self.tools = tools
        self.max_steps = max_steps

    async def run(
        self,
        question: str,
        media_path: Path,
        kind: MediaKind,
        *,
        width: int | None = None,
        height: int | None = None,
        history: list[ChatTurn] | None = None,
    ) -> AsyncIterator[TraceEvent]:
        run_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        yield _event(run_id, "run_started", message=question, model=self.planner.model_in_use)

        try:
            specs = await self.tools.list_tools()
        except Exception as exc:
            yield _event(run_id, "error", message=f"Could not reach the vision tools: {exc}")
            yield _event(run_id, "done", latency_ms=_ms(started))
            return

        contents: list[types.Content] = [
            types.Content(role="user" if turn.role == "user" else "model", parts=[types.Part(text=turn.text)])
            for turn in (history or [])[-6:]
        ]
        contents.append(
            types.Content(role="user", parts=[types.Part(text=build_user_message(question, media_path, kind, width,
                                                                                   height))])
        )  # fmt: skip

        answer: FinalAnswer | None = None
        for step in range(1, self.max_steps + 1):
            force = step == self.max_steps
            yield _event(run_id, "thinking", step=step, message="Planning next step" if not force else "Wrapping up")
            try:
                response = await self.planner.generate(contents, specs, SYSTEM_PROMPT, force_answer=force)
            except LLMError as exc:
                yield _event(run_id, "error", step=step, message=str(exc))
                break

            candidate = response.candidates[0].content if response.candidates else None
            calls = response.function_calls or []
            if candidate is None or not calls:
                text = (response.text or "").strip() if candidate else ""
                answer = FinalAnswer(answer=text or "I could not produce an answer.")
                break
            contents.append(candidate)  # keep the model turn verbatim (preserves Gemini thought signatures)

            response_parts: list[types.Part] = []
            for call in calls:
                name, args = call.name or "", dict(call.args or {})
                if name == FINAL_ANSWER_TOOL:
                    answer = _parse_final(args)
                    break
                call_id = call.id or uuid.uuid4().hex[:8]
                try:
                    safe_args = enforce_media_arg(name, args, media_path, kind)
                except ValueError as exc:
                    yield _event(run_id, "tool_result", step=step, call_id=call_id, tool=name, ok=False,
                                 summary=str(exc), latency_ms=0.0)  # fmt: skip
                    response_parts.append(types.Part.from_function_response(name=name, response={"error": str(exc)}))
                    continue

                shown_args = {k: v for k, v in safe_args.items() if k not in MEDIA_ARG.values()}
                yield _event(run_id, "tool_call", step=step, call_id=call_id, tool=name, args=shown_args)
                t0 = time.perf_counter()
                result = await self.tools.call_tool(name, safe_args)
                latency = _ms(t0)
                if result.ok and result.data is not None:
                    summary = summarize(name, result.data)
                    yield _event(run_id, "tool_result", step=step, call_id=call_id, tool=name, ok=True,
                                 summary=summary, latency_ms=latency, data=result.data)  # fmt: skip
                    payload = {"step": step, "result": compact_for_llm(name, result.data)}
                else:
                    yield _event(run_id, "tool_result", step=step, call_id=call_id, tool=name, ok=False,
                                 summary=result.error, latency_ms=latency)  # fmt: skip
                    payload = {"step": step, "error": result.error}
                response_parts.append(types.Part.from_function_response(name=name, response=payload))

            if answer is not None:
                break
            contents.append(types.Content(role="user", parts=response_parts))

        if answer is not None:
            yield _event(run_id, "answer", answer=answer, model=self.planner.model_in_use)
        yield _event(run_id, "done", latency_ms=_ms(started), model=self.planner.model_in_use)


def _parse_final(args: dict[str, Any]) -> FinalAnswer:
    try:
        return FinalAnswer.model_validate(args)
    except ValidationError:
        logger.warning("final_answer args failed validation: %s", json.dumps(args)[:300])
        return FinalAnswer(answer=str(args.get("answer", "")) or "I could not produce an answer.")


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)

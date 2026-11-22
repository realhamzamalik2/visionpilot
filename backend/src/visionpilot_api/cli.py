"""Ask the agent a question from the terminal and watch the trace (no frontend needed).

uv run visionpilot-ask D:\\photos\\menu.jpg "Read the Korean text and translate it"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from visionpilot_api.agent import Agent
from visionpilot_api.config import get_settings
from visionpilot_api.llm import GeminiPlanner, LLMError
from visionpilot_api.mcp_client import MCPToolClient
from visionpilot_api.media import MediaError, kind_for


def _quiet_connection_resets(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
    # Windows' Proactor loop logs a harmless ConnectionResetError when a server closes an idle HTTP connection.
    if isinstance(context.get("exception"), ConnectionResetError):
        return
    loop.default_exception_handler(context)


async def _ask(media: Path, question: str) -> int:
    asyncio.get_running_loop().set_exception_handler(_quiet_connection_resets)
    settings = get_settings()
    try:
        planner = GeminiPlanner(
            settings.gemini_api_key.get_secret_value(), settings.gemini_model, settings.gemini_fallback_model,
            max_retries=settings.gemini_max_retries, timeout_s=settings.gemini_timeout_s,
        )  # fmt: skip
    except LLMError as exc:
        print(f"[error] {exc}")
        return 1
    tools = MCPToolClient(settings.mcp_url, settings.mcp_call_timeout_s)
    tools.start()
    exit_code = 0
    try:
        async for event in Agent(planner, tools, settings.agent_max_steps).run(question, media, kind_for(media)):
            match event.type:
                case "run_started":
                    print(f"> {question}  (planner: {event.model})")
                case "thinking":
                    print(f"  step {event.step}: {event.message}...")
                case "tool_call":
                    print(f"    -> {event.tool}({json.dumps(event.args, ensure_ascii=False)})")
                case "tool_result":
                    mark = "ok " if event.ok else "ERR"
                    print(f"    <- [{mark}] {event.tool}: {event.summary}  ({event.latency_ms:.0f} ms)")
                case "answer" if event.answer:
                    print("\n" + event.answer.answer)
                    for ev in event.answer.evidence:
                        print(f"   evidence: step {ev.step} items {ev.item_indices} -> {ev.label}")
                case "error":
                    exit_code = 1
                    print(f"[error] {event.message}")
                case "done":
                    print(f"\n(done in {(event.latency_ms or 0) / 1000:.1f}s)")
    finally:
        await tools.close()
    return exit_code


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visionpilot-ask", description=__doc__)
    parser.add_argument("media", type=Path, help="Image or video file")
    parser.add_argument("question")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    media = args.media.expanduser().resolve()
    try:
        kind_for(media)
    except MediaError as exc:
        raise SystemExit(str(exc)) from exc
    if not media.is_file():
        raise SystemExit(f"File not found: {media}")
    raise SystemExit(asyncio.run(_ask(media, args.question)))


if __name__ == "__main__":
    main()

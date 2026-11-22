"""Run the eval set: real Gemini planner + real MCP tools, scored for tool selection and answer correctness.

uv run visionpilot-eval                # evals/questions.jsonl -> evals/results/latest.{json,md}
uv run visionpilot-eval --limit 5 --delay 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from visionpilot_api.agent import Agent
from visionpilot_api.config import REPO_ROOT, get_settings
from visionpilot_api.llm import GeminiPlanner
from visionpilot_api.mcp_client import MCPToolClient
from visionpilot_api.media import MediaStore

EVALS_DIR = REPO_ROOT / "evals"


@dataclass
class EvalCase:
    id: str
    media: str
    question: str
    expect: dict[str, Any]


@dataclass
class EvalResult:
    id: str
    question: str
    tools_called: list[str]
    answer: str
    tool_ok: bool
    answer_ok: bool
    latency_s: float
    error: str | None = None
    failed_checks: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.tool_ok and self.answer_ok


def load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as fh:
        return [EvalCase(**json.loads(line)) for line in fh if line.strip()]


def check_tools(expect: dict[str, Any], called: list[str]) -> list[str]:
    failures = []
    missing = [t for t in expect.get("tools_all", []) if t not in called]
    if missing:
        failures.append(f"missing tools {missing}")
    options = expect.get("tools_any")
    if options and not any(t in called for t in options):
        failures.append(f"none of {options} called")
    return failures


def check_answer(expect: dict[str, Any], answer: str) -> list[str]:
    text = answer.lower()
    failures = []
    for needle in expect.get("answer_contains_all", []):
        if needle.lower() not in text:
            failures.append(f"missing '{needle}'")
    options = expect.get("answer_contains_any")
    if options and not any(o.lower() in text for o in options):
        failures.append(f"none of {options}")
    pattern = expect.get("answer_regex")
    if pattern and not re.search(pattern, answer, flags=re.IGNORECASE):
        failures.append(f"no match for /{pattern}/")
    number = expect.get("answer_number")
    if number:
        values = [int(n) for n in re.findall(r"(?<![\d.,])\d+(?![\d.,]\d)", answer.replace(",", ""))]
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
        values += [v for w, v in words.items() if re.search(rf"\b{w}\b", text)]
        if not any(number["min"] <= v <= number["max"] for v in values):
            failures.append(f"no number in [{number['min']}, {number['max']}]")
    return failures


async def run_case(agent: Agent, media: MediaStore, case: EvalCase) -> EvalResult:
    path, kind = media.resolve(case.media)
    start = time.perf_counter()
    called: list[str] = []
    answer, error = "", None
    async for event in agent.run(case.question, path, kind):
        if event.type == "tool_call" and event.tool:
            called.append(event.tool)
        elif event.type == "answer" and event.answer:
            answer = event.answer.answer
        elif event.type == "error":
            error = event.message
    tool_failures = check_tools(case.expect, called)
    answer_failures = check_answer(case.expect, answer) if answer else ["no answer"]
    return EvalResult(
        id=case.id,
        question=case.question,
        tools_called=called,
        answer=answer,
        tool_ok=not tool_failures,
        answer_ok=not answer_failures,
        latency_s=round(time.perf_counter() - start, 1),
        error=error,
        failed_checks=tool_failures + answer_failures,
    )


def summarize(results: list[EvalResult], model: str) -> dict[str, Any]:
    n = len(results) or 1
    latencies = sorted(r.latency_s for r in results)
    return {
        "date": datetime.now().isoformat(timespec="minutes"),
        "planner": model,
        "questions": len(results),
        "tool_selection_accuracy": round(sum(r.tool_ok for r in results) / n, 3),
        "answer_accuracy": round(sum(r.answer_ok for r in results) / n, 3),
        "task_success": round(sum(r.success for r in results) / n, 3),
        "mean_tool_calls": round(statistics.mean(len(r.tools_called) for r in results), 2) if results else 0,
        "median_latency_s": statistics.median(latencies) if latencies else 0,
        "p90_latency_s": latencies[int(0.9 * (len(latencies) - 1))] if latencies else 0,
    }


def to_markdown(summary: dict[str, Any], results: list[EvalResult]) -> str:
    lines = [
        f"Planner `{summary['planner']}`, {summary['questions']} questions, run {summary['date']}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tool selection accuracy | {summary['tool_selection_accuracy']:.0%} |",
        f"| Answer accuracy | {summary['answer_accuracy']:.0%} |",
        f"| Task success (both) | {summary['task_success']:.0%} |",
        f"| Mean tool calls per question | {summary['mean_tool_calls']} |",
        f"| Median / p90 latency | {summary['median_latency_s']} s / {summary['p90_latency_s']} s |",
        "",
        "| Case | Tools called | Pass | Notes |",
        "|---|---|---|---|",
    ]
    for r in results:
        mark = "yes" if r.success else "no"
        notes = "; ".join(r.failed_checks) or ""
        lines.append(f"| {r.id} | {', '.join(r.tools_called) or 'none'} | {mark} | {notes} |")
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    cases = load_cases(Path(args.questions))
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    cases = cases[: args.limit] if args.limit else cases

    planner = GeminiPlanner(
        settings.gemini_api_key.get_secret_value(), settings.gemini_model, settings.gemini_fallback_model,
        max_retries=settings.gemini_max_retries, timeout_s=settings.gemini_timeout_s,
    )  # fmt: skip
    tools = MCPToolClient(settings.mcp_url, settings.mcp_call_timeout_s)
    tools.start()
    media = MediaStore(settings.upload_dir, settings.samples_dir, settings.max_upload_mb)
    agent = Agent(planner, tools, settings.agent_max_steps)

    results: list[EvalResult] = []
    try:
        for i, case in enumerate(cases, 1):
            result = await run_case(agent, media, case)
            results.append(result)
            status = "PASS" if result.success else "FAIL"
            print(
                f"[{i:>2}/{len(cases)}] {status} {case.id:<16} {result.latency_s:>6.1f}s  tools={result.tools_called}"
            )
            if not result.success:
                print(
                    f"        {'; '.join(result.failed_checks)}" + (f" | error: {result.error}" if result.error else "")
                )
            if i < len(cases):
                await asyncio.sleep(args.delay)  # stay under the free tier's requests-per-minute limit
    finally:
        await tools.close()

    summary = summarize(results, planner.model_in_use)
    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": [asdict(r) | {"success": r.success} for r in results]}
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "latest.md").write_text(to_markdown(summary, results), encoding="utf-8")
    print("\n" + to_markdown(summary, results).split("\n\n")[1])
    print(f"Saved {out_dir / 'latest.md'}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visionpilot-eval", description=__doc__)
    parser.add_argument("--questions", default=str(EVALS_DIR / "questions.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="Run only these case ids")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between questions")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
